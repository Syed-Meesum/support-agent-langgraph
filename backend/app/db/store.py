"""Case persistence and the human-review queue.

One row per case holds the whole `CaseState` as JSON - that is
`cases.state_snapshot` here - and `case_messages` / `audit_log` are the tables a
detail view reads. Both are implemented: the snapshot is the source of truth, the
two child tables are written from it on every save so the queue and the audit view
are plain SQL.

The DDL is portable (TEXT columns, `on conflict ... do update`, `excluded`) so one
implementation serves both SQLite locally and the documented Postgres/Supabase
schema. Payloads are stored as JSON text rather than `jsonb` for that reason; see
`app/db/schema.sql` for the doc's native Postgres schema.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterator, Mapping, Sequence

from pydantic import BaseModel

from app.db.settings import (
    BACKEND_POSTGRES,
    BACKEND_SQLITE,
    backend,
    case_db_path,
    database_url,
)

_DDL = (
    """
    create table if not exists cases (
        id text primary key,
        customer_id text,
        category text,
        status text not null,
        customer_message text,
        escalation_reason text,
        case_brief text,
        state_snapshot text not null,
        created_at text not null,
        updated_at text not null
    )
    """,
    "create index if not exists cases_status_idx on cases (status)",
    """
    create table if not exists case_messages (
        case_id text not null,
        seq integer not null,
        role text not null,
        content text not null,
        created_at text not null,
        primary key (case_id, seq)
    )
    """,
    """
    create table if not exists audit_log (
        case_id text not null,
        seq integer not null,
        node_name text not null,
        event_type text not null,
        payload text not null,
        created_at text not null,
        primary key (case_id, seq)
    )
    """,
)

_CASE_UPSERT = """
    insert into cases (
        id, customer_id, category, status, customer_message,
        escalation_reason, case_brief, state_snapshot, created_at, updated_at
    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    on conflict (id) do update set
        customer_id = excluded.customer_id,
        category = excluded.category,
        status = excluded.status,
        customer_message = excluded.customer_message,
        escalation_reason = excluded.escalation_reason,
        case_brief = excluded.case_brief,
        state_snapshot = excluded.state_snapshot,
        updated_at = excluded.updated_at
"""

_QUEUE_COLUMNS = (
    "id, customer_id, category, status, escalation_reason, case_brief, updated_at"
)


class StoreError(RuntimeError):
    """Raised when the configured case store cannot be opened."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    """Serialize the few non-JSON types a CaseState contains.

    Public because the API serves the same shapes it persists - one definition of
    "what a case looks like on the wire", rather than a second serializer in
    `app/api` that could drift from this one.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"cannot persist {type(value).__name__} to JSON")


def _dumps(value: Any) -> str:
    return json.dumps(value, default=json_default, ensure_ascii=False)


def serialize(state: Mapping[str, Any]) -> dict[str, Any]:
    """A `CaseState` as JSON-safe primitives. Used by `CaseStore.save` and the API."""
    return json.loads(_dumps(dict(state)))


class CaseStore:
    """One row per case, plus derived message and audit rows.

    Construct through `open_case_store()`; the connection lifecycle belongs to the
    caller, not to the store.
    """

    def __init__(self, conn: Any, *, dialect: str) -> None:
        self._conn = conn
        self._dialect = dialect
        self._postgres = dialect == BACKEND_POSTGRES

    # -- plumbing ---------------------------------------------------------

    def _sql(self, statement: str) -> str:
        # Positional placeholders so one set of queries serves both drivers.
        return statement.replace("?", "%s") if self._postgres else statement

    def _query(self, statement: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(self._sql(statement), tuple(params))
        return [dict(row) for row in cur.fetchall()]

    def _write(self, statement: str, params: Sequence[Any] = ()) -> None:
        cur = self._conn.cursor()
        cur.execute(self._sql(statement), tuple(params))

    def setup(self) -> None:
        """Create the tables if they are missing. Safe to call repeatedly."""
        cur = self._conn.cursor()
        for statement in _DDL:
            cur.execute(statement)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- writes -----------------------------------------------------------

    def save(self, state: Mapping[str, Any]) -> None:
        """Upsert one case snapshot and rewrite its derived rows."""
        case_id = str(state["case_id"])
        now = _utcnow()
        brief = state.get("case_brief")
        self._write(
            _CASE_UPSERT,
            (
                case_id,
                state.get("customer_id"),
                state.get("category"),
                state.get("status") or "in_progress",
                state.get("customer_message"),
                state.get("escalation_reason"),
                _dumps(brief) if brief else None,
                _dumps(dict(state)),
                now,
                now,
            ),
        )

        messages = state.get("conversation_history") or []
        self._write("delete from case_messages where case_id = ?", (case_id,))
        if messages:
            cur = self._conn.cursor()
            cur.executemany(
                self._sql(
                    "insert into case_messages (case_id, seq, role, content, created_at) "
                    "values (?, ?, ?, ?, ?)"
                ),
                [
                    (
                        case_id,
                        index,
                        _message_field(message, "role"),
                        _message_field(message, "content"),
                        now,
                    )
                    for index, message in enumerate(messages)
                ],
            )

        audit = state.get("audit_log") or []
        self._write("delete from audit_log where case_id = ?", (case_id,))
        if audit:
            cur = self._conn.cursor()
            cur.executemany(
                self._sql(
                    "insert into audit_log (case_id, seq, node_name, event_type, payload, created_at) "
                    "values (?, ?, ?, ?, ?, ?)"
                ),
                [
                    (
                        case_id,
                        index,
                        str(entry.get("node", "")),
                        str(entry.get("event_type", "")),
                        _dumps(entry.get("payload") or {}),
                        now,
                    )
                    for index, entry in enumerate(audit)
                ],
            )

        self._conn.commit()

    # -- reads ------------------------------------------------------------

    def get_row(self, case_id: str) -> dict[str, Any] | None:
        """Queue-shaped row for one case: the brief, not the whole snapshot."""
        rows = self._query(f"select {_QUEUE_COLUMNS} from cases where id = ?", (case_id,))
        return _brief_row(rows[0]) if rows else None

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        """The persisted `CaseState` for one case, or None."""
        rows = self._query("select state_snapshot from cases where id = ?", (case_id,))
        return json.loads(rows[0]["state_snapshot"]) if rows else None

    def list_by_status(self, status: str) -> list[dict[str, Any]]:
        rows = self._query(
            f"select {_QUEUE_COLUMNS} from cases where status = ? order by updated_at",
            (status,),
        )
        return [_brief_row(row) for row in rows]

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """The most recently touched cases, newest first. Drives the UI's list."""
        rows = self._query(
            f"select {_QUEUE_COLUMNS} from cases order by updated_at desc limit ?",
            (int(limit),),
        )
        return [_brief_row(row) for row in rows]

    def count_by_status(self) -> dict[str, int]:
        """Case counts per status, for the UI header and the docs' summary line."""
        rows = self._query("select status, count(*) as n from cases group by status")
        return {str(row["status"]): int(row["n"]) for row in rows}

    def messages(self, case_id: str) -> list[dict[str, Any]]:
        return self._query(
            "select seq, role, content, created_at from case_messages "
            "where case_id = ? order by seq",
            (case_id,),
        )

    def audit_trail(self, case_id: str) -> list[dict[str, Any]]:
        rows = self._query(
            "select seq, node_name, event_type, payload, created_at from audit_log "
            "where case_id = ? order by seq",
            (case_id,),
        )
        for row in rows:
            row["payload"] = json.loads(row["payload"])
        return rows

    def count(self) -> int:
        return int(self._query("select count(*) as n from cases")[0]["n"])


def _message_field(message: Any, field: str) -> str:
    """Read a field from either a `Message` model or a plain dict."""
    if isinstance(message, Mapping):
        return str(message.get(field, ""))
    return str(getattr(message, field, ""))


def _brief_row(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("case_brief")
    return {
        "case_id": row["id"],
        "customer_id": row.get("customer_id"),
        "category": row.get("category"),
        "status": row.get("status"),
        "escalation_reason": row.get("escalation_reason"),
        "case_brief": json.loads(raw) if raw else None,
        "updated_at": row.get("updated_at"),
    }


@contextmanager
def open_connection() -> Iterator[tuple[Any, str]]:
    """Yield `(connection, dialect)` for the configured backend.

    This is the only place a connection is opened, so the case store and the seed
    script cannot disagree about which database they are talking to, or how.
    """
    if backend() == BACKEND_POSTGRES:
        with _postgres_connection() as pair:
            yield pair
    else:
        with _sqlite_connection() as pair:
            yield pair


@contextmanager
def open_case_store() -> Iterator[CaseStore]:
    """Yield a `CaseStore` for the configured backend, closing it on exit.

    Use as a context manager - one connection per unit of work::

        with open_case_store() as store:
            store.save(state)
    """
    with open_connection() as (conn, dialect):
        store = CaseStore(conn, dialect=dialect)
        store.setup()
        yield store


@contextmanager
def _sqlite_connection() -> Iterator[tuple[Any, str]]:
    path = case_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False so a web server's thread pool can share the store.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn, BACKEND_SQLITE
    finally:
        conn.close()


@contextmanager
def _postgres_connection() -> Iterator[tuple[Any, str]]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise StoreError(
            "DATABASE_URL is set but psycopg is not installed. "
            'Run: pip install "psycopg[binary]>=3.2"'
        ) from exc

    url = database_url()
    if url is None:  # pragma: no cover - guarded by backend()
        raise StoreError("DATABASE_URL is not set")

    with psycopg.connect(url, row_factory=dict_row) as conn:
        yield conn, BACKEND_POSTGRES
