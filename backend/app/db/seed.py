"""Seed the fixture tables from `eval/cases/fixtures.json`.

The project's data model defines `customers`, `charges`, `subscriptions` and
`knowledge_chunks`, and this script is meant to run automatically on container start
when the database is empty (checked with a marker row, not a flag).

**What reads these rows, honestly.** The tools in `app/tools/` are mocked and read
`fixtures.json` directly - that is what "mocked" means in this project, and it is
why the test suite runs with no database at all. These tables are the documented
schema's fixture half, seeded so the Postgres/Supabase path is real rather than
vaguely described: `app/tools/fixtures.py` names this script as where the same
shapes go, so swapping `FixtureStore` for these tables is one file, not a redesign.
`knowledge_chunks` is populated for the same reason - retrieval currently runs on
the local Chroma index (`app/rag/retrieval.py`), and the pgvector column is
left null until that swap, rather than filling it with an embedding that does not
match its declared dimension (the schema declares `vector(1536)`; the local ONNX
model is 384 - assumption #6 in README.md).

    python -m app.db.seed            # seed if empty
    python -m app.db.seed --force    # clear and re-seed
    python -m app.db.seed --status   # what is in there now
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from app.db.settings import BACKEND_POSTGRES, backend
from app.db.store import open_connection
from app.tools.fixtures import DEFAULT_FIXTURES_PATH

TABLES = ("customers", "charges", "subscriptions", "knowledge_chunks")

# Portable DDL: TEXT/NUMERIC only, so one set of statements creates the schema on
# both SQLite and Postgres. `app/db/schema.sql` keeps the doc's native Postgres
# version, including the `uuid` types and the `vector` column, for Supabase users.
_DDL = (
    """
    create table if not exists customers (
        id text primary key,
        name text not null,
        email text not null,
        tier text not null default 'standard',
        account_status text not null default 'active'
    )
    """,
    """
    create table if not exists charges (
        id text primary key,
        customer_id text not null,
        order_id text,
        amount numeric(10, 2) not null,
        status text not null,
        created_at text not null
    )
    """,
    """
    create table if not exists subscriptions (
        id text primary key,
        customer_id text not null,
        plan text not null,
        status text not null,
        price numeric(10, 2) not null,
        renewal_date text
    )
    """,
    """
    create table if not exists knowledge_chunks (
        id text primary key,
        source text not null,
        content text not null,
        embedding text
    )
    """,
)

INSERT_CUSTOMER = (
    "insert into customers (id, name, email, tier, account_status) values (?, ?, ?, ?, ?)"
)
INSERT_CHARGE = (
    "insert into charges (id, customer_id, order_id, amount, status, created_at) "
    "values (?, ?, ?, ?, ?, ?)"
)
INSERT_SUBSCRIPTION = (
    "insert into subscriptions (id, customer_id, plan, status, price, renewal_date) "
    "values (?, ?, ?, ?, ?, ?)"
)
INSERT_CHUNK = (
    "insert into knowledge_chunks (id, source, content) values (?, ?, ?)"
)


@dataclass
class SeedReport:
    """What the seeder did, so the caller can print one honest line."""

    skipped: bool
    inserted: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.inserted.values())

    def describe(self) -> str:
        if self.skipped:
            return "fixture tables already seeded (marker row present); nothing to do"
        counts = ", ".join(f"{name}={count}" for name, count in self.inserted.items())
        return f"seeded {self.total} row(s): {counts}"


class _Connection:
    """A `?`-placeholder shim, the same trick `CaseStore` uses for both drivers."""

    def __init__(self, conn: Any, dialect: str) -> None:
        self._conn = conn
        self._postgres = dialect == BACKEND_POSTGRES

    def _sql(self, statement: str) -> str:
        return statement.replace("?", "%s") if self._postgres else statement

    def execute(self, statement: str, params: Sequence[Any] = ()) -> None:
        self._conn.cursor().execute(self._sql(statement), tuple(params))

    def executemany(self, statement: str, rows: Sequence[Sequence[Any]]) -> None:
        cur = self._conn.cursor()
        cur.executemany(self._sql(statement), [tuple(row) for row in rows])

    def query(self, statement: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Rows as plain dicts, for callers that need to read back what was seeded."""
        cur = self._conn.cursor()
        cur.execute(self._sql(statement), tuple(params))
        return [dict(row) for row in cur.fetchall()]

    def count(self, table: str) -> int:
        return int(self.query(f"select count(*) as n from {table}")[0]["n"])

    def commit(self) -> None:
        self._conn.commit()


@contextmanager
def open_seed_connection() -> Iterator[_Connection]:
    with open_connection() as (conn, dialect):
        yield _Connection(conn, dialect)


def create_tables(db: _Connection) -> None:
    for statement in _DDL:
        db.execute(statement)
    db.commit()


def is_seeded(db: _Connection) -> bool:
    """The doc's marker check: any customer row means the fixtures are in."""
    return db.count("customers") > 0


def _knowledge_chunk_rows() -> list[tuple[str, str, str]]:
    """The same chunks the Chroma index holds, as (id, source, content).

    The same *ids* too, because both come from `retrieval.chunk_id` - so the table
    and the index describe the same chunks and cannot drift.
    """
    from app.rag.retrieval import chunk_id, knowledge_chunks

    return [
        (
            chunk_id(chunk.source, chunk.position, chunk.content),
            chunk.source,
            chunk.content,
        )
        for chunk in knowledge_chunks()
    ]


def seed(
    *,
    fixtures_path: Path | None = None,
    force: bool = False,
    include_knowledge: bool = True,
) -> SeedReport:
    """Insert the fixture rows. A no-op when the tables are already populated.

    `force` clears the four tables first, so a re-run is idempotent rather than
    duplicated. `include_knowledge=False` skips the knowledge chunks, which is what
    the tests use to stay independent of the RAG docs.
    """
    path = fixtures_path or DEFAULT_FIXTURES_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))

    with open_seed_connection() as conn:
        create_tables(conn)
        if is_seeded(conn) and not force:
            return SeedReport(skipped=True, inserted={})
        if force:
            for table in TABLES:
                conn.execute(f"delete from {table}")

        inserted: dict[str, int] = {}

        customers = raw.get("customers", [])
        conn.executemany(
            INSERT_CUSTOMER,
            [
                (
                    row["id"],
                    row["name"],
                    row["email"],
                    row.get("tier", "standard"),
                    row.get("account_status", "active"),
                )
                for row in customers
            ],
        )
        inserted["customers"] = len(customers)

        charges = raw.get("charges", [])
        conn.executemany(
            INSERT_CHARGE,
            [
                (
                    row["id"],
                    row["customer_id"],
                    row.get("order_id"),
                    str(row["amount"]),
                    row["status"],
                    row["created_at"],
                )
                for row in charges
            ],
        )
        inserted["charges"] = len(charges)

        subscriptions = raw.get("subscriptions", [])
        conn.executemany(
            INSERT_SUBSCRIPTION,
            [
                (
                    row["id"],
                    row["customer_id"],
                    row["plan"],
                    row["status"],
                    str(row["price"]),
                    row.get("renewal_date"),
                )
                for row in subscriptions
            ],
        )
        inserted["subscriptions"] = len(subscriptions)

        chunks = _knowledge_chunk_rows() if include_knowledge else []
        if chunks:
            conn.executemany(INSERT_CHUNK, chunks)
        inserted["knowledge_chunks"] = len(chunks)

        conn.commit()
        return SeedReport(skipped=False, inserted=inserted)


def ensure_seeded(**kwargs: Any) -> SeedReport:
    """Seed if empty. Called at API startup; never raises on a missing database."""
    try:
        return seed(**kwargs)
    except Exception as exc:  # the API is still useful without the fixture tables
        print(f"warning: fixture seeding skipped: {exc}", file=sys.stderr)
        return SeedReport(skipped=True, inserted={})


def status() -> dict[str, int]:
    with open_seed_connection() as db:
        create_tables(db)
        return {table: db.count(table) for table in TABLES}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="clear and re-seed")
    parser.add_argument("--status", action="store_true", help="print row counts and exit")
    parser.add_argument("--no-knowledge", action="store_true",
                        help="skip the RAG knowledge chunks")
    args = parser.parse_args(argv)

    if args.status:
        print(f"backend: {backend()}")
        for table, count in status().items():
            print(f"  {table:<18} {count}")
        return 0

    report = seed(force=args.force, include_knowledge=not args.no_knowledge)
    print(f"backend: {backend()}")
    print(report.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
