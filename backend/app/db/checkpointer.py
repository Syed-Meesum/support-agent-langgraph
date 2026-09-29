"""Durable graph checkpoints, so a paused case can actually be resumed.

Real LangGraph checkpointing, rather than
hand-rolled JSON snapshots, is what makes "a human approves and the
graph continues from where it stopped" a fact rather than a claim. This module
supplies the checkpointer and picks its backend from the environment:

- `DATABASE_URL` set  -> `PostgresSaver` (the Supabase path in the doc)
- otherwise           -> `SqliteSaver` at `CHECKPOINT_PATH`

Both are real, durable, and support `graph.update_state(...)` + `invoke(None)`,
which is the resume mechanism `app.graph.hitl` uses. The SQLite default means the
project needs no services to run, and the test suite stays offline.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from app.db.settings import (
    BACKEND_POSTGRES,
    backend,
    checkpoint_path,
    database_url,
)

# Pydantic models we put inside CaseState. langgraph's serializer msgpacks
# anything that is not plain JSON, and refuses to guess at import time: without
# this it warns today and will block later. Registering the exact types we store
# keeps the checkpoint readable by a future langgraph.
ALLOWED_MSGPACK_MODULES = [("app.graph.state", "Message")]


class CheckpointerError(RuntimeError):
    """Raised when the configured checkpointer backend cannot be opened."""


def _serde() -> Any:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)


@contextmanager
def open_checkpointer() -> Iterator[Any]:
    """Yield a checkpointer, chosen from the environment.

    Use as a context manager - the connection is closed on exit::

        with open_checkpointer() as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            ...
    """
    serde = _serde()
    if backend() == BACKEND_POSTGRES:
        yield from _open_postgres(serde)
    else:
        yield from _open_sqlite(serde)


def _open_sqlite(serde: Any) -> Iterator[Any]:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise CheckpointerError(
            "langgraph-checkpoint-sqlite is not installed. "
            'Run: pip install "langgraph-checkpoint-sqlite>=3,<5"'
        ) from exc

    path = checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False so the saver can be used from the thread pool a
    # web server runs sync endpoints on.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        saver = SqliteSaver(conn, serde=serde)
        saver.setup()
        yield saver
    finally:
        conn.close()


def _open_postgres(serde: Any) -> Iterator[Any]:
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise CheckpointerError(
            "DATABASE_URL is set but langgraph-checkpoint-postgres is not installed. "
            'Run: pip install "langgraph-checkpoint-postgres>=3,<5" "psycopg[binary]>=3.2"'
        ) from exc

    url = database_url()
    if url is None:  # pragma: no cover - guarded by backend()
        raise CheckpointerError("DATABASE_URL is not set")

    # `serde` is supported by recent langgraph-checkpoint-postgres; on an older
    # build, fall back to its default serializer rather than failing outright.
    try:
        manager = PostgresSaver.from_conn_string(url, serde=serde)
    except TypeError:  # pragma: no cover - version guard
        manager = PostgresSaver.from_conn_string(url)
    with manager as saver:
        saver.setup()
        yield saver
