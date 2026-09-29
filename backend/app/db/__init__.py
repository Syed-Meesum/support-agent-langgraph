"""Persistence: durable graph checkpoints and the case store.

Two concerns, deliberately separate:

- `checkpointer` - LangGraph's own checkpoint tables, keyed by `thread_id`. This
  is what makes a paused case resumable mid-graph.
- `store` - the app's `cases` / `case_messages` / `audit_log` tables. This is what
  the human-review queue and the detail view read.

Both default to local SQLite and switch to the documented Supabase Postgres when
`DATABASE_URL` is set, so the project runs anywhere and still has a real
production path.
"""

from app.db.checkpointer import CheckpointerError, open_checkpointer
from app.db.settings import (
    BACKEND_POSTGRES,
    BACKEND_SQLITE,
    backend,
    case_db_path,
    checkpoint_path,
    database_url,
)
from app.db.store import (
    CaseStore,
    StoreError,
    json_default,
    open_case_store,
    open_connection,
    serialize,
)

__all__ = [
    "open_checkpointer",
    "CheckpointerError",
    "open_case_store",
    "open_connection",
    "CaseStore",
    "StoreError",
    "serialize",
    "json_default",
    "backend",
    "database_url",
    "checkpoint_path",
    "case_db_path",
    "BACKEND_POSTGRES",
    "BACKEND_SQLITE",
]
