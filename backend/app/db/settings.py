"""Where the persistence layer reads its configuration from.

One place, so the checkpointer, the case store and the tests all agree on which
paths/env vars matter. Deliberately dependency-free: nothing here imports the
graph package, so `app.db` can be imported from anywhere without a cycle.
"""

from __future__ import annotations

import os
from pathlib import Path

# The documented production backend: a Supabase Postgres instance, which is also
# what `PostgresSaver` needs. Absent -> local SQLite.
DATABASE_URL_ENV = "DATABASE_URL"

# Local fallbacks. Both live under backend/.data/ (gitignored) so a clone still
# runs, and every step of the project stays testable with no services running.
CHECKPOINT_PATH_ENV = "CHECKPOINT_PATH"
CASE_DB_PATH_ENV = "CASE_DB_PATH"

BACKEND_SQLITE = "sqlite"
BACKEND_POSTGRES = "postgres"

# backend/app/db/settings.py -> backend/app -> backend
BACKEND_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_ROOT / ".data"
DEFAULT_CHECKPOINT_PATH = DATA_DIR / "checkpoints.sqlite"
DEFAULT_CASE_DB_PATH = DATA_DIR / "cases.sqlite"


def database_url() -> str | None:
    """The Supabase/Postgres connection string, when one is configured."""
    value = os.environ.get(DATABASE_URL_ENV)
    return value.strip() if value and value.strip() else None


def backend() -> str:
    """`postgres` when DATABASE_URL is set, otherwise `sqlite`."""
    return BACKEND_POSTGRES if database_url() else BACKEND_SQLITE


def checkpoint_path() -> Path:
    configured = os.environ.get(CHECKPOINT_PATH_ENV)
    return Path(configured) if configured else DEFAULT_CHECKPOINT_PATH


def case_db_path() -> Path:
    configured = os.environ.get(CASE_DB_PATH_ENV)
    return Path(configured) if configured else DEFAULT_CASE_DB_PATH
