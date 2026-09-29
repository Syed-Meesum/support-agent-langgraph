"""Environment configuration.

Loads the repo-root `.env` if it exists, so a local run picks up
ANTHROPIC_API_KEY without exporting it by hand.

One `.env`, deliberately: Docker Compose reads the same root file for its
`${VAR}` substitutions, so there is a single place to put
secrets rather than one per service.

Process environment always wins - the file is loaded with `override=False`, so
`docker compose` env vars and an explicit `export` both take precedence.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# backend/app/config.py -> backend/app -> backend -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"


@lru_cache(maxsize=1)
def load_env() -> bool:
    """Load the root .env once. Returns True if a file was present."""
    if not ENV_FILE.is_file():
        return False
    load_dotenv(ENV_FILE, override=False)
    return True


def reload_env() -> None:
    """Forget that the .env was loaded. Used by tests."""
    load_env.cache_clear()
