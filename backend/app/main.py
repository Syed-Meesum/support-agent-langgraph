"""Uvicorn entry point: `uvicorn app.main:app`.

    cd backend
    uvicorn app.main:app --reload --port 8000

Interactive schema at http://localhost:8000/docs.
"""

from __future__ import annotations

from app.api import create_app
from app.config import load_env

# Read the repo-root .env before the first request, so a local run finds
# ANTHROPIC_API_KEY without exporting it (same behaviour as the CLI).
load_env()

app = create_app()

__all__ = ["app"]
