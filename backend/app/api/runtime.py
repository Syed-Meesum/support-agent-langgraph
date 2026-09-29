"""The API's long-lived graph, checkpointer and case store.

A server cannot open and close the persistence layers per request the way the CLI
does, so this holds one of each for the process's lifetime. `open_checkpointer()`
and `open_case_store()` are context managers; `start()`/`stop()` enter and exit
them, and FastAPI's lifespan calls them (`app/api/app.py`).

**The graph is built lazily.** `build_graph()` without an injected LLM calls
`get_llm()`, which raises when `ANTHROPIC_API_KEY` is missing. Building it at
startup would make the whole app fail to boot for a reviewer who just wants to
look at `/docs` - so `/api/health` reports `model_ready: false` and only the
case-running endpoints return 503 until a key is present.

**Everything that touches state is serialized behind one lock.** `SqliteSaver`
holds a single connection, and a case run is a few seconds of work; running two at
once would interleave writes on that connection. For a single-reviewer demo,
correct-but-serial is the right trade, and it is the honest one to explain: this
is not a concurrent service, and nothing here claims to be.
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import Any

from app.config import load_env
from app.db import open_case_store, open_checkpointer
from app.db.settings import backend, case_db_path, checkpoint_path
from app.graph import hitl
from app.graph.build import build_graph
from app.graph.llm import API_KEY_ENV, model_name


def new_case_id() -> str:
    """A fresh case id. Also the LangGraph `thread_id`, so it must be unique."""
    return "case_" + uuid.uuid4().hex[:12]


class GraphRuntime:
    """Owns the persistence layers and the compiled graph."""

    def __init__(self) -> None:
        self._checkpointer_cm: Any = None
        self._store_cm: Any = None
        self.checkpointer: Any = None
        self.store: Any = None
        self._graph: Any = None
        self.lock = threading.RLock()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._store_cm is not None:
            return
        self._checkpointer_cm = open_checkpointer()
        self.checkpointer = self._checkpointer_cm.__enter__()
        try:
            self._store_cm = open_case_store()
            self.store = self._store_cm.__enter__()
        except BaseException:
            self._checkpointer_cm.__exit__(None, None, None)
            self._checkpointer_cm = None
            self.checkpointer = None
            raise

    def stop(self) -> None:
        # Store first: the store's context exits after the checkpointer's opened,
        # so it unwinds first.
        for name in ("_store_cm", "_checkpointer_cm"):
            manager = getattr(self, name)
            if manager is not None:
                manager.__exit__(None, None, None)
                setattr(self, name, None)
        self.store = None
        self.checkpointer = None
        self._graph = None

    @property
    def graph(self) -> Any:
        """The compiled graph, built on first use so a missing key is not fatal."""
        if self._graph is None:
            self._graph = build_graph(checkpointer=self.checkpointer)
        return self._graph

    # -- introspection -----------------------------------------------------

    def model_ready(self) -> bool:
        load_env()
        return bool(os.environ.get(API_KEY_ENV))

    def health(self) -> dict[str, Any]:
        local = backend() == "sqlite"
        return {
            "status": "ok",
            "persistence_backend": backend(),
            "checkpoint_path": str(checkpoint_path()) if local else None,
            "case_db_path": str(case_db_path()) if local else None,
            "model": model_name(),
            "model_ready": self.model_ready(),
            "cases_by_status": self._status_counts(),
        }

    # -- case operations ---------------------------------------------------

    def run_case(
        self, *, customer_id: str, message: str, case_id: str | None = None
    ) -> dict[str, Any]:
        """Run a fresh case to its first stopping point and persist it."""
        with self.lock:
            return hitl.run_case(
                graph=self.graph,
                store=self.store,
                case_id=case_id or new_case_id(),
                customer_id=customer_id,
                message=message,
            )

    def approve(
        self, *, case_id: str, reviewer: str, notes: str = ""
    ) -> dict[str, Any]:
        with self.lock:
            return hitl.approve(
                graph=self.graph,
                store=self.store,
                case_id=case_id,
                reviewer=reviewer,
                notes=notes,
            )

    def reject(
        self,
        *,
        case_id: str,
        reviewer: str,
        notes: str = "",
        alternative_action: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            return hitl.reject(
                graph=self.graph,
                store=self.store,
                case_id=case_id,
                reviewer=reviewer,
                notes=notes,
                alternative_action=alternative_action,
            )

    def case(self, case_id: str) -> dict[str, Any] | None:
        with self.lock:
            return hitl.get_case(self.store, case_id)

    def queue(self) -> list[dict[str, Any]]:
        with self.lock:
            return hitl.list_queue(self.store)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.lock:
            return self.store.list_recent(limit)

    def audit_trail(self, case_id: str) -> list[dict[str, Any]]:
        with self.lock:
            return self.store.audit_trail(case_id)

    def _status_counts(self) -> dict[str, int]:
        if self.store is None:
            return {}
        with self.lock:
            return self.store.count_by_status()
