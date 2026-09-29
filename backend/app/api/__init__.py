"""The FastAPI service: cases, the human-review queue, and eval results.

`create_app()` builds the app; `app.main` exposes it as `app` for uvicorn.
"""

from app.api.app import create_app
from app.api.runtime import GraphRuntime, new_case_id
from app.api.schemas import CaseDetail, QueueRow

__all__ = ["create_app", "GraphRuntime", "new_case_id", "CaseDetail", "QueueRow"]
