"""The HTTP API: run a case, work the human-review queue, read eval results.

These endpoints are thin by design. Everything they call already existed as a
function with a test: `app/graph/hitl.py` owns the run/queue/approve/reject
contract, `app/db/` owns persistence, and `eval/runner.py` owns the eval run. The
API's job is to name them over HTTP and turn their exceptions into status codes.

| Endpoint | What it does |
|---|---|
| `GET  /api/health` | which persistence backend, which model, whether a key is set |
| `GET  /api/demos` | the ready-made cases the UI offers as buttons |
| `GET  /api/cases` | recent cases, newest first |
| `POST /api/cases` | run a case to its first stopping point |
| `GET  /api/cases/{id}` | the full case state, including the audit trail |
| `GET  /api/cases/{id}/audit` | just the persisted audit rows |
| `GET  /api/queue` | cases in `awaiting_human`, with their case briefs |
| `POST /api/cases/{id}/approve` | resume the graph and execute the action |
| `POST /api/cases/{id}/reject` | resume at the reply with the human's alternative |
| `GET  /api/evals` | the last eval run `eval/runner.py` wrote |
| `POST /api/evals/run` | run the eval suite now and return the metrics table |

Status codes worth knowing: **503** when the model is not configured (no
`ANTHROPIC_API_KEY`), **409** when a review is attempted on a case that cannot
accept one, **404** for an unknown case.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.runtime import GraphRuntime
from app.api.schemas import (
    CaseDetail,
    DemoCase,
    ErrorResponse,
    EvalReportResponse,
    EvalRunRequest,
    HealthResponse,
    QueueRow,
    RejectRequest,
    ReviewRequest,
    RunCaseRequest,
)
from app.db import CheckpointerError, StoreError
from app.graph.hitl import HumanReviewError
from app.graph.llm import MissingAPIKeyError, model_name

# backend/app/api/app.py -> backend/app/api -> backend/app -> backend
BACKEND_ROOT = Path(__file__).resolve().parents[2]
EVAL_REPORT_PATH = BACKEND_ROOT / "eval" / "results" / "latest.json"

# The UI is same-origin behind the Docker nginx, so CORS is only needed for the
# Vite dev server. Overridable for anyone serving the UI from elsewhere.
DEFAULT_CORS_ORIGINS = ("http://localhost:3000", "http://localhost:5173")

runtime = GraphRuntime()


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS")
    if not raw:
        return list(DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI) -> Iterator[None]:
    # Seed the fixture tables on start if they are empty.
    # Best-effort by design - the API is still useful without them, and the tools
    # read fixtures.json, so a seeding failure must not stop the service booting.
    from app.db.seed import ensure_seeded

    runtime.start()
    ensure_seeded()
    try:
        yield
    finally:
        runtime.stop()


def _service_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


def create_app() -> FastAPI:
    app = FastAPI(
        title="Autonomous Support Agent",
        summary=(
            "A LangGraph billing-support agent with a deterministic policy gate, "
            "a human-in-the-loop review queue, and a measured eval suite."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": "autonomous-support-agent",
            "docs": "/docs",
            "health": "/api/health",
        }

    # -- health and demo presets ------------------------------------------

    @app.get("/api/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        return HealthResponse.model_validate(runtime.health())

    @app.get("/api/demos", response_model=list[DemoCase], tags=["meta"])
    def demos() -> list[DemoCase]:
        from app.graph.run import DEMO_CASES

        return [
            DemoCase(
                id=key,
                customer_id=case["customer_id"],
                message=case["message"],
                expectation=case["expectation"],
            )
            for key, case in DEMO_CASES.items()
        ]

    # -- cases -------------------------------------------------------------

    @app.post(
        "/api/cases",
        response_model=CaseDetail,
        responses={503: {"model": ErrorResponse, "description": "No model configured"}},
        tags=["cases"],
    )
    def run_case(body: RunCaseRequest) -> CaseDetail:
        """Run a case. Returns it paused in the queue, or already resolved."""
        try:
            state = runtime.run_case(
                customer_id=body.customer_id,
                message=body.message,
                case_id=body.case_id,
            )
        except MissingAPIKeyError as exc:
            raise _service_unavailable(exc) from exc
        except (StoreError, CheckpointerError) as exc:
            raise _service_unavailable(exc) from exc
        return CaseDetail.from_state(state)

    @app.get("/api/cases", response_model=list[QueueRow], tags=["cases"])
    def list_cases(limit: int = 20) -> list[QueueRow]:
        return [QueueRow.model_validate(row) for row in runtime.recent(limit)]

    @app.get("/api/cases/{case_id}", response_model=CaseDetail, tags=["cases"])
    def get_case(case_id: str) -> CaseDetail:
        state = runtime.case(case_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"no case {case_id!r}")
        return CaseDetail.from_state(state)

    @app.get("/api/cases/{case_id}/audit", tags=["cases"])
    def get_audit(case_id: str) -> list[dict[str, Any]]:
        if runtime.case(case_id) is None:
            raise HTTPException(status_code=404, detail=f"no case {case_id!r}")
        return runtime.audit_trail(case_id)

    # -- the human-review queue -------------------------------------------

    @app.get("/api/queue", response_model=list[QueueRow], tags=["review"])
    def queue() -> list[QueueRow]:
        return [QueueRow.model_validate(row) for row in runtime.queue()]

    @app.post(
        "/api/cases/{case_id}/approve",
        response_model=CaseDetail,
        responses={409: {"model": ErrorResponse, "description": "Not awaiting a human"}},
        tags=["review"],
    )
    def approve(case_id: str, body: ReviewRequest) -> CaseDetail:
        """Approve: the proposed action executes, then the graph resumes to reply."""
        try:
            state = runtime.approve(
                case_id=case_id, reviewer=body.reviewer, notes=body.notes
            )
        except HumanReviewError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except MissingAPIKeyError as exc:
            raise _service_unavailable(exc) from exc
        return CaseDetail.from_state(state)

    @app.post(
        "/api/cases/{case_id}/reject",
        response_model=CaseDetail,
        responses={409: {"model": ErrorResponse, "description": "Not awaiting a human"}},
        tags=["review"],
    )
    def reject(case_id: str, body: RejectRequest) -> CaseDetail:
        """Reject: nothing executes and the human's resolution is relayed."""
        try:
            state = runtime.reject(
                case_id=case_id,
                reviewer=body.reviewer,
                notes=body.notes,
                alternative_action=body.alternative_action,
            )
        except HumanReviewError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except MissingAPIKeyError as exc:
            raise _service_unavailable(exc) from exc
        return CaseDetail.from_state(state)

    # -- eval results ------------------------------------------------------

    @app.get("/api/evals", response_model=EvalReportResponse, tags=["evals"])
    def evals() -> EvalReportResponse:
        """The last eval run, as written by `python -m eval.runner`."""
        if not EVAL_REPORT_PATH.is_file():
            return EvalReportResponse(
                available=False,
                path=str(EVAL_REPORT_PATH),
                hint=(
                    "No eval run recorded yet. From backend/: "
                    "python -m eval.runner --offline   (or --live with an API key)"
                ),
            )
        return EvalReportResponse(
            available=True,
            path=str(EVAL_REPORT_PATH),
            report=json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8")),
        )

    @app.post("/api/evals/run", response_model=EvalReportResponse, tags=["evals"])
    def run_evals(body: EvalRunRequest) -> EvalReportResponse:
        """Run the eval suite in-process and refresh `/api/evals`.

        Synchronous: the offline suite is a scripted agent over 39 cases and takes
        a few seconds. A live run makes roughly 200 model calls and is much
        slower, so the UI defaults to offline and says which one it ran.
        """
        # Imported here so the API package does not depend on the eval package
        # unless someone actually asks for an eval run.
        from eval import runner as eval_runner
        from eval.metrics import RunReport
        from eval.schemas import load_cases

        if body.mode == eval_runner.LIVE and not runtime.model_ready():
            raise HTTPException(
                status_code=503,
                detail=(
                    "ANTHROPIC_API_KEY is not set, so a live eval run is not "
                    "possible. Use mode='offline' or set the key."
                ),
            )

        cases = load_cases()
        if body.limit:
            cases = cases[: body.limit]

        llm = None
        if body.mode == eval_runner.LIVE:
            try:
                llm = eval_runner.get_llm()
            except MissingAPIKeyError as exc:  # pragma: no cover - guarded above
                raise _service_unavailable(exc) from exc

        judge = eval_runner.build_judge(body.mode, llm)
        report = RunReport(
            run_label="api",
            mode=body.mode,
            judge=judge.name,
            model=model_name() if body.mode == eval_runner.LIVE else None,
        )
        report.results = eval_runner.run_cases(cases, mode=body.mode, judge=judge)
        report.finish()
        eval_runner.write_report(report, EVAL_REPORT_PATH)
        return EvalReportResponse(
            available=True,
            path=str(EVAL_REPORT_PATH),
            report=report.model_dump(mode="json"),
        )

    return app


__all__ = ["create_app", "runtime", "EVAL_REPORT_PATH"]
