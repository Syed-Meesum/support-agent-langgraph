"""Request and response models for the HTTP API.

Typed responses are not just tidiness here: they give the FastAPI schema at
`/docs` a description of every field of a case, which is the fastest way for
someone reviewing this repo to see what the system actually produces.

`CaseDetail` mirrors `CaseState` exactly. If a field is
added to the state, this is the second place it must be added - a test asserts the
two stay in step.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.db.store import serialize


class MessageOut(BaseModel):
    role: str
    content: str


class AuditEntry(BaseModel):
    node: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class CaseDetail(BaseModel):
    """One case's full state - the documented `CaseState`, serialized."""

    case_id: str
    customer_id: str | None = None
    customer_message: str | None = None
    status: str
    category: str | None = None
    sentiment: str | None = None
    confidence: float | None = None
    customer_context: dict[str, Any] | None = None
    knowledge_hits: list[dict[str, Any]] = Field(default_factory=list)
    investigation_notes: list[str] = Field(default_factory=list)
    proposed_action: dict[str, Any] | None = None
    policy_result: dict[str, Any] | None = None
    verified: bool = False
    action_result: dict[str, Any] | None = None
    escalation_reason: str | None = None
    case_brief: dict[str, Any] | None = None
    human_decision: dict[str, Any] | None = None
    final_response: str | None = None
    conversation_history: list[MessageOut] = Field(default_factory=list)
    audit_log: list[AuditEntry] = Field(default_factory=list)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "CaseDetail":
        return cls.model_validate(serialize(state))


class QueueRow(BaseModel):
    """A queued case as the reviewer's list shows it."""

    case_id: str
    customer_id: str | None = None
    category: str | None = None
    status: str
    escalation_reason: str | None = None
    case_brief: dict[str, Any] | None = None
    updated_at: str | None = None


class RunCaseRequest(BaseModel):
    customer_id: str = Field(description="The customer the case is about.")
    message: str = Field(min_length=1, description="What the customer said.")
    case_id: str | None = Field(
        default=None,
        description="Optional explicit case id. Generated when omitted.",
    )


class ReviewRequest(BaseModel):
    reviewer: str = Field(min_length=1, description="Who is deciding.")
    notes: str = Field(default="", description="The reviewer's notes.")


class RejectRequest(ReviewRequest):
    alternative_action: str | None = Field(
        default=None,
        description="What to do instead, e.g. 'partial_refund'. Relayed to the customer.",
    )


class HealthResponse(BaseModel):
    status: str
    persistence_backend: str
    checkpoint_path: str | None = None
    case_db_path: str | None = None
    model: str
    model_ready: bool = Field(
        description="False when ANTHROPIC_API_KEY is missing; running a case would 503."
    )
    cases_by_status: dict[str, int] = Field(default_factory=dict)


class DemoCase(BaseModel):
    id: str
    customer_id: str
    message: str
    expectation: str


class EvalReportResponse(BaseModel):
    """The last eval run, whatever `eval/runner.py` wrote."""

    available: bool
    path: str
    report: dict[str, Any] | None = None
    hint: str | None = Field(
        default=None, description="How to produce a report when none exists yet."
    )


class EvalRunRequest(BaseModel):
    mode: str = Field(
        default="offline",
        pattern="^(offline|live)$",
        description="offline = scripted agent (no key); live = the real model.",
    )
    limit: int | None = Field(
        default=None, ge=1, description="Run only the first N cases. All when omitted."
    )


class ErrorResponse(BaseModel):
    detail: str
