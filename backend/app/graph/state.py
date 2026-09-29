"""CaseState - the graph's shared state.

`audit_log` and `conversation_history` use `operator.add` reducers so each node
appends only its own new entries instead of rewriting the whole list.

`customer_id` sits alongside the graph's own fields: every tool call needs it, and
the `cases` table already carries it, so the case record is what supplies it.

`case_brief` and `human_decision` are the step-6 (HITL) additions: the brief is
what a human reviewer sees in the queue, and the decision is what the reviewer
writes back before the graph resumes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict

# Case statuses. `awaiting_human` is the step-6 pause: the run stops with a case
# brief and `app.graph.hitl` resumes the same thread once a reviewer decides.
IN_PROGRESS = "in_progress"
RESOLVED = "resolved"
ESCALATED = "escalated"
AWAITING_HUMAN = "awaiting_human"


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["customer", "agent", "human_reviewer"]
    content: str


class CaseState(TypedDict, total=False):
    case_id: str
    customer_id: str
    customer_message: str
    conversation_history: Annotated[list[Message], operator.add]
    category: str | None            # "refund" | "cancellation"
    sentiment: str | None
    confidence: float | None
    customer_context: dict[str, Any] | None
    knowledge_hits: list[dict[str, Any]]
    investigation_notes: list[str]
    proposed_action: dict[str, Any] | None
    policy_result: dict[str, Any] | None
    verified: bool
    action_result: dict[str, Any] | None
    escalation_reason: str | None
    case_brief: dict[str, Any] | None   # what the human reviewer sees
    human_decision: dict[str, Any] | None  # written back before a resume
    final_response: str | None
    status: str                     # IN_PROGRESS | RESOLVED | ESCALATED | AWAITING_HUMAN
    audit_log: Annotated[list[dict[str, Any]], operator.add]


def new_case_state(
    case_id: str,
    customer_id: str,
    customer_message: str,
) -> CaseState:
    """Build a fresh state for one case.

    Seed every key explicitly so the reducers have a starting value and the
    persisted snapshot is complete rather than sparse.
    """
    return CaseState(
        case_id=case_id,
        customer_id=customer_id,
        customer_message=customer_message,
        conversation_history=[Message(role="customer", content=customer_message)],
        category=None,
        sentiment=None,
        confidence=None,
        customer_context=None,
        knowledge_hits=[],
        investigation_notes=[],
        proposed_action=None,
        policy_result=None,
        verified=False,
        action_result=None,
        escalation_reason=None,
        case_brief=None,
        human_decision=None,
        final_response=None,
        status=IN_PROGRESS,
        audit_log=[],
    )
