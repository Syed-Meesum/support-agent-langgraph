"""Human-in-the-loop: run a case, queue it, and resume it after a review.

This is the service layer behind `GET /api/queue`,
`POST /api/cases/{id}/approve` and `POST /api/cases/{id}/reject` in `app/api/`. The
transport is deliberately thin, so these functions are the whole contract: they take
a compiled graph (with a checkpointer), a `CaseStore`, and plain arguments.

The resume itself is real LangGraph mechanics, not a re-run:

- The graph is checkpointed against `thread_id = case_id`, so every node write is
  durable (see `app/db/checkpointer.py`).
- An escalation ends the run with `status = "awaiting_human"` and a case brief.
- `approve`/`reject` inject the reviewer's decision with `graph.update_state(...)`
  and then `graph.invoke(None, config)`, which continues from the checkpoint
  rather than replaying `router`/`investigate`/`decide`.

`update_state`'s `as_node` chooses *where* the resume continues from, which is why
an approval skips back through the gate and a rejection skips ahead to the reply.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.graph import nodes
from app.graph.state import AWAITING_HUMAN, RESOLVED, Message, new_case_state

__all__ = [
    "AWAITING_HUMAN",
    "RESOLVED",
    "HumanReviewError",
    "thread_config",
    "run_case",
    "list_queue",
    "get_case",
    "approve",
    "reject",
]

HUMAN_APPROVED = nodes.HUMAN_APPROVED
HUMAN_REJECTED = nodes.HUMAN_REJECTED


class HumanReviewError(RuntimeError):
    """Raised when a review is attempted on a case that cannot accept one."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def thread_config(case_id: str) -> dict[str, Any]:
    """LangGraph's thread id for one case. The case id *is* the thread."""
    return {"configurable": {"thread_id": case_id}}


def run_case(
    *,
    graph: Any,
    store: Any,
    case_id: str,
    customer_id: str,
    message: str,
) -> dict[str, Any]:
    """Run a fresh case to its first stopping point, then persist it.

    Returns the resulting state. `status == AWAITING_HUMAN` means it is now in the
    queue; `RESOLVED` means it finished on its own.
    """
    state = new_case_state(case_id, customer_id, message)
    result = graph.invoke(state, thread_config(case_id))
    store.save(result)
    return result


def list_queue(store: Any) -> list[dict[str, Any]]:
    """Cases waiting on a human, each with its case brief."""
    return store.list_by_status(AWAITING_HUMAN)


def get_case(store: Any, case_id: str) -> dict[str, Any] | None:
    """The persisted `CaseState` for one case, or None."""
    return store.get_case(case_id)


def approve(
    *,
    graph: Any,
    store: Any,
    case_id: str,
    reviewer: str,
    notes: str = "",
) -> dict[str, Any]:
    """Approve a queued case: the proposed action actually executes, then replies.

    The approval overrides `policy_check` and `verify` by design - that is the
    point of a human override - so the graph resumes at `act_or_escalate`.
    """
    _require_awaiting_human(store, case_id, "approve")
    decision = {
        "decision": HUMAN_APPROVED,
        "reviewer": reviewer,
        "notes": notes,
        "reviewed_at": _now(),
    }
    updates = _merge(_review_record(decision), {"human_decision": decision})
    # `verify` is the only node whose successor is the executor, which is exactly
    # the resume point an approval needs.
    return _resume(graph, store, case_id, updates, as_node="verify")


def reject(
    *,
    graph: Any,
    store: Any,
    case_id: str,
    reviewer: str,
    notes: str = "",
    alternative_action: str | None = None,
) -> dict[str, Any]:
    """Reject a queued case: nothing executes; the reviewer's resolution is sent.

    Resumes directly at `respond`, so the customer hears the human's decision
    rather than the agent's original proposal.
    """
    _require_awaiting_human(store, case_id, "reject")
    decision = {
        "decision": HUMAN_REJECTED,
        "reviewer": reviewer,
        "notes": notes,
        "alternative_action": alternative_action,
        "reviewed_at": _now(),
    }
    updates = _merge(
        _review_record(decision),
        {"human_decision": decision},
        nodes.human_rejection_outcome(decision),
    )
    # `act_or_escalate`'s only successor is `respond`, so resuming as that node
    # skips the executor entirely.
    return _resume(graph, store, case_id, updates, as_node="act_or_escalate")


def _merge(*parts: dict[str, Any]) -> dict[str, Any]:
    """Combine state updates, concatenating list-valued keys.

    `audit_log` and `conversation_history` are `operator.add` reducers, so two
    contributors must be concatenated here rather than clobbering each other -
    which is exactly how the reject path would otherwise lose its review entry.
    """
    merged: dict[str, Any] = {}
    for part in parts:
        for key, value in part.items():
            if isinstance(merged.get(key), list) and isinstance(value, list):
                merged[key] = merged[key] + value
            else:
                merged[key] = value
    return merged


def _resume(
    graph: Any,
    store: Any,
    case_id: str,
    updates: dict[str, Any],
    *,
    as_node: str,
) -> dict[str, Any]:
    config = thread_config(case_id)
    try:
        graph.update_state(config, updates, as_node=as_node)
    except ValueError as exc:
        # LangGraph raises this when the graph was compiled without a
        # checkpointer - there is no checkpoint to resume from.
        raise HumanReviewError(
            f"case {case_id!r} cannot be resumed: the graph has no checkpointer. "
            "Build it with build_graph(checkpointer=...) from app.db.open_checkpointer()."
        ) from exc
    result = graph.invoke(None, config)
    store.save(result)
    return result


def _require_awaiting_human(store: Any, case_id: str, action: str) -> dict[str, Any]:
    row = store.get_row(case_id)
    if row is None:
        raise HumanReviewError(f"there is no case {case_id!r} to {action}")
    if row.get("status") != AWAITING_HUMAN:
        raise HumanReviewError(
            f"case {case_id!r} is {row.get('status')!r}, not {AWAITING_HUMAN!r}: "
            f"nothing to {action}"
        )
    return row


def _review_record(decision: dict[str, Any]) -> dict[str, Any]:
    """The conversation entry and audit entry a review adds to the case."""
    return {
        "conversation_history": [
            Message(role="human_reviewer", content=_review_note(decision))
        ],
        "audit_log": [
            {
                "node": "human_review",
                "event_type": "human_decision",
                "payload": decision,
            }
        ],
    }


def _review_note(decision: dict[str, Any]) -> str:
    verb = "approved" if decision["decision"] == HUMAN_APPROVED else "rejected"
    reviewer = decision.get("reviewer") or "a human reviewer"
    parts = [f"{reviewer} {verb} the proposed action."]
    if decision.get("notes"):
        parts.append(f"Notes: {decision['notes']}")
    if decision.get("alternative_action"):
        parts.append(f"Alternative resolution: {decision['alternative_action']}.")
    return " ".join(parts)
