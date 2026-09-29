"""The seven graph nodes.

Five nodes call the LLM (router, investigate, decide, verify, respond).
`policy_check` is pure code. `act_or_escalate` executes the approved action, or
pauses the case for a human (`status = awaiting_human`).

Every node appends to `audit_log` so the demo trail can show, per case, which
tool was called, what the model proposed, which policy rule fired, and whether
verification passed.
"""

from __future__ import annotations

from typing import Any

from app.graph.prompts import (
    decide_messages,
    investigate_messages,
    respond_messages,
    router_messages,
    verify_messages,
)
from app.graph.schemas import (
    ActionProposal,
    InvestigationNotes,
    RouterDecision,
    VerificationResult,
    to_proposed_action,
)
from app.graph.state import AWAITING_HUMAN, RESOLVED, CaseState, Message
from app.policy import evaluate_policy_dict
from app.tools import action_tools, customer_tools, knowledge_tools
from app.tools.errors import ToolError

MAX_KNOWLEDGE_HITS = 3

# What a reviewer writes into `human_decision`.
HUMAN_APPROVED = "approved"
HUMAN_REJECTED = "rejected"


def _audit(node: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """One audit_log entry. Mirrors the audit_log table."""
    return {"node": node, "event_type": event_type, "payload": payload}


def _structured(llm: Any, schema: Any, messages: Any) -> Any:
    return llm.with_structured_output(schema).invoke(messages)


def _text(message: Any) -> str:
    """Pull plain text out of an AIMessage, whose content may be blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()
    return str(content).strip()


# --------------------------------------------------------------------------
# 1. router
# --------------------------------------------------------------------------


def router_node(state: CaseState, *, llm: Any) -> dict[str, Any]:
    """LLM call. Classifies category + sentiment and writes them to state."""
    decision = _structured(
        llm, RouterDecision, router_messages(state["customer_message"])
    )
    return {
        "category": decision.category,
        "sentiment": decision.sentiment,
        "confidence": decision.confidence,
        "audit_log": [
            _audit("router", "llm_call", {"decision": decision.model_dump()})
        ],
    }


# --------------------------------------------------------------------------
# 2. investigate
# --------------------------------------------------------------------------


def investigate_node(state: CaseState, *, llm: Any) -> dict[str, Any]:
    """Calls the four read tools, then an LLM synthesises the notes.

    A failing tool is recorded in `customer_context["tool_errors"]` and the node
    carries on: the policy engine treats a recorded tool error as an escalation
    trigger, so the case still ends somewhere honest.
    """
    customer_id = state["customer_id"]
    context: dict[str, Any] = {}
    tool_errors: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    def _record(tool: str, ok: bool, payload: dict[str, Any]) -> None:
        audit.append(_audit("investigate", "tool_call", {"tool": tool, "ok": ok, **payload}))

    def _fail(tool: str, exc: ToolError, key: str | None = None) -> None:
        error = exc.to_dict()["error"]
        tool_errors.append(error)
        if key is not None:
            context[key] = None
        _record(tool, False, {"error": error})

    try:
        customer = customer_tools.get_customer(customer_id)
        context["customer"] = customer.model_dump(mode="json")
        _record("get_customer", True, {"result": context["customer"]})
    except ToolError as exc:
        _fail("get_customer", exc, "customer")

    try:
        charges = customer_tools.get_charges(customer_id)
        context["charges"] = [c.model_dump(mode="json") for c in charges]
        _record("get_charges", True, {"result": context["charges"]})
    except ToolError as exc:
        _fail("get_charges", exc, "charges")

    try:
        subscription = customer_tools.get_subscription(customer_id)
        context["subscription"] = (
            subscription.model_dump(mode="json") if subscription is not None else None
        )
        _record("get_subscription", True, {"result": context["subscription"]})
    except ToolError as exc:
        _fail("get_subscription", exc, "subscription")

    knowledge_hits: list[dict[str, Any]] = []
    try:
        hits = knowledge_tools.search_knowledge_base(
            state["customer_message"], top_k=MAX_KNOWLEDGE_HITS
        )
        knowledge_hits = [hit.model_dump(mode="json") for hit in hits]
        _record("search_knowledge_base", True, {"result": knowledge_hits})
    except ToolError as exc:
        _fail("search_knowledge_base", exc)

    context["tool_errors"] = tool_errors

    notes = _structured(
        llm,
        InvestigationNotes,
        investigate_messages(state["customer_message"], context, knowledge_hits),
    )
    audit.append(
        _audit("investigate", "llm_call", {"investigation_notes": notes.notes})
    )

    return {
        "customer_context": context,
        "knowledge_hits": knowledge_hits,
        "investigation_notes": notes.notes,
        "audit_log": audit,
    }


# --------------------------------------------------------------------------
# 3. decide
# --------------------------------------------------------------------------


def decide_node(state: CaseState, *, llm: Any) -> dict[str, Any]:
    """LLM call. Proposes exactly one action. Deliberately says nothing about
    whether it is allowed - that is policy_check's job."""
    proposal: ActionProposal = _structured(
        llm,
        ActionProposal,
        decide_messages(
            state["customer_message"],
            state.get("customer_context") or {},
            state.get("investigation_notes") or [],
        ),
    )
    action = to_proposed_action(proposal)
    return {
        "proposed_action": action,
        "audit_log": [_audit("decide", "llm_call", {"proposed_action": action})],
    }


# --------------------------------------------------------------------------
# 4. policy_check  (pure code, no LLM)
# --------------------------------------------------------------------------


def policy_check_node(state: CaseState) -> dict[str, Any]:
    result = evaluate_policy_dict(
        state.get("proposed_action") or {},
        state.get("customer_context"),
        sentiment=state.get("sentiment"),
        confidence=state.get("confidence"),
    )
    return {
        "policy_result": result,
        "audit_log": [
            _audit(
                "policy_check",
                "policy_decision",
                {"policy_result": result, "decided_by": "deterministic_policy_engine"},
            )
        ],
    }


# --------------------------------------------------------------------------
# 5. verify
# --------------------------------------------------------------------------


def verify_node(state: CaseState, *, llm: Any) -> dict[str, Any]:
    """LLM call. Only reached when policy approved. Answers the narrow question
    of whether the evidence supports this exact action."""
    policy_result = state.get("policy_result") or {}
    if not policy_result.get("approved"):
        # Unreachable on a correctly wired graph; refuse rather than confirm.
        return {
            "verified": False,
            "audit_log": [
                _audit(
                    "verify",
                    "llm_call",
                    {"verified": False, "reason": "policy did not approve; verification not reached"},
                )
            ],
        }

    verdict: VerificationResult = _structured(
        llm,
        VerificationResult,
        verify_messages(
            state["customer_message"],
            state.get("customer_context") or {},
            state.get("investigation_notes") or [],
            state.get("proposed_action") or {},
        ),
    )
    return {
        "verified": verdict.verified,
        "audit_log": [
            _audit("verify", "llm_call", {"verified": verdict.verified, "reason": verdict.reason})
        ],
    }


# --------------------------------------------------------------------------
# 6. act_or_escalate
# --------------------------------------------------------------------------


def _last_verify_reason(state: CaseState) -> str | None:
    for entry in reversed(state.get("audit_log") or []):
        if entry.get("node") == "verify" and entry.get("event_type") == "llm_call":
            payload = entry.get("payload") or {}
            if payload.get("verified") is False and payload.get("reason"):
                return str(payload["reason"])
    return None


def _case_brief(state: CaseState, reason: str) -> dict[str, Any]:
    """The concise brief a human reviewer sees in the queue.

    Built entirely from state that already exists - a human review never
    re-investigates the case.
    """
    notes = state.get("investigation_notes") or []
    return {
        "problem_summary": state.get("customer_message") or "",
        "evidence_summary": "; ".join(str(n) for n in notes)
        if notes
        else "no evidence could be grounded",
        "proposed_action": state.get("proposed_action") or {},
        "reason_for_review": reason,
    }


def human_rejection_outcome(human_decision: dict[str, Any]) -> dict[str, Any]:
    """The state a reviewer's rejection produces.

    Deliberately writes no `action_result`: nothing was executed, and the reply
    must not claim otherwise. The reviewer's resolution travels in
    `human_decision`, which `respond` is told to relay verbatim.

    Shared by the node (if a rejection ever reaches it) and by
    `app.graph.hitl.reject`, which resumes the graph with exactly this.
    """
    return {
        "status": RESOLVED,
        "audit_log": [
            _audit(
                "act_or_escalate",
                "human_decision",
                {"branch": "human_rejected", "human_decision": human_decision},
            )
        ],
    }


def act_or_escalate_node(state: CaseState) -> dict[str, Any]:
    """Executes the approved, verified action - or pauses for a human.

    Reached from policy_check (rejected / requires_human) and from verify (both
    outcomes), so the branch is decided here from state. When the case needs a
    human it writes `status = "awaiting_human"` plus a `case_brief` and the run
    ends there: the checkpoint *is* the pause, and the resume service picks the
    case up from the queue.

    A `human_decision` in state means this run is a resume, and an approval
    deliberately overrides policy_check and verify - a person is allowed to
    authorise what the gate refused.
    """
    human_decision = state.get("human_decision") or {}
    human_approved = human_decision.get("decision") == HUMAN_APPROVED

    policy_result = state.get("policy_result") or {}
    action = state.get("proposed_action") or {}
    action_type = action.get("type")
    params = action.get("params") or {}

    def escalated(reason: str) -> dict[str, Any]:
        brief = _case_brief(state, reason)
        return {
            "escalation_reason": reason,
            "case_brief": brief,
            "status": AWAITING_HUMAN,
            "audit_log": [
                _audit(
                    "act_or_escalate",
                    "policy_decision",
                    {"branch": AWAITING_HUMAN, "reason": reason, "case_brief": brief},
                )
            ],
        }

    def executed(
        result: dict[str, Any],
        detail: str,
        *,
        branch: str = "action",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"branch": branch, "result": result, "detail": detail}
        if human_approved:
            payload["human_decision"] = human_decision
        return {
            "action_result": result,
            "status": RESOLVED,
            "audit_log": [_audit("act_or_escalate", "tool_call", payload)],
        }

    # A reviewed case resumes here. Handle a rejection too, so the node is total
    # even if a resume lands on it rather than on respond.
    if human_decision.get("decision") == HUMAN_REJECTED:
        return human_rejection_outcome(human_decision)

    # A human approval overrides both gates by design; anything else must pass them.
    if human_approved:
        branch = "human_approved"
    else:
        # Not permitted: policy rejected it, or policy deferred to a human.
        if not policy_result.get("approved"):
            return escalated(
                policy_result.get("reason")
                or "the policy engine did not approve the proposed action"
            )

        # Permitted but not evidence-backed.
        if not state.get("verified"):
            return escalated(
                _last_verify_reason(state)
                or "verification could not confirm the evidence supports the proposed action"
            )
        branch = "action"

    # A no-op action needs no tool call.
    if action_type == "no_action":
        return executed(
            {"success": True, "action": "no_action", "detail": "no action was required"},
            "no_action executes nothing",
            branch=branch,
        )

    try:
        if action_type in ("refund", "partial_refund"):
            charge_id = params.get("charge_id")
            amount = params.get("amount")
            if not charge_id or amount is None:
                return escalated(
                    "the approved refund is missing the charge or amount it needs"
                )
            refund = action_tools.issue_refund(
                case_id=state["case_id"],
                customer_id=state["customer_id"],
                charge_id=str(charge_id),
                amount=amount,
                reason=str(action.get("rationale") or "approved refund"),
            )
            return executed(
                {**refund.model_dump(mode="json"), "action": action_type},
                "refund issued",
                branch=branch,
            )

        if action_type == "cancel_subscription":
            subscription_id = params.get("subscription_id")
            if not subscription_id:
                return escalated(
                    "the approved cancellation is missing the subscription it needs"
                )
            cancellation = action_tools.cancel_subscription(
                customer_id=state["customer_id"],
                subscription_id=str(subscription_id),
            )
            return executed(
                {**cancellation.model_dump(mode="json"), "action": action_type},
                "subscription cancelled",
                branch=branch,
            )
    except ToolError as exc:
        return escalated(f"the approved action could not be executed: {exc.message}")

    return escalated(
        f"no executor exists for approved action {action_type!r}"
    )


# --------------------------------------------------------------------------
# 7. respond
# --------------------------------------------------------------------------


def respond_node(state: CaseState, *, llm: Any) -> dict[str, Any]:
    """LLM call. Composes the customer-facing reply from what actually happened."""
    outcome: dict[str, Any] = {}
    if state.get("human_decision"):
        outcome["human_review"] = state["human_decision"]
    if state.get("action_result") is not None:
        outcome["executed_action"] = state["action_result"]
    # "Still escalating" is only true while the case is still waiting on a human.
    # A case that has since been reviewed and resolved keeps its escalation_reason
    # as history - telling the model about it would contradict the outcome.
    if state.get("status") == AWAITING_HUMAN and state.get("escalation_reason"):
        outcome["escalated_to_human"] = {"reason": state["escalation_reason"]}

    message = _text(
        llm.invoke(respond_messages(state.get("sentiment"), outcome))
    )
    return {
        "final_response": message,
        "conversation_history": [Message(role="agent", content=message)],
        "audit_log": [_audit("respond", "llm_call", {"final_response": message})],
    }
