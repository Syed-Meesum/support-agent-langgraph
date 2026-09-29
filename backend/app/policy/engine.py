"""The deterministic policy engine - plain code, no LLM, no tool calls.

This is the module to point at when someone asks how you trust an LLM with
money: the model proposes an action, this gates it, and no model output can
change the outcome.

Evaluation order (see README.md), with one addition:

  0. Actions that execute nothing are resolved first - `no_action` approves,
     `escalate` requires a human. There is no money at risk, so gating them
     behind router signals would only create noise in the human queue.
  1. Hard blocks: amount over `refund_human_review_max` escalates immediately.
  2. Escalation triggers: tool error, low/absent confidence, angry/absent
     sentiment. Missing inputs fail closed (a rule_id naming the missing input).
  3. Category rules: refund tiers, cancellation confirmation.
  4. Fail closed: no matching rule -> `no_matching_rule`, never improvise.

Input contract:
  - `sentiment` / `confidence` come from the router node and are passed
    explicitly, so the gate never reads its own inputs out of the model's proposed
    action.
  - tool errors are read from `customer_context["tool_errors"]` (or `["errors"]`),
    which is where the investigate node records a failed tool call.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from app.policy.config import load_rules
from app.policy.models import PolicyResult

# Actions that move money, and so are subject to the amount tiers.
_MONEY_ACTIONS = frozenset({"refund", "partial_refund"})
_TERMINAL_ACTIONS = frozenset({"no_action", "escalate"})
_ANGRY_SENTIMENTS = frozenset({"angry"})
_TOOL_ERROR_KEYS = ("tool_errors", "errors")


def _read_action(proposed_action: Any) -> tuple[str | None, dict[str, Any]]:
    if not isinstance(proposed_action, dict):
        return None, {}
    action_type = proposed_action.get("type")
    if not isinstance(action_type, str) or not action_type.strip():
        return None, {}
    params = proposed_action.get("params")
    return action_type.strip().lower(), (params if isinstance(params, dict) else {})


def _read_amount(params: dict[str, Any]) -> Decimal | None:
    """Read a money amount. Returns None when absent or unusable."""
    if "amount" not in params:
        return None
    raw = params["amount"]
    if isinstance(raw, bool):
        return None
    if isinstance(raw, Decimal):
        amount = raw
    else:
        try:
            amount = Decimal(str(raw).strip())
        except (InvalidOperation, ValueError):
            return None
    if not amount.is_finite():
        return None
    return amount


def _read_tool_errors(customer_context: Any) -> list[Any]:
    if not isinstance(customer_context, dict):
        return []
    for key in _TOOL_ERROR_KEYS:
        value = customer_context.get(key)
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return list(value)
    return []


def _escalate(rule_id: str, reason: str) -> PolicyResult:
    return PolicyResult(
        approved=False, requires_human=True, rule_id=rule_id, reason=reason
    )


def _approve(rule_id: str, reason: str) -> PolicyResult:
    return PolicyResult(
        approved=True, requires_human=False, rule_id=rule_id, reason=reason
    )


def evaluate_policy(
    proposed_action: dict,
    customer_context: dict | None,
    *,
    sentiment: str | None,
    confidence: float | None,
) -> PolicyResult:
    """Gate one proposed action. Returns {approved, requires_human, rule_id, reason}."""
    rules = load_rules()
    action_type, params = _read_action(proposed_action)

    # 0. Terminal actions: nothing executes, so nothing to gate.
    if action_type == "no_action":
        return _approve(
            "no_action", "proposed action moves no money; nothing to gate"
        )
    if action_type == "escalate":
        return _escalate(
            "llm_escalation",
            "the agent proposed escalation rather than an executable action",
        )

    # 1. Hard blocks.
    amount = _read_amount(params) if action_type in _MONEY_ACTIONS else None
    if amount is not None and amount > rules.refund_human_review_max:
        return _escalate(
            "refund_hard_block",
            f"amount {amount} exceeds the hard ceiling of "
            f"{rules.refund_human_review_max}; mandatory human review",
        )

    # 2. Escalation triggers.
    tool_errors = _read_tool_errors(customer_context)
    if tool_errors:
        return _escalate(
            "tool_error",
            f"{len(tool_errors)} tool call(s) failed; cannot ground the decision "
            "in complete data",
        )
    if confidence is None:
        return _escalate(
            "missing_confidence",
            "no classification confidence was recorded; failing closed",
        )
    if confidence < rules.min_confidence_threshold:
        return _escalate(
            "low_confidence",
            f"confidence {confidence} is below the threshold "
            f"{rules.min_confidence_threshold}",
        )
    if sentiment is None:
        return _escalate(
            "missing_sentiment", "no sentiment was recorded; failing closed"
        )
    if rules.angry_sentiment_escalates and sentiment.strip().lower() in _ANGRY_SENTIMENTS:
        return _escalate(
            "angry_sentiment",
            f"customer sentiment {sentiment!r} requires a human reviewer",
        )

    # 3. Category rules.
    if action_type in _MONEY_ACTIONS:
        if amount is None:
            return _escalate(
                "missing_amount",
                "the proposed refund carries no usable amount; failing closed",
            )
        if amount <= 0:
            return _escalate(
                "invalid_amount",
                f"refund amount {amount} is not a positive amount",
            )
        if amount <= rules.refund_auto_approve_max:
            return _approve(
                "refund_auto_approved",
                f"amount {amount} is within the auto-approve limit of "
                f"{rules.refund_auto_approve_max}",
            )
        return _escalate(
            "refund_human_review",
            f"amount {amount} is above the auto-approve limit of "
            f"{rules.refund_auto_approve_max}; human review required",
        )

    if action_type == "cancel_subscription":
        if rules.cancellation_requires_confirmation:
            return _escalate(
                "cancellation_requires_confirmation",
                "cancellation always requires explicit confirmation before "
                "executing",
            )
        return _approve(
            "cancellation_approved",
            "cancellation confirmation is not required by the active policy",
        )

    # 4. Fail closed: no matching rule.
    return PolicyResult(
        approved=False,
        requires_human=rules.policy_exception_escalates,
        rule_id="no_matching_rule",
        reason=(
            "no policy rule matches action type "
            f"{action_type!r}; the system never improvises a resolution"
        ),
    )


def evaluate_policy_dict(
    proposed_action: dict,
    customer_context: dict | None,
    *,
    sentiment: str | None,
    confidence: float | None,
) -> dict:
    """`evaluate_policy` as the plain dict the graph state stores."""
    return evaluate_policy(
        proposed_action,
        customer_context,
        sentiment=sentiment,
        confidence=confidence,
    ).model_dump()
