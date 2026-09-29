"""The deterministic offline agent, used when there is no ANTHROPIC_API_KEY.

This is a *scripted* stand-in for Claude, not a model and not an approximation of
one. Its choices come from the case file's `scripted` block, so an offline run
measures the harness, the tools, the policy engine, the graph wiring and the
metric maths - every part of the system except the model. It says nothing about
model quality, and `eval/runner.py` prints that distinction in its header so the
numbers can never be mistaken for a benchmark.

Two design choices are worth knowing:

- The scripted agent for an injection case deliberately OBEYS the injected
  instruction (`scripted.action` proposes the $5000 refund). A pass therefore
  proves something real: `policy_check` blocks it. If the scripted agent refused
  the injection itself, the run would prove nothing about the guard.
- The scripted reply is written from what the case *expects* to happen, so a
  resolved case claims a refund and an escalated case promises a human. A
  scripted reply that over-claimed would be caught by the hallucination judge -
  which is exactly the failure that judge exists to detect.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from app.graph.schemas import (
    ActionProposal,
    InvestigationNotes,
    RouterDecision,
    VerificationResult,
)
from eval.schemas import EvalCase, ScriptedPlan

DEFAULT_NOTE = "evidence from the tool calls in customer_context; nothing further could be grounded"
HUMAN_REPLY = (
    "Thanks for flagging this - I've passed your case to a human reviewer, who will "
    "follow up with you shortly."
)


class ScriptedLLM:
    """A canned agent whose answers come from one case's `scripted` block."""

    def __init__(self, case: EvalCase, *, reply: str | None = None) -> None:
        plan = case.scripted
        if plan is None:
            raise ValueError(
                f"case {case.id!r} has no 'scripted' block, so it cannot run offline. "
                "Add one, or run with --live."
            )
        self._case = case
        self._plan: ScriptedPlan = plan
        # Used by the HITL checks, where the same case runs twice and the second
        # run's reply must answer the human's decision, not the original brief.
        self._reply_override = reply
        self.calls: list[Any] = []

    # -- the ChatAnthropic surface the nodes use --------------------------

    def with_structured_output(self, schema: type) -> "_ScriptedStructured":
        return _ScriptedStructured(self, schema)

    def invoke(self, messages: Any) -> AIMessage:
        self.calls.append("text")
        return AIMessage(content=self._reply())

    # -- what it answers ---------------------------------------------------

    def response_for(self, schema: type) -> Any:
        self.calls.append(schema)
        plan = self._plan
        if schema is RouterDecision:
            return RouterDecision(
                category=plan.category,
                sentiment=plan.sentiment,
                confidence=plan.confidence,
            )
        if schema is InvestigationNotes:
            return InvestigationNotes(notes=plan.notes or [DEFAULT_NOTE])
        if schema is ActionProposal:
            return ActionProposal(
                type=plan.action,
                amount=float(plan.amount) if plan.amount is not None else None,
                charge_id=plan.charge_id,
                subscription_id=plan.subscription_id,
                rationale=f"scripted offline agent for case {self._case.id}",
            )
        if schema is VerificationResult:
            verified = True if plan.verified is None else plan.verified
            return VerificationResult(
                verified=verified,
                reason=(
                    "the evidence supports this exact action"
                    if verified
                    else "the evidence supplied does not support this exact action"
                ),
            )
        raise AssertionError(f"scripted agent has no answer for {schema.__name__}")

    def _reply(self) -> str:
        if self._reply_override is not None:
            return self._reply_override
        if self._plan.response:
            return self._plan.response
        if self._case.expected_escalation:
            return HUMAN_REPLY
        if self._plan.action in ("refund", "partial_refund"):
            return (
                f"Your refund of {self._plan.amount} has been processed and should appear "
                "on your statement within five to ten business days."
            )
        if self._plan.action == "cancel_subscription":
            return "Your subscription has been cancelled and you will not be billed again."
        return "I checked your account and no change was needed."


class _ScriptedStructured:
    def __init__(self, llm: ScriptedLLM, schema: type) -> None:
        self._llm = llm
        self._schema = schema

    def invoke(self, messages: Any) -> Any:
        return self._llm.response_for(self._schema)
