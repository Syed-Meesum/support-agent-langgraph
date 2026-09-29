"""Structured-output schemas for the five LLM nodes.

Each LLM node returns one of these rather than free text, so a node can never
return something the graph cannot read. The decide node emits `ActionProposal`,
which is then converted into the documented
`{type, params, rationale}` shape by `to_proposed_action`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ActionType = Literal[
    "refund",
    "partial_refund",
    "no_action",
    "cancel_subscription",
    "escalate",
]

Sentiment = Literal["angry", "negative", "neutral", "positive"]


class RouterDecision(BaseModel):
    """Node 1 output: classification only."""

    model_config = ConfigDict(frozen=True)

    category: Literal["refund", "cancellation"] = Field(
        description="Primary intent of the customer message."
    )
    sentiment: Sentiment = Field(description="Tone of the customer message.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in this classification. Use a low value when the message "
            "is ambiguous or does not clearly match either category."
        ),
    )


class InvestigationNotes(BaseModel):
    """Node 2 output: evidence-grounded findings, nothing speculative."""

    model_config = ConfigDict(frozen=True)

    notes: list[str] = Field(
        description=(
            "Findings, each citing the specific field it came from, e.g. "
            "'charges: two succeeded charges of 29.99 on order ord_7781 "
            "(ch_5001, ch_5002)'. No speculation. If a needed value is absent, "
            "say that it is absent."
        )
    )


class ActionProposal(BaseModel):
    """Node 3 output. Note: the model proposes; it does not decide if it is allowed."""

    model_config = ConfigDict(frozen=True)

    type: ActionType = Field(description="Exactly one action to take.")
    amount: float | None = Field(
        default=None,
        description=(
            "Amount to refund, in the customer's currency. Required for 'refund' "
            "and 'partial_refund'; must equal an amount found in the evidence."
        ),
    )
    charge_id: str | None = Field(
        default=None,
        description="The charge being refunded. Required for refund actions.",
    )
    subscription_id: str | None = Field(
        default=None,
        description="The subscription being cancelled. Required for 'cancel_subscription'.",
    )
    rationale: str = Field(
        description="Why this action, referencing the investigation notes."
    )


class VerificationResult(BaseModel):
    """Node 5 output: does the evidence support this exact action?"""

    model_config = ConfigDict(frozen=True)

    verified: bool = Field(
        description=(
            "True only if the evidence clearly supports this exact action at this "
            "exact amount for this exact customer. Anything less than a clear yes "
            "must be False."
        )
    )
    reason: str = Field(description="One line explaining the verdict.")


def to_proposed_action(proposal: ActionProposal) -> dict[str, Any]:
    """Convert the model's proposal into the documented state shape.

    Money is formatted to two decimals as a string so the policy engine parses an
    exact decimal rather than inheriting a float's binary approximation.
    """
    params: dict[str, Any] = {}
    if proposal.amount is not None:
        params["amount"] = f"{proposal.amount:.2f}"
    if proposal.charge_id:
        params["charge_id"] = proposal.charge_id
    if proposal.subscription_id:
        params["subscription_id"] = proposal.subscription_id
    return {
        "type": proposal.type,
        "params": params,
        "rationale": proposal.rationale,
    }
