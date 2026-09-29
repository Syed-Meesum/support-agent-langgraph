"""Typed policy config and result.

The result shape is fixed by the design:
{approved, requires_human, rule_id, reason}  ->  PolicyResult.model_dump()
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_CENTS = Decimal("0.01")


def _to_money(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value.quantize(_CENTS)
    return Decimal(str(value)).quantize(_CENTS)


class PolicyRules(BaseModel):
    """The gate's rules, from policy_rules.yaml."""

    model_config = ConfigDict(frozen=True)

    refund_auto_approve_max: Decimal
    refund_human_review_max: Decimal
    cancellation_requires_confirmation: bool
    min_confidence_threshold: float = Field(ge=0.0, le=1.0)
    angry_sentiment_escalates: bool
    policy_exception_escalates: bool

    @field_validator(
        "refund_auto_approve_max", "refund_human_review_max", mode="before"
    )
    @classmethod
    def _quantize_money(cls, value: object) -> Decimal:
        try:
            return _to_money(value)
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"not a valid money amount: {value!r}") from exc

    @field_validator("refund_auto_approve_max", "refund_human_review_max")
    @classmethod
    def _non_negative(cls, value: Decimal) -> Decimal:
        if value < 0:
            raise ValueError("refund thresholds must not be negative")
        return value

    @model_validator(mode="after")
    def _thresholds_are_ordered(self) -> "PolicyRules":
        if self.refund_auto_approve_max > self.refund_human_review_max:
            raise ValueError(
                "refund_auto_approve_max must not exceed refund_human_review_max"
            )
        return self


class PolicyResult(BaseModel):
    """The decision the graph stores as `policy_result`.

    Invariant: `approved=True` means the action may proceed to `verify`. Anything
    that needs a human is `approved=False, requires_human=True` - so the
    `policy_check --approved--> verify` edge is unambiguous.
    """

    model_config = ConfigDict(frozen=True)

    approved: bool
    requires_human: bool
    rule_id: str
    reason: str
