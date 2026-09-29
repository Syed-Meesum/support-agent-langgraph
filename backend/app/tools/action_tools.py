"""Mocked action tools: issue_refund, cancel_subscription.

These are only called from act_or_escalate, and only after policy_check and
verify have passed (or a human has approved). They are idempotent where money
is involved so a replayed graph run cannot refund twice.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from app.tools.errors import ToolError
from app.tools.fixtures import get_store
from app.tools.models import CancellationResult, RefundResult

_CENTS = Decimal("0.01")

# idempotency_key -> the result we already produced for it
_REFUNDS: dict[str, RefundResult] = {}


def _require_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolError("invalid_argument", f"{name} must be a non-empty string")
    return value.strip()


def _money(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ToolError("invalid_amount", "amount must be a number, not a boolean")
    if isinstance(value, Decimal):
        amount = value
    elif isinstance(value, (int, float)):
        amount = Decimal(str(value))
    elif isinstance(value, str):
        try:
            amount = Decimal(value)
        except InvalidOperation as exc:
            raise ToolError("invalid_amount", f"amount {value!r} is not a number") from exc
    else:
        raise ToolError("invalid_amount", f"amount has unsupported type {type(value).__name__}")
    if not amount.is_finite():
        raise ToolError("invalid_amount", "amount must be a finite number")
    return amount.quantize(_CENTS)


def issue_refund(
    case_id: str,
    customer_id: str,
    charge_id: str,
    amount: Any,
    reason: str,
) -> RefundResult:
    """Record a mocked refund against a charge.

    `case_id` is required to build the idempotency key
    (``f"{case_id}:{charge_id}"``) so a retried run returns the first result
    instead of refunding twice.
    """
    case_id = _require_id(case_id, "case_id")
    customer_id = _require_id(customer_id, "customer_id")
    charge_id = _require_id(charge_id, "charge_id")
    reason = _require_id(reason, "reason")

    store = get_store()
    store.require_customer(customer_id)
    charge = store.require_charge(charge_id)

    # Do not disclose or act on another customer's charge.
    if charge.customer_id != customer_id:
        raise ToolError(
            "charge_not_found",
            f"no charge with id {charge_id!r} for customer {customer_id!r}",
            charge_id=charge_id,
            customer_id=customer_id,
        )
    if charge.status != "succeeded":
        raise ToolError(
            "charge_not_refundable",
            f"charge {charge_id!r} has status {charge.status!r}, only succeeded charges can be refunded",
            charge_id=charge_id,
            status=charge.status,
        )

    amount = _money(amount)
    if amount <= 0:
        raise ToolError("invalid_amount", "amount must be greater than zero", amount=str(amount))
    if amount > charge.amount:
        raise ToolError(
            "invalid_amount",
            f"amount {amount} exceeds the charge amount {charge.amount}",
            amount=str(amount),
            charge_amount=str(charge.amount),
        )

    idempotency_key = f"{case_id}:{charge_id}"
    existing = _REFUNDS.get(idempotency_key)
    if existing is not None:
        return existing.model_copy(update={"idempotent_replay": True})

    refund_id = "rf_" + uuid.uuid5(uuid.NAMESPACE_URL, idempotency_key).hex[:12]
    result = RefundResult(
        success=True,
        refund_id=refund_id,
        customer_id=customer_id,
        charge_id=charge_id,
        amount=amount,
        reason=reason,
        idempotency_key=idempotency_key,
    )
    _REFUNDS[idempotency_key] = result
    return result


def cancel_subscription(customer_id: str, subscription_id: str) -> CancellationResult:
    """Record a mocked subscription cancellation."""
    customer_id = _require_id(customer_id, "customer_id")
    subscription_id = _require_id(subscription_id, "subscription_id")

    store = get_store()
    store.require_customer(customer_id)
    subscription = store.require_subscription(subscription_id)

    if subscription.customer_id != customer_id:
        raise ToolError(
            "subscription_not_found",
            f"no subscription with id {subscription_id!r} for customer {customer_id!r}",
            subscription_id=subscription_id,
            customer_id=customer_id,
        )
    if subscription.status != "active":
        raise ToolError(
            "subscription_not_cancelable",
            f"subscription {subscription_id!r} has status {subscription.status!r}, only active subscriptions can be cancelled",
            subscription_id=subscription_id,
            status=subscription.status,
        )

    return CancellationResult(
        success=True,
        customer_id=customer_id,
        subscription_id=subscription_id,
        status="canceled",
        canceled_at=datetime.now(timezone.utc),
    )


def reset_refunds() -> None:
    """Clear the mocked refund ledger. Used by tests."""
    _REFUNDS.clear()
