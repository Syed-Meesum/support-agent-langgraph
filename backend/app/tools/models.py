"""Typed inputs/outputs for the four mocked tools and the two action tools."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict

AccountStatus = Literal["active", "suspended", "closed"]
ChargeStatus = Literal["succeeded", "failed", "refunded"]
SubscriptionStatus = Literal["active", "canceled", "past_due"]


class _Frozen(BaseModel):
    """Base for tool outputs: immutable value objects."""

    model_config = ConfigDict(frozen=True)


class Customer(_Frozen):
    id: str
    name: str
    email: str
    tier: str
    account_status: AccountStatus


class Charge(_Frozen):
    id: str
    customer_id: str
    amount: Decimal
    status: ChargeStatus
    created_at: datetime
    order_id: str | None = None


class Subscription(_Frozen):
    id: str
    customer_id: str
    plan: str
    status: SubscriptionStatus
    price: Decimal
    renewal_date: date


class KnowledgeHit(_Frozen):
    content: str
    source: str
    score: float


class RefundResult(_Frozen):
    success: bool
    refund_id: str
    customer_id: str
    charge_id: str
    amount: Decimal
    reason: str
    idempotency_key: str
    idempotent_replay: bool = False


class CancellationResult(_Frozen):
    success: bool
    customer_id: str
    subscription_id: str
    status: SubscriptionStatus
    canceled_at: datetime
