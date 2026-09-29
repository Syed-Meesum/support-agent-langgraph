"""Mocked read tools: get_customer, get_charges, get_subscription."""

from __future__ import annotations

from app.tools.errors import ToolError
from app.tools.fixtures import get_store
from app.tools.models import Charge, Customer, Subscription


def _require_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolError("invalid_argument", f"{name} must be a non-empty string")
    return value.strip()


def get_customer(customer_id: str) -> Customer:
    """Return the customer record, or raise if the customer does not exist."""
    customer_id = _require_id(customer_id, "customer_id")
    return get_store().require_customer(customer_id)


def get_charges(customer_id: str) -> list[Charge]:
    """Return the customer's charges, newest first.

    An existing customer with no charges yields an empty list - that is real
    data, not a fabricated default. An unknown customer raises.
    """
    customer_id = _require_id(customer_id, "customer_id")
    store = get_store()
    store.require_customer(customer_id)
    return store.charges_for(customer_id)


def get_subscription(customer_id: str) -> Subscription | None:
    """Return the customer's subscription, or None if they have none."""
    customer_id = _require_id(customer_id, "customer_id")
    store = get_store()
    store.require_customer(customer_id)
    return store.subscription_for(customer_id)
