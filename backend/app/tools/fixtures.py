"""Loads the mock fixture data that backs the tools.

The fixtures live in backend/eval/cases/fixtures.json (see README.md).
Later these rows are seeded into Postgres by app/db/seed.py; the tools read the
same shapes from here in the meantime.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from app.tools.errors import ToolError
from app.tools.models import Charge, Customer, Subscription

DEFAULT_FIXTURES_PATH = (
    Path(__file__).resolve().parents[2] / "eval" / "cases" / "fixtures.json"
)
FIXTURES_PATH_ENV = "SUPPORT_FIXTURES_PATH"


class FixtureStore:
    """In-memory lookup over the fixture rows."""

    def __init__(
        self,
        customers: list[Customer],
        charges: list[Charge],
        subscriptions: list[Subscription],
    ) -> None:
        self.customers = customers
        self.charges = charges
        self.subscriptions = subscriptions
        self._customers_by_id = {c.id: c for c in customers}

    @classmethod
    def from_dict(cls, raw: dict) -> "FixtureStore":
        return cls(
            customers=[Customer.model_validate(row) for row in raw.get("customers", [])],
            charges=[Charge.model_validate(row) for row in raw.get("charges", [])],
            subscriptions=[
                Subscription.model_validate(row) for row in raw.get("subscriptions", [])
            ],
        )

    @classmethod
    def from_path(cls, path: Path) -> "FixtureStore":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ToolError(
                "fixtures_not_found",
                f"fixture file does not exist: {path}",
                path=str(path),
            ) from exc
        return cls.from_dict(raw)

    def require_customer(self, customer_id: str) -> Customer:
        customer = self._customers_by_id.get(customer_id)
        if customer is None:
            raise ToolError(
                "customer_not_found",
                f"no customer with id {customer_id!r}",
                customer_id=customer_id,
            )
        return customer

    def charges_for(self, customer_id: str) -> list[Charge]:
        rows = [c for c in self.charges if c.customer_id == customer_id]
        return sorted(rows, key=lambda c: c.created_at, reverse=True)

    def require_charge(self, charge_id: str) -> Charge:
        for charge in self.charges:
            if charge.id == charge_id:
                return charge
        raise ToolError(
            "charge_not_found",
            f"no charge with id {charge_id!r}",
            charge_id=charge_id,
        )

    def subscription_for(self, customer_id: str) -> Subscription | None:
        for sub in self.subscriptions:
            if sub.customer_id == customer_id:
                return sub
        return None

    def require_subscription(self, subscription_id: str) -> Subscription:
        for sub in self.subscriptions:
            if sub.id == subscription_id:
                return sub
        raise ToolError(
            "subscription_not_found",
            f"no subscription with id {subscription_id!r}",
            subscription_id=subscription_id,
        )


@lru_cache(maxsize=1)
def get_store() -> FixtureStore:
    """Return the process-wide fixture store, loaded once."""
    path = Path(os.environ.get(FIXTURES_PATH_ENV, DEFAULT_FIXTURES_PATH))
    return FixtureStore.from_path(path)


def reset_store() -> None:
    """Drop the cached store. Used by tests that swap fixture files."""
    get_store.cache_clear()
