"""Typed eval cases, loaded from `eval/cases/cases.json`.

A case declares what it expects, and `eval.runner` compares the graph's real
behaviour against it. Two of the expectation fields accept either one value or a
list, because a *correct* agent has more than one honest way to handle several of
these scenarios - a hard-blocked refund may be proposed then blocked, or the
agent may escalate on its own. Pinning one exact action would make the suite
assert a prompt's phrasing rather than the system's behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.graph.schemas import ActionType, Sentiment

CASES_PATH = Path(__file__).resolve().parent / "cases" / "cases.json"


class ScriptedPlan(BaseModel):
    """What the deterministic offline agent does for one case.

    Read only by `eval/scripted_llm.py`, only in offline mode. A live run ignores
    this entirely and the real model decides.
    """

    model_config = ConfigDict(frozen=True)

    category: Literal["refund", "cancellation"]
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    action: ActionType
    amount: str | None = Field(
        default=None, description="Money amount, as the exact 2dp string the graph uses."
    )
    charge_id: str | None = None
    subscription_id: str | None = None
    verified: bool | None = Field(
        default=None,
        description="What the scripted verify node answers. Defaults to True.",
    )
    notes: list[str] = Field(default_factory=list)
    response: str | None = Field(
        default=None, description="Override the scripted reply. Defaults to a grounded sentence."
    )


class EvalCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    scenario_name: str
    input_message: str
    seed_data: dict[str, Any]
    expected_action: list[str]
    expected_rule_id: list[str]
    expected_escalation: bool
    adversarial: bool = False
    why: str = ""
    scripted: ScriptedPlan | None = None

    @field_validator("expected_action", "expected_rule_id", mode="before")
    @classmethod
    def _as_list(cls, value: Any) -> Any:
        """Accept a bare string as a one-element list."""
        if isinstance(value, str):
            return [value]
        return value

    @property
    def customer_id(self) -> str:
        customer_id = self.seed_data.get("customer_id")
        if not isinstance(customer_id, str) or not customer_id:
            raise ValueError(f"case {self.id!r} has no seed_data.customer_id")
        return customer_id

    @property
    def case_id(self) -> str:
        """The id this case runs under. Stable, so runs are comparable."""
        return f"eval_{self.id}"


def load_cases(path: Path | None = None) -> list[EvalCase]:
    """Load and validate every case, in file order."""
    raw = json.loads((path or CASES_PATH).read_text(encoding="utf-8"))
    return [EvalCase.model_validate(row) for row in raw["cases"]]
