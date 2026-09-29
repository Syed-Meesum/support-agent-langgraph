"""The hallucination check: is every claim in the reply traceable to evidence?

This is a second, cheap judge pass over each reply - the check the design asks
for. Two judges are implemented:

- `AmountGroundingJudge` - deterministic, and the default for offline runs and
  tests. It extracts every monetary amount from each sentence of the reply and
  fails the sentence if that amount never appears in the evidence. For a billing
  agent this is the sharp edge: an invented refund amount or an invented total is
  the claim that actually costs money, and it is checkable without a model. It is
  narrower than an LLM judge by design, and says so in its `name`.
- `LLMJudge` - the doc's version. Asks the model, per sentence, whether every
  factual claim is supported by the evidence blob.

Both return the same `JudgeResult`, so the metrics are computed identically and
the two can be compared on the same run.

The evidence blob is deliberately the raw material the agent was given and what
it actually did - `customer_context`, `knowledge_hits`, `investigation_notes`,
`policy_result`, `action_result`, `escalation_reason`, any human decision - not
the agent's own reply, which would make the check circular.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Protocol

from pydantic import BaseModel, Field

# Money as a customer would see it: "$29.99", "29.99", "1,200.00". Plus large bare
# integers ("refund 5000"), which is how an injected amount usually arrives.
_AMOUNT_PATTERNS = (
    re.compile(r"\$\s?\d[\d,]*(?:\.\d{1,2})?"),
    re.compile(r"(?<![\w.])\d[\d,]*\.\d{2}(?![\w])"),
    re.compile(r"(?<![\w.])\d[\d,]{2,}(?![\w])"),
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class JudgeResult:
    """Per-sentence grounding verdict for one reply."""

    sentences: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def supported_count(self) -> int:
        return len(self.sentences) - len(self.unsupported)

    @property
    def rate(self) -> float:
        """Fraction of sentences that could not be traced to evidence."""
        if not self.sentences:
            return 0.0
        return len(self.unsupported) / len(self.sentences)


class Judge(Protocol):
    name: str

    def judge(self, response: str, evidence: str) -> JudgeResult: ...


# --------------------------------------------------------------------------
# the evidence blob
# --------------------------------------------------------------------------


def evidence_blob(state: Mapping[str, Any]) -> str:
    """Everything the agent was given plus what it actually did.

    Deliberately excludes `final_response` - including the text under judgement
    would make the check trivially circular.
    """
    import json

    parts: dict[str, Any] = {
        "customer_message": state.get("customer_message"),
        "customer_context": state.get("customer_context"),
        "knowledge_hits": state.get("knowledge_hits"),
        "investigation_notes": state.get("investigation_notes"),
        "proposed_action": state.get("proposed_action"),
        "policy_result": state.get("policy_result"),
        "action_result": state.get("action_result"),
        "escalation_reason": state.get("escalation_reason"),
        "human_decision": state.get("human_decision"),
    }
    return json.dumps(parts, default=str, ensure_ascii=False)


def _amounts_in(text: str) -> set[Decimal]:
    found: set[Decimal] = set()
    for pattern in _AMOUNT_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(0).replace("$", "").replace(",", "").strip()
            try:
                found.add(Decimal(raw))
            except (InvalidOperation, ValueError):
                continue
    return found


def split_sentences(response: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(response or "") if part.strip()]


# --------------------------------------------------------------------------
# the judges
# --------------------------------------------------------------------------


class AmountGroundingJudge:
    """Deterministic: no monetary amount may appear that the evidence lacks."""

    name = "amount-grounding (deterministic)"

    def judge(self, response: str, evidence: str) -> JudgeResult:
        known = _amounts_in(evidence)
        result = JudgeResult()
        for sentence in split_sentences(response):
            result.sentences.append(sentence)
            invented = sorted(amount for amount in _amounts_in(sentence) if amount not in known)
            if invented:
                result.unsupported.append(sentence)
                result.reasons.append(
                    "amount not present in the evidence: "
                    + ", ".join(str(amount) for amount in invented)
                )
        return result


class _SentenceVerdict(BaseModel):
    sentence: str = Field(description="The sentence being checked, copied verbatim.")
    supported: bool = Field(
        description=(
            "True only if every factual claim in the sentence is directly traceable "
            "to the evidence. A claim that merely sounds plausible is not supported."
        )
    )
    reason: str = Field(description="One short line saying what is or is not supported.")


class _HallucinationVerdict(BaseModel):
    sentences: list[_SentenceVerdict] = Field(
        description="One entry per sentence of the reply, in order."
    )


JUDGE_SYSTEM_PROMPT = """\
You are auditing a customer-support agent's reply against the evidence the agent
was given. Split the reply into sentences. For each sentence, decide whether
every factual claim in it can be traced to a specific value in the evidence blob.

Rules:
- A claim is supported only if the evidence contains it. Plausible-sounding
  details that the evidence does not contain are NOT supported.
- General reassurance ("a person will follow up") is supported if the evidence
  shows the case escalated, and unsupported if it did not.
- Timing promises ("within five to ten business days") are supported if the reply
  also states an action the evidence records as executed, because they describe
  the consequence of that action rather than a separate fact.
- Quote each sentence verbatim, in order, with no omissions.
"""


class LLMJudge:
    """The doc's version: one model pass, per-sentence yes/no, against evidence."""

    name = "LLM judge"

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def judge(self, response: str, evidence: str) -> JudgeResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        verdict: _HallucinationVerdict = self._llm.with_structured_output(
            _HallucinationVerdict
        ).invoke(
            [
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"EVIDENCE:\n{evidence}\n\n"
                        f"AGENT REPLY TO AUDIT:\n{response or '(empty)'}"
                    )
                ),
            ]
        )
        result = JudgeResult()
        for item in verdict.sentences:
            result.sentences.append(item.sentence)
            if not item.supported:
                result.unsupported.append(item.sentence)
                result.reasons.append(item.reason)
        return result
