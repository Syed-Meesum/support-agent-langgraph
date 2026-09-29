"""Unit tests for the deterministic policy engine."""

from __future__ import annotations

from decimal import Decimal

import pytest
import yaml

from app.policy import PolicyRules, evaluate_policy, evaluate_policy_dict
from app.policy.config import (
    POLICY_RULES_PATH_ENV,
    load_rules,
    read_rules,
    reload_rules,
)
from app.policy.models import PolicyResult

CONFIRMED = {"sentiment": "neutral", "confidence": 0.95}


def refund_action(amount, action_type="refund"):
    return {
        "type": action_type,
        "params": {"amount": amount, "charge_id": "ch_5002"},
        "rationale": "duplicate charge",
    }


def cancel_action():
    return {
        "type": "cancel_subscription",
        "params": {"subscription_id": "sub_9005"},
        "rationale": "customer asked to cancel",
    }


def gate(action, context=None, **overrides):
    signals = {**CONFIRMED, **overrides}
    return evaluate_policy(action, context, **signals)


@pytest.fixture
def rules_file(tmp_path, monkeypatch):
    """Point the engine at a temporary rules file."""

    def _write(**overrides):
        config = {
            "refund_auto_approve_max": 50.00,
            "refund_human_review_max": 500.00,
            "cancellation_requires_confirmation": True,
            "min_confidence_threshold": 0.80,
            "angry_sentiment_escalates": True,
            "policy_exception_escalates": True,
        }
        config.update(overrides)
        path = tmp_path / "rules.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        monkeypatch.setenv(POLICY_RULES_PATH_ENV, str(path))
        reload_rules()
        return path

    yield _write
    reload_rules()


# --------------------------------------------------------------------------
# config loading
# --------------------------------------------------------------------------


class TestRulesConfig:
    def test_ships_with_the_documented_defaults(self):
        rules = load_rules()
        assert rules.refund_auto_approve_max == Decimal("50.00")
        assert rules.refund_human_review_max == Decimal("500.00")
        assert rules.cancellation_requires_confirmation is True
        assert rules.min_confidence_threshold == 0.80
        assert rules.angry_sentiment_escalates is True
        assert rules.policy_exception_escalates is True

    def test_amounts_are_exact_decimals(self):
        rules = load_rules()
        assert isinstance(rules.refund_auto_approve_max, Decimal)
        # 50.00 must not become 49.999999999999998 via float arithmetic
        assert rules.refund_auto_approve_max == Decimal("50")

    def test_env_override_is_picked_up(self, rules_file):
        rules_file(refund_auto_approve_max=25.00)
        assert load_rules().refund_auto_approve_max == Decimal("25.00")

    def test_inverted_thresholds_are_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "refund_auto_approve_max": 600.00,
                    "refund_human_review_max": 500.00,
                    "cancellation_requires_confirmation": True,
                    "min_confidence_threshold": 0.80,
                    "angry_sentiment_escalates": True,
                    "policy_exception_escalates": True,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            read_rules(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError):
            read_rules(tmp_path / "nope.yaml")

    def test_malformed_yaml_raises(self, tmp_path):
        path = tmp_path / "broken.yaml"
        path.write_text("refund_auto_approve_max: [unclosed\n", encoding="utf-8")
        with pytest.raises(ValueError):
            read_rules(path)

    def test_non_mapping_raises(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ValueError):
            read_rules(path)

    def test_missing_key_raises(self, tmp_path):
        path = tmp_path / "partial.yaml"
        path.write_text(
            yaml.safe_dump({"refund_auto_approve_max": 50.0}), encoding="utf-8"
        )
        with pytest.raises(ValueError):
            read_rules(path)

    def test_out_of_range_confidence_raises(self, rules_file):
        path = rules_file(min_confidence_threshold=1.5)
        with pytest.raises(ValueError):
            read_rules(path)


# --------------------------------------------------------------------------
# refund tiers and boundaries
# --------------------------------------------------------------------------


class TestRefundTiers:
    def test_clean_refund_is_auto_approved(self):
        result = gate(refund_action("29.99"))
        assert result.approved is True
        assert result.requires_human is False
        assert result.rule_id == "refund_auto_approved"
        assert "29.99" in result.reason

    @pytest.mark.parametrize("amount", [0.01, "1.00", 49.99, "50.00"])
    def test_at_or_below_the_limit_is_auto_approved(self, amount):
        result = gate(refund_action(amount))
        assert result.approved is True
        assert result.rule_id == "refund_auto_approved"

    @pytest.mark.parametrize("amount", ["50.01", 100.00, "499.99", "500.00"])
    def test_just_over_the_limit_needs_a_human(self, amount):
        result = gate(refund_action(amount))
        assert result.approved is False
        assert result.requires_human is True
        assert result.rule_id == "refund_human_review"

    @pytest.mark.parametrize("amount", ["500.01", 600.00, "5000.00"])
    def test_over_the_hard_ceiling_is_blocked(self, amount):
        result = gate(refund_action(amount))
        assert result.approved is False
        assert result.requires_human is True
        assert result.rule_id == "refund_hard_block"

    def test_partial_refunds_use_the_same_tiers(self):
        assert gate(refund_action("30.00", "partial_refund")).approved is True
        assert gate(refund_action("300.00", "partial_refund")).rule_id == (
            "refund_human_review"
        )
        assert gate(refund_action("900.00", "partial_refund")).rule_id == (
            "refund_hard_block"
        )

    def test_float_and_string_amounts_are_equivalent(self):
        assert gate(refund_action(29.99)).rule_id == gate(refund_action("29.99")).rule_id

    @pytest.mark.parametrize(
        "action",
        [
            {"type": "refund", "params": {}, "rationale": "r"},
            {"type": "refund", "params": {"amount": "not-a-number"}, "rationale": "r"},
            {"type": "refund", "params": {"amount": None}, "rationale": "r"},
            {"type": "refund", "params": {"amount": True}, "rationale": "r"},
            {"type": "refund", "params": {"amount": "NaN"}, "rationale": "r"},
        ],
    )
    def test_missing_or_unusable_amount_fails_closed(self, action):
        result = gate(action)
        assert result.approved is False
        assert result.requires_human is True
        assert result.rule_id == "missing_amount"

    @pytest.mark.parametrize("amount", [0, "0.00", "-1.00"])
    def test_non_positive_amount_fails_closed(self, amount):
        result = gate(refund_action(amount))
        assert result.rule_id == "invalid_amount"
        assert result.requires_human is True


# --------------------------------------------------------------------------
# escalation triggers
# --------------------------------------------------------------------------


class TestEscalationTriggers:
    def test_low_confidence_escalates(self):
        result = gate(refund_action("10.00"), confidence=0.5)
        assert result.rule_id == "low_confidence"
        assert result.requires_human is True
        assert result.approved is False

    def test_confidence_exactly_at_the_threshold_passes(self):
        assert gate(refund_action("10.00"), confidence=0.80).approved is True

    def test_missing_confidence_fails_closed(self):
        result = gate(refund_action("10.00"), confidence=None)
        assert result.rule_id == "missing_confidence"
        assert result.requires_human is True

    def test_angry_sentiment_escalates(self):
        result = gate(refund_action("10.00"), sentiment="angry")
        assert result.rule_id == "angry_sentiment"
        assert result.requires_human is True

    def test_angry_sentiment_is_case_insensitive(self):
        assert gate(refund_action("10.00"), sentiment="Angry").rule_id == (
            "angry_sentiment"
        )

    def test_missing_sentiment_fails_closed(self):
        result = gate(refund_action("10.00"), sentiment=None)
        assert result.rule_id == "missing_sentiment"
        assert result.requires_human is True

    def test_neutral_and_positive_sentiment_do_not_escalate(self):
        for sentiment in ("neutral", "positive", "frustrated"):
            assert gate(refund_action("10.00"), sentiment=sentiment).approved is True

    def test_angry_sentiment_can_be_configured_off(self, rules_file):
        rules_file(angry_sentiment_escalates=False)
        assert gate(refund_action("10.00"), sentiment="angry").approved is True

    def test_tool_error_escalates(self):
        result = gate(refund_action("10.00"), context={"tool_errors": [{"code": "x"}]})
        assert result.rule_id == "tool_error"
        assert result.requires_human is True

    def test_tool_errors_are_also_read_from_an_errors_key(self):
        result = gate(refund_action("10.00"), context={"errors": ["boom"]})
        assert result.rule_id == "tool_error"

    def test_an_empty_error_list_is_not_a_trigger(self):
        result = gate(refund_action("10.00"), context={"tool_errors": []})
        assert result.approved is True

    def test_missing_or_odd_context_is_tolerated(self):
        assert gate(refund_action("10.00"), context=None).approved is True
        assert gate(refund_action("10.00"), context="not-a-dict").approved is True

    def test_hard_blocks_beat_every_other_trigger(self):
        result = gate(
            refund_action("600.00"),
            context={"tool_errors": [{"code": "x"}]},
            sentiment="angry",
            confidence=0.10,
        )
        assert result.rule_id == "refund_hard_block"

    def test_tool_errors_beat_confidence_and_sentiment(self):
        result = gate(
            refund_action("10.00"),
            context={"tool_errors": [{"code": "x"}]},
            sentiment="angry",
            confidence=0.10,
        )
        assert result.rule_id == "tool_error"

    def test_confidence_is_checked_before_sentiment(self):
        result = gate(refund_action("10.00"), sentiment="angry", confidence=0.10)
        assert result.rule_id == "low_confidence"

    def test_triggers_beat_the_category_rule(self):
        result = gate(refund_action("10.00"), sentiment="angry")
        assert result.rule_id == "angry_sentiment"


# --------------------------------------------------------------------------
# other action types
# --------------------------------------------------------------------------


class TestOtherActions:
    def test_cancellation_always_requires_confirmation(self):
        result = gate(cancel_action())
        assert result.approved is False
        assert result.requires_human is True
        assert result.rule_id == "cancellation_requires_confirmation"

    def test_cancellation_can_be_approved_when_confirmation_is_off(self, rules_file):
        rules_file(cancellation_requires_confirmation=False)
        result = gate(cancel_action())
        assert result.approved is True
        assert result.requires_human is False
        assert result.rule_id == "cancellation_approved"

    def test_no_action_is_approved_without_gating(self):
        result = gate({"type": "no_action", "params": {}, "rationale": "nothing to do"})
        assert result.approved is True
        assert result.requires_human is False
        assert result.rule_id == "no_action"

    def test_no_action_is_not_blocked_by_missing_signals(self):
        # Nothing executes, so absent router signals are not a reason to wake a human.
        result = gate({"type": "no_action", "params": {}}, sentiment=None, confidence=None)
        assert result.approved is True

    def test_agent_proposed_escalation_goes_to_a_human(self):
        result = gate({"type": "escalate", "params": {}, "rationale": "unclear"})
        assert result.approved is False
        assert result.requires_human is True
        assert result.rule_id == "llm_escalation"

    @pytest.mark.parametrize(
        "action",
        [
            {"type": "send_invoice", "params": {}},
            {"type": "refunds", "params": {"amount": "10.00"}},
            {"params": {"amount": "10.00"}},
            {"type": "", "params": {}},
            "not-a-dict",
            None,
        ],
    )
    def test_unknown_actions_fail_closed(self, action):
        result = gate(action)
        assert result.approved is False
        assert result.requires_human is True
        assert result.rule_id == "no_matching_rule"

    def test_policy_exception_can_refuse_instead_of_escalating(self, rules_file):
        rules_file(policy_exception_escalates=False)
        result = gate({"type": "send_invoice", "params": {}})
        assert result.approved is False
        assert result.requires_human is False
        assert result.rule_id == "no_matching_rule"


# --------------------------------------------------------------------------
# engine-level guarantees
# --------------------------------------------------------------------------


class TestEngineGuarantees:
    def test_approved_never_coexists_with_requires_human(self):
        actions = [
            refund_action("10.00"),
            refund_action("100.00"),
            refund_action("600.00"),
            cancel_action(),
            {"type": "no_action", "params": {}},
            {"type": "escalate", "params": {}},
            {"type": "weird", "params": {}},
        ]
        for action in actions:
            result = gate(action)
            assert not (result.approved and result.requires_human), action

    def test_every_result_names_a_rule_and_a_reason(self):
        for action in [refund_action("10.00"), refund_action("600.00"), {"type": "x"}]:
            result = gate(action)
            assert result.rule_id
            assert result.reason

    def test_engine_never_reads_its_own_inputs_from_the_action(self):
        # A model that tries to grant itself confidence in params must be ignored.
        action = refund_action("10.00")
        action["params"].update({"confidence": 0.99, "sentiment": "neutral"})
        result = gate(action, sentiment="angry", confidence=0.10)
        assert result.rule_id == "low_confidence"

    def test_engine_is_deterministic(self):
        action = refund_action("29.99")
        first = gate(action)
        second = gate(action)
        assert first == second

    def test_result_dumps_to_the_documented_four_keys(self):
        payload = evaluate_policy_dict(
            refund_action("29.99"), None, sentiment="neutral", confidence=0.95
        )
        assert set(payload) == {"approved", "requires_human", "rule_id", "reason"}
        assert payload["approved"] is True

    def test_result_is_immutable(self):
        result = gate(refund_action("29.99"))
        assert isinstance(result, PolicyResult)
        with pytest.raises(Exception):
            result.approved = False  # type: ignore[misc]

    def test_rules_are_validated_on_load(self):
        assert isinstance(load_rules(), PolicyRules)

    def test_engine_needs_no_api_key_and_no_network(self, monkeypatch):
        # The trust story depends on this module staying pure code: it must not
        # need a model provider, a database, or any outbound call to decide.
        for key in ("ANTHROPIC_API_KEY", "LANGSMITH_API_KEY", "DATABASE_URL"):
            monkeypatch.delenv(key, raising=False)
        assert gate(refund_action("29.99")).approved is True

    def test_engine_pulls_in_no_model_or_network_client(self):
        import app.policy.engine as engine

        forbidden = {
            "anthropic",
            "openai",
            "langchain",
            "langgraph",
            "requests",
            "httpx",
            "chromadb",
        }
        assert forbidden.isdisjoint(set(engine.__dict__))
