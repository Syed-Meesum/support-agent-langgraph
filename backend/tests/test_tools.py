"""Unit tests for the four mocked tools and the two action tools."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from app.tools import action_tools, customer_tools, knowledge_tools
from app.tools.errors import ToolError


@pytest.fixture(autouse=True)
def _clean_refund_ledger():
    action_tools.reset_refunds()
    yield
    action_tools.reset_refunds()


# --------------------------------------------------------------------------
# get_customer
# --------------------------------------------------------------------------


class TestGetCustomer:
    def test_returns_typed_customer(self):
        customer = customer_tools.get_customer("cust_1001")
        assert customer.id == "cust_1001"
        assert customer.name == "Dana Reyes"
        assert customer.email == "dana.reyes@example.com"
        assert customer.tier == "pro"
        assert customer.account_status == "active"

    def test_unknown_customer_raises(self):
        with pytest.raises(ToolError) as excinfo:
            customer_tools.get_customer("cust_does_not_exist")
        assert excinfo.value.code == "customer_not_found"
        assert excinfo.value.to_dict()["error"]["code"] == "customer_not_found"

    @pytest.mark.parametrize("bad_id", ["", "   ", None, 42])
    def test_blank_id_raises(self, bad_id):
        with pytest.raises(ToolError) as excinfo:
            customer_tools.get_customer(bad_id)  # type: ignore[arg-type]
        assert excinfo.value.code == "invalid_argument"


# --------------------------------------------------------------------------
# get_charges
# --------------------------------------------------------------------------


class TestGetCharges:
    def test_returns_all_charges_newest_first(self):
        charges = customer_tools.get_charges("cust_1001")
        assert len(charges) == 3
        created = [c.created_at for c in charges]
        assert created == sorted(created, reverse=True)

    def test_duplicate_pair_is_present(self):
        charges = customer_tools.get_charges("cust_1001")
        succeeded = [c for c in charges if c.status == "succeeded"]
        assert len(succeeded) == 2
        assert {c.amount for c in succeeded} == {Decimal("29.99")}
        assert len({c.order_id for c in succeeded}) == 1

    def test_amounts_are_decimal(self):
        charge = customer_tools.get_charges("cust_1002")[0]
        assert charge.amount == Decimal("15.00")
        assert isinstance(charge.amount, Decimal)

    def test_customer_with_no_charges_returns_empty_list(self):
        assert customer_tools.get_charges("cust_1007") == []

    def test_unknown_customer_raises(self):
        with pytest.raises(ToolError) as excinfo:
            customer_tools.get_charges("cust_does_not_exist")
        assert excinfo.value.code == "customer_not_found"


# --------------------------------------------------------------------------
# get_subscription
# --------------------------------------------------------------------------


class TestGetSubscription:
    def test_returns_active_subscription(self):
        sub = customer_tools.get_subscription("cust_1001")
        assert sub is not None
        assert sub.id == "sub_9001"
        assert sub.plan == "Pro Monthly"
        assert sub.status == "active"
        assert sub.price == Decimal("29.99")

    def test_customer_without_subscription_returns_none(self):
        assert customer_tools.get_subscription("cust_1007") is None

    def test_non_active_subscription_is_returned_as_is(self):
        sub = customer_tools.get_subscription("cust_1010")
        assert sub is not None
        assert sub.status == "past_due"

    def test_unknown_customer_raises(self):
        with pytest.raises(ToolError) as excinfo:
            customer_tools.get_subscription("cust_does_not_exist")
        assert excinfo.value.code == "customer_not_found"


# --------------------------------------------------------------------------
# search_knowledge_base
# --------------------------------------------------------------------------


class TestSearchKnowledgeBase:
    """Tool-level contract only - retrieval quality lives in test_retrieval.py."""

    def test_returns_typed_hits_with_similarity_scores(self):
        hits = knowledge_tools.search_knowledge_base("refund auto approval threshold")
        assert hits
        assert "refund-policy.md" in {hit.source for hit in hits}
        assert all(hit.content.strip() for hit in hits)
        # scores are cosine similarities now, so they are bounded
        assert all(0.0 <= hit.score <= 1.0 for hit in hits)

    def test_refund_threshold_text_is_retrievable(self):
        hits = knowledge_tools.search_knowledge_base(
            "refunds above 500 require human review"
        )
        blob = " ".join(hit.content for hit in hits)
        assert "500.00" in blob

    def test_duplicate_query_retrieves_duplicate_policy(self):
        hits = knowledge_tools.search_knowledge_base("duplicate charge refund")
        assert "duplicate-charge-policy.md" in {hit.source for hit in hits}

    def test_cancellation_query_retrieves_cancellation_policy(self):
        hits = knowledge_tools.search_knowledge_base("cancel my subscription")
        assert "cancellation-policy.md" in {hit.source for hit in hits}

    def test_top_k_is_respected(self):
        assert len(knowledge_tools.search_knowledge_base("refund", top_k=1)) == 1

    def test_scores_descend(self):
        hits = knowledge_tools.search_knowledge_base("refund policy threshold human review")
        assert [hit.score for hit in hits] == sorted(
            (hit.score for hit in hits), reverse=True
        )

    def test_irrelevant_query_returns_empty_list(self):
        assert knowledge_tools.search_knowledge_base("zzzqqq xylophone") == []

    def test_blank_query_raises(self):
        with pytest.raises(ToolError) as excinfo:
            knowledge_tools.search_knowledge_base("   ")
        assert excinfo.value.code == "invalid_argument"

    def test_non_positive_top_k_raises(self):
        with pytest.raises(ToolError) as excinfo:
            knowledge_tools.search_knowledge_base("refund", top_k=0)
        assert excinfo.value.code == "invalid_argument"


# --------------------------------------------------------------------------
# issue_refund
# --------------------------------------------------------------------------


class TestIssueRefund:
    def test_happy_path(self):
        result = action_tools.issue_refund(
            case_id="case_1",
            customer_id="cust_1001",
            charge_id="ch_5002",
            amount="29.99",
            reason="duplicate charge",
        )
        assert result.success is True
        assert result.refund_id.startswith("rf_")
        assert result.amount == Decimal("29.99")
        assert result.idempotency_key == "case_1:ch_5002"
        assert result.idempotent_replay is False

    def test_float_amount_is_accepted_exactly(self):
        result = action_tools.issue_refund(
            case_id="case_1",
            customer_id="cust_1001",
            charge_id="ch_5002",
            amount=29.99,
            reason="duplicate charge",
        )
        assert result.amount == Decimal("29.99")

    def test_replaying_the_same_case_is_idempotent(self):
        first = action_tools.issue_refund(
            "case_1", "cust_1001", "ch_5002", "29.99", "duplicate charge"
        )
        second = action_tools.issue_refund(
            "case_1", "cust_1001", "ch_5002", "29.99", "duplicate charge"
        )
        assert second.refund_id == first.refund_id
        assert second.idempotent_replay is True
        assert first.idempotent_replay is False

    def test_a_different_case_does_not_replay(self):
        first = action_tools.issue_refund(
            "case_1", "cust_1001", "ch_5002", "29.99", "duplicate charge"
        )
        second = action_tools.issue_refund(
            "case_2", "cust_1001", "ch_5002", "29.99", "duplicate charge"
        )
        assert second.refund_id != first.refund_id

    def test_amount_above_the_charge_raises(self):
        with pytest.raises(ToolError) as excinfo:
            action_tools.issue_refund(
                "case_1", "cust_1001", "ch_5002", "30.00", "duplicate charge"
            )
        assert excinfo.value.code == "invalid_amount"

    @pytest.mark.parametrize("bad_amount", ["0", "0.00", "-5.00", "not-a-number", True])
    def test_non_positive_or_invalid_amount_raises(self, bad_amount):
        with pytest.raises(ToolError) as excinfo:
            action_tools.issue_refund(
                "case_1", "cust_1001", "ch_5002", bad_amount, "duplicate charge"
            )
        assert excinfo.value.code == "invalid_amount"

    def test_charge_of_another_customer_raises(self):
        with pytest.raises(ToolError) as excinfo:
            action_tools.issue_refund(
                "case_1", "cust_1002", "ch_5002", "29.99", "duplicate charge"
            )
        assert excinfo.value.code == "charge_not_found"

    def test_failed_charge_is_not_refundable(self):
        with pytest.raises(ToolError) as excinfo:
            action_tools.issue_refund(
                "case_1", "cust_1001", "ch_5003", "29.99", "failed attempt"
            )
        assert excinfo.value.code == "charge_not_refundable"

    def test_unknown_charge_raises(self):
        with pytest.raises(ToolError) as excinfo:
            action_tools.issue_refund(
                "case_1", "cust_1001", "ch_nope", "29.99", "duplicate charge"
            )
        assert excinfo.value.code == "charge_not_found"

    def test_unknown_customer_raises(self):
        with pytest.raises(ToolError) as excinfo:
            action_tools.issue_refund(
                "case_1", "cust_nope", "ch_5002", "29.99", "duplicate charge"
            )
        assert excinfo.value.code == "customer_not_found"

    def test_blank_reason_raises(self):
        with pytest.raises(ToolError) as excinfo:
            action_tools.issue_refund(
                "case_1", "cust_1001", "ch_5002", "29.99", "   "
            )
        assert excinfo.value.code == "invalid_argument"


# --------------------------------------------------------------------------
# cancel_subscription
# --------------------------------------------------------------------------


class TestCancelSubscription:
    def test_happy_path(self):
        result = action_tools.cancel_subscription("cust_1005", "sub_9005")
        assert result.success is True
        assert result.subscription_id == "sub_9005"
        assert result.status == "canceled"
        assert result.canceled_at.tzinfo is not None

    def test_unknown_subscription_raises(self):
        with pytest.raises(ToolError) as excinfo:
            action_tools.cancel_subscription("cust_1005", "sub_nope")
        assert excinfo.value.code == "subscription_not_found"

    def test_subscription_of_another_customer_raises(self):
        with pytest.raises(ToolError) as excinfo:
            action_tools.cancel_subscription("cust_1005", "sub_9001")
        assert excinfo.value.code == "subscription_not_found"

    def test_non_active_subscription_is_not_cancelable(self):
        with pytest.raises(ToolError) as excinfo:
            action_tools.cancel_subscription("cust_1010", "sub_9010")
        assert excinfo.value.code == "subscription_not_cancelable"

    def test_unknown_customer_raises(self):
        with pytest.raises(ToolError) as excinfo:
            action_tools.cancel_subscription("cust_nope", "sub_9005")
        assert excinfo.value.code == "customer_not_found"


# --------------------------------------------------------------------------
# cross-cutting: outputs must survive the JSON round trip the graph persists
# --------------------------------------------------------------------------


def test_all_tool_outputs_are_json_serializable():
    payloads = [
        customer_tools.get_customer("cust_1001").model_dump(mode="json"),
        [c.model_dump(mode="json") for c in customer_tools.get_charges("cust_1001")],
        customer_tools.get_subscription("cust_1001").model_dump(mode="json"),  # type: ignore[union-attr]
        [
            hit.model_dump(mode="json")
            for hit in knowledge_tools.search_knowledge_base("refund policy")
        ],
        action_tools.issue_refund(
            "case_1", "cust_1001", "ch_5002", "29.99", "duplicate charge"
        ).model_dump(mode="json"),
        action_tools.cancel_subscription("cust_1005", "sub_9005").model_dump(mode="json"),
    ]
    assert json.loads(json.dumps(payloads))


def test_tool_error_payload_shape():
    error = ToolError("customer_not_found", "no customer", customer_id="cust_x")
    assert error.to_dict() == {
        "ok": False,
        "error": {
            "code": "customer_not_found",
            "message": "no customer",
            "customer_id": "cust_x",
        },
    }
