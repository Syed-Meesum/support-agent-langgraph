"""HITL + persistence tests: the queue, the resume flow, and the store.

Everything runs offline against a temp SQLite checkpointer and case store, using
the same stub LLM `test_graph.py` uses. No API key, no services.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import pytest

from app.db import backend, open_case_store, open_checkpointer
from app.db.settings import CASE_DB_PATH_ENV, CHECKPOINT_PATH_ENV, DATABASE_URL_ENV
from app.graph import hitl
from app.graph import run as runner
from app.graph.build import build_graph
from app.graph.state import AWAITING_HUMAN, RESOLVED, Message
from app.tools import action_tools
from test_graph import StubLLM, cancel, refund

# cust_1003's 600.00 duplicate - over the hard ceiling, so it always escalates.
CEILING_CASE = {"customer_id": "cust_1003", "message": "Charged twice for my 600 plan, refund all of it."}
CANCEL_CASE = {"customer_id": "cust_1005", "message": "Please cancel my subscription."}
RESOLVING_CASE = {
    "customer_id": "cust_1001",
    "message": "I was charged twice for my $29.99 subscription this month, please refund me.",
}


@pytest.fixture(autouse=True)
def _isolated_persistence(tmp_path, monkeypatch):
    """Point both persistence layers at a temp dir and use SQLite, not Postgres."""
    monkeypatch.setenv(CHECKPOINT_PATH_ENV, str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv(CASE_DB_PATH_ENV, str(tmp_path / "cases.sqlite"))
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)
    action_tools.reset_refunds()
    yield
    action_tools.reset_refunds()


@contextmanager
def persistent(stub):
    """A compiled graph + case store over the same temp persistence."""
    with open_checkpointer() as checkpointer, open_case_store() as store:
        yield build_graph(llm=stub, checkpointer=checkpointer), store


def run_ceiling_case(graph, store, case_id="case_hitl"):
    return hitl.run_case(graph=graph, store=store, case_id=case_id, **CEILING_CASE)


# --------------------------------------------------------------------------
# the store and the queue
# --------------------------------------------------------------------------


class TestQueueAndPersistence:
    def test_sqlite_is_the_default_backend(self):
        # No DATABASE_URL in the test env, so nothing external is needed.
        assert backend() == "sqlite"

    def test_a_resolved_case_is_persisted_but_never_queued(self):
        with persistent(StubLLM()) as (graph, store):
            state = hitl.run_case(
                graph=graph,
                store=store,
                case_id="case_ok",
                customer_id=RESOLVING_CASE["customer_id"],
                message=RESOLVING_CASE["message"],
            )
            assert state["status"] == RESOLVED
            assert store.count() == 1
            assert hitl.list_queue(store) == []
            assert store.get_row("case_ok")["status"] == RESOLVED

    def test_an_escalated_case_lands_in_the_queue_with_its_brief(self):
        with persistent(StubLLM(action=refund(600.00, "ch_5020"))) as (graph, store):
            state = run_ceiling_case(graph, store)
            assert state["status"] == AWAITING_HUMAN

            queue = hitl.list_queue(store)
            assert [row["case_id"] for row in queue] == ["case_hitl"]
            row = queue[0]
            assert row["customer_id"] == "cust_1003"
            assert row["category"] == "refund"
            assert row["case_brief"]["proposed_action"]["type"] == "refund"
            assert row["case_brief"]["reason_for_review"] == state["escalation_reason"]
            # the brief is what the reviewer reads: all four documented fields
            assert set(row["case_brief"]) == {
                "problem_summary",
                "evidence_summary",
                "proposed_action",
                "reason_for_review",
            }

    def test_the_snapshot_round_trips_through_the_store(self):
        with persistent(StubLLM()) as (graph, store):
            state = hitl.run_case(
                graph=graph,
                store=store,
                case_id="case_round",
                customer_id=RESOLVING_CASE["customer_id"],
                message=RESOLVING_CASE["message"],
            )
            snapshot = store.get_case("case_round")
            assert snapshot["case_id"] == state["case_id"]
            assert snapshot["final_response"] == state["final_response"]
            assert snapshot["action_result"]["success"] is True
            assert snapshot["policy_result"]["rule_id"] == "refund_auto_approved"
            assert [m["role"] for m in snapshot["conversation_history"]] == ["customer", "agent"]
            assert snapshot["audit_log"]

    def test_messages_and_audit_rows_are_written_from_the_snapshot(self):
        with persistent(StubLLM()) as (graph, store):
            hitl.run_case(
                graph=graph,
                store=store,
                case_id="case_rows",
                customer_id=RESOLVING_CASE["customer_id"],
                message=RESOLVING_CASE["message"],
            )
            messages = store.messages("case_rows")
            assert [m["role"] for m in messages] == ["customer", "agent"]
            assert messages[0]["content"] == RESOLVING_CASE["message"]

            audit = store.audit_trail("case_rows")
            assert {row["node_name"] for row in audit} == {
                "router",
                "investigate",
                "decide",
                "policy_check",
                "verify",
                "act_or_escalate",
                "respond",
            }
            # payloads come back as structured JSON, not strings
            assert all(isinstance(row["payload"], dict) for row in audit)

    def test_the_store_survives_reopening(self):
        with persistent(StubLLM()) as (graph, store):
            hitl.run_case(
                graph=graph,
                store=store,
                case_id="case_reopen",
                customer_id=RESOLVING_CASE["customer_id"],
                message=RESOLVING_CASE["message"],
            )
        # a new connection to the same file still sees the case
        with open_case_store() as reopened:
            assert reopened.get_case("case_reopen")["status"] == RESOLVED
            assert reopened.count() == 1

    @pytest.mark.parametrize("action", ["approve", "reject"])
    def test_a_case_that_is_not_waiting_on_a_human_cannot_be_reviewed(self, action):
        with persistent(StubLLM()) as (graph, store):
            hitl.run_case(
                graph=graph,
                store=store,
                case_id="case_done",
                customer_id=RESOLVING_CASE["customer_id"],
                message=RESOLVING_CASE["message"],
            )
            with pytest.raises(hitl.HumanReviewError, match="nothing to"):
                getattr(hitl, action)(graph=graph, store=store, case_id="case_done", reviewer="sam")

    def test_an_unknown_case_cannot_be_reviewed(self):
        with persistent(StubLLM()) as (graph, store):
            with pytest.raises(hitl.HumanReviewError, match="no case"):
                hitl.approve(graph=graph, store=store, case_id="case_missing", reviewer="sam")


# --------------------------------------------------------------------------
# approve: the human overrides the gate and the action really runs
# --------------------------------------------------------------------------


class TestApprove:
    def test_approving_executes_the_action_the_gate_refused(self):
        with persistent(StubLLM(action=refund(600.00, "ch_5020"))) as (graph, store):
            queued = run_ceiling_case(graph, store)
            # the gate said no
            assert queued["policy_result"]["approved"] is False
            assert queued["policy_result"]["requires_human"] is True
            assert queued["action_result"] is None

            state = hitl.approve(
                graph=graph, store=store, case_id="case_hitl", reviewer="alice"
            )
            assert state["status"] == RESOLVED
            assert state["verified"] is False  # verify was skipped, not passed
            assert state["action_result"]["success"] is True
            assert state["action_result"]["refund_id"].startswith("rf_")
            assert state["action_result"]["amount"] == "600.00"
            assert state["final_response"]

    def test_approving_clears_the_queue_and_updates_the_snapshot(self):
        with persistent(StubLLM(action=refund(600.00, "ch_5020"))) as (graph, store):
            run_ceiling_case(graph, store)
            hitl.approve(graph=graph, store=store, case_id="case_hitl", reviewer="alice")
            assert hitl.list_queue(store) == []
            assert store.get_case("case_hitl")["status"] == RESOLVED

    def test_the_audit_trail_records_who_approved_what(self):
        with persistent(StubLLM(action=refund(600.00, "ch_5020"))) as (graph, store):
            run_ceiling_case(graph, store)
            state = hitl.approve(
                graph=graph,
                store=store,
                case_id="case_hitl",
                reviewer="alice",
                notes="confirmed duplicate",
            )
            reviews = [
                entry
                for entry in state["audit_log"]
                if entry["event_type"] == "human_decision"
                and entry["node"] == "human_review"
            ]
            assert len(reviews) == 1
            assert reviews[0]["payload"]["decision"] == "approved"
            assert reviews[0]["payload"]["reviewer"] == "alice"
            assert reviews[0]["payload"]["notes"] == "confirmed duplicate"

            executions = [
                entry
                for entry in state["audit_log"]
                if entry["node"] == "act_or_escalate" and entry["event_type"] == "tool_call"
            ]
            assert executions[-1]["payload"]["branch"] == "human_approved"

    def test_the_reviewers_note_reaches_the_conversation(self):
        with persistent(StubLLM(action=refund(600.00, "ch_5020"))) as (graph, store):
            run_ceiling_case(graph, store)
            state = hitl.approve(
                graph=graph, store=store, case_id="case_hitl", reviewer="alice", notes="checked"
            )
            # The customer first hears "a person is looking at this" (the holding
            # reply the escalating run ends on), then hears the outcome.
            assert [m.role for m in state["conversation_history"]] == [
                "customer",
                "agent",
                "human_reviewer",
                "agent",
            ]
            note = state["conversation_history"][2].content
            assert "alice approved" in note
            assert "checked" in note

    def test_resuming_does_not_replay_the_earlier_nodes(self):
        # The point of real checkpointing: the second run continues from the
        # checkpoint instead of re-classifying and re-investigating the case.
        stub = StubLLM(action=refund(600.00, "ch_5020"))
        with persistent(stub) as (graph, store):
            run_ceiling_case(graph, store)
            before = len(stub.calls)
            hitl.approve(graph=graph, store=store, case_id="case_hitl", reviewer="alice")
            assert stub.calls[before:] == ["text"]  # only `respond` ran

    def test_the_post_approval_reply_is_not_told_the_case_is_still_escalating(self):
        # A resolved case keeps its escalation_reason as history; passing it to the
        # reply as "still escalating" would make the customer-facing message
        # contradict the refund that was just issued.
        stub = StubLLM(action=refund(600.00, "ch_5020"))
        with persistent(stub) as (graph, store):
            run_ceiling_case(graph, store)
            hitl.approve(graph=graph, store=store, case_id="case_hitl", reviewer="alice")
            prompt = stub.last_messages[-1][1]
            assert "human_review" in prompt
            assert "executed_action" in prompt
            assert "escalated_to_human" not in prompt

    def test_the_escalating_reply_is_told_a_human_is_reviewing_it(self):
        stub = StubLLM(action=refund(600.00, "ch_5020"))
        with persistent(stub) as (graph, store):
            run_ceiling_case(graph, store)
            prompt = stub.last_messages[-1][1]
            assert "escalated_to_human" in prompt
            assert "human_review" not in prompt

    def test_a_cancellation_can_be_approved(self):
        # cancellation_requires_confirmation means this path always queues.
        with persistent(StubLLM(action=cancel("sub_9005"))) as (graph, store):
            queued = hitl.run_case(
                graph=graph, store=store, case_id="case_cancel", **CANCEL_CASE
            )
            assert queued["status"] == AWAITING_HUMAN
            assert queued["policy_result"]["rule_id"] == "cancellation_requires_confirmation"

            state = hitl.approve(
                graph=graph, store=store, case_id="case_cancel", reviewer="bob"
            )
            assert state["status"] == RESOLVED
            assert state["action_result"]["success"] is True
            assert state["action_result"]["subscription_id"] == "sub_9005"


# --------------------------------------------------------------------------
# reject: nothing executes, the human's resolution is sent
# --------------------------------------------------------------------------


class TestReject:
    def test_rejecting_never_executes_the_action(self):
        with persistent(StubLLM(action=refund(600.00, "ch_5020"))) as (graph, store):
            run_ceiling_case(graph, store)
            state = hitl.reject(
                graph=graph, store=store, case_id="case_hitl", reviewer="bob", notes="not a duplicate"
            )
            assert state["status"] == RESOLVED
            assert state["action_result"] is None

            # Nothing hid in a tool: re-issuing the same refund is not a replay.
            probe = action_tools.issue_refund("case_hitl", "cust_1003", "ch_5020", "600.00", "probe")
            assert probe.idempotent_replay is False

    def test_the_rejection_and_the_alternative_are_recorded(self):
        stub = StubLLM(action=refund(600.00, "ch_5020"))
        with persistent(stub) as (graph, store):
            run_ceiling_case(graph, store)
            state = hitl.reject(
                graph=graph,
                store=store,
                case_id="case_hitl",
                reviewer="bob",
                notes="only the second charge is owed",
                alternative_action="partial_refund",
            )
            decision = state["human_decision"]
            assert decision["decision"] == "rejected"
            assert decision["reviewer"] == "bob"
            assert decision["alternative_action"] == "partial_refund"
            # the reply was composed after the decision, from the reviewer's notes
            assert state["conversation_history"][-1].role == "agent"
            assert state["final_response"] == stub.text
            assert hitl.list_queue(store) == []

    def test_a_rejection_is_audited_under_the_human_node(self):
        with persistent(StubLLM(action=refund(600.00, "ch_5020"))) as (graph, store):
            run_ceiling_case(graph, store)
            state = hitl.reject(graph=graph, store=store, case_id="case_hitl", reviewer="bob")
            branches = [
                entry["payload"].get("branch")
                for entry in state["audit_log"]
                if entry["event_type"] == "human_decision"
            ]
            assert "human_rejected" in branches
            reviewed = [
                entry for entry in state["audit_log"] if entry["node"] == "human_review"
            ]
            assert reviewed and reviewed[0]["payload"]["decision"] == "rejected"


# --------------------------------------------------------------------------
# rendering: the queue and the resumed trail
# --------------------------------------------------------------------------


class TestRendering:
    def test_render_queue_shows_each_brief_and_the_resume_hints(self):
        with persistent(StubLLM(action=refund(600.00, "ch_5020"))) as (graph, store):
            run_ceiling_case(graph, store)
            text = runner.render_queue(store)
            assert "human-review queue: 1 case(s)" in text
            assert "case case_hitl" in text
            assert "600.00" in text
            assert "--review approve" in text
            assert "--review reject" in text

    def test_render_queue_says_empty_when_nothing_is_waiting(self):
        with persistent(StubLLM()) as (_graph, store):
            text = runner.render_queue(store)
            assert "human-review queue: 0 case(s)" in text
            assert "(empty)" in text

    def test_render_trail_shows_the_brief_before_review_and_the_decision_after(self):
        with persistent(StubLLM(action=refund(600.00, "ch_5020"))) as (graph, store):
            run_ceiling_case(graph, store)
            queued = runner.render_trail(store.get_case("case_hitl"))
            assert "status: awaiting_human" in queued
            assert "case brief" in queued
            assert "reason_for_review:" in queued

            state = hitl.approve(
                graph=graph, store=store, case_id="case_hitl", reviewer="alice", notes="checked"
            )
            text = runner.render_trail(state)
            assert "status: resolved" in text
            assert "human decision" in text
            assert "decision=approved" in text
            assert "reviewer=alice" in text
            assert "[human_review/human_decision] approved by alice" in text


# --------------------------------------------------------------------------
# the checkpointer itself
# --------------------------------------------------------------------------


class TestCheckpointerWiring:
    def test_the_graph_must_have_a_checkpointer_to_resume(self):
        # A store alone is not enough: the resume needs a checkpoint.
        with open_case_store() as store:
            graph = build_graph(llm=StubLLM(action=refund(600.00, "ch_5020")))
            run_ceiling_case(graph, store)
            with pytest.raises(hitl.HumanReviewError, match="no checkpointer"):
                hitl.approve(graph=graph, store=store, case_id="case_hitl", reviewer="alice")

    def test_a_case_can_be_resumed_after_the_process_restarts(self):
        # Two separate checkpointer/store lifetimes over the same files - which is
        # what a restart looks like.
        with persistent(StubLLM(action=refund(600.00, "ch_5020"))) as (graph, store):
            run_ceiling_case(graph, store)

        with persistent(StubLLM(action=refund(600.00, "ch_5020"))) as (graph, store):
            assert [row["case_id"] for row in hitl.list_queue(store)] == ["case_hitl"]
            state = hitl.approve(
                graph=graph, store=store, case_id="case_hitl", reviewer="alice"
            )
            assert state["status"] == RESOLVED
            assert state["action_result"]["success"] is True

    def test_the_checkpoint_keeps_the_typed_conversation(self):
        with persistent(StubLLM()) as (graph, store):
            hitl.run_case(
                graph=graph,
                store=store,
                case_id="case_typed",
                customer_id=RESOLVING_CASE["customer_id"],
                message=RESOLVING_CASE["message"],
            )
            # Reading the checkpoint deserializes it; a strict warning filter turns
            # "unregistered type" (which would mean an unreadable checkpoint later)
            # into a failure here.
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                snapshot = graph.get_state(hitl.thread_config("case_typed"))
            history = snapshot.values["conversation_history"]
            assert [m.role for m in history] == ["customer", "agent"]
            assert isinstance(history[0], Message)

    def test_the_thread_id_is_the_case_id(self):
        assert hitl.thread_config("case_42") == {"configurable": {"thread_id": "case_42"}}
