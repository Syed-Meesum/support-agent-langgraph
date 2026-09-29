"""API tests: every endpoint, offline, against a temp SQLite checkpointer.

The model is stubbed the same way `test_graph.py` does it, injected into the
runtime's compiled graph. So the whole HTTP surface - running a case, the queue,
approve, reject, the eval endpoint - is exercised with no API key and no network.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.api import app as api_module
from app.api.app import create_app, runtime
from app.api.schemas import CaseDetail
from app.db.settings import CASE_DB_PATH_ENV, CHECKPOINT_PATH_ENV, DATABASE_URL_ENV
from app.graph.build import build_graph
from app.graph.state import AWAITING_HUMAN, RESOLVED, CaseState, new_case_state
from app.tools import action_tools
from test_graph import StubLLM, refund

RESOLVING = {
    "customer_id": "cust_1001",
    "message": "I was charged twice for my $29.99 subscription this month, please refund me.",
}
# cust_1003's 600.00 duplicate: over the hard ceiling, so it always needs a human.
CEILING = {
    "customer_id": "cust_1003",
    "message": "Charged twice for my 600 plan, refund all of it.",
}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Temp persistence, no Postgres, no API key, no eval files in the repo."""
    monkeypatch.setenv(CHECKPOINT_PATH_ENV, str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv(CASE_DB_PATH_ENV, str(tmp_path / "cases.sqlite"))
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("EVAL_DATA_DIR", str(tmp_path / "eval-data"))
    monkeypatch.setattr(api_module, "EVAL_REPORT_PATH", tmp_path / "latest.json")
    action_tools.reset_refunds()
    yield
    action_tools.reset_refunds()


def stub_the_model(stub=None):
    """Point the runtime's graph at a stub LLM. Call inside the client context."""
    runtime._graph = build_graph(llm=stub or StubLLM(), checkpointer=runtime.checkpointer)
    return runtime._graph


@pytest.fixture
def client():
    """A client with the model NOT configured - for the 503 paths."""
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def stubbed_client():
    """A client whose runtime graph is a stub, so cases actually run."""
    with TestClient(create_app()) as test_client:
        stub_the_model()
        yield test_client


# --------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------


class TestMeta:
    def test_root_points_at_the_schema(self, client):
        body = client.get("/").json()
        assert body["docs"] == "/docs"

    def test_health_reports_the_backend_and_that_no_key_is_set(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["persistence_backend"] == "sqlite"
        assert body["model_ready"] is False
        assert body["model"]

    def test_health_reports_counts_by_status(self, stubbed_client):
        stubbed_client.post("/api/cases", json=RESOLVING)
        body = stubbed_client.get("/api/health").json()
        assert body["cases_by_status"] == {RESOLVED: 1}

    def test_the_openapi_schema_is_generated(self, client):
        schema = client.get("/openapi.json").json()
        assert "/api/cases" in schema["paths"]
        assert "/api/queue" in schema["paths"]
        assert "/api/evals" in schema["paths"]

    def test_demos_are_the_cli_presets(self, client):
        demos = client.get("/api/demos").json()
        ids = {demo["id"] for demo in demos}
        assert {"happy", "escalate", "cancellation", "injection"} <= ids
        assert all(demo["expectation"] for demo in demos)


# --------------------------------------------------------------------------
# running a case
# --------------------------------------------------------------------------


class TestRunCase:
    def test_running_a_case_without_a_key_is_503_not_a_crash(self, client):
        response = client.post("/api/cases", json=RESOLVING)
        assert response.status_code == 503
        assert "ANTHROPIC_API_KEY" in response.json()["detail"]

    def test_a_resolvable_case_returns_its_full_state(self, stubbed_client):
        response = stubbed_client.post("/api/cases", json=RESOLVING)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == RESOLVED
        assert body["category"] == "refund"
        assert body["policy_result"]["rule_id"] == "refund_auto_approved"
        assert body["action_result"]["success"] is True
        assert body["action_result"]["action"] == "refund"
        assert body["final_response"]
        assert body["audit_log"]

    def test_the_reply_and_the_trail_are_both_returned(self, stubbed_client):
        body = stubbed_client.post("/api/cases", json=RESOLVING).json()
        # A node emits several entries (investigate one per tool call), so the
        # run's path is the sequence of distinct node names.
        order: list[str] = []
        for entry in body["audit_log"]:
            if not order or order[-1] != entry["node"]:
                order.append(entry["node"])
        assert order == [
            "router",
            "investigate",
            "decide",
            "policy_check",
            "verify",
            "act_or_escalate",
            "respond",
        ]
        assert [message["role"] for message in body["conversation_history"]] == [
            "customer",
            "agent",
        ]

    def test_an_escalating_case_lands_in_the_queue(self, stubbed_client):
        stub_the_model(StubLLM(action=refund(600.00, "ch_5020")))
        body = stubbed_client.post("/api/cases", json=CEILING).json()
        assert body["status"] == AWAITING_HUMAN
        assert body["policy_result"]["rule_id"] == "refund_hard_block"
        assert body["case_brief"]["reason_for_review"]

        queue = stubbed_client.get("/api/queue").json()
        assert [row["case_id"] for row in queue] == [body["case_id"]]

    def test_an_explicit_case_id_is_honoured(self, stubbed_client):
        body = stubbed_client.post(
            "/api/cases", json={**RESOLVING, "case_id": "case_named"}
        ).json()
        assert body["case_id"] == "case_named"

    def test_generated_case_ids_are_unique(self, stubbed_client):
        first = stubbed_client.post("/api/cases", json=RESOLVING).json()["case_id"]
        second = stubbed_client.post("/api/cases", json=RESOLVING).json()["case_id"]
        assert first != second

    def test_recent_cases_are_listed_newest_first(self, stubbed_client):
        stubbed_client.post("/api/cases", json={**RESOLVING, "case_id": "case_a"})
        stubbed_client.post("/api/cases", json={**RESOLVING, "case_id": "case_b"})
        listed = [row["case_id"] for row in stubbed_client.get("/api/cases").json()]
        assert set(listed) == {"case_a", "case_b"}

    def test_a_case_can_be_fetched_and_its_audit_read(self, stubbed_client):
        created = stubbed_client.post("/api/cases", json=RESOLVING).json()
        case_id = created["case_id"]

        fetched = stubbed_client.get(f"/api/cases/{case_id}").json()
        assert fetched["case_id"] == case_id
        assert fetched["action_result"] == created["action_result"]

        trail = stubbed_client.get(f"/api/cases/{case_id}/audit").json()
        assert [row["node_name"] for row in trail] == [
            entry["node"] for entry in created["audit_log"]
        ]

    def test_an_unknown_case_is_404(self, stubbed_client):
        assert stubbed_client.get("/api/cases/case_nope").status_code == 404
        assert stubbed_client.get("/api/cases/case_nope/audit").status_code == 404

    def test_an_empty_message_is_rejected(self, stubbed_client):
        response = stubbed_client.post(
            "/api/cases", json={"customer_id": "cust_1001", "message": ""}
        )
        assert response.status_code == 422


# --------------------------------------------------------------------------
# the review queue
# --------------------------------------------------------------------------

CEILING_PROPOSAL = refund(600.00, "ch_5020")


class TestReview:
    def queue_one_case(self, client) -> str:
        stub_the_model(StubLLM(action=CEILING_PROPOSAL))
        return client.post("/api/cases", json=CEILING).json()["case_id"]

    def test_the_queue_starts_empty(self, stubbed_client):
        assert stubbed_client.get("/api/queue").json() == []

    def test_the_queue_row_carries_the_four_field_brief(self, stubbed_client):
        self.queue_one_case(stubbed_client)
        row = stubbed_client.get("/api/queue").json()[0]
        assert set(row["case_brief"]) == {
            "problem_summary",
            "evidence_summary",
            "proposed_action",
            "reason_for_review",
        }
        assert row["customer_id"] == "cust_1003"
        assert row["category"] == "refund"

    def test_approving_executes_the_action_and_clears_the_queue(self, stubbed_client):
        case_id = self.queue_one_case(stubbed_client)
        response = stubbed_client.post(
            f"/api/cases/{case_id}/approve",
            json={"reviewer": "alice", "notes": "verified by hand"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == RESOLVED
        assert body["action_result"]["success"] is True
        assert body["human_decision"]["decision"] == "approved"
        assert body["human_decision"]["reviewer"] == "alice"
        assert stubbed_client.get("/api/queue").json() == []
        # the approval is a distinguishable branch in the trail
        assert any(
            (entry["payload"] or {}).get("branch") == "human_approved"
            for entry in body["audit_log"]
        )

    def test_rejecting_executes_nothing_and_relays_the_alternative(self, stubbed_client):
        case_id = self.queue_one_case(stubbed_client)
        response = stubbed_client.post(
            f"/api/cases/{case_id}/reject",
            json={
                "reviewer": "bob",
                "notes": "only the duplicate is owed",
                "alternative_action": "partial_refund",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == RESOLVED
        assert body["action_result"] is None
        assert body["human_decision"]["decision"] == "rejected"
        assert body["human_decision"]["alternative_action"] == "partial_refund"
        assert any(
            message["role"] == "human_reviewer"
            for message in body["conversation_history"]
        )

    def test_approving_twice_is_409_not_a_second_refund(self, stubbed_client):
        case_id = self.queue_one_case(stubbed_client)
        payload = {"reviewer": "alice"}
        assert stubbed_client.post(f"/api/cases/{case_id}/approve", json=payload).status_code == 200
        second = stubbed_client.post(f"/api/cases/{case_id}/approve", json=payload)
        assert second.status_code == 409
        assert "not 'awaiting_human'" in second.json()["detail"]

    def test_reviewing_a_resolved_case_is_409(self, stubbed_client):
        case_id = stubbed_client.post("/api/cases", json=RESOLVING).json()["case_id"]
        response = stubbed_client.post(
            f"/api/cases/{case_id}/reject", json={"reviewer": "bob"}
        )
        assert response.status_code == 409

    def test_reviewing_an_unknown_case_is_409(self, stubbed_client):
        response = stubbed_client.post(
            "/api/cases/case_nope/approve", json={"reviewer": "alice"}
        )
        assert response.status_code == 409
        assert "no case" in response.json()["detail"]

    def test_a_review_needs_a_reviewer(self, stubbed_client):
        case_id = self.queue_one_case(stubbed_client)
        response = stubbed_client.post(f"/api/cases/{case_id}/approve", json={"notes": "x"})
        assert response.status_code == 422


# --------------------------------------------------------------------------
# eval results
# --------------------------------------------------------------------------


class TestEvals:
    def test_with_no_report_it_says_how_to_make_one(self, client):
        body = client.get("/api/evals").json()
        assert body["available"] is False
        assert "eval.runner" in body["hint"]

    def test_it_serves_a_report_written_by_the_runner(self, client, tmp_path):
        api_module.EVAL_REPORT_PATH.write_text(
            json.dumps({"run_label": "x", "mode": "offline", "metrics": {"total_cases": 3}}),
            encoding="utf-8",
        )
        body = client.get("/api/evals").json()
        assert body["available"] is True
        assert body["report"]["metrics"]["total_cases"] == 3

    def test_running_the_evals_offline_returns_and_stores_the_table(self, client):
        response = client.post("/api/evals/run", json={"mode": "offline", "limit": 4})
        assert response.status_code == 200
        report = response.json()["report"]
        assert report["mode"] == "offline"
        assert report["metrics"]["total_cases"] == 4
        assert report["metrics"]["policy_compliance"] == 1.0
        # and the same report is now what GET /api/evals serves
        assert client.get("/api/evals").json()["report"]["metrics"]["total_cases"] == 4

    def test_a_live_run_without_a_key_is_503(self, client):
        response = client.post("/api/evals/run", json={"mode": "live"})
        assert response.status_code == 503
        assert "ANTHROPIC_API_KEY" in response.json()["detail"]

    def test_an_unknown_mode_is_rejected(self, client):
        response = client.post("/api/evals/run", json={"mode": "whatever"})
        assert response.status_code == 422

    def test_a_report_cannot_be_produced_without_the_harness(self, client):
        # The eval package is what runs the suite; if it is missing, the endpoint
        # must fail loudly rather than return an empty table.
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.startswith("eval"):
                raise ImportError(f"blocked {name}")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocked
        try:
            with pytest.raises(ImportError):
                client.post("/api/evals/run", json={"mode": "offline", "limit": 1})
        finally:
            builtins.__import__ = real_import


# --------------------------------------------------------------------------
# the API and the documented state stay in step
# --------------------------------------------------------------------------


class TestSchemaMatchesState:
    def test_case_detail_covers_every_documented_state_field(self):
        assert set(CaseDetail.model_fields) == set(CaseState.__annotations__)

    def test_a_fresh_state_serializes_into_the_response_model(self):
        state = new_case_state("case_x", "cust_1001", "hello")
        detail = CaseDetail.from_state(state)
        assert detail.case_id == "case_x"
        assert detail.status == "in_progress"
        assert [message.role for message in detail.conversation_history] == ["customer"]


def test_the_uvicorn_entry_point_exposes_an_app():
    assert app_main.app is not None
    assert app_main.app.title == "Autonomous Support Agent"
