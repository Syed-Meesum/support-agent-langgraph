"""Eval harness tests: the dataset, the judges, the metric maths, and one real run.

The end-to-end test drives `eval.runner.main` in offline mode, so it exercises the
whole harness - tools, policy engine, graph, judging, metrics, report writing -
with no API key and no network. That is the point of the scripted agent, and this
is the test that would catch the harness silently rotting.
"""

from __future__ import annotations

import json

import pytest

from app.graph.state import AWAITING_HUMAN, ESCALATED, RESOLVED
from eval import runner
from eval.judge import (
    AmountGroundingJudge,
    JudgeResult,
    LLMJudge,
    evidence_blob,
    split_sentences,
)
from eval.metrics import (
    CaseResult,
    build_case_result,
    escalation_ok,
    injection_ok,
    policy_ok,
    resolution_ok,
    summarize,
)
from eval.schemas import load_cases
from eval.scripted_llm import ScriptedLLM


# --------------------------------------------------------------------------
# the dataset
# --------------------------------------------------------------------------


class TestDataset:
    def test_the_dataset_loads_and_is_the_documented_size(self):
        cases = load_cases()
        # 30-50 cases is the intended size of the dataset.
        assert 30 <= len(cases) <= 50

    def test_case_ids_are_unique(self):
        ids = [case.id for case in load_cases()]
        assert len(ids) == len(set(ids))

    def test_every_case_declares_the_documented_fields(self):
        for case in load_cases():
            assert case.scenario_name
            assert case.input_message
            assert case.customer_id
            assert case.expected_action
            assert case.expected_rule_id
            assert isinstance(case.expected_escalation, bool)

    def test_every_case_can_run_offline(self):
        # A case without a scripted block would make `--offline` fail partway
        # through a run rather than up front.
        missing = [case.id for case in load_cases() if case.scripted is None]
        assert missing == []

    def test_the_documented_scenario_categories_are_all_present(self):
        cases = load_cases()
        ids = {case.id for case in cases}

        # Each of the scenario categories, by case id.
        assert "dup_29_99_auto_approved" in ids  # clean auto-approvable refund
        assert {"dup_49_99_under_limit", "dup_50_01_requires_review"} <= ids  # boundary pair
        assert "dup_600_00_hard_block" in ids  # should escalate
        assert "cancel_active_subscription" in ids  # a cancellation case
        assert "injection_refund_5000" in ids  # a prompt injection
        assert "no_charges_on_record" in ids  # missing/ambiguous data
        assert "hitl_review_then_approve" in ids  # the resume case

    def test_there_are_adversarial_cases_on_both_sides_of_the_ceiling(self):
        adversarial = [case for case in load_cases() if case.adversarial]
        assert len(adversarial) >= 3
        # One adversarial case must be expected to resolve, so the metric cannot
        # be passed by simply escalating everything adversarial.
        assert any(not case.expected_escalation for case in adversarial)

    def test_both_threshold_boundaries_are_covered_by_a_pair(self):
        by_id = {case.id: case for case in load_cases()}
        under = by_id["dup_49_99_under_limit"]
        over = by_id["dup_50_01_requires_review"]
        assert under.expected_rule_id == ["refund_auto_approved"]
        assert over.expected_rule_id == ["refund_human_review"]
        assert under.expected_escalation is False
        assert over.expected_escalation is True


# --------------------------------------------------------------------------
# the judges
# --------------------------------------------------------------------------


class TestAmountGroundingJudge:
    def test_a_grounded_amount_passes(self):
        verdict = AmountGroundingJudge().judge(
            "Your refund of 29.99 has been processed.",
            json.dumps({"action_result": {"amount": "29.99"}}),
        )
        assert verdict.unsupported == []
        assert verdict.rate == 0.0

    def test_an_invented_amount_is_caught(self):
        verdict = AmountGroundingJudge().judge(
            "You will receive 59.98 in total.",
            json.dumps({"action_result": {"amount": "29.99"}}),
        )
        assert verdict.unsupported == ["You will receive 59.98 in total."]
        assert verdict.rate == 1.0

    def test_one_bad_sentence_among_good_ones_is_the_only_one_flagged(self):
        evidence = json.dumps({"customer_context": {"charges": [{"amount": "12.00"}]}})
        verdict = AmountGroundingJudge().judge(
            "Your refund of 12.00 has been processed. I have also credited 5000 as a gift.",
            evidence,
        )
        assert len(verdict.sentences) == 2
        assert verdict.unsupported == [
            "I have also credited 5000 as a gift."
        ]
        assert verdict.rate == 0.5

    def test_a_dollar_sign_and_thousands_separator_are_understood(self):
        verdict = AmountGroundingJudge().judge(
            "We refunded $1,200.00 to you.",
            json.dumps({"action_result": {"amount": "1200.00"}}),
        )
        assert verdict.unsupported == []

    def test_timing_promises_are_not_treated_as_amounts(self):
        verdict = AmountGroundingJudge().judge(
            "It should appear within five to ten business days.",
            json.dumps({"action_result": {"amount": "29.99"}}),
        )
        assert verdict.unsupported == []

    def test_ids_in_the_evidence_do_not_leak_in_as_amounts(self):
        # ord_7781 and ch_5002 must not be read as the amounts 7781 / 5002.
        verdict = AmountGroundingJudge().judge(
            "Your refund of 29.99 is on the way.",
            "charges: ch_5002 and ord_7781 both 29.99",
        )
        assert verdict.unsupported == []

    def test_split_sentences_handles_newlines(self):
        assert split_sentences("One. Two!\nThree?") == ["One.", "Two!", "Three?"]


class _JudgeLLM:
    """Returns a canned hallucination verdict; records the call."""

    def __init__(self, verdict) -> None:
        self._verdict = verdict
        self.calls = 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.calls += 1
        self.messages = messages
        return self._verdict


class TestLLMJudge:
    def test_it_reports_unsupported_sentences_from_the_model(self):
        from eval.judge import _HallucinationVerdict, _SentenceVerdict

        llm = _JudgeLLM(
            _HallucinationVerdict(
                sentences=[
                    _SentenceVerdict(sentence="A.", supported=True, reason="in evidence"),
                    _SentenceVerdict(sentence="B.", supported=False, reason="invented"),
                ]
            )
        )
        verdict = LLMJudge(llm).judge("A. B.", evidence="{}")
        assert verdict.sentences == ["A.", "B."]
        assert verdict.unsupported == ["B."]
        assert verdict.reasons == ["invented"]
        assert llm.calls == 1

    def test_the_evidence_is_what_the_prompt_contains(self):
        from eval.judge import _HallucinationVerdict

        llm = _JudgeLLM(_HallucinationVerdict(sentences=[]))
        LLMJudge(llm).judge("A.", evidence="SENTINEL-EVIDENCE")
        text = " ".join(str(message.content) for message in llm.messages)
        assert "SENTINEL-EVIDENCE" in text


class TestEvidenceBlob:
    def test_it_excludes_the_reply_under_judgement(self):
        blob = evidence_blob(
            {
                "action_result": {"amount": "29.99"},
                "final_response": "I refunded you 999.99.",
            }
        )
        assert "29.99" in blob
        assert "999.99" not in blob


# --------------------------------------------------------------------------
# the metric maths
# --------------------------------------------------------------------------


def result(
    *,
    status: str,
    expected_escalation: bool,
    adversarial: bool = False,
    rule_id: str | None = "refund_auto_approved",
    expected_rule_id: list[str] | None = None,
    executed: bool = False,
    hallucinations: int = 0,
    sentences: int = 1,
) -> CaseResult:
    return CaseResult(
        case_id="c",
        scenario_name="s",
        adversarial=adversarial,
        expected_escalation=expected_escalation,
        expected_action=["refund"],
        expected_rule_id=expected_rule_id or ["refund_auto_approved"],
        status=status,
        actual_rule_id=rule_id,
        executed=executed,
        resolution_ok=resolution_ok(status, expected_escalation),
        policy_ok=policy_ok(rule_id, expected_rule_id or ["refund_auto_approved"]),
        escalation_ok=escalation_ok(status, expected_escalation),
        hallucinated=["x"] * hallucinations,
        sentence_count=sentences,
    )


class TestMetricHelpers:
    @pytest.mark.parametrize(
        "status,expected,ok",
        [
            (RESOLVED, False, True),
            (RESOLVED, True, False),
            (AWAITING_HUMAN, True, True),
            (AWAITING_HUMAN, False, False),
            (ESCALATED, True, True),
        ],
    )
    def test_escalation_ok_in_both_directions(self, status, expected, ok):
        assert escalation_ok(status, expected) is ok

    def test_resolution_ok_requires_resolved_for_a_non_escalation_case(self):
        assert resolution_ok(RESOLVED, False) is True
        assert resolution_ok(AWAITING_HUMAN, False) is False
        # An escalation case is "resolved" by escalating, not by resolving.
        assert resolution_ok(AWAITING_HUMAN, True) is True
        assert resolution_ok(RESOLVED, True) is False

    def test_policy_ok_accepts_any_expected_rule(self):
        assert policy_ok("a", ["a", "b"]) is True
        assert policy_ok("c", ["a", "b"]) is False
        assert policy_ok(None, ["a"]) is False

    def test_injection_ok_rejects_an_unapproved_execution(self):
        # The architectural rule: nothing executes without the gate's approval.
        assert injection_ok(
            {"policy_result": {"approved": False}, "action_result": {"success": True}}
        ) is False
        assert injection_ok(
            {"policy_result": {"approved": True}, "action_result": {"success": True}}
        ) is True
        assert injection_ok({"status": AWAITING_HUMAN}) is True
        assert injection_ok({"status": RESOLVED}) is False


class TestSummarize:
    def test_all_passing_gives_full_marks(self):
        metrics = summarize(
            [
                result(status=RESOLVED, expected_escalation=False),
                result(status=AWAITING_HUMAN, expected_escalation=True),
            ]
        )
        assert metrics.resolution_rate == 1.0
        assert metrics.policy_compliance == 1.0
        assert metrics.escalation_accuracy == 1.0
        assert metrics.hallucination_rate == 0.0
        assert metrics.failures == []
        assert metrics.passed_cases == 2

    def test_resolution_rate_is_denominated_on_resolvable_cases_only(self):
        metrics = summarize(
            [
                result(status=RESOLVED, expected_escalation=False),
                result(status=AWAITING_HUMAN, expected_escalation=False),
                # correctly escalated - must not enter the resolution denominator
                result(status=AWAITING_HUMAN, expected_escalation=True),
            ]
        )
        assert metrics.resolvable_cases == 2
        assert metrics.resolved_cases == 1
        assert metrics.resolution_rate == 0.5
        # the middle case escalated when it should have resolved, so it is the
        # one case that fails escalation accuracy - the other two are correct
        assert metrics.escalation_accuracy == pytest.approx(2 / 3)

    def test_hallucination_rate_is_sentences_not_cases(self):
        metrics = summarize(
            [
                result(status=RESOLVED, expected_escalation=False, hallucinations=1, sentences=4),
                result(status=RESOLVED, expected_escalation=False, hallucinations=0, sentences=6),
            ]
        )
        assert metrics.hallucinated_sentences == 1
        assert metrics.judged_sentences == 10
        assert metrics.hallucination_rate == 0.1

    def test_injection_resistance_only_counts_adversarial_cases(self):
        metrics = summarize(
            [
                result(status=RESOLVED, expected_escalation=False, adversarial=False),
                result(status=AWAITING_HUMAN, expected_escalation=True, adversarial=True),
            ]
        )
        assert metrics.adversarial_cases == 1
        assert metrics.injection_resistance == 1.0

    def test_failures_name_the_case_and_the_reason(self):
        metrics = summarize(
            [
                result(
                    status=RESOLVED,
                    expected_escalation=True,
                    rule_id="refund_auto_approved",
                    expected_rule_id=["refund_human_review"],
                )
            ]
        )
        assert len(metrics.failures) == 1
        failure = metrics.failures[0]
        assert "c:" in failure
        assert "expected_escalation=True" in failure
        assert "expected one of ['refund_human_review']" in failure

    def test_an_error_is_reported_verbatim(self):
        broken = result(status=RESOLVED, expected_escalation=False)
        broken.error = "ValueError: boom"
        metrics = summarize([broken])
        assert metrics.failures == ["c: ValueError: boom"]


class TestBuildCaseResult:
    def test_it_maps_a_finished_state(self):
        case = load_cases()[0]
        state = {
            "status": RESOLVED,
            "proposed_action": {"type": "refund", "params": {"amount": "29.99"}},
            "policy_result": {"approved": True, "rule_id": "refund_auto_approved"},
            "action_result": {"success": True, "amount": "29.99"},
            "final_response": "Done.",
        }
        built = build_case_result(
            case=case, state=state, judge_result=JudgeResult(sentences=["Done."])
        )
        assert built.status == RESOLVED
        assert built.actual_action == "refund"
        assert built.actual_rule_id == "refund_auto_approved"
        assert built.executed is True
        assert built.sentence_count == 1
        assert built.passed is True


# --------------------------------------------------------------------------
# the scripted agent
# --------------------------------------------------------------------------


class TestScriptedLLM:
    def test_it_refuses_a_case_with_no_scripted_block(self):
        case = load_cases()[0].model_copy(update={"scripted": None})
        with pytest.raises(ValueError, match="no 'scripted' block"):
            ScriptedLLM(case)

    def test_it_answers_every_schema_the_nodes_ask_for(self):
        from app.graph.schemas import (
            ActionProposal,
            InvestigationNotes,
            RouterDecision,
            VerificationResult,
        )

        llm = ScriptedLLM(load_cases()[0])
        for schema in (RouterDecision, InvestigationNotes, ActionProposal, VerificationResult):
            assert isinstance(llm.with_structured_output(schema).invoke([]), schema)

    def test_the_reply_is_a_grounded_refund_claim_when_the_case_resolves(self):
        case = load_cases()[0]
        reply = ScriptedLLM(case).invoke([]).content
        assert case.scripted.amount in reply

    def test_the_reply_promises_a_human_when_the_case_escalates(self):
        case = next(c for c in load_cases() if c.id == "dup_600_00_hard_block")
        reply = ScriptedLLM(case).invoke([]).content
        assert "human reviewer" in reply


# --------------------------------------------------------------------------
# the runner, end to end, offline
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _scratch_eval_persistence(tmp_path, monkeypatch):
    """Keep eval scratch files (and the demo's) out of the repo during tests."""
    monkeypatch.setenv(runner.EVAL_DATA_DIR_ENV, str(tmp_path / "eval-data"))
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv("CASE_DB_PATH", str(tmp_path / "cases.sqlite"))


class TestRunnerOffline:
    def test_a_limited_offline_run_produces_the_metrics_table(self, capsys):
        exit_code = runner.main(["--offline", "--limit", "5", "--no-json", "-q"])
        out = capsys.readouterr().out.strip()
        assert exit_code == 0
        for metric in (
            "Resolution rate",
            "Policy compliance",
            "Escalation accuracy",
            "Hallucination rate",
            "Injection resistance",
        ):
            assert metric in out

    def test_an_offline_run_says_it_is_not_a_model_measurement(self, capsys):
        runner.main(["--offline", "--limit", "2", "--no-json", "-q"])
        out = capsys.readouterr().out
        assert "NOT a model measurement" in out

    def test_the_whole_dataset_passes_offline(self, capsys):
        """The scripted agent is derived from the expectations, so it must agree."""
        exit_code = runner.main(["--offline", "--no-json", "-q"])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "no failures" in out
        assert "cases passing every check: 39/39" in out

    def test_it_writes_a_json_report_the_api_can_serve(self, tmp_path, capsys):
        target = tmp_path / "report.json"
        runner.main(["--offline", "--limit", "3", "--json", str(target), "-q"])
        written = json.loads(target.read_text(encoding="utf-8"))
        assert written["mode"] == "offline"
        assert written["judge"] == "amount-grounding (deterministic)"
        assert written["metrics"]["total_cases"] == 3
        assert len(written["results"]) == 3

    def test_a_single_case_can_be_selected(self, capsys):
        runner.main(
            ["--offline", "--case", "dup_600_00_hard_block", "--no-json", "-q"]
        )
        out = capsys.readouterr().out
        assert "cases        1" in out

    def test_an_unknown_case_id_is_an_error(self, capsys):
        assert runner.main(["--offline", "--case", "nope", "--no-json", "-q"]) == 2
        assert "no eval case with id 'nope'" in capsys.readouterr().err

    def test_live_without_a_key_refuses_rather_than_skipping(self, monkeypatch, capsys):
        from app.config import reload_env

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        reload_env()
        assert runner.main(["--live", "--no-json", "-q"]) == 2
        assert "ANTHROPIC_API_KEY" in capsys.readouterr().err

    def test_mode_defaults_to_offline_without_a_key(self, monkeypatch):
        from app.config import reload_env

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        reload_env()
        assert runner.resolve_mode(None) == runner.OFFLINE


class TestRunnerHitlChecks:
    def test_every_hitl_check_passes_offline(self):
        case = next(c for c in load_cases() if c.id == runner.HITL_CASE_ID)
        checks = runner.run_hitl_checks(case, mode=runner.OFFLINE)
        failed = [name for name, ok, _ in checks if not ok]
        assert failed == []
        assert len(checks) == 12

    def test_the_check_covers_approve_and_reject(self):
        case = next(c for c in load_cases() if c.id == runner.HITL_CASE_ID)
        names = " ".join(name for name, _, _ in runner.run_hitl_checks(case, mode=runner.OFFLINE))
        assert "approval executes" in names
        assert "rejection executes nothing" in names
        assert "resume did not replay" in names

    def test_repeated_runs_are_identical(self):
        """A stale checkpoint must not change the second run's outcome."""
        case = next(c for c in load_cases() if c.id == runner.HITL_CASE_ID)
        first = runner.run_hitl_checks(case, mode=runner.OFFLINE)
        second = runner.run_hitl_checks(case, mode=runner.OFFLINE)
        assert [ok for _, ok, _ in first] == [ok for _, ok, _ in second]
        assert all(ok for _, ok, _ in second)


class TestReportRendering:
    def test_an_empty_denominator_prints_na_not_a_fake_zero(self):
        from eval.metrics import RunReport

        report = RunReport(run_label="t", mode=runner.OFFLINE, judge="j")
        report.results = [result(status=AWAITING_HUMAN, expected_escalation=True)]
        report.finish()
        rendered = runner.render_report(report)
        # No case should auto-resolve and none is adversarial, so both of those
        # lines must read n/a rather than inventing a 0%.
        resolution_line = next(
            line for line in rendered.splitlines() if line.startswith("  Resolution rate")
        )
        injection_line = next(
            line for line in rendered.splitlines() if line.startswith("  Injection resistance")
        )
        assert "n/a" in resolution_line
        assert "n/a" in injection_line
