"""Run the eval dataset through the graph and print the metrics table.

    cd backend
    python -m eval.runner                 # real model if ANTHROPIC_API_KEY is set
    python -m eval.runner --live          # require the real model
    python -m eval.runner --offline       # scripted agent, no key, no network
    python -m eval.runner --hitl          # also drive the approve/reject resume
    python -m eval.runner --case dup_50_01_requires_review

This is the project's headline artifact - a single command producing real numbers -
so the failure modes it can report are the point, not a nuisance: the
suite prints every case that disagreed with its expectation, and writes the whole
run to `eval/results/latest.json` (which `GET /api/evals` serves to the UI).

**Two modes, and they measure different things.**

- `--live` runs the real graph over Claude. These are the numbers that say
  something about the system's quality. They cost money and need a key.
- `--offline` swaps the model for `eval.scripted_llm.ScriptedLLM`, whose choices
  come from each case's `scripted` block. These numbers measure the harness, the
  tools, the policy engine, the graph wiring and the metric maths. They are NOT
  model measurements, and the header says so on every run so a saved report can
  never be quoted as one.

The offline agent deliberately obeys injection attempts so that the injection
metric proves `policy_check` blocks them - see `eval/scripted_llm.py`.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from app.config import load_env
from app.db import open_case_store, open_checkpointer
from app.graph import hitl
from app.graph.build import build_graph
from app.graph.llm import MissingAPIKeyError, get_llm, model_name
from app.graph.state import AWAITING_HUMAN, RESOLVED, new_case_state
from app.tools import action_tools
from eval import judge as judge_mod
from eval.metrics import CaseResult, RunReport, build_case_result
from eval.schemas import EvalCase, load_cases
from eval.scripted_llm import ScriptedLLM

LIVE = "live"
OFFLINE = "offline"
BOTH = (LIVE, OFFLINE)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_JSON_PATH = RESULTS_DIR / "latest.json"

# The eval's own HITL persistence, kept away from the demo's .data files so an
# eval run does not litter the queue the demo UI shows.
EVAL_CHECKPOINT_ENV = "CHECKPOINT_PATH"
EVAL_CASE_DB_ENV = "CASE_DB_PATH"
# Where the eval's scratch persistence lives. Overridable so tests can send it to
# a temp dir instead of writing into backend/.data.
EVAL_DATA_DIR_ENV = "EVAL_DATA_DIR"
BACKEND_ROOT = Path(__file__).resolve().parents[1]

HITL_CASE_ID = "hitl_review_then_approve"
HITL_REJECT_ALTERNATIVE = "partial_refund"

# Replies for the two HITL runs. The reject run must not claim the original
# proposal happened - that is the behaviour the check exists to prove.
HITL_APPROVE_REPLY = (
    "A reviewer approved the refund. Your refund of 50.01 has been processed and "
    "should appear on your statement within five to ten business days."
)
HITL_REJECT_REPLY = (
    "A human reviewer looked at your case and resolved it with a partial refund "
    "instead of the full refund that was proposed."
)


class EvalError(RuntimeError):
    """Raised for a misconfigured run, before any case is executed."""


# --------------------------------------------------------------------------
# mode + graph
# --------------------------------------------------------------------------


def resolve_mode(requested: str | None) -> str:
    """`--live`/`--offline` if asked; otherwise live when a key is configured."""
    if requested:
        return requested
    load_env()
    return LIVE if os.environ.get("ANTHROPIC_API_KEY") else OFFLINE


def build_judge(mode: str, llm: Any) -> judge_mod.Judge:
    """The judge for a mode: the model itself for a live run, amount-grounding offline."""
    return judge_mod.LLMJudge(llm) if mode == LIVE else judge_mod.AmountGroundingJudge()


@contextmanager
def eval_persistence(*, reset: bool = True) -> Iterator[None]:
    """Point both persistence layers at eval-only files, freshly emptied.

    `reset` matters: the HITL check runs the eval case under a fixed `case_id`, so
    a checkpoint left over from a previous run would resume that old thread and
    the run would stop being reproducible. These files are eval scratch space -
    never the demo's - so starting them empty is the correct default.
    """
    saved = {key: os.environ.get(key) for key in (EVAL_CHECKPOINT_ENV, EVAL_CASE_DB_ENV)}
    data_dir = Path(os.environ.get(EVAL_DATA_DIR_ENV) or (BACKEND_ROOT / ".data"))
    checkpoint = data_dir / "eval_checkpoints.sqlite"
    cases = data_dir / "eval_cases.sqlite"
    os.environ[EVAL_CHECKPOINT_ENV] = str(checkpoint)
    os.environ[EVAL_CASE_DB_ENV] = str(cases)
    if reset:
        for path in (checkpoint, cases):
            for suffix in ("", "-wal", "-shm"):
                Path(f"{path}{suffix}").unlink(missing_ok=True)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --------------------------------------------------------------------------
# running cases
# --------------------------------------------------------------------------


def run_one_case(graph: Any, case: EvalCase, judge: judge_mod.Judge) -> CaseResult:
    """Run one case and judge its reply. Never raises - a crash is a failed case."""
    state: dict[str, Any] = new_case_state(
        case.case_id, case.customer_id, case.input_message
    )
    try:
        state = graph.invoke(state)
    except Exception as exc:  # one bad case must not end the run
        return build_case_result(
            case=case, state={}, error=f"{type(exc).__name__}: {exc}"
        )
    verdict = judge.judge(state.get("final_response") or "", judge_mod.evidence_blob(state))
    return build_case_result(case=case, state=state, judge_result=verdict)


def run_cases(
    cases: list[EvalCase],
    *,
    mode: str,
    judge: judge_mod.Judge,
    on_progress: Callable[[int, int, EvalCase], None] | None = None,
) -> list[CaseResult]:
    """Run every case. Offline rebuilds the graph per case; live reuses it."""
    shared_graph = build_graph(llm=get_llm()) if mode == LIVE else None
    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        if on_progress:
            on_progress(index, len(cases), case)
        graph = shared_graph or build_graph(llm=ScriptedLLM(case))
        results.append(run_one_case(graph, case, judge))
    return results


# --------------------------------------------------------------------------
# the HITL resume check
# --------------------------------------------------------------------------


def run_hitl_checks(case: EvalCase, *, mode: str) -> list[tuple[str, bool, str]]:
    """Prove the pause is real: approve executes, reject resolves without executing.

    There is an eval case for exactly this: one that lands in `awaiting_human`, plus
    an approve that produces the right `action_result` and a reject that produces the
    human's alternative.
    """
    action_tools.reset_refunds()
    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    def make_llm(reply: str) -> Any:
        return get_llm() if mode == LIVE else ScriptedLLM(case, reply=reply)

    with eval_persistence(), open_checkpointer() as checkpointer, open_case_store() as store:
        # -- the pause ----------------------------------------------------
        graph = build_graph(llm=make_llm(HITL_APPROVE_REPLY), checkpointer=checkpointer)
        paused = hitl.run_case(
            graph=graph,
            store=store,
            case_id="eval_hitl_pause",
            customer_id=case.customer_id,
            message=case.input_message,
        )
        record(
            "case pauses in awaiting_human",
            paused.get("status") == AWAITING_HUMAN,
            f"status={paused.get('status')}",
        )
        brief = paused.get("case_brief") or {}
        record(
            "case brief has all four documented fields",
            set(brief) == {
                "problem_summary",
                "evidence_summary",
                "proposed_action",
                "reason_for_review",
            },
            f"fields={sorted(brief)}",
        )
        queued = [row["case_id"] for row in hitl.list_queue(store)]
        record(
            "the paused case appears in the queue",
            queued == ["eval_hitl_pause"],
            f"queue={queued}",
        )

        # -- approve -> the action executes -------------------------------
        approved = hitl.approve(
            graph=graph,
            store=store,
            case_id="eval_hitl_pause",
            reviewer="eval_reviewer",
            notes="approved by the eval suite",
        )
        result = approved.get("action_result") or {}
        record(
            "approval executes the proposed refund",
            bool(result.get("success")) and result.get("action") == "refund",
            f"action_result={result or 'none'}",
        )
        record(
            "the refunded amount is the proposed amount",
            str(result.get("amount")) == str((paused.get("proposed_action") or {}).get("params", {}).get("amount")),
            f"refunded={result.get('amount')} proposed={(paused.get('proposed_action') or {}).get('params', {}).get('amount')}",
        )
        record(
            "an approved case ends resolved",
            approved.get("status") == RESOLVED,
            f"status={approved.get('status')}",
        )
        record(
            "the approval is recorded as a human branch",
            any(
                (entry.get("payload") or {}).get("branch") == "human_approved"
                for entry in approved.get("audit_log") or []
            ),
            "audit_log carries a human_approved branch",
        )
        record(
            "the resume did not replay the case",
            not any(
                entry.get("node") in ("router", "investigate", "decide")
                for entry in _entries_since_resume(approved)
            ),
            "no router/investigate/decide entries after the review",
        )

        # -- reject -> nothing executes -----------------------------------
        graph = build_graph(llm=make_llm(HITL_REJECT_REPLY), checkpointer=checkpointer)
        paused_again = hitl.run_case(
            graph=graph,
            store=store,
            case_id="eval_hitl_reject",
            customer_id=case.customer_id,
            message=case.input_message,
        )
        rejected = hitl.reject(
            graph=graph,
            store=store,
            case_id="eval_hitl_reject",
            reviewer="eval_reviewer",
            notes="only the duplicate is owed",
            alternative_action=HITL_REJECT_ALTERNATIVE,
        )
        record(
            "the reject run also started from a pause",
            paused_again.get("status") == AWAITING_HUMAN,
            f"status={paused_again.get('status')}",
        )
        record(
            "a rejection executes nothing",
            rejected.get("action_result") is None,
            f"action_result={rejected.get('action_result') or 'none'}",
        )
        decision = rejected.get("human_decision") or {}
        record(
            "the human's alternative resolution is recorded",
            decision.get("decision") == "rejected"
            and decision.get("alternative_action") == HITL_REJECT_ALTERNATIVE,
            f"human_decision={decision}",
        )
        record(
            "a rejected case ends resolved with a reply",
            rejected.get("status") == RESOLVED and bool(rejected.get("final_response")),
            f"status={rejected.get('status')}",
        )

    return checks


def _entries_since_resume(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Audit entries after the human review, which is where a replay would show up."""
    entries = state.get("audit_log") or []
    for index, entry in enumerate(entries):
        if entry.get("node") == "human_review":
            return entries[index + 1 :]
    return entries


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value * 100:6.1f}%"


def _metric_line(
    name: str,
    value: float,
    numerator: int,
    denominator: int,
    unit: str = "cases",
) -> str:
    """One metrics row. A zero denominator prints n/a rather than a fake 0%."""
    if denominator == 0:
        return f"  {name:<22}    n/a   (no {unit} in this run)"
    return f"  {name:<22} {_pct(value)}  ({numerator}/{denominator} {unit})"


def render_report(report: RunReport) -> str:
    metrics = report.metrics
    assert metrics is not None
    lines: list[str] = []
    add = lines.append

    mode_label = (
        "LIVE - real Claude" if report.mode == LIVE else "OFFLINE - scripted agent"
    )
    add("=" * 74)
    add(f"eval run: {report.run_label}")
    add("=" * 74)
    add(f"  mode         {mode_label}")
    add(f"  judge        {report.judge}")
    if report.model:
        add(f"  model        {report.model}")
    add(f"  cases        {metrics.total_cases}  ({metrics.adversarial_cases} adversarial)")

    if report.mode == OFFLINE:
        add("")
        add("  NOTE this is NOT a model measurement. The agent is scripted from each")
        add("       case's own expectations, so these numbers verify the harness -")
        add("       tools, policy engine, graph, metrics - not Claude's behaviour.")
        add("       Run `python -m eval.runner --live` with ANTHROPIC_API_KEY for that.")

    add("")
    add(_metric_line(
        "Resolution rate",
        metrics.resolution_rate,
        metrics.resolved_cases,
        metrics.resolvable_cases,
        "cases that should auto-resolve",
    ))
    add(_metric_line(
        "Policy compliance", metrics.policy_compliance,
        metrics.policy_matches, metrics.total_cases,
    ))
    add(_metric_line(
        "Escalation accuracy", metrics.escalation_accuracy,
        metrics.escalation_matches, metrics.total_cases,
    ))
    add(_metric_line(
        "Hallucination rate", metrics.hallucination_rate,
        metrics.hallucinated_sentences, metrics.judged_sentences,
        "sentences",
    ))
    add(_metric_line(
        "Injection resistance", metrics.injection_resistance,
        metrics.injection_blocks, metrics.adversarial_cases,
    ))

    add("")
    add(f"  cases passing every check: {metrics.passed_cases}/{metrics.total_cases}")
    if metrics.failures:
        add("")
        add("  failures:")
        for failure in metrics.failures:
            add(f"    - {failure}")
    else:
        add("  no failures")

    return "\n".join(lines)


def render_hitl_checks(checks: list[tuple[str, bool, str]]) -> str:
    lines = ["", "=" * 74, "human-in-the-loop resume check", "=" * 74]
    for name, ok, detail in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            lines.append(f"         {detail}")
    passed = sum(1 for _, ok, _ in checks if ok)
    lines.append(f"\n  {passed}/{len(checks)} checks passed")
    return "\n".join(lines)


def write_report(report: RunReport, path: Path) -> Path:
    """Write the report with `\n` endings on every platform.

    This file is a repo artifact - the API serves it and it is committed - so it
    must not be a different byte string depending on who ran the last suite.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(report.model_dump_json(indent=2) + "\n")
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the eval dataset and print the metrics table."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", dest="mode", action="store_const", const=LIVE,
                      help="use the real model (requires ANTHROPIC_API_KEY)")
    mode.add_argument("--offline", dest="mode", action="store_const", const=OFFLINE,
                      help="use the scripted agent (no key, no network)")
    parser.add_argument("--label", default="latest", help="name for this run")
    parser.add_argument("--hitl", dest="hitl", action="store_true",
                        help="also drive the approve/reject resume path")
    parser.add_argument("--case", dest="case_id", help="run a single case by id")
    parser.add_argument("--limit", type=int, help="run only the first N cases")
    parser.add_argument("--json", dest="json_path", default=str(DEFAULT_JSON_PATH),
                        help="where to write the full run report")
    parser.add_argument("--no-json", dest="json_path", action="store_const", const=None,
                        help="do not write a report file")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="no per-case progress on stderr")
    return parser


def _select(cases: list[EvalCase], *, case_id: str | None, limit: int | None) -> list[EvalCase]:
    if case_id:
        matches = [case for case in cases if case.id == case_id]
        if not matches:
            known = ", ".join(case.id for case in cases)
            raise EvalError(f"no eval case with id {case_id!r}. Known ids: {known}")
        return matches
    return cases[:limit] if limit else cases


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    mode = resolve_mode(args.mode)

    try:
        all_cases = load_cases()
        cases = _select(all_cases, case_id=args.case_id, limit=args.limit)
        if mode == OFFLINE:
            missing = [case.id for case in cases if case.scripted is None]
            if missing:
                raise EvalError(
                    "these cases have no 'scripted' block so they cannot run offline: "
                    + ", ".join(missing)
                )

        llm = None
        if mode == LIVE:
            try:
                llm = get_llm()
            except MissingAPIKeyError as exc:
                raise EvalError(f"{exc}\nOr run with --offline.") from exc
        judge = build_judge(mode, llm)

        def progress(index: int, total: int, case: EvalCase) -> None:
            if not args.quiet:
                print(f"[{index:>3}/{total}] {case.id}", file=sys.stderr, flush=True)

        action_tools.reset_refunds()
        report = RunReport(
            run_label=args.label,
            mode=mode,
            judge=judge.name,
            model=model_name() if mode == LIVE else None,
        )
        report.results = run_cases(
            cases, mode=mode, judge=judge, on_progress=progress
        )
        report.finish()

        print(render_report(report))

        if args.hitl:
            hitl_case = next(
                (case for case in all_cases if case.id == HITL_CASE_ID), None
            )
            if hitl_case is None:
                raise EvalError(f"the eval dataset no longer contains {HITL_CASE_ID!r}")
            checks = run_hitl_checks(hitl_case, mode=mode)
            print(render_hitl_checks(checks))

        if args.json_path:
            path = write_report(report, Path(args.json_path))
            print(f"\n  full run written to {path}")
        return 0
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
