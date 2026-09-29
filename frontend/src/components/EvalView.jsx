import { useCallback, useEffect, useState } from "react";

import { api } from "../api.js";
import { METRIC_ROWS, formatTime, percent } from "../format.js";

// The metrics table a recruiter reads first - served by the API from whatever
// `python -m eval.runner` last wrote, and re-runnable from here.
//
// The mode is shown loudly on purpose. An offline run uses a scripted agent, and
// its numbers say nothing about model quality; presenting them as a benchmark
// would be dishonest, so the UI cannot do it by accident.

export default function EvalView() {
  const [report, setReport] = useState(null);
  const [hint, setHint] = useState(null);
  const [error, setError] = useState(null);
  const [running, setRunning] = useState(null);
  const [onlyFailures, setOnlyFailures] = useState(false);

  const load = useCallback(async () => {
    try {
      const body = await api.evals();
      setReport(body.available ? body.report : null);
      setHint(body.available ? null : body.hint);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function run(mode) {
    setRunning(mode);
    setError(null);
    try {
      const body = await api.runEvals({ mode });
      setReport(body.report);
      setHint(null);
      setOnlyFailures(false);
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(null);
    }
  }

  const metrics = report?.metrics;
  const rows = (report?.results ?? []).filter((row) =>
    onlyFailures ? !passed(row) : true,
  );

  return (
    <div className="panel eval-panel">
      <div className="eval-head">
        <div>
          <h2>Eval results</h2>
          {report ? (
            <p className="muted small">
              run <code>{report.run_label}</code> · {report.results.length} cases ·{" "}
              judge <code>{report.judge}</code>
              {report.model ? (
                <>
                  {" "}
                  · model <code>{report.model}</code>
                </>
              ) : null}
              <br />
              finished {formatTime(report.completed_at)}
            </p>
          ) : null}
        </div>
        <div className="eval-actions">
          <button type="button" onClick={() => run("offline")} disabled={!!running}>
            {running === "offline" ? "Running…" : "Run offline suite"}
          </button>
          <button
            type="button"
            className="primary"
            onClick={() => run("live")}
            disabled={!!running}
          >
            {running === "live" ? "Running…" : "Run live suite"}
          </button>
        </div>
      </div>

      {error ? <p className="error">{error}</p> : null}

      {!report ? (
        <p className="muted">
          {hint ??
            "No eval run recorded yet. Run the offline suite above, or from backend/: "}
          {!hint ? <code>python -m eval.runner --offline</code> : null}
        </p>
      ) : (
        <>
          {report.mode === "offline" ? (
            <p className="warn">
              This is an <strong>offline</strong> run against a scripted agent, not
              Claude. It verifies the harness — tools, policy engine, graph, metrics
              — and is not a measure of model quality. Use{" "}
              <em>Run live suite</em> for that.
            </p>
          ) : (
            <p className="notice">
              Live run against <code>{report.model}</code> — these are real model
              measurements.
            </p>
          )}

          <table className="metrics">
            <thead>
              <tr>
                <th>Metric</th>
                <th className="num">Value</th>
                <th>Counts</th>
              </tr>
            </thead>
            <tbody>
              {METRIC_ROWS.map((row) => {
                const value = metrics?.[row.key];
                const good = row.lowerIsBetter ? value === 0 : value === 1;
                return (
                  <tr key={row.key}>
                    <td>{row.label}</td>
                    <td className={`num ${good ? "good" : value >= 0.9 ? "ok" : "bad"}`}>
                      {percent(value)}
                    </td>
                    <td className="muted small">{metrics ? row.counts(metrics) : ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <h3 className="spaced">
            Cases <span className="count">{metrics?.passed_cases}</span> passing of{" "}
            {metrics?.total_cases}
          </h3>

          {metrics?.failures?.length ? (
            <>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={onlyFailures}
                  onChange={(event) => setOnlyFailures(event.target.checked)}
                />
                show only the cases that failed a check
              </label>
              <ul className="failures">
                {metrics.failures.map((failure, index) => (
                  <li key={index}>
                    <code>{failure}</code>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p className="notice">
              Every case matched its expectation on resolution, policy rule and
              escalation.
            </p>
          )}

          <div className="table-scroll">
            <table className="cases">
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Expected → actual rule</th>
                  <th>Escalation</th>
                  <th>Status</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.case_id} className={passed(row) ? "" : "failed"}>
                    <td>
                      <strong>{row.case_id}</strong>
                      <br />
                      <span className="muted small">{row.scenario_name}</span>
                      {row.adversarial ? <span className="tag">adversarial</span> : null}
                    </td>
                    <td>
                      <code>{row.expected_rule_id.join(" | ")}</code>
                      <br />
                      <span className={row.policy_ok ? "ok" : "bad"}>
                        {row.actual_rule_id}
                      </span>
                    </td>
                    <td className="small">
                      expected {String(row.expected_escalation)}
                      <br />
                      <span className={row.escalation_ok ? "ok" : "bad"}>
                        got {row.status}
                      </span>
                    </td>
                    <td className="small">{row.actual_action ?? "—"}</td>
                    <td className="small">
                      {row.error ? (
                        <span className="bad">{row.error}</span>
                      ) : row.hallucinated.length > 0 ? (
                        <span className="bad">
                          {row.hallucinated.length} unsupported sentence(s)
                        </span>
                      ) : (
                        <span className="ok">ok</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function passed(row) {
  if (row.error) return false;
  if (!row.resolution_ok || !row.policy_ok || !row.escalation_ok) return false;
  if (row.adversarial && !row.injection_ok) return false;
  return true;
}
