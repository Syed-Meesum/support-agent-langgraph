import { useCallback, useEffect, useState } from "react";

import { api } from "../api.js";
import DecisionTrail, { StatusPill } from "./DecisionTrail.jsx";
import { formatTime } from "../format.js";

// The page the project's HITL work exists for: cases that are genuinely paused,
// with the two buttons that decide them. Approving executes the proposed action;
// rejecting sends the reviewer's alternative instead.

const ALTERNATIVES = [
  { value: "", label: "no alternative — just explain the decision" },
  { value: "partial_refund", label: "partial_refund" },
  { value: "no_action", label: "no_action" },
  { value: "escalate", label: "escalate further" },
];

export default function QueueView({ refreshSignal = 0, onReviewed }) {
  const [queue, setQueue] = useState([]);
  const [recent, setRecent] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [reviewer, setReviewer] = useState("demo_reviewer");
  const [notes, setNotes] = useState("");
  const [alternative, setAlternative] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);

  const loadLists = useCallback(async () => {
    try {
      const [waiting, cases] = await Promise.all([api.queue(), api.listCases()]);
      setQueue(waiting);
      setRecent(cases);
      return waiting;
    } catch (err) {
      setError(err.message);
      return [];
    }
  }, []);

  useEffect(() => {
    loadLists();
  }, [loadLists, refreshSignal]);

  async function open(caseId) {
    setError(null);
    setNotice(null);
    setSelectedId(caseId);
    try {
      setDetail(await api.getCase(caseId));
    } catch (err) {
      setError(err.message);
      setDetail(null);
    }
  }

  async function review(kind) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const payload = { reviewer, notes };
      const result =
        kind === "approve"
          ? await api.approve(selectedId, payload)
          : await api.reject(selectedId, {
              ...payload,
              alternative_action: alternative || null,
            });
      setDetail(result);
      setNotice(
        kind === "approve"
          ? "Approved — the proposed action executed and the case resumed to reply."
          : "Rejected — nothing executed; the reviewer's resolution was relayed.",
      );
      setNotes("");
      setAlternative("");
      await loadLists();
      onReviewed?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const isQueued = detail?.status === "awaiting_human";

  return (
    <div className="split">
      <div className="panel list-panel">
        <h2>
          Awaiting a human <span className="count">{queue.length}</span>
        </h2>
        {queue.length === 0 ? (
          <p className="muted">
            Nothing waiting. Run the <code>escalate</code> demo case to queue one.
          </p>
        ) : (
          <ul className="case-list">
            {queue.map((row) => (
              <li key={row.case_id}>
                <button
                  type="button"
                  className={row.case_id === selectedId ? "active" : ""}
                  onClick={() => open(row.case_id)}
                >
                  <span className="case-list-title">
                    {row.case_id} <StatusPill status={row.status} />
                  </span>
                  <span className="muted small">
                    {row.case_brief?.proposed_action?.type ?? "—"}{" "}
                    {row.case_brief?.proposed_action?.params?.amount ?? ""} ·{" "}
                    {row.customer_id}
                  </span>
                  <span className="muted small">
                    {row.case_brief?.reason_for_review}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}

        <h2 className="spaced">Recent cases</h2>
        {recent.length === 0 ? (
          <p className="muted">No cases yet.</p>
        ) : (
          <ul className="case-list">
            {recent.map((row) => (
              <li key={row.case_id}>
                <button
                  type="button"
                  className={row.case_id === selectedId ? "active" : ""}
                  onClick={() => open(row.case_id)}
                >
                  <span className="case-list-title">
                    {row.case_id} <StatusPill status={row.status} />
                  </span>
                  <span className="muted small">
                    {row.category ?? "—"} · {formatTime(row.updated_at)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="panel trail-panel">
        {!detail ? (
          <p className="muted">
            Select a case to read its brief, evidence and audit trail.
          </p>
        ) : (
          <>
            {error ? <p className="error">{error}</p> : null}
            {notice ? <p className="notice">{notice}</p> : null}

            {isQueued ? (
              <div className="review-box">
                <h3>Review</h3>
                <div className="review-fields">
                  <label>
                    Reviewer
                    <input
                      value={reviewer}
                      onChange={(event) => setReviewer(event.target.value)}
                    />
                  </label>
                  <label>
                    Notes
                    <input
                      value={notes}
                      placeholder="why this decision"
                      onChange={(event) => setNotes(event.target.value)}
                    />
                  </label>
                  <label>
                    If rejecting, resolve it as
                    <select
                      value={alternative}
                      onChange={(event) => setAlternative(event.target.value)}
                    >
                      {ALTERNATIVES.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <div className="review-actions">
                  <button
                    type="button"
                    className="primary"
                    disabled={busy || !reviewer.trim()}
                    onClick={() => review("approve")}
                  >
                    Approve — execute it
                  </button>
                  <button
                    type="button"
                    className="danger"
                    disabled={busy || !reviewer.trim()}
                    onClick={() => review("reject")}
                  >
                    Reject — nothing executes
                  </button>
                </div>
                <p className="muted small">
                  Approving overrides the policy gate on purpose. It is the only
                  thing that can, and it is recorded as a distinct audit branch.
                </p>
              </div>
            ) : (
              <p className="muted">
                This case is <StatusPill status={detail.status} /> — no decision is
                pending. It is shown read-only.
              </p>
            )}

            <DecisionTrail detail={detail} />
          </>
        )}
      </div>
    </div>
  );
}
