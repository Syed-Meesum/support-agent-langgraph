import { useEffect, useState } from "react";

import { api } from "../api.js";
import DecisionTrail from "./DecisionTrail.jsx";

// Run one case end to end and show the full decision trail. The demo buttons are
// the backend's own CLI presets, so the UI cannot drift from the documented
// examples in the README.

const CUSTOM = "__custom__";

export default function ChatView({ onCaseFinished }) {
  const [demos, setDemos] = useState([]);
  const [selected, setSelected] = useState(CUSTOM);
  const [customerId, setCustomerId] = useState("cust_1001");
  const [message, setMessage] = useState(
    "I was charged twice for my $29.99 subscription this month, please refund me.",
  );
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    api
      .demos()
      .then((rows) => {
        setDemos(rows);
        const happy = rows.find((row) => row.id === "happy");
        if (happy) applyDemo(happy);
      })
      .catch((err) => setError(err.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function applyDemo(demo) {
    setSelected(demo.id);
    setCustomerId(demo.customer_id);
    setMessage(demo.message);
  }

  const activeDemo = demos.find((row) => row.id === selected);

  async function run(event) {
    event.preventDefault();
    setRunning(true);
    setError(null);
    try {
      const result = await api.runCase({ customer_id: customerId, message });
      setDetail(result);
      onCaseFinished?.(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="split">
      <form className="panel run-panel" onSubmit={run}>
        <h2>Run a case</h2>

        <label>
          Demo case
          <select
            value={selected}
            onChange={(event) => {
              const id = event.target.value;
              const demo = demos.find((row) => row.id === id);
              if (demo) applyDemo(demo);
              else setSelected(id);
            }}
          >
            {demos.map((demo) => (
              <option key={demo.id} value={demo.id}>
                {demo.id}
              </option>
            ))}
            <option value={CUSTOM}>custom…</option>
          </select>
        </label>
        {activeDemo ? <p className="muted small">{activeDemo.expectation}</p> : null}

        <label>
          Customer id
          <input
            value={customerId}
            onChange={(event) => setCustomerId(event.target.value)}
            required
          />
        </label>

        <label>
          Customer message
          <textarea
            rows={5}
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            required
          />
        </label>

        <button type="submit" disabled={running || !message.trim()}>
          {running ? "Running the graph…" : "Run case"}
        </button>

        {error ? <p className="error">{error}</p> : null}

        {detail ? (
          <div className="panel-note">
            {detail.status === "awaiting_human"
              ? "This case paused for a human — see the Review queue tab."
              : "This case resolved without a human."}
          </div>
        ) : null}
      </form>

      <div className="panel trail-panel">
        {detail ? (
          <DecisionTrail detail={detail} />
        ) : (
          <p className="muted">
            Run a case to see its router decision, tool calls, the policy rule that
            fired, verification, the outcome, and the full audit trail.
          </p>
        )}
      </div>
    </div>
  );
}
