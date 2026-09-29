import { useCallback, useEffect, useState } from "react";

import { api } from "./api.js";
import ChatView from "./components/ChatView.jsx";
import EvalView from "./components/EvalView.jsx";
import QueueView from "./components/QueueView.jsx";

const TABS = [
  { id: "chat", label: "Chat" },
  { id: "queue", label: "Review queue" },
  { id: "evals", label: "Eval results" },
];

export default function App() {
  const [tab, setTab] = useState("chat");
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [refreshSignal, setRefreshSignal] = useState(0);

  const loadHealth = useCallback(async () => {
    try {
      setHealth(await api.health());
      setHealthError(null);
    } catch (err) {
      setHealthError(err.message);
    }
  }, []);

  useEffect(() => {
    loadHealth();
  }, [loadHealth]);

  const bumped = useCallback(() => {
    setRefreshSignal((value) => value + 1);
    loadHealth();
  }, [loadHealth]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>Autonomous Support Agent</h1>
          <p className="muted small">
            LangGraph · deterministic policy gate · human-in-the-loop review ·
            measured evals
          </p>
        </div>

        <nav className="tabs">
          {TABS.map((entry) => (
            <button
              key={entry.id}
              type="button"
              className={entry.id === tab ? "active" : ""}
              onClick={() => setTab(entry.id)}
            >
              {entry.label}
            </button>
          ))}
        </nav>

        <div className="health">
          {healthError ? (
            <span className="bad small">backend unreachable</span>
          ) : health ? (
            <>
              <span
                className={health.model_ready ? "ok small" : "warn small"}
                title={health.model_ready ? "model configured" : "set ANTHROPIC_API_KEY"}
              >
                {health.model_ready ? "model ready" : "no API key"}
              </span>
              <span className="muted small">
                {health.persistence_backend} · {health.model}
              </span>
            </>
          ) : (
            <span className="muted small">checking backend…</span>
          )}
        </div>
      </header>

      {healthError ? (
        <p className="error banner">
          Could not reach the API ({healthError}). Start the backend with{" "}
          <code>uvicorn app.main:app --port 8000</code> from <code>backend/</code>,
          or run <code>docker compose up</code>.
        </p>
      ) : null}

      {health && !health.model_ready ? (
        <p className="warn banner">
          No <code>ANTHROPIC_API_KEY</code> is configured, so running a case returns
          503. Add it to <code>.env</code> at the repo root. The eval results tab
          still works in offline mode.
        </p>
      ) : null}

      <main>
        {tab === "chat" ? <ChatView onCaseFinished={bumped} /> : null}
        {tab === "queue" ? (
          <QueueView refreshSignal={refreshSignal} onReviewed={bumped} />
        ) : null}
        {tab === "evals" ? <EvalView /> : null}
      </main>

      <footer className="footer muted small">
        Every action passes the deterministic policy gate. Nothing executes without
        it — except a human approval, which is recorded as its own audit branch.
      </footer>
    </div>
  );
}
