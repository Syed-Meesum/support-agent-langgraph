// The decision trail: everything that happened in one case, in the order a
// reviewer needs it. This is the component that renders the audit view, and it is
// deliberately explicit about which steps were decided by code and which by the
// model - that separation is the point of the project.

const STATUS_LABELS = {
  in_progress: "in progress",
  resolved: "resolved",
  escalated: "escalated",
  awaiting_human: "awaiting human",
};

export function StatusPill({ status }) {
  const known = Object.prototype.hasOwnProperty.call(STATUS_LABELS, status);
  return (
    <span className={`pill pill-${known ? status : "unknown"}`}>
      {STATUS_LABELS[status] ?? status}
    </span>
  );
}

function Section({ title, note, children }) {
  return (
    <section className="trail-section">
      <h4>
        {title}
        {note ? <span className="section-note">{note}</span> : null}
      </h4>
      {children}
    </section>
  );
}

function Amount({ children }) {
  return <code className="amount">{children}</code>;
}

function ProposedAction({ action }) {
  if (!action || !action.type) return <p className="muted">no action proposed</p>;
  const params = action.params ?? {};
  return (
    <>
      <p>
        <strong>{action.type}</strong>
        {Object.entries(params).map(([key, value]) => (
          <span key={key} className="param">
            {" "}
            {key}=
            <Amount>{String(value)}</Amount>
          </span>
        ))}
      </p>
      {action.rationale ? <p className="muted">{action.rationale}</p> : null}
    </>
  );
}

function PolicyCheck({ policy }) {
  if (!policy) return <p className="muted">not reached</p>;
  return (
    <>
      <p className="rule">
        <code>{policy.rule_id}</code>
        <span
          className={`pill pill-${policy.approved ? "resolved" : "awaiting_human"}`}
        >
          {policy.approved ? "approved" : "requires human"}
        </span>
      </p>
      {policy.reason ? <p className="muted">{policy.reason}</p> : null}
    </>
  );
}

function Outcome({ detail }) {
  if (detail.action_result) {
    const result = detail.action_result;
    return (
      <p>
        Executed <strong>{result.action ?? "action"}</strong>
        {result.amount ? (
          <>
            {" "}
            of <Amount>{String(result.amount)}</Amount>
          </>
        ) : null}
        {result.refund_id ? (
          <>
            {" "}
            (<code>{result.refund_id}</code>)
          </>
        ) : null}
      </p>
    );
  }
  if (detail.escalation_reason) {
    return <p className="muted">{detail.escalation_reason}</p>;
  }
  return <p className="muted">nothing was executed</p>;
}

function CaseBrief({ brief }) {
  const rows = [
    ["problem", brief.problem_summary],
    ["evidence", brief.evidence_summary],
    ["reason for review", brief.reason_for_review],
  ];
  return (
    <dl className="brief">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value || "—"}</dd>
        </div>
      ))}
      <div>
        <dt>proposed</dt>
        <dd>
          <ProposedAction action={brief.proposed_action} />
        </dd>
      </div>
    </dl>
  );
}

export default function DecisionTrail({ detail }) {
  const context = detail.customer_context ?? {};
  const charges = context.charges ?? [];
  const subscription = context.subscription;
  const toolErrors = context.tool_errors ?? [];

  return (
    <div className="trail">
      <header className="trail-header">
        <div>
          <h3>{detail.case_id}</h3>
          <p className="muted">
            customer <code>{detail.customer_id}</code>
          </p>
        </div>
        <div className="trail-header-right">
          <StatusPill status={detail.status} />
          <p className="muted">
            router: {detail.category ?? "—"} / {detail.sentiment ?? "—"} /{" "}
            {detail.confidence ?? "—"}
          </p>
        </div>
      </header>

      <Section title="Customer message">
        <blockquote>{detail.customer_message}</blockquote>
      </Section>

      <Section title="Investigation notes" note="every claim cites a field">
        {detail.investigation_notes.length === 0 ? (
          <p className="muted">none</p>
        ) : (
          <ul>
            {detail.investigation_notes.map((note, index) => (
              <li key={index}>{note}</li>
            ))}
          </ul>
        )}
        <p className="muted small">
          {charges.length} charge(s) on file
          {subscription ? ` · subscription ${subscription.id} (${subscription.status})` : " · no subscription"}
        </p>
        {toolErrors.length > 0 ? (
          <p className="warn small">
            {toolErrors.length} tool call(s) failed — policy escalates on this
          </p>
        ) : null}
      </Section>

      {detail.knowledge_hits.length > 0 ? (
        <Section title="Policy docs retrieved" note="vector search">
          <ul className="hits">
            {detail.knowledge_hits.map((hit, index) => (
              <li key={index}>
                <code>{hit.score?.toFixed?.(3) ?? hit.score}</code> {hit.source}
              </li>
            ))}
          </ul>
        </Section>
      ) : null}

      <Section title="Proposed action" note="model">
        <ProposedAction action={detail.proposed_action} />
      </Section>

      <Section title="Policy check" note="pure code — no LLM">
        <PolicyCheck policy={detail.policy_result} />
      </Section>

      <Section title="Verification" note="model, reached only if approved">
        <p>
          <strong>{detail.verified ? "verified" : "not verified"}</strong>
          {!detail.verified && detail.policy_result?.approved
            ? " — the evidence did not support the action, so it did not execute"
            : null}
        </p>
      </Section>

      <Section title="Outcome">
        <Outcome detail={detail} />
      </Section>

      {detail.case_brief ? (
        <Section title="Case brief" note="what the human reviewer saw">
          <CaseBrief brief={detail.case_brief} />
        </Section>
      ) : null}

      {detail.human_decision ? (
        <Section title="Human decision" note="written before the resume">
          <p>
            <strong>{detail.human_decision.decision}</strong> by{" "}
            {detail.human_decision.reviewer}
            {detail.human_decision.alternative_action ? (
              <>
                {" "}
                → <code>{detail.human_decision.alternative_action}</code>
              </>
            ) : null}
          </p>
          {detail.human_decision.notes ? (
            <p className="muted">{detail.human_decision.notes}</p>
          ) : null}
        </Section>
      ) : null}

      <Section title="Reply to the customer">
        <blockquote className="reply">{detail.final_response}</blockquote>
      </Section>

      <Section title="Audit trail" note={`${detail.audit_log.length} entries`}>
        <ol className="audit">
          {detail.audit_log.map((entry, index) => (
            <li key={index}>
              <span className="audit-node">{entry.node}</span>
              <span className="audit-event">{entry.event_type}</span>
              <span className="audit-payload">
                {summarize(entry.payload)}
              </span>
            </li>
          ))}
        </ol>
      </Section>
    </div>
  );
}

function summarize(payload = {}) {
  const keys = Object.keys(payload);
  if (keys.length === 0) return "";
  const shown = keys
    .slice(0, 3)
    .map((key) => `${key}=${short(payload[key])}`)
    .join("  ");
  return keys.length > 3 ? `${shown}  (+${keys.length - 3})` : shown;
}

function short(value) {
  if (value === null || value === undefined) return String(value);
  if (typeof value === "object") {
    const text = JSON.stringify(value);
    return text.length > 80 ? `${text.slice(0, 77)}...` : text;
  }
  return String(value);
}
