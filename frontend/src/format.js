// Small display helpers shared by the views.

export function formatTime(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString();
}

export function percent(value, digits = 1) {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return `${(value * 100).toFixed(digits)}%`;
}

// The report's metrics object, in the order policy-and-eval.md §2 lists them.
// `lowerIsBetter` is only true for the hallucination rate, which is a rate of
// failure rather than a rate of success.
export const METRIC_ROWS = [
  {
    key: "resolution_rate",
    label: "Resolution rate",
    counts: (m) => `${m.resolved_cases}/${m.resolvable_cases} cases that should auto-resolve`,
  },
  {
    key: "policy_compliance",
    label: "Policy compliance",
    counts: (m) => `${m.policy_matches}/${m.total_cases} cases`,
  },
  {
    key: "escalation_accuracy",
    label: "Escalation accuracy",
    counts: (m) => `${m.escalation_matches}/${m.total_cases} cases`,
  },
  {
    key: "hallucination_rate",
    label: "Hallucination rate",
    lowerIsBetter: true,
    counts: (m) => `${m.hallucinated_sentences}/${m.judged_sentences} sentences`,
  },
  {
    key: "injection_resistance",
    label: "Injection resistance",
    counts: (m) => `${m.injection_blocks}/${m.adversarial_cases} adversarial cases`,
  },
];
