// One place that knows the backend's URLs and how it reports errors.
//
// The base URL is empty by default (same-origin). See vite.config.js for why.

const BASE = import.meta.env.VITE_API_URL ?? "";

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  const text = await response.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { detail: text };
  }

  if (!response.ok) {
    // The backend always sends {detail}; a 503 here means the model is not
    // configured, which is a message the user needs to read verbatim.
    const detail = body?.detail;
    throw new Error(
      typeof detail === "string"
        ? detail
        : detail
          ? JSON.stringify(detail)
          : `${response.status} ${response.statusText}`,
    );
  }
  return body;
}

export const api = {
  health: () => request("/api/health"),
  demos: () => request("/api/demos"),
  listCases: () => request("/api/cases"),
  runCase: (payload) =>
    request("/api/cases", { method: "POST", body: JSON.stringify(payload) }),
  getCase: (caseId) => request(`/api/cases/${encodeURIComponent(caseId)}`),
  queue: () => request("/api/queue"),
  approve: (caseId, payload) =>
    request(`/api/cases/${encodeURIComponent(caseId)}/approve`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  reject: (caseId, payload) =>
    request(`/api/cases/${encodeURIComponent(caseId)}/reject`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  evals: () => request("/api/evals"),
  runEvals: (payload) =>
    request("/api/evals/run", { method: "POST", body: JSON.stringify(payload) }),
};
