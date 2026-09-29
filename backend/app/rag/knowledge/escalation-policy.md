# Escalation Policy

## When to escalate

A case is escalated to a human reviewer when any of the following is true:

- The proposed refund exceeds the auto-approval threshold.
- No policy rule matches the proposed action.
- The verification step cannot confirm the evidence supports the exact action.
- The customer sentiment is angry, or the case is a repeat complaint.
- A required tool call failed or returned incomplete data.

## What escalation means

Escalation is not a refusal. The case is paused with a brief for a human, who can
approve the proposed action, reject it, or substitute a different resolution. The
graph resumes after the human decision.

## Fail closed

When the correct action is unclear, escalate. The system never improvises a
resolution that no policy rule covers.
