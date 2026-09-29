"""Errors every tool raises instead of inventing a value."""

from __future__ import annotations

from typing import Any


class ToolError(Exception):
    """Structured error raised when required data is missing or invalid.

    Tools never fabricate a default. If a fixture lookup fails, they raise this,
    and the caller (the graph) is responsible for turning it into an escalation.
    """

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                **self.details,
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ToolError(code={self.code!r}, message={self.message!r})"
