"""The LLM the graph's five LLM nodes call."""

from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic

from app.config import load_env

# Claude 4.6-and-later model IDs are dateless and pinned (see Anthropic's model
# ID docs), so this is a fixed snapshot rather than a moving alias.
DEFAULT_MODEL = "claude-sonnet-5"
MODEL_ENV = "ANTHROPIC_MODEL"
API_KEY_ENV = "ANTHROPIC_API_KEY"


class MissingAPIKeyError(RuntimeError):
    """Raised when a node needs the LLM but no key is configured."""


def model_name() -> str:
    load_env()
    return os.environ.get(MODEL_ENV) or DEFAULT_MODEL


def get_llm() -> ChatAnthropic:
    """Build the chat model. Errors loudly rather than failing mid-run."""
    load_env()
    if not os.environ.get(API_KEY_ENV):
        raise MissingAPIKeyError(
            f"{API_KEY_ENV} is not set. Add it to the .env file at the repo root "
            "(copy .env.example) or export it before running the graph."
        )
    return ChatAnthropic(
        model=model_name(),
        temperature=0,
        max_tokens=1500,
        timeout=90,
        max_retries=2,
    )
