"""Loads and caches policy_rules.yaml."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import ValidationError

from app.policy.models import PolicyRules

DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "policy_rules.yaml"
POLICY_RULES_PATH_ENV = "POLICY_RULES_PATH"


def rules_path() -> Path:
    return Path(os.environ.get(POLICY_RULES_PATH_ENV, DEFAULT_RULES_PATH))


def read_rules(path: Path) -> PolicyRules:
    """Parse and validate a rules file. Raises ValueError if it is unusable."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"policy rules file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"policy rules file is not valid YAML: {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"policy rules file must contain a mapping: {path}")

    try:
        return PolicyRules.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"policy rules file is invalid: {path}: {exc}") from exc


@lru_cache(maxsize=1)
def load_rules() -> PolicyRules:
    """Return the active policy rules, loaded once per process."""
    return read_rules(rules_path())


def reload_rules() -> None:
    """Drop the cached rules. Used by tests and by config-change reload."""
    load_rules.cache_clear()
