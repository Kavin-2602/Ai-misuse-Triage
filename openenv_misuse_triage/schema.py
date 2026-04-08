"""
schema.py - Output schema definition and validation for AI Misuse Triage Environment.

Defines the agent's required output structure, valid label sets, and validation logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Valid value sets
# ---------------------------------------------------------------------------

VALID_RISK_LABELS = {"benign", "suspicious", "harmful", "low", "medium", "high"}
VALID_CATEGORIES = {"phishing", "scam", "prompt_injection", "medical", "harassment", "writing_assistance", "other"}
VALID_ACTIONS = {"allow", "warn", "escalate", "block"}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class AgentDecision:
    """Represents a validated agent decision for a single episode."""

    risk_label: str   # benign | suspicious | harmful
    category: str     # phishing | scam | prompt_injection | medical | harassment | other
    action: str       # allow | warn | escalate | block
    rationale: str    # short free-text explanation

    def to_dict(self) -> dict[str, str]:
        return {
            "risk_label": self.risk_label,
            "category": self.category,
            "action": self.action,
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class SchemaValidationError(ValueError):
    """Raised when an agent output fails schema validation."""
    pass


def validate_decision(raw: Any) -> AgentDecision:
    """
    Parse and validate raw agent output (dict or JSON string).

    Args:
        raw: A dict or JSON-encoded string representing the agent's decision.

    Returns:
        A validated AgentDecision instance.

    Raises:
        SchemaValidationError: with a descriptive message on any failure.
    """
    # --- Step 1: Parse if needed ---
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"Output is not valid JSON: {exc}") from exc
    elif isinstance(raw, dict):
        data = raw
    else:
        raise SchemaValidationError(
            f"Expected a dict or JSON string, got {type(raw).__name__}."
        )

    # --- Step 2: Required keys ---
    required_keys = {"risk_label", "category", "action", "rationale"}
    missing = required_keys - data.keys()
    if missing:
        raise SchemaValidationError(f"Missing required keys: {sorted(missing)}")

    # --- Step 3: Type checks ---
    for key in required_keys:
        if not isinstance(data[key], str):
            raise SchemaValidationError(
                f"Field '{key}' must be a string, got {type(data[key]).__name__}."
            )

    # --- Step 4: Value set validation ---
    if data["risk_label"] not in VALID_RISK_LABELS:
        raise SchemaValidationError(
            f"Invalid risk_label '{data['risk_label']}'. "
            f"Must be one of: {sorted(VALID_RISK_LABELS)}"
        )

    if data["category"] not in VALID_CATEGORIES:
        raise SchemaValidationError(
            f"Invalid category '{data['category']}'. "
            f"Must be one of: {sorted(VALID_CATEGORIES)}"
        )

    if data["action"] not in VALID_ACTIONS:
        raise SchemaValidationError(
            f"Invalid action '{data['action']}'. "
            f"Must be one of: {sorted(VALID_ACTIONS)}"
        )

    # --- Step 5: Rationale non-empty ---
    if not data["rationale"].strip():
        raise SchemaValidationError("Field 'rationale' must not be empty.")

    return AgentDecision(
        risk_label=data["risk_label"],
        category=data["category"],
        action=data["action"],
        rationale=data["rationale"].strip(),
    )


def safe_validate_decision(raw: Any) -> tuple[AgentDecision | None, str | None]:
    """
    Attempt validation without raising; returns (decision, error_message).

    Returns:
        (AgentDecision, None) on success.
        (None, error_string) on failure.
    """
    try:
        return validate_decision(raw), None
    except SchemaValidationError as exc:
        return None, str(exc)
