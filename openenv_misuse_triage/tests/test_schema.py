"""
test_schema.py - Unit tests for schema validation.
"""

import json
import pytest

from openenv_misuse_triage.schema import (
    validate_decision,
    safe_validate_decision,
    SchemaValidationError,
    VALID_RISK_LABELS,
    VALID_CATEGORIES,
    VALID_ACTIONS,
)


# ---------------------------------------------------------------------------
# Valid inputs
# ---------------------------------------------------------------------------

VALID_DICT = {
    "risk_label": "harmful",
    "category": "phishing",
    "action": "block",
    "rationale": "Detected phishing content.",
}

VALID_JSON_STR = json.dumps(VALID_DICT)


class TestValidInputs:
    def test_valid_dict(self):
        d = validate_decision(VALID_DICT)
        assert d.risk_label == "harmful"
        assert d.category == "phishing"
        assert d.action == "block"
        assert d.rationale == "Detected phishing content."

    def test_valid_json_string(self):
        d = validate_decision(VALID_JSON_STR)
        assert d.risk_label == "harmful"

    def test_all_valid_risk_labels(self):
        for label in VALID_RISK_LABELS:
            inp = {**VALID_DICT, "risk_label": label}
            d = validate_decision(inp)
            assert d.risk_label == label

    def test_all_valid_categories(self):
        for cat in VALID_CATEGORIES:
            inp = {**VALID_DICT, "category": cat}
            d = validate_decision(inp)
            assert d.category == cat

    def test_all_valid_actions(self):
        for act in VALID_ACTIONS:
            inp = {**VALID_DICT, "action": act}
            d = validate_decision(inp)
            assert d.action == act

    def test_rationale_whitespace_stripped(self):
        inp = {**VALID_DICT, "rationale": "  some text  "}
        d = validate_decision(inp)
        assert d.rationale == "some text"

    def test_to_dict_roundtrip(self):
        d = validate_decision(VALID_DICT)
        assert d.to_dict() == VALID_DICT


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

class TestInvalidInputs:
    def test_not_valid_json_string(self):
        with pytest.raises(SchemaValidationError, match="not valid JSON"):
            validate_decision("{bad json")

    def test_wrong_type(self):
        with pytest.raises(SchemaValidationError, match="Expected a dict"):
            validate_decision(42)

    def test_missing_risk_label(self):
        inp = {k: v for k, v in VALID_DICT.items() if k != "risk_label"}
        with pytest.raises(SchemaValidationError, match="Missing required keys"):
            validate_decision(inp)

    def test_missing_multiple_keys(self):
        with pytest.raises(SchemaValidationError, match="Missing required keys"):
            validate_decision({})

    def test_invalid_risk_label(self):
        inp = {**VALID_DICT, "risk_label": "dangerous"}
        with pytest.raises(SchemaValidationError, match="Invalid risk_label"):
            validate_decision(inp)

    def test_invalid_category(self):
        inp = {**VALID_DICT, "category": "unknown_category"}
        with pytest.raises(SchemaValidationError, match="Invalid category"):
            validate_decision(inp)

    def test_invalid_action(self):
        inp = {**VALID_DICT, "action": "delete"}
        with pytest.raises(SchemaValidationError, match="Invalid action"):
            validate_decision(inp)

    def test_empty_rationale(self):
        inp = {**VALID_DICT, "rationale": "   "}
        with pytest.raises(SchemaValidationError, match="must not be empty"):
            validate_decision(inp)

    def test_non_string_field(self):
        inp = {**VALID_DICT, "risk_label": 1}
        with pytest.raises(SchemaValidationError, match="must be a string"):
            validate_decision(inp)


# ---------------------------------------------------------------------------
# Safe validate
# ---------------------------------------------------------------------------

class TestSafeValidate:
    def test_returns_decision_on_success(self):
        decision, err = safe_validate_decision(VALID_DICT)
        assert decision is not None
        assert err is None

    def test_returns_error_on_failure(self):
        decision, err = safe_validate_decision({"risk_label": "bad"})
        assert decision is None
        assert err is not None
        assert "Missing" in err or "Invalid" in err
