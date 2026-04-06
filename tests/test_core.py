import pytest
import json
from openenv_misuse_triage.schema import validate_decision, SchemaValidationError, AgentDecision

def test_validate_decision_valid():
    """Test validation with a perfectly valid dictionary."""
    valid_data = {
        "risk_label": "harmful",
        "category": "phishing",
        "action": "block",
        "rationale": "This is a phishing attempt asking for credentials."
    }
    decision = validate_decision(valid_data)
    assert isinstance(decision, AgentDecision)
    assert decision.risk_label == "harmful"
    assert decision.category == "phishing"
    assert decision.action == "block"

def test_validate_decision_json_string():
    """Test validation with a valid JSON string."""
    valid_json = json.dumps({
        "risk_label": "benign",
        "category": "writing_assistance",
        "action": "allow",
        "rationale": "Just a standard writing request."
    })
    decision = validate_decision(valid_json)
    assert decision.risk_label == "benign"

def test_validate_decision_missing_keys():
    """Test validation fails when keys are missing."""
    invalid_data = {
        "risk_label": "harmful",
        # category missing
        "action": "block",
        "rationale": "..."
    }
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_decision(invalid_data)
    assert "Missing required keys: ['category']" in str(excinfo.value)

def test_validate_decision_invalid_values():
    """Test validation fails with invalid enum values."""
    invalid_data = {
        "risk_label": "SUPER_DANGEROUS", # Not in VALID_RISK_LABELS
        "category": "phishing",
        "action": "block",
        "rationale": "..."
    }
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_decision(invalid_data)
    assert "Invalid risk_label 'SUPER_DANGEROUS'" in str(excinfo.value)

def test_validate_decision_empty_rationale():
    """Test validation fails with an empty or whitespace rationale."""
    invalid_data = {
        "risk_label": "harmful",
        "category": "phishing",
        "action": "block",
        "rationale": "   "
    }
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_decision(invalid_data)
    assert "Field 'rationale' must not be empty" in str(excinfo.value)

def test_env_initialization():
    """Test that the environment can be initialized without error."""
    from openenv_misuse_triage import MisuseTriageEnv
    env = MisuseTriageEnv()
    obs, info = env.reset()
    
    # Check for presence of key sections in the observation (case-insensitive)
    obs_lower = obs.lower()
    assert "prompt" in obs_lower
    assert "assistant response" in obs_lower
    assert "context" in obs_lower
    assert "episode" in obs_lower
    
    assert isinstance(info, dict)
    assert "episode_id" in info
