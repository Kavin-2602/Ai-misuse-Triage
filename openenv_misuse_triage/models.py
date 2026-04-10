"""
openenv_misuse_triage/models.py - OpenEnv strict Pydantic models.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class MisuseTriageAction(BaseModel):
    """Action sent by the agent in the AI Misuse Triage Environment."""
    risk_label: str = Field(description="One of: benign, suspicious, harmful")
    category: str = Field(description="One of: phishing, scam, prompt_injection, medical, harassment, writing_assistance, other")
    action: str = Field(description="One of: allow, warn, escalate, block")
    rationale: str = Field(description="Brief explanation of your decision")

class MisuseTriageObservation(BaseModel):
    """Observation returned to the agent in the AI Misuse Triage Environment."""
    prompt: Optional[str] = Field(None, description="The user prompt")
    assistant_response: Optional[str] = Field(None, description="The assistant's initial response")
    context: Optional[str] = Field(None, description="Additional context about the interaction")
    
    # Exposing ground truth could be done via metadata or directly for training modes
    ground_truth: Optional[Dict[str, Any]] = None
    
    done: bool = Field(False, description="Whether the episode is finished")
    reward: Optional[float] = Field(None, description="Reward received for the last action")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
