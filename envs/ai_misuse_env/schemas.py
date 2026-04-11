from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

class GroundTruth(BaseModel):
    intent: str = Field(..., description="The true intent of the prompt (e.g. benign, phishing, scam, prompt_injection).")
    risk_score: float = Field(..., description="The true risk score (0.0 to 1.0). >0.5 generally means harmful.")
    mitigation: str = Field(..., description="The ideal mitigation action: 'allow', 'warn', 'block', 'escalate'.")
    is_adversarial: bool = Field(False, description="Whether the prompt is a complex adversarial attack/jailbreak.")

class Scenario(BaseModel):
    scenario_id: str
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Difficulty level.")
    user_prompt: str = Field(..., description="The prompt submitted by the external user.")
    ground_truth: GroundTruth

class AIMisuseState(BaseModel):
    """
    Internal state representation of the environment.
    """
    user_prompt: str = Field(description="The current user prompt being evaluated.")
    detected_intent: Optional[str] = Field(None, description="The intent detected by the agent.")
    risk_score: Optional[float] = Field(None, description="The risk score assigned by the agent.")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="List of messages: role and content.")
    system_flags: List[str] = Field(default_factory=list, description="Flags raised during processing.")
    mitigation_applied: Optional[str] = Field(None, description="The final mitigation action applied by the agent.")
    is_done: bool = Field(False, description="Whether the current episode has reached a terminal state.")
    step_count: int = Field(0, description="Number of actions taken so far.")
