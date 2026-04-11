"""
Action definitions for AI Misuse Triage Environment.
"""
from typing import Tuple, Dict, Any
from pydantic import BaseModel, Field
from .schemas import AIMisuseState

class Action(BaseModel):
    """Base Action class. Subclasses represent specific agent actions."""
    
    def execute(self, state: AIMisuseState) -> Tuple[AIMisuseState, float, bool, str]:
        """
        Executes the action on the given state.
        Returns:
            Tuple[AIMisuseState, reward_delta, is_done, feedback_message]
        """
        raise NotImplementedError

class DetectIntentAction(Action):
    """Agent detects the intent of the prompt."""
    intent: str = Field(..., description="The detected intent.")
    
    def execute(self, state: AIMisuseState) -> Tuple[AIMisuseState, float, bool, str]:
        if state.detected_intent is not None:
            return state, -0.05, False, "Intent already detected. Redundant action."
        
        state.detected_intent = self.intent
        state.conversation_history.append({"role": "system", "content": f"[Action] Intent detected: {self.intent}"})
        return state, 0.0, False, f"Successfully set intent to {self.intent}."

class ClassifyRiskAction(Action):
    """Agent assigns a risk score to the prompt."""
    score: float = Field(..., description="Risk score from 0.0 to 1.0.")
    
    def execute(self, state: AIMisuseState) -> Tuple[AIMisuseState, float, bool, str]:
        if state.detected_intent is None:
            return state, -0.1, False, "Cannot classify risk before detecting intent."
        if state.risk_score is not None:
            return state, -0.05, False, "Risk score already assigned. Redundant action."
            
        state.risk_score = self.score
        state.conversation_history.append({"role": "system", "content": f"[Action] Risk score classified: {self.score}"})
        return state, 0.0, False, f"Successfully set risk score to {self.score}."

class DecideMitigationAction(Action):
    """Agent decides on a mitigation strategy."""
    mitigation: str = Field(..., description="Must be one of: 'allow', 'warn', 'block'.")
    
    def execute(self, state: AIMisuseState) -> Tuple[AIMisuseState, float, bool, str]:
        if state.risk_score is None:
            return state, -0.1, False, "Cannot decide mitigation before classifying risk."
        if getattr(state, "mitigation_applied", None) is not None:
            return state, -0.05, False, "Mitigation already decided. Redundant action."
            
        if self.mitigation not in ["allow", "warn", "block"]:
            return state, -0.1, False, f"Invalid mitigation strategy: {self.mitigation}"
            
        state.mitigation_applied = self.mitigation
        state.conversation_history.append({"role": "system", "content": f"[Action] Mitigation applied: {self.mitigation}"})
        
        # This is a terminal action if we're not responding to the user directly
        state.is_done = True
        return state, 0.0, True, f"Successfully applied mitigation: {self.mitigation}."

class GenerateSafeResponseAction(Action):
    """Agent responds to the user (e.g. providing safe alternative or direct answer)."""
    response: str = Field(..., description="The response sent back to the user.")
    
    def execute(self, state: AIMisuseState) -> Tuple[AIMisuseState, float, bool, str]:
        if state.mitigation_applied is None:
            # We can respond without explicit mitigation if it's implicitly 'allow', but for strictness:
            return state, -0.1, False, "Should decide mitigation before generating response."
            
        state.conversation_history.append({"role": "assistant", "content": self.response})
        state.is_done = True
        return state, 0.0, True, "Response sent to user. Episode completed."

class EscalateReviewAction(Action):
    """Agent sends the prompt to a human reviewer."""
    reason: str = Field(..., description="Reason for escalation.")
    
    def execute(self, state: AIMisuseState) -> Tuple[AIMisuseState, float, bool, str]:
        if state.detected_intent is None:
             # Escalating without attempting to understand is slightly penalized
             reward = -0.05
        else:
             reward = 0.0
             
        state.mitigation_applied = "escalate"
        state.system_flags.append(f"Escalated: {self.reason}")
        state.conversation_history.append({"role": "system", "content": f"[Action] Escalated to human: {self.reason}"})
        state.is_done = True
        
        return state, reward, True, f"Escalated for human review: {self.reason}"
