"""
Core environment logic matching Meta OpenEnv standard.
"""
from typing import Tuple, Dict, Any, Optional
import copy

from .schemas import AIMisuseState, Scenario
from .scenarios import generate_random_scenario, load_scenarios
from .reward import calculate_step_reward, calculate_final_reward
from .actions import (
    Action, DetectIntentAction, ClassifyRiskAction,
    DecideMitigationAction, GenerateSafeResponseAction, EscalateReviewAction
)

class AIMisuseEnv:
    """
    OpenEnv-compatible MDP for AI Misuse Triage.
    Has reset(), step(action), and get_state() methods.
    """
    
    def __init__(self):
        self._current_state: Optional[AIMisuseState] = None
        self._current_scenario: Optional[Scenario] = None
        self._action_registry = {
            "detect_intent": DetectIntentAction,
            "classify_risk": ClassifyRiskAction,
            "decide_mitigation": DecideMitigationAction,
            "generate_safe_response": GenerateSafeResponseAction,
            "escalate_review": EscalateReviewAction
        }
        
    def reset(self, scenario_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Resets the environment with a new scenario.
        Returns the initial text observation and info dict.
        """
        all_scenarios = load_scenarios()
        if scenario_id:
            self._current_scenario = next((s for s in all_scenarios if s.scenario_id == scenario_id), None)
            if not self._current_scenario:
                raise ValueError(f"Scenario {scenario_id} not found.")
        else:
            self._current_scenario = generate_random_scenario()
            
        self._current_state = AIMisuseState(
            user_prompt=self._current_scenario.user_prompt
        )
        
        info = {
            "scenario_id": self._current_scenario.scenario_id,
            "difficulty": self._current_scenario.difficulty
        }
        
        obs = self._render_observation()
        return obs, info
        
    def step(self, action_dict: Dict[str, Any]) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Takes an action and transitions the state.
        action_dict format: {"action_name": "detect_intent", "intent": "phishing"}
        """
        if self._current_state is None:
            raise RuntimeError("Environment must be reset before calling step().")
            
        if self._current_state.is_done:
            return self._render_observation(), 0.0, True, False, {"error": "Episode is already done."}
            
        action_name = action_dict.get("action_name")
        if action_name not in self._action_registry:
            # Invalid action
            return self._render_observation(), -0.5, False, False, {"error": f"Invalid action: {action_name}"}
            
        action_cls = self._action_registry[action_name]
        try:
            # Drop action_name to parse the kwargs
            kwargs = {k: v for k, v in action_dict.items() if k != "action_name"}
            action_obj = action_cls(**kwargs)
        except Exception as e:
            # Pydantic validation failure
            return self._render_observation(), -0.5, False, False, {"error": f"Action validation failed: {str(e)}"}
            
        # Execute action
        new_state, action_reward, is_done, feedback = action_obj.execute(self._current_state)
        self._current_state = new_state
        self._current_state.step_count += 1
        
        # Calculate reward
        reward = calculate_step_reward(self._current_state, action_reward)
        info = {"feedback": feedback}
        
        if is_done:
            final_reward, final_info = calculate_final_reward(self._current_state, self._current_scenario)
            reward += final_reward
            info.update(final_info)
            info["ground_truth"] = self._current_scenario.ground_truth.dict()
            
        return self._render_observation(), reward, is_done, False, info
        
    def state(self) -> Dict[str, Any]:
        """Returns the fully serialized Pydantic state."""
        if not self._current_state:
            return {}
        return self._current_state.dict()
        
    def _render_observation(self) -> str:
        """Draws the text-based interface and action history for the agent to observe."""
        if not self._current_state:
            return "[Empty Env]"
            
        lines = [
            f"--- AI MISUSE TRIAGE (Step {self._current_state.step_count}) ---",
            f"USER PROMPT: {self._current_state.user_prompt}",
            "-" * 40
        ]
        
        if self._current_state.conversation_history:
            lines.append("HISTORY:")
            for msg in self._current_state.conversation_history:
                lines.append(f"[{msg['role'].upper()}] {msg['content']}")
                
        lines.append("-" * 40)
        lines.append("CURRENT MEMORY:")
        lines.append(f"Detected Intent: {self._current_state.detected_intent or 'None'}")
        lines.append(f"Risk Score: {self._current_state.risk_score if self._current_state.risk_score is not None else 'None'}")
        lines.append(f"Mitigation Applied: {self._current_state.mitigation_applied or 'None'}")
        
        if self._current_state.is_done:
            lines.append(">>> EPISODE TERMINATED <<<")
            
        return "\n".join(lines)
