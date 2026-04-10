"""
openenv_misuse_triage/__init__.py - Compat wrapper for evaluator scripts
"""

import logging

try:
    from openenv_misuse_triage.models import MisuseTriageAction
    from server.misuse_triage_environment import MisuseTriageEnvironment
    from openenv_misuse_triage.tasks import make_observation
except ImportError:
    pass

class MisuseTriageEnv:
    """
    Backward compatibility wrapper to ensure inference.py scripts 
    continue to work without modifying the agent evaluation files.
    """
    def __init__(self, shuffle=False, seed=0, **kwargs):
        self.env = MisuseTriageEnvironment()
        self.seed = seed
        self.shuffle = shuffle
        
    def reset(self, seed=None, **kwargs):
        obs = self.env.reset(seed=seed or self.seed, shuffle=self.shuffle)
        
        # reconstruct episode dictionary for legacy string template compatibility
        ep = {
            "episode_id": obs.metadata.get("episode_id", "local") if obs.metadata else "local",
            "prompt": obs.prompt or "",
            "assistant_response": obs.assistant_response or "",
            "context": obs.context or "",
            "ground_truth": obs.ground_truth or {}
        }
        str_observation = make_observation(ep)
        
        info = {
            "episode_id": ep["episode_id"],
            "episode_index": 0,
            "total_episodes": 25, 
        }
        return str_observation, info
        
    def step(self, action_dict):
        # Graceful fallback for testing scripts
        if not isinstance(action_dict, dict):
            action_dict = {}
            
        action = MisuseTriageAction(
            risk_label=action_dict.get("risk_label", "benign"),
            category=action_dict.get("category", "other"),
            action=action_dict.get("action", "allow"),
            rationale=action_dict.get("rationale", "")
        )
        
        obs = self.env.step(action)
        
        ep = {
            "episode_id": obs.metadata.get("episode_id", "local") if obs.metadata else "local",
            "prompt": obs.prompt or "",
            "assistant_response": obs.assistant_response or "",
            "context": obs.context or "",
            "ground_truth": obs.ground_truth or {}
        }
        str_observation = make_observation(ep) if not obs.done else ""
        
        info = {
            "episode_id": ep["episode_id"],
            "grade_result": {},
            "feedback": "",
            "episodes_remaining": 0
        }
        return str_observation, obs.reward, obs.done, False, info
    
    def state(self):
        return {"done": False}
    
    def close(self):
        pass
