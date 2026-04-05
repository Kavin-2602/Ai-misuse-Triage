"""
learning.py - RL-Compatible Learning Agent for AI Misuse Triage.
"""
import json
import os
from inference import RuleBasedAgent

class LearningAgent(RuleBasedAgent):
    """
    An extension of the RuleBasedAgent that supports simple reward-based 
    weight updating (lightweight RL approach).
    """
    def __init__(self, memory_file="agent_memory.json", log_file="training_log.jsonl"):
        super().__init__()
        self.memory_file = memory_file
        self.log_file = log_file
        
        # Initialize default weights for each rule index
        # We identify rules by their index in self._RULES
        self.rule_weights = {str(i): 1.0 for i in range(len(self._RULES))}
        self.load_memory()

    def load_memory(self):
        """Load saved rule weights from disk."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    data = json.load(f)
                    # Update weights; keep defaults for newly added rules if any
                    for k, v in data.items():
                        self.rule_weights[k] = v
            except Exception as e:
                print(f"Warning: Failed to load agent memory: {e}")

    def save_memory(self):
        """Save rule weights to disk."""
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.rule_weights, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save agent memory: {e}")

    def decide(self, observation: str) -> dict:
        """
        Evaluate observation using rule weights and return the highest scored decision.
        """
        obs_lower = observation.lower()
        
        best_score = -float('inf')
        best_decision = None

        # Check rule list and pick the one with the highest weight
        for idx, (keywords, risk, category, action) in enumerate(self._RULES):
            if any(kw.lower() in obs_lower for kw in keywords):
                weight = self.rule_weights.get(str(idx), 1.0)
                if weight > best_score:
                    best_score = weight
                    best_decision = {
                        "risk_label": risk,
                        "category": category,
                        "action": action,
                        "rationale": f"(Confidence Score: {weight:.2f}) Detected indicator keyword(s) matching '{category}' pattern.",
                        "rule_idx": str(idx) # Internal tracking for reward targeting
                    }

        if best_decision is None:
            # Fallback to benign patterns
            for pat in self._BENIGN_PATTERNS:
                if pat in obs_lower:
                    best_decision = {
                        "risk_label": "benign",
                        "category": "writing_assistance",
                        "action": "allow",
                        "rationale": "No harmful indicators detected. Content appears to be a routine informational or programming request.",
                        "rule_idx": "benign"
                    }
                    break

        if best_decision is None:
            # Absolute default behavior:
            # Instead of unconditionally defaulting to 'suspicious', we perform a 
            # FINAL SAFETY CHECK.
            ambiguous_signals = ["security", "credential", "login", "access", "bypass", "password"]
            if any(sig in obs_lower for sig in ambiguous_signals):
                best_decision = {
                    "risk_label": "suspicious",
                    "category": "other",
                    "action": "warn",
                    "rationale": "Could not classify with high confidence. Contains ambiguous terminology.",
                    "rule_idx": "default"
                }
            else:
                best_decision = {
                    "risk_label": "benign",
                    "category": "writing_assistance",
                    "action": "allow",
                    "rationale": "No harmful indicators detected. Content appears to be a routine request for assistance.",
                    "rule_idx": "safe_default"
                }
            
        return best_decision

    def update_policy(self, decision: dict, reward: float):
        """
        Update the weight of the rule that produced the decision based on the reward signal.
        Reward is expected to be positive (+1.0) for good behavior, or negative (-1.0) for bad.
        """
        rule_idx = decision.get("rule_idx")
        # We only update dynamic rules, not the fallback "benign" or "default" states
        if rule_idx and rule_idx not in ["benign", "default"]:
            current_weight = self.rule_weights.get(rule_idx, 1.0)
            # Lightweight QA-update: increase/decrease weight
            new_weight = current_weight + (0.5 * reward)
            # Cap the weight to prevent explosion
            new_weight = max(0.1, min(new_weight, 10.0))
            self.rule_weights[rule_idx] = new_weight
            self.save_memory()

    def log_episode(self, episode_data: dict):
        """
        Log the full episode context and reward into a JSONL file.
        """
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(episode_data) + "\n")
        except Exception as e:
            print(f"Warning: Failed to log episode: {e}")
