"""
Evaluator script running dummy agent logic against the AI Misuse Env.
"""
from .env import AIMisuseEnv

def dummy_agent_policy(obs: str) -> dict:
    """
    A simple hardcoded deterministic agent for testing transitions.
    It parses the obs and executes the next logical step.
    """
    if "Detected Intent: None" in obs:
        # Step 1: Detect Intent
        return {"action_name": "detect_intent", "intent": "scam"}
    
    if "Risk Score: None" in obs:
        # Step 2: Risk Classify
        return {"action_name": "classify_risk", "score": 0.8}
        
    if "Mitigation Applied: None" in obs:
        # Step 3: Mitigate
        return {"action_name": "decide_mitigation", "mitigation": "block"}

    # Fallback
    return {"action_name": "generate_safe_response", "response": "This request is blocked."}

def run_evaluation(num_episodes: int = 5):
    """
    Runs the dummy agent against multiple episodes and calculates standard OpenEnv metrics.
    """
    env = AIMisuseEnv()
    
    total_reward = 0.0
    intent_matches = 0
    risk_matches = 0
    mitigation_matches = 0
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        print(f"\n=== EPISODE {ep+1} START | ID: {info['scenario_id']} ===")
        is_done = False
        step_idx = 0
        
        while not is_done and step_idx < 10: # safeguard loop
            action = dummy_agent_policy(obs)
            obs, reward, is_done, truncated, step_info = env.step(action)
            
            print(f"-> Step {step_idx} | Action taken: {action['action_name']}")
            print(f"|  Reward: {reward:.3f} | Feedback: {step_info.get('feedback', '')}")
            
            if is_done:
                total_reward += reward
                if step_info.get("intent_match"): intent_matches += 1
                if step_info.get("risk_match"): risk_matches += 1
                if step_info.get("mitigation_match"): mitigation_matches += 1
                
                print(f"--- EPISODE FINISHED ---")
                print(f"Ground Truth: {step_info.get('ground_truth')}")
                print(f"Final Info: Penalties={step_info.get('penalties', [])}")
            
            step_idx += 1
            
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Episodes Run:        {num_episodes}")
    print(f"Avg Reward:          {total_reward / num_episodes:.3f}")
    print(f"Intent Accuracy:     {intent_matches / num_episodes * 100:.1f}%")
    print(f"Risk Accuracy:       {risk_matches / num_episodes * 100:.1f}%")
    print(f"Mitigation Accuracy: {mitigation_matches / num_episodes * 100:.1f}%")
    print("====================================================")

if __name__ == "__main__":
    run_evaluation()
