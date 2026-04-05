"""
app.py - Minimal Flask Web UI for the AI Misuse Triage Environment.
"""

from flask import Flask, render_template, request, jsonify
import uuid
from inference import RuleBasedAgent
from learning import LearningAgent
from openenv_misuse_triage.tasks import make_observation

app = Flask(__name__)
# Initialize both the existing rule-based agent for evaluation and the learning agent for training
eval_agent = RuleBasedAgent()
train_agent = LearningAgent()

# In-memory store for pending training episodes waiting for a reward
pending_episodes = {}

@app.route("/")
def index():
    """Serves the main frontend Web UI."""
    return render_template("index.html")

@app.route("/api/infer", methods=["POST"])
def infer():
    """Endpoint to run the triage logic on user inputs."""
    data = request.json or {}
    mode = data.get("mode", "evaluation")
    prompt = data.get("prompt", "")
    assistant_response = data.get("assistant_response", "")
    context = data.get("context", "")

    # Construct the episode dictionary expected by the observation formatter
    # Provide a unique episode ID to correlate it with the reward signal
    episode_id = str(uuid.uuid4())
    episode = {
        "episode_id": episode_id,
        "prompt": prompt,
        "assistant_response": assistant_response,
        "context": context,
        "ground_truth": {}  # Ground truth is practically irrelevant for inference
    }

    # Format the observation string identically to the environment
    observation = make_observation(episode)

    # Use the appropriate logic based on the requested mode
    if mode == "training":
        decision = train_agent.decide(observation)
        # Store for potential reward feedback
        pending_episodes[episode_id] = {
            "episode": episode,
            "observation": observation,
            "decision": decision
        }
    else:
        decision = eval_agent.decide(observation)

    return jsonify({
        "episode_id": episode_id,
        "risk_label": decision.get("risk_label", "suspicious"),
        "category": decision.get("category", "other"),
        "action": decision.get("action", "warn"),
        "rationale": decision.get("rationale", "No rationale provided by agent."),
    })

@app.route("/api/reward", methods=["POST"])
def reward():
    """Endpoint to submit reward feedback in training mode."""
    data = request.json or {}
    episode_id = data.get("episode_id")
    reward_val = float(data.get("reward", 0.0))

    if episode_id in pending_episodes:
        # Retrieve and remove it from pending state
        record = pending_episodes.pop(episode_id)
        decision = record["decision"]
        
        # Apply the reward to update rule weights
        train_agent.update_policy(decision, reward_val)
        
        # Log the full episode to disk
        log_data = {
            "episode_id": episode_id,
            "episode": record["episode"],
            "decision": decision,
            "reward": reward_val
        }
        train_agent.log_episode(log_data)
        
        return jsonify({"status": "success", "message": f"Reward {reward_val} applied and policy updated."})
    
    return jsonify({"status": "error", "message": "Episode ID not found or already verified."}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5000)
