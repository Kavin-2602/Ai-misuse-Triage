"""
server/app.py - OpenEnv Server Entry Point for AI Misuse Triage Environment.
"""
import uuid
import sys
import os

# Add root folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openenv.core.env_server.http_server import create_app

from server.misuse_triage_environment import MisuseTriageEnvironment
from openenv_misuse_triage.models import MisuseTriageAction, MisuseTriageObservation

# Initialize auxiliary agents if present in environment
try:
    from inference import RuleBasedAgent
    eval_agent = RuleBasedAgent()
except ImportError:
    eval_agent = None

try:
    from learning import LearningAgent
    train_agent = LearningAgent()
except ImportError:
    train_agent = None

# 1. Create the OpenEnv platform backend APIs
# This adds /reset, /step, /state, /schema automatically.
app = create_app(
    MisuseTriageEnvironment,
    MisuseTriageAction,
    MisuseTriageObservation,
    env_name="misuse_triage",
    max_concurrent_envs=10
)

# 2. Port the Web UI elements
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

pending_episodes = {}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serves the main frontend Web UI."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/infer")
@app.post("/api/infer")
async def infer(request: Request):
    """Endpoint to run the triage logic on user inputs."""
    data = await request.json()
    mode = data.get("mode", "evaluation")
    prompt = data.get("prompt", "")
    assistant_response = data.get("assistant_response", "")
    context = data.get("context", "")

    episode_id = str(uuid.uuid4())
    episode = {
        "episode_id": episode_id,
        "prompt": prompt,
        "assistant_response": assistant_response,
        "context": context,
        "ground_truth": {}
    }

    try:
        from openenv_misuse_triage.tasks import make_observation
        observation = make_observation(episode)
    except:
        observation = ""

    decision = {}
    if mode == "training" and train_agent:
        decision = train_agent.decide(observation)
        pending_episodes[episode_id] = {
            "episode": episode,
            "observation": observation,
            "decision": decision
        }
    elif eval_agent:
        decision = eval_agent.decide(observation)

    return JSONResponse(content={
        "episode_id": episode_id,
        "risk_label": decision.get("risk_label", "suspicious"),
        "category": decision.get("category", "other"),
        "action": decision.get("action", "warn"),
        "rationale": decision.get("rationale", "No rationale provided by agent."),
    })

@app.post("/api/reward")
async def reward(request: Request):
    """Endpoint to submit reward feedback in training mode."""
    data = await request.json()
    episode_id = data.get("episode_id")
    reward_val = float(data.get("reward", 0.0))

    if episode_id in pending_episodes:
        record = pending_episodes.pop(episode_id)
        decision = record["decision"]
        if train_agent:
            train_agent.update_policy(decision, reward_val)
            train_agent.log_episode({
                "episode_id": episode_id,
                "episode": record["episode"],
                "decision": decision,
                "reward": reward_val
            })
        return JSONResponse(content={"status": "success", "message": f"Reward {reward_val} applied."})

    return JSONResponse(status_code=404, content={"status": "error", "message": "Episode ID not found."})

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)