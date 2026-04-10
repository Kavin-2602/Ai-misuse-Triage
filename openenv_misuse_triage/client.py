"""
openenv_misuse_triage/client.py - OpenEnv Client for AI Misuse Triage Environment.
"""

import json
from typing import Dict, Optional, Any
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from models import MisuseTriageAction, MisuseTriageObservation
except ImportError:
    from .models import MisuseTriageAction, MisuseTriageObservation

class MisuseTriageClientEnv(EnvClient[MisuseTriageAction, MisuseTriageObservation, State]):
    """
    Client for the AI Misuse Triage Environment.
    Use this to programmatically interact with a hosted web-socket environment.
    """

    def _step_payload(self, action: MisuseTriageAction) -> Dict:
        """Serialize action to dict before sending over the wire."""
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[MisuseTriageObservation]:
        """Deserialize server response."""
        obs_data = payload.get("observation", {})
        observation = MisuseTriageObservation(
            prompt=obs_data.get("prompt"),
            assistant_response=obs_data.get("assistant_response"),
            context=obs_data.get("context"),
            ground_truth=obs_data.get("ground_truth"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {})
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: Dict) -> State:
        """Deserialize state payload."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0)
        )
