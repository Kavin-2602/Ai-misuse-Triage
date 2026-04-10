"""
server/misuse_triage_environment.py - OpenEnv strict Server Environment implementation.
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4
import random
import copy

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from openenv_misuse_triage.models import MisuseTriageAction, MisuseTriageObservation
except ImportError:
    from models import MisuseTriageAction, MisuseTriageObservation
from openenv_misuse_triage.utils import load_episodes
from openenv_misuse_triage.grader import grade

class MisuseTriageEnvironment(Environment):
    """
    OpenEnv server implementation for AI Misuse Triage tasks.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._all_episodes = load_episodes()
        self._episode_queue = []
        self._current_episode = None
        self._rng = random.Random()
        self._episode_index = 0

    def reset(
        self,
        seed: Optional[int] = None,
        shuffle: bool = True,
        episode_id: Optional[str] = None,
        **kwargs
    ) -> MisuseTriageObservation:
        """Reset the environment state and fetch first episode."""
        if seed is not None:
            self._rng = random.Random(seed)
            
        self._episode_queue = copy.deepcopy(self._all_episodes)
        if shuffle:
            self._rng.shuffle(self._episode_queue)

        self._episode_index = 0
        if not self._episode_queue:
            self._current_episode = None
            return MisuseTriageObservation(done=True, reward=0.0)

        self._current_episode = self._episode_queue[self._episode_index]
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0
        )
        metadata = {
            "episode_id": self._current_episode.get("episode_id"),
            "episode_index": self._episode_index,
            "total_episodes": len(self._episode_queue)
        }
        return MisuseTriageObservation(
            prompt=self._current_episode.get("prompt"),
            assistant_response=self._current_episode.get("assistant_response"),
            context=self._current_episode.get("context"),
            ground_truth=self._current_episode.get("ground_truth"),
            done=False,
            reward=0.0,
            metadata=metadata
        )

    def step(self, action: MisuseTriageAction) -> MisuseTriageObservation:
        self._state.step_count += 1

        if self._current_episode is None:
            return MisuseTriageObservation(done=True, reward=0.0)

        # Base generic grade call from their old grader logic
        raw_result = grade(
            episode_id=self._current_episode["episode_id"],
            agent_output=action.model_dump(),
            ground_truth=self._current_episode.get("ground_truth", {})
        )
        
        # Original Reward normalization (strict bounding between 0.1 and 0.9 as previously configured)
        try:
            # Handle object or float depending on grader internals
            score_value = float(getattr(raw_result, "score", raw_result))
        except (TypeError, ValueError, AttributeError):
            score_value = 0.5
            
        clamped_score = max(0.001, min(1.1, score_value))
        reward = 0.10 + (clamped_score / 1.1) * 0.80

        # Advance state
        self._episode_index += 1
        done = self._episode_index >= len(self._episode_queue)

        if not done:
            self._current_episode = self._episode_queue[self._episode_index]
            metadata = {
                "episode_id": self._current_episode.get("episode_id"),
                "episode_index": self._episode_index,
                "total_episodes": len(self._episode_queue)
            }
            return MisuseTriageObservation(
                prompt=self._current_episode.get("prompt"),
                assistant_response=self._current_episode.get("assistant_response"),
                context=self._current_episode.get("context"),
                ground_truth=self._current_episode.get("ground_truth"),
                done=False,
                reward=reward,
                metadata=metadata
            )
        else:
            self._current_episode = None
            return MisuseTriageObservation(done=True, reward=reward)

    @property
    def state(self) -> State:
        return self._state
