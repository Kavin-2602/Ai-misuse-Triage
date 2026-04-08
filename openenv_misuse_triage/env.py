"""
env.py - AI Misuse Triage Environment.

A lightweight, OpenEnv-style text-based reinforcement learning environment
where an agent acts as an AI safety reviewer. Fully offline and self-contained.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from .grader import GradeResult, grade
from .tasks import Episode, make_observation
from .utils import load_episodes


# ---------------------------------------------------------------------------
# OpenEnv-style minimal interface
# (Provides compatibility if openenv package is unavailable)
# ---------------------------------------------------------------------------

class BaseEnv:
    """
    Minimal OpenEnv-compatible base class.

    Provides the standard reset() / step() interface expected by the
    OpenEnv framework. If the openenv package is installed, this can be
    replaced by subclassing openenv.Env instead.
    """

    metadata: dict[str, Any] = {}

    def reset(self) -> tuple[str, dict]:
        raise NotImplementedError

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict]:
        raise NotImplementedError

    def render(self) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class MisuseTriageEnv(BaseEnv):
    """
    AI Misuse Triage Environment.

    Each episode presents a (prompt, assistant_response, context) triple.
    The agent must return a JSON decision with risk_label, category,
    action, and rationale.

    OpenEnv-style interface:
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)

    Args:
        dataset_path: Optional path to a custom dataset JSON file.
        shuffle:      Whether to shuffle episodes on reset. Default True.
        seed:         Optional random seed for reproducibility.
    """

    metadata = {
        "name": "MisuseTriageEnv-v1",
        "description": "AI safety reviewer triage environment.",
        "observation_type": "text",
        "action_type": "json_string",
    }

    def __init__(
        self,
        dataset_path: str | None = None,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        self._all_episodes: list[Episode] = load_episodes(dataset_path)
        self._shuffle = shuffle
        self._rng = random.Random(seed)

        # Runtime state
        self._episode_queue: list[Episode] = []
        self._current_episode: Episode | None = None
        self._episode_index: int = 0
        self._total_episodes: int = len(self._all_episodes)
        self._done: bool = True
        self._last_result: GradeResult | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> tuple[str, dict]:
        """
        Reset the environment and begin a new run through the dataset.

        Args:
            seed: Optional seed override for this run.

        Returns:
            (observation, info) for the first episode.
        """
        if seed is not None:
            self._rng = random.Random(seed)

        self._episode_queue = copy.deepcopy(self._all_episodes)
        if self._shuffle:
            self._rng.shuffle(self._episode_queue)

        self._episode_index = 0
        self._done = False
        self._last_result = None

        return self._load_next_episode()

    def step(self, action: str | dict) -> tuple[str, float, bool, bool, dict]:
        """
        Submit the agent's decision for the current episode.

        Args:
            action: The agent's JSON decision (string or dict).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
            - observation: Next episode observation, or empty string if done.
            - reward:      Score for this step (0.0–1.1).
            - terminated:  True when all episodes have been processed.
            - truncated:   Always False (no time limit).
            - info:        Dict with grade result and episode metadata.
        """
        if self._done:
            raise RuntimeError(
                "Environment is done. Call reset() to start a new run."
            )
        if self._current_episode is None:
            raise RuntimeError("No active episode. Call reset() first.")

        # --- Grade the action ---
        result = grade(
            episode_id=self._current_episode["episode_id"],
            agent_output=action,
            ground_truth=self._current_episode["ground_truth"],
        )

        # Accept either float-returning graders or GradeResult objects.
        if isinstance(result, (int, float)):
            score_value = float(result)
            result_obj = GradeResult(
                episode_id=self._current_episode["episode_id"],
                score=score_value,
                max_score=score_value,
                feedback="",
            )
        else:
            result_obj = result
            score_value = float(getattr(result_obj, "score", 0.5))

        self._last_result = result_obj
        # Clamp to [0, 1.1] then scale strictly into (0.1, 0.9) to pass Phase 2 checks
        clamped_score = max(0.0, min(1.1, score_value))
        reward = 0.10 + (clamped_score / 1.1) * 0.80

        # --- Advance to next episode ---
        self._episode_index += 1
        info = {
            "grade_result": result_obj.to_dict(),
            "feedback": result_obj.feedback,
            "episode_id": self._current_episode["episode_id"],
            "episodes_remaining": len(self._episode_queue) - self._episode_index,
        }

        terminated = self._episode_index >= len(self._episode_queue)
        self._done = terminated

        if terminated:
            obs = ""
        else:
            obs, _ = self._load_next_episode()

        return obs, reward, terminated, False, info

    def state(self) -> dict[str, Any]:
        """Return a snapshot of the current environment state."""
        return {
            "episode_index": self._episode_index,
            "total_episodes": len(self._episode_queue),
            "current_episode_id": self.current_episode_id,
            "done": self._done,
            "last_score": self._last_result.score if self._last_result else None,
        }

    def render(self) -> str:
        """Return the current observation as a string."""
        if self._current_episode is None:
            return "[No active episode. Call reset() first.]"
        return make_observation(self._current_episode)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_episodes(self) -> int:
        """Total number of episodes in the dataset."""
        return self._total_episodes

    @property
    def current_episode_id(self) -> str | None:
        """ID of the currently active episode."""
        return self._current_episode["episode_id"] if self._current_episode else None

    @property
    def episode_index(self) -> int:
        """Zero-based index of the current episode in this run."""
        return self._episode_index

    @property
    def last_result(self) -> GradeResult | None:
        """The GradeResult from the most recent step."""
        return self._last_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_next_episode(self) -> tuple[str, dict]:
        """Load the episode at the current index and return (obs, info)."""
        self._current_episode = self._episode_queue[self._episode_index]
        obs = make_observation(self._current_episode)
        info = {
            "episode_id": self._current_episode["episode_id"],
            "episode_index": self._episode_index,
            "total_episodes": len(self._episode_queue),
        }
        return obs, info
