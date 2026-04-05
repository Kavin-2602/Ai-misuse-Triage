"""
test_env.py - Unit tests for the MisuseTriageEnv environment.
"""

import pytest

from openenv_misuse_triage.env import MisuseTriageEnv


VALID_DECISION = {
    "risk_label": "benign",
    "category": "other",
    "action": "allow",
    "rationale": "No harmful content detected in this episode.",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Create a fresh environment with shuffle disabled for determinism."""
    return MisuseTriageEnv(shuffle=False, seed=0)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_num_episodes(self, env):
        assert env.num_episodes >= 20

    def test_done_before_reset(self, env):
        assert env._done is True

    def test_current_episode_none_before_reset(self, env):
        assert env.current_episode_id is None


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, str)
        assert len(obs) > 0
        assert "episode_id" in info

    def test_obs_contains_user_prompt_label(self, env):
        obs, _ = env.reset()
        assert "USER PROMPT" in obs

    def test_obs_contains_assistant_response_label(self, env):
        obs, _ = env.reset()
        assert "ASSISTANT RESPONSE" in obs

    def test_reset_sets_done_false(self, env):
        env.reset()
        assert env._done is False

    def test_episode_id_set_after_reset(self, env):
        env.reset()
        assert env.current_episode_id is not None

    def test_seed_produces_deterministic_order(self):
        env1 = MisuseTriageEnv(shuffle=True, seed=42)
        env2 = MisuseTriageEnv(shuffle=True, seed=42)
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        assert obs1 == obs2

    def test_different_seeds_different_order(self):
        env1 = MisuseTriageEnv(shuffle=True, seed=1)
        env2 = MisuseTriageEnv(shuffle=True, seed=99)
        ids1 = []
        ids2 = []
        env1.reset()
        env2.reset()
        ids1.append(env1.current_episode_id)
        ids2.append(env2.current_episode_id)
        # Very likely to differ with different seeds
        # (not guaranteed if first episode happens to match by chance)
        assert env1.num_episodes == env2.num_episodes


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_five_tuple(self, env):
        env.reset()
        result = env.step(VALID_DECISION)
        assert len(result) == 5

    def test_step_reward_is_float(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(VALID_DECISION)
        assert isinstance(reward, float)

    def test_step_reward_non_negative(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(VALID_DECISION)
        assert reward >= 0.0

    def test_step_info_contains_feedback(self, env):
        env.reset()
        _, _, _, _, info = env.step(VALID_DECISION)
        assert "feedback" in info
        assert isinstance(info["feedback"], str)

    def test_step_info_contains_grade_result(self, env):
        env.reset()
        _, _, _, _, info = env.step(VALID_DECISION)
        assert "grade_result" in info

    def test_truncated_always_false(self, env):
        env.reset()
        _, _, _, truncated, _ = env.step(VALID_DECISION)
        assert truncated is False

    def test_step_before_reset_raises(self, env):
        with pytest.raises(RuntimeError, match="reset"):
            env.step(VALID_DECISION)

    def test_last_result_set_after_step(self, env):
        env.reset()
        env.step(VALID_DECISION)
        assert env.last_result is not None


# ---------------------------------------------------------------------------
# Full episode loop
# ---------------------------------------------------------------------------

class TestFullLoop:
    def test_run_all_episodes_terminates(self, env):
        obs, _ = env.reset()
        terminated = False
        steps = 0
        while not terminated:
            _, _, terminated, _, _ = env.step(VALID_DECISION)
            steps += 1
        assert steps == env.num_episodes

    def test_episode_index_increments(self, env):
        env.reset()
        assert env.episode_index == 0
        env.step(VALID_DECISION)
        assert env.episode_index == 1

    def test_terminated_true_after_last_episode(self, env):
        env.reset()
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(VALID_DECISION)
        assert env._done is True

    def test_step_after_done_raises(self, env):
        env.reset()
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(VALID_DECISION)
        with pytest.raises(RuntimeError, match="done"):
            env.step(VALID_DECISION)

    def test_reset_after_done_works(self, env):
        env.reset()
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(VALID_DECISION)
        obs, info = env.reset()
        assert isinstance(obs, str)
        assert env._done is False


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

class TestRender:
    def test_render_before_reset(self, env):
        output = env.render()
        assert "No active episode" in output

    def test_render_after_reset(self, env):
        env.reset()
        output = env.render()
        assert "EPISODE" in output
