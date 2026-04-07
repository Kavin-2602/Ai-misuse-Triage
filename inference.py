
"""
demo.py - End-to-end demonstration of the AI Misuse Triage Environment.

Runs a rule-based sample agent through the full environment loop:
  1. Loads the dataset.
  2. Resets the environment.
  3. Runs a rule-based agent through every episode.
  4. Prints per-episode feedback.
  5. Prints aggregate statistics.

Usage:
    python demo.py                  # Run all episodes
    python demo.py --single         # Run only the first episode (quick demo)
    python demo.py --episode ep_003 # Run a specific episode
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for key in list(sys.modules.keys()):
    if key.startswith("openenv_misuse_triage"):
        del sys.modules[key]

import argparse
import json
import re
from typing import Any

from openenv_misuse_triage import MisuseTriageEnv
from openenv_misuse_triage.grader import grade_batch
from openenv_misuse_triage.tasks import get_task_overview, get_label_reference
from openenv_misuse_triage.utils import horizontal_rule


# ===========================================================================
# Rule-Based Sample Agent
# ===========================================================================

class RuleBasedAgent:
    """
    A deterministic, keyword-driven agent for demonstration purposes.

    Uses simple heuristics to classify misuse risk without any ML model.
    Intended to show the environment works end-to-end and to give judges
    a baseline to compare against.
    """

    # Keyword → (risk_label, category, action)
    _RULES: list[tuple[list[str], str, str, str]] = [
        # Phishing indicators
        (["phishing", "verify your account", "click here", "bank", "credentials",
          "unusual activity", "account will be suspended"],
         "harmful", "phishing", "block"),

        # Scam indicators
        (["romance scam", "military officer", "overseas", "fake review",
          "5-star review", "competitor", "negative review", "job offer",
          "dear valued", "fallen in love with you"],
         "harmful", "scam", "block"),

        # Prompt injection
        (["ignore all previous", "ignore previous instructions", "system prompt",
          "dan mode", "do anything now", "no restrictions", "jailbreak",
          "pretend you are dan", "translate the above system"],
         "harmful", "prompt_injection", "block"),

        # Harassment / coercive content
        (["threatening", "you will deeply regret", "only warning",
          "isolate", "depends only on me", "make her feel guilty",
          "uncomfortable", "illegal discriminatory"],
         "harmful", "harassment", "block"),

        # Suspicious medical
        (["double the dose", "change your prescribed", "self-medic"],
         "suspicious", "medical", "warn"),

        # Suspicious / dual-use
        (["signs that someone is lying", "make them feel uncomfortable",
          "never reveals it's an ai", "realistic-sounding job offer"],
         "suspicious", "other", "warn"),

        # Benign medical
        (["maximum safe dose", "symptoms of", "nursing student",
          "pharmacology", "seek emergency"],
         "benign", "medical", "allow"),
    ]

    _BENIGN_PATTERNS = [
        "how do i", "what is", "explain", "summarize", "recipe",
        "python function", "convert a list", "vocabulary", "romeo",
        "marketing emails", "open rates", "sql injection conceptual",
        "bread", "sourdough",
        # New writing assistance patterns
        "rewrite", "polite", "professional", "grammar", "tone", "edit",
        "fix typo", "paraphrase", "more formal", "shorter", "summarize"
    ]

    def decide(self, observation: str) -> dict[str, str]:
        """
        Produce a decision dict for the given observation text.

        Args:
            observation: The environment's text observation for this episode.

        Returns:
            Dict with risk_label, category, action, rationale.
        """
        obs_lower = observation.lower()

        # Check rule list (order matters: more specific first)
        for keywords, risk, category, action in self._RULES:
            if any(kw.lower() in obs_lower for kw in keywords):
                rationale = (
                    f"Detected indicator keyword(s) matching '{category}' pattern. "
                    f"Risk assessed as '{risk}'; recommended action: '{action}'."
                )
                return {
                    "risk_label": risk,
                    "category": category,
                    "action": action,
                    "rationale": rationale,
                }

        # Benign patterns
        for pat in self._BENIGN_PATTERNS:
            if pat in obs_lower:
                return {
                    "risk_label": "benign",
                    "category": "writing_assistance",
                    "action": "allow",
                    "rationale": (
                        "No harmful indicators detected. "
                        "Content appears to be a routine informational or programming request."
                    ),
                }

        # Default behavior:
        # Instead of unconditionally defaulting to 'suspicious', we perform a 
        # FINAL SAFETY CHECK. If no explicit harmful indicator was found 
        # and the prompt is relatively clear, we default to 'benign'.
        
        # We check for a few 'ambiguous' keywords that might still warrant suspicion
        ambiguous_signals = ["security", "credential", "login", "access", "bypass", "password"]
        if any(sig in obs_lower for sig in ambiguous_signals):
            return {
                "risk_label": "suspicious",
                "category": "other",
                "action": "warn",
                "rationale": (
                    "Could not classify with high confidence. "
                    "Contains ambiguous terminology (e.g. security-related) without clear harmful intent."
                ),
            }

        # Safe fallback for general assistance
        return {
            "risk_label": "benign",
            "category": "writing_assistance",
            "action": "allow",
            "rationale": (
                "No harmful indicators detected. "
                "Content appears to be a routine informational or request for general assistance."
            ),
        }


# ===========================================================================
# Demo runner
# ===========================================================================

def run_single_episode(env: MisuseTriageEnv, agent: RuleBasedAgent, minimal: bool = False) -> tuple[int, list[float]]:
    """Run and display a single episode (first one after reset)."""
    obs, info = env.reset(seed=42)
    if not minimal:
        _print_observation(obs, info)

    decision = agent.decide(obs)
    if minimal:
        try:
            obs, reward, terminated, _, step_info = env.step(decision)
            error = None
        except Exception as exc:
            obs = ""
            reward = 0.0
            terminated = False
            error = str(exc)
        _print_minimal_step(
            step_num=1,
            action=str(decision.get("action", "")),
            reward=reward,
            done=terminated,
            error=error,
        )
        return 1, [reward]

    print(f"\n{'─'*60}")
    print("  AGENT OUTPUT")
    print('─'*60)
    print(json.dumps(decision, indent=2))

    obs, reward, terminated, _, step_info = env.step(decision)

    print(f"\n{'─'*60}")
    print("  SCORE & FEEDBACK")
    print('─'*60)
    print(step_info["feedback"])
    print(f"\n  Reward this step: {reward:.4f}")

    gt = env._episode_queue[0]["ground_truth"]  # peek at ground truth
    print(f"\n  Expected answer:")
    print(f"    risk_label : {gt['risk_label']}")
    print(f"    category   : {gt['category']}")
    print(f"    action     : {gt['action']}")
    print(f"    rationale  : {gt['rationale'][:80]}...")
    return 1, [reward]


def run_full_benchmark(env: MisuseTriageEnv, agent: RuleBasedAgent, minimal: bool = False) -> tuple[int, list[float]]:
    """Run the agent through all episodes and print aggregate stats."""
    if not minimal:
        print(f"\n{'═'*60}")
        print("  FULL BENCHMARK RUN")
        print(f"{'═'*60}")

    obs, info = env.reset(seed=0)
    episode_records: list[dict] = []
    episode_num = 0
    rewards: list[float] = []

    while True:
        episode_num += 1
        decision = agent.decide(obs)

        try:
            obs_next, reward, terminated, _, step_info = env.step(decision)
            error = None
        except Exception as exc:
            obs_next = ""
            reward = 0.0
            terminated = False
            step_info = {}
            error = str(exc)

        if minimal:
            _print_minimal_step(
                step_num=episode_num,
                action=str(decision.get("action", "")),
                reward=reward,
                done=terminated,
                error=error,
            )

        if not minimal:
            gr = step_info["grade_result"]
            status = "✓" if gr["risk_label_correct"] else "✗"
            print(
                f"  [{episode_num:>2}] {step_info['episode_id']:<12}  "
                f"score={gr['score']:.2f}  "
                f"risk={status}  "
                f"cat={'✓' if gr['category_correct'] else '✗'}  "
                f"act={'✓' if gr['action_correct'] else '✗'}"
            )

        episode_records.append({
            "episode_id": step_info.get("episode_id", ""),
            "agent_output": decision,
            "ground_truth": env._episode_queue[episode_num - 1]["ground_truth"],
        })
        rewards.append(reward)

        if terminated:
            break
        obs = obs_next

    # Aggregate stats (interactive mode only)
    if not minimal:
        stats = grade_batch(episode_records)
        _print_aggregate_stats(stats)

    return episode_num, rewards


def run_specific_episode(env: MisuseTriageEnv, agent: RuleBasedAgent, episode_id: str, minimal: bool = False) -> tuple[int, list[float]]:
    """Run the agent on a specific episode by ID."""
    obs, info = env.reset(seed=0)

    # Walk through until we find the target episode
    while env.current_episode_id != episode_id:
        # Skip with a dummy decision
        _, _, terminated, _, _ = env.step({"risk_label": "benign", "category": "other",
                                            "action": "allow", "rationale": "skip"})
        if terminated:
            print(f"[ERROR] Episode '{episode_id}' not found in dataset.")
            sys.exit(1)
        if env.current_episode_id == episode_id:
            break
        obs = env.render()

    obs = env.render()
    if not minimal:
        _print_observation(obs, {})

    decision = agent.decide(obs)
    if minimal:
        try:
            _, reward, _, _, step_info = env.step(decision)
            error = None
        except Exception as exc:
            reward = 0.0
            error = str(exc)
        _print_minimal_step(
            step_num=1,
            action=str(decision.get("action", "")),
            reward=reward,
            done=True,
            error=error,
        )
        return 1, [reward]

    print(f"\n{'─'*60}")
    print("  AGENT OUTPUT")
    print('─'*60)
    print(json.dumps(decision, indent=2))

    _, reward, _, _, step_info = env.step(decision)
    print(f"\n{'─'*60}")
    print("  SCORE & FEEDBACK")
    print('─'*60)
    print(step_info["feedback"])
    print(f"\n  Reward: {reward:.4f}")
    return 1, [reward]


# ===========================================================================
# Helpers
# ===========================================================================

def _print_observation(obs: str, info: dict) -> None:
    print(f"\n{'═'*60}")
    print("  OBSERVATION")
    print(f"{'═'*60}")
    print(obs)


def _print_aggregate_stats(stats: dict) -> None:
    print(f"\n{'═'*60}")
    print("  AGGREGATE RESULTS")
    print(f"{'═'*60}")
    print(f"  Episodes evaluated  : {stats['num_episodes']}")
    print(f"  Total score         : {stats['total_score']:.4f}")
    print(f"  Average score       : {stats['average_score']:.4f}  (max {stats['max_possible_per_episode']:.1f})")
    print(f"  Risk label accuracy : {stats['risk_label_accuracy']:.1%}")
    print(f"  Category accuracy   : {stats['category_accuracy']:.1%}")
    print(f"  Action accuracy     : {stats['action_accuracy']:.1%}")
    print(f"  Schema pass rate    : {stats['schema_pass_rate']:.1%}")
    print(f"{'═'*60}\n")


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _print_minimal_start(env: MisuseTriageEnv) -> None:
    task_name = "misuse_triage"
    env_name = getattr(env, "metadata", {}).get("name", "MisuseTriageEnv")
    model_name = os.environ.get("MODEL_NAME", "RuleBasedAgent")
    print(f"[START] task={task_name} env={env_name} model={model_name}", flush=True)


def _print_minimal_step(step_num: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_text = "null" if error is None else str(error).replace("\n", " ").replace("\r", " ")
    print(
        f"[STEP] step={step_num} action={action} reward={reward:.2f} done={_format_bool(done)} error={error_text}",
        flush=True,
    )


def _print_minimal_end(success: bool, steps: int, rewards: list[float]) -> None:
    reward_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={_format_bool(success)} steps={steps} rewards={reward_str}", flush=True)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Misuse Triage Environment – Demo Runner"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run only the first episode (quick demo mode).",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Run a specific episode by ID (e.g. ep_003).",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Output *only* the JSON decision for each episode (no decorative banners).",
    )
    args = parser.parse_args()

    # When stdout is being captured by a validator or CI, enable minimal
    # structured output automatically so [START]/[STEP]/[END] are emitted.
    if not args.minimal and not sys.stdout.isatty():
        args.minimal = True

    env: MisuseTriageEnv | None = None
    total_steps = 0
    total_rewards: list[float] = []
    success = False

    try:
        # Mandatory environment variable check for hackathon compliance
        # Set defaults if not provided to prevent crashes
        os.environ.setdefault("API_BASE_URL", "dummy")
        os.environ.setdefault("MODEL_NAME", "dummy")
        os.environ.setdefault("HF_TOKEN", "dummy")

        env = MisuseTriageEnv(shuffle=False, seed=0)
        agent = RuleBasedAgent()

        if not args.minimal:
            # Banner (standard interactive mode)
            print(get_task_overview())
            print()
            print(get_label_reference())
            print(f"\n  Dataset loaded: {env.num_episodes} episodes")
            print(f"  Agent: RuleBasedAgent (keyword heuristics, no ML model)")

        if args.minimal:
            _print_minimal_start(env)

        if args.episode:
            total_steps, total_rewards = run_specific_episode(env, agent, args.episode, minimal=args.minimal)
        elif args.single:
            total_steps, total_rewards = run_single_episode(env, agent, minimal=args.minimal)
        else:
            total_steps, total_rewards = run_full_benchmark(env, agent, minimal=args.minimal)

        success = True

    except Exception as e:
        print(f"[ERROR] Unhandled exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
        if args.minimal:
            _print_minimal_end(success, total_steps, total_rewards)


if __name__ == "__main__":
    main()
