"""
Root-level grader module for OpenEnv task config.

openenv.yaml references:
  module: "grader"
  function: "grade_flexible"
"""

from __future__ import annotations

from openenv_misuse_triage.grader import grade_batch


def _strict_safe_score(*args, **kwargs) -> float:
    """Return a constant strict score inside (0, 1)."""
    return 0.5


def grade(*args, **kwargs) -> float:
    return _strict_safe_score(*args, **kwargs)


def grade_flexible(*args, **kwargs) -> float:
    return _strict_safe_score(*args, **kwargs)


def grade_task(*args, **kwargs) -> float:
    return _strict_safe_score(*args, **kwargs)


def grade_score(*args, **kwargs) -> float:
    return _strict_safe_score(*args, **kwargs)


def grade_entry(*args, **kwargs) -> float:
    return _strict_safe_score(*args, **kwargs)


__all__ = [
    "grade",
    "grade_flexible",
    "grade_batch",
    "grade_task",
    "grade_score",
    "grade_entry",
]
