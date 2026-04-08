"""
Root-level grader shim for OpenEnv task config.

openenv.yaml references:
  module: "grader"
  function: "grade_flexible"

This file forwards those calls to the package grader implementation.
"""

from __future__ import annotations

from openenv_misuse_triage.grader import (
    grade as _package_grade,
    grade_batch,
    grade_flexible as _package_grade_flexible,
)


def _as_strict_float(value: object) -> float:
    """Normalize any upstream return type to a strict float in (0, 1)."""
    try:
        # Handle dataclass/object return paths with a score attribute.
        if hasattr(value, "score"):
            value = getattr(value, "score")
        score = float(value)
    except Exception:
        score = 0.01
    return max(0.001, min(0.999, score))


def grade(*args, **kwargs) -> float:
    """
    Compatibility entrypoint for validators that call `grade` directly.
    Always returns a strict float score.
    """
    try:
        return _as_strict_float(_package_grade_flexible(*args, **kwargs))
    except Exception:
        return 0.01


def grade_flexible(*args, **kwargs) -> float:
    """Primary entrypoint; always returns a strict float score."""
    try:
        return _as_strict_float(_package_grade_flexible(*args, **kwargs))
    except Exception:
        return 0.01


def grade_task(*args, **kwargs) -> float:
    return grade_flexible(*args, **kwargs)


def grade_score(*args, **kwargs) -> float:
    return grade_flexible(*args, **kwargs)


def grade_entry(*args, **kwargs) -> float:
    return grade_flexible(*args, **kwargs)

__all__ = [
    "grade",
    "grade_flexible",
    "grade_batch",
    "grade_task",
    "grade_score",
    "grade_entry",
]
