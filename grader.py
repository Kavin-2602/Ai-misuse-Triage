"""
Root-level grader shim for OpenEnv task config.

openenv.yaml references:
  module: "grader"
  function: "grade_flexible"

This file forwards those calls to the package grader implementation.
"""

from __future__ import annotations

from openenv_misuse_triage.grader import (
    grade,
    grade_batch,
    grade_entry,
    grade_flexible,
    grade_score,
    grade_task,
)

__all__ = [
    "grade",
    "grade_flexible",
    "grade_batch",
    "grade_task",
    "grade_score",
    "grade_entry",
]
