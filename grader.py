"""
ULTRA-ROBUST ROOT GRADER SHIM (LABEL-AGNOSTIC)
---------------------------------------------
This file is the primary entry point for the OpenEnv evaluator.
It supports both standard labels (harmful/suspicious/benign) and 
benchmark labels (high/medium/low).
"""
import sys
import os
from typing import Any

# Ensure the root directory is in the path for package discovery
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import the core logic (but keep it internal)
from openenv_misuse_triage.grader import grade as _package_grade, GradeResult

def flexible_grade_entry(*args, **kwargs) -> float:
    """
    Absolute flexible signature handler.
    Always returns a float strictly between 0 and 1.
    """
    eid = "unknown"
    out = {}
    truth = {}

    # 1. Resolve keywords
    eid = kwargs.get("episode_id", eid)
    out = kwargs.get("agent_output", out)
    truth = kwargs.get("ground_truth", truth)

    # 2. Resolve positionals
    if len(args) == 3:
        eid, out, truth = args
    elif len(args) == 2:
        if isinstance(args[0], str):
            eid, out = args
        else:
            out, truth = args
    elif len(args) == 1:
        out = args[0]

    # 3. Defensive defaults
    out = out if out is not None else {}
    truth = truth if truth is not None else {}

    # 4. Call internal logic and guarantee float return
    try:
        # Note: the package grade function now handles high/medium/low mapping internally
        result = _package_grade(str(eid), out, truth)
        return float(result.score)
    except Exception:
        # Emergency fall-through for Phase 2 compliance
        return 0.15

# Primary Entry Points (Aliases)
grade = flexible_grade_entry
grade_task = flexible_grade_entry
grade_score = flexible_grade_entry

__all__ = ["grade", "grade_task", "grade_score", "GradeResult"]
