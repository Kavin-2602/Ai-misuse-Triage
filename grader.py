"""
Root grader shim to ensure OpenEnv evaluator can discover the grading logic.
"""
from openenv_misuse_triage.grader import grade_score, grade, GradeResult

__all__ = ["grade_score", "grade", "GradeResult"]
