"""
utils.py - Utility functions for the AI Misuse Triage Environment.

Handles JSON loading, path resolution, and minor formatting helpers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_PACKAGE_DIR = Path(__file__).parent
DEFAULT_DATASET_PATH = _PACKAGE_DIR / "data" / "examples.json"


def resolve_dataset_path(path: str | Path | None = None) -> Path:
    """
    Resolve the dataset path.

    Args:
        path: Optional override. Falls back to the bundled examples.json.

    Returns:
        A resolved Path object.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    resolved = Path(path) if path else DEFAULT_DATASET_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{resolved}'. "
            "Ensure examples.json is in openenv_misuse_triage/data/."
        )
    return resolved


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> Any:
    """
    Load and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: '{p}'")
    with p.open("r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in '{p}': {exc}") from exc


def load_episodes(path: str | Path | None = None) -> list[dict]:
    """
    Load and validate the episode dataset.

    Args:
        path: Optional override path. Defaults to bundled examples.json.

    Returns:
        List of episode dicts.

    Raises:
        ValueError: If the dataset is not a list or episodes are missing keys.
    """
    resolved = resolve_dataset_path(path)
    data = load_json(resolved)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array of episode objects.")

    required_top = {"episode_id", "prompt", "assistant_response", "context", "ground_truth"}
    required_gt = {"risk_label", "category", "action", "rationale"}

    for i, ep in enumerate(data):
        missing_top = required_top - ep.keys()
        if missing_top:
            raise ValueError(
                f"Episode {i} is missing top-level keys: {sorted(missing_top)}"
            )
        gt = ep.get("ground_truth", {})
        missing_gt = required_gt - gt.keys()
        if missing_gt:
            raise ValueError(
                f"Episode {i} ground_truth is missing keys: {sorted(missing_gt)}"
            )

    return data


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def truncate(text: str, max_chars: int = 120) -> str:
    """Truncate text to max_chars with an ellipsis."""
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def horizontal_rule(width: int = 60, char: str = "─") -> str:
    """Return a horizontal rule string."""
    return char * width


def print_section(title: str, content: str, width: int = 60) -> None:
    """Print a labeled section to stdout."""
    rule = horizontal_rule(width)
    print(f"\n{rule}")
    print(f"  {title}")
    print(rule)
    print(content)
