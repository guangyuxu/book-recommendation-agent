"""Compare Books capability: weigh two or more named books against each other."""

from __future__ import annotations

from typing import Any

from ..llm import HEAVY
from ._shared import run_text


def run(state: dict[str, Any]) -> dict[str, Any]:
    """Compare the named books for this child (fit, difficulty, themes) and give a verdict."""
    return {
        "comparison": run_text(
            state,
            "You are a children's-book analyst. The user named two or more books. Compare them "
            "for this child across fit, reading difficulty, themes, and values, then give a "
            "clear recommendation on which to read first and why.",
            strategy=HEAVY,
        )
    }
