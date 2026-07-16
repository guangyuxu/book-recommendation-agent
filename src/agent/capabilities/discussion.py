"""Reading Discussion capability: generate post-reading discussion questions."""

from __future__ import annotations

from typing import Any

from ..llm import STANDARD
from ._shared import run_text


def run(state: dict[str, Any]) -> dict[str, Any]:
    """Generate age-appropriate discussion/reflection questions about a book for the child."""
    return {"questions": run_text(state, "discussion.questions", strategy=STANDARD)}
