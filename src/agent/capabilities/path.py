"""Reading Path Planning capability: a staged plan between reading levels or books."""

from __future__ import annotations

from typing import Any

from ..llm import STANDARD
from ._shared import run_text


def run(state: dict[str, Any]) -> dict[str, Any]:
    """Produce a staged reading path to move the child toward the next level or target."""
    return {"reading_path": run_text(state, "path.plan", strategy=STANDARD)}
