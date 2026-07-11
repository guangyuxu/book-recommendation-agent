"""Reading Path Planning capability: a staged plan between reading levels or books."""

from __future__ import annotations

from typing import Any

from ._shared import run_text


def run(state: dict[str, Any]) -> dict[str, Any]:
    """Produce a staged reading path to move the child toward the next level or target."""
    return {
        "reading_path": run_text(
            state,
            "You are a reading coach. Design a staged reading path for this child: a sequence "
            "of books or book types that bridges from where they are now to the goal, with "
            "what each stage builds and roughly how to know they're ready to move on.",
        )
    }
