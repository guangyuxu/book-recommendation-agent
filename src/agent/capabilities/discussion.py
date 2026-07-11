"""Reading Discussion capability: generate post-reading discussion questions."""

from __future__ import annotations

from ._shared import run_text


def run(state: dict) -> dict:
    """Generate age-appropriate discussion/reflection questions about a book for the child."""
    return {
        "questions": run_text(
            state,
            "You are a reading mentor. Generate a short set of open-ended discussion and "
            "reflection questions to guide this child's thinking about the book they read. "
            "Tune the depth to the child's age and reading level.",
        )
    }
