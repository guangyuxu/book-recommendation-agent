"""Content Creation capability: draft articles, copy, or social posts on request."""

from __future__ import annotations

from typing import Any

from ..llm import STANDARD
from ._shared import run_text


def run(state: dict[str, Any]) -> dict[str, Any]:
    """Draft the requested content (article, copy, social post), informed by the context."""
    return {
        "draft": run_text(
            state,
            "You are a writing assistant for a family reading context. Draft the content the "
            "user asked for (article, copy, social post, etc.). Match the requested format, "
            "length, and tone; keep it usable as-is.",
            strategy=STANDARD,
        )
    }
