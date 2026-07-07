"""Content Creation capability: draft articles, copy, or social posts on request."""

from __future__ import annotations

from ._shared import run_text


def run(state: dict) -> dict:
    """Draft the requested content (article, copy, social post), informed by the context."""
    return {
        "draft": run_text(
            state,
            "You are a writing assistant for a family reading context. Draft the content the "
            "user asked for (article, copy, social post, etc.). Match the requested format, "
            "length, and tone; keep it usable as-is.",
        )
    }
