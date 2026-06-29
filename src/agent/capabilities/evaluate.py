"""Evaluate Book capability: assess whether one named book suits the target child."""

from __future__ import annotations

from ._shared import run_text


def run(state: dict) -> dict:
    """Assess a single book's fit, themes, values, and reading difficulty for the child."""
    return run_text(
        state,
        "You are a children's-book analyst. The user named a book. Evaluate whether it suits "
        "this child: its themes, values, reading difficulty, and any content to be aware of. "
        "Be concrete and balanced.",
    )
