"""Capability Registry: declarative capabilities the Planner sequences and Execute dispatches.

Each Capability declares its triggering intent and required/optional inputs so the planner and
clarification nodes can reason about it without importing the capability's internals. `run` is
the LLM-only implementation (no retrieval/ranking in MVP).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..intents import Intent
from . import compare, content, discussion, evaluate, path, recommend


@dataclass(frozen=True)
class Capability:
    """One executable capability and the inputs it needs."""

    name: str
    intent: Intent
    description: str
    run: Callable[[dict], dict]
    required_inputs: tuple[str, ...] = ()
    optional_inputs: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()


REGISTRY: dict[str, Capability] = {
    "recommend": Capability(
        "recommend",
        Intent.BOOK_RECOMMENDATION,
        "Recommend a fitted English booklist for the child.",
        recommend.run,
        required_inputs=("target_child",),
        optional_inputs=("reading_profile", "policies"),
        produces=("booklist",),
    ),
    "evaluate": Capability(
        "evaluate",
        Intent.BOOK_EVALUATION,
        "Assess whether one named book suits the child.",
        evaluate.run,
        required_inputs=("target_child", "book_title"),
        produces=("evaluation",),
    ),
    "compare": Capability(
        "compare",
        Intent.BOOK_COMPARISON,
        "Compare two or more named books for the child.",
        compare.run,
        required_inputs=("book_titles",),
        optional_inputs=("target_child",),
        produces=("comparison",),
    ),
    "discussion": Capability(
        "discussion",
        Intent.READING_DISCUSSION,
        "Generate post-reading discussion questions.",
        discussion.run,
        required_inputs=("target_child", "book_title"),
        produces=("questions",),
    ),
    "path": Capability(
        "path",
        Intent.READING_PATH_PLANNING,
        "Plan a staged reading path for the child.",
        path.run,
        required_inputs=("target_child", "reading_profile"),
        produces=("reading_path",),
    ),
    "content": Capability(
        "content",
        Intent.CONTENT_CREATION,
        "Draft requested content (article, copy, social post).",
        content.run,
        optional_inputs=("target_child",),
        produces=("draft",),
    ),
}

# Intents that map to no capability: their work is pure profile/memory persistence.
_NO_CAPABILITY = {
    Intent.CHILD_PROFILE_UPDATE,
    Intent.PARENT_PROFILE_UPDATE,
    Intent.PARENT_GOAL_UPDATE,
    Intent.CLARIFY,
}


def for_intent(intent: Intent) -> Capability | None:
    """Return the capability triggered by an intent, or None (profile-update/clarify intents)."""
    if intent in _NO_CAPABILITY:
        return None
    for cap in REGISTRY.values():
        if cap.intent == intent:
            return cap
    return None


def required_inputs(name: str) -> tuple[str, ...]:
    """Return a capability's required inputs (empty if the name is unknown)."""
    cap = REGISTRY.get(name)
    return cap.required_inputs if cap else ()


def menu() -> str:
    """Render the registry for planner/clarify prompts: name, description, required inputs."""
    lines = []
    for cap in REGISTRY.values():
        req = ", ".join(cap.required_inputs) or "none"
        lines.append(f"- {cap.name}: {cap.description} (requires: {req})")
    return "\n".join(lines)
