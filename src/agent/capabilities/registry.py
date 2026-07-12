"""Capability Registry: declarative capabilities the Planner lists and Execute dispatches.

Each Capability declares its triggering intent and required inputs so the planner and
clarification nodes can reason about it without importing the capability's internals.
Capabilities are independent (no capability consumes another's output), so Execute fans them out
in parallel. `run` is the LLM-only implementation (no retrieval/ranking in MVP).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..intents import Intent
from . import compare, content, discussion, evaluate, path, recommend


@dataclass(frozen=True)
class Capability:
    """One executable capability and the inputs it needs.

    `required_inputs` is the only remaining resource vocabulary: the clarify node checks each
    against state/ambient to decide whether an input is missing. Capabilities are independent --
    none consumes another's output -- so there is no `produces` and no producer -> consumer edge.

    A capability is executed one of two ways, and declares which here:
      - `run`: a single-shot LLM function (state -> result dict). Execute wraps it in a node.
      - `graph`: a compiled LangGraph subgraph wired in directly as its own node (e.g. recommend's
        generate/validate self-critique loop). It appends {name: result} to execute's `results`
        fan-in channel itself, the same contribution shape `run` capabilities produce.
    Exactly one of the two is set.
    """

    name: str
    intent: Intent
    description: str
    run: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    required_inputs: tuple[str, ...] = ()
    graph: Any = None


REGISTRY: dict[str, Capability] = {
    "recommend": Capability(
        "recommend",
        Intent.BOOK_RECOMMENDATION,
        "Recommend a fitted English booklist for the child.",
        required_inputs=("target_child",),
        graph=recommend.recommend_graph,
    ),
    "evaluate": Capability(
        "evaluate",
        Intent.BOOK_EVALUATION,
        "Assess whether one named book suits the child.",
        required_inputs=("target_child", "books"),
        graph=evaluate.evaluate_subgraph,
    ),
    "compare": Capability(
        "compare",
        Intent.BOOK_COMPARISON,
        "Compare two or more named books for the child.",
        compare.run,
        required_inputs=("books",),
    ),
    "discussion": Capability(
        "discussion",
        Intent.READING_DISCUSSION,
        "Generate post-reading discussion questions.",
        discussion.run,
        required_inputs=("target_child", "books"),
    ),
    "path": Capability(
        "path",
        Intent.READING_PATH_PLANNING,
        "Plan a staged reading path for the child.",
        path.run,
        required_inputs=("target_child", "reading_profile"),
    ),
    "content": Capability(
        "content",
        Intent.CONTENT_CREATION,
        "Draft requested content (article, copy, social post).",
        content.run,
    ),
}

# Resources that come from context/state (load_context + understand), not from a capability's
# output. The clarify node treats these as preconditions checked against state (see
# ambient_satisfied): a required input already met from ambient does not trigger a question.
#   target_child      -- resolved via understand/load_context (state["target_child_id"])
#   reading_profile   -- the child's stored profile (state["children"][id]["reading_profile"])
#   policies          -- family reading policies (state["policies"])
#   books             -- books the user named this turn (understanding.mentioned_books)
AMBIENT: frozenset[str] = frozenset(
    {"target_child", "reading_profile", "policies", "books"}
)

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
