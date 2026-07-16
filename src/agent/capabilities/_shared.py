"""Shared helpers for capabilities: profile/policy briefs and a generic LLM text runner.

Capabilities are LLM-only in the MVP (no retrieval/ranking/vector search). They read the
already-loaded context from state -- they never touch the database.
"""

from __future__ import annotations

from typing import Any

from .. import prompts
from ..llm import STANDARD, Strategy


def target_child(state: dict[str, Any]) -> dict[str, Any] | None:
    """Return the resolved target child's dict (with nested reading_profile), or None."""
    cid = state.get("target_child_id")
    children = state.get("children") or {}
    return children.get(cid) if cid else None


def _kv_lines(pairs: list[tuple[str, object]]) -> list[str]:
    out = []
    for key, value in pairs:
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        out.append(f"- {key}: {value}")
    return out


def child_brief(state: dict[str, Any]) -> str:
    """Render the target child's profile + reading profile as a compact prompt block."""
    child = target_child(state)
    if not child:
        return "(no specific child resolved)"
    rp = child.get("reading_profile") or {}
    lines = _kv_lines(
        [
            ("name", child.get("display_name")),
            ("gender", child.get("gender")),
            ("age", child.get("age")),
            ("grade", child.get("grade")),
            ("reading_language", child.get("reading_language")),
            ("reading_level_note", rp.get("reading_level_note")),
            ("cefr_level", rp.get("cefr_level")),
            ("lexile", rp.get("lexile")),
            ("current_stage", rp.get("current_stage")),
            ("interests", rp.get("interests")),
            ("preferred_genres", rp.get("preferred_genres")),
            ("disliked_genres", rp.get("disliked_genres")),
            ("liked_themes", rp.get("liked_themes")),
            ("avoid_topics", rp.get("avoid_topics")),
            ("summary", rp.get("summary")),
            ("notes", child.get("notes")),
        ]
    )
    return "\n".join(lines) if lines else "(child on file, but profile is sparse)"


def policies_brief(state: dict[str, Any]) -> str:
    """Render the family's active reading policies (goals / constraints / topics to avoid)."""
    goals: list[str] = []
    constraints: list[str] = []
    avoid: list[str] = []
    for p in state.get("policies") or []:
        goals += p.get("goals") or []
        constraints += p.get("constraints") or []
        avoid += p.get("avoid_topics") or []
    lines = _kv_lines(
        [("goals", goals), ("constraints", constraints), ("avoid_topics", avoid)]
    )
    return "\n".join(lines) if lines else "(no reading policies on file)"


def mentioned_books(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the books the user named this turn (from the understanding)."""
    return (state.get("understanding") or {}).get("mentioned_books") or []


def run_text(
    state: dict[str, Any],
    prompt_id: str,
    *,
    strategy: Strategy | None = None,
) -> str:
    """Run a one-shot LLM call over the conversation with a capability's registry prompt.

    `prompt_id` is a `<namespace>.<key>` in the prompt registry whose template embeds the child
    and policy briefs (passed as pre-serialized vars -- never raw rows). Pass `strategy=HEAVY` for
    deep-analysis capabilities (compare) or leave unset to use STANDARD. Returns the reply text;
    callers wrap it under their produced-resource key, e.g.
    `{"comparison": run_text(..., strategy=HEAVY)}`.
    """
    _strategy = strategy if strategy is not None else STANDARD
    system = prompts.render(
        prompt_id,
        child_brief=child_brief(state),
        policies_brief=policies_brief(state),
    )
    reply = _strategy.chain().invoke(
        [*system, *state["messages"]], config=prompts.config(prompt_id)
    )
    return str(reply.content)
