"""Conversation Understanding: natural language -> structured Understanding. No business logic.

The LLM reports only message evidence (including which child the MESSAGE points to, as
`child_ref`). Reconciling that evidence with the pinned/active child is a separate,
deterministic step -- `resolve_child` -- so the "which child" decision is testable and does not
silently let a stale pin override an explicit mention.
"""

from __future__ import annotations

from typing import Any, cast

from langchain.messages import SystemMessage

from ..intents import intent_menu
from ..language import normalize_language
from ..llm import STANDARD
from ..state import FlowState
from .schemas import Understanding

_structured = STANDARD.structured(Understanding)


def resolve_child(
    ref: dict[str, Any], children: dict[str, dict[str, Any]], preset: str | None
) -> tuple[str | None, bool, bool]:
    """Reconcile the message's child evidence with the pinned child.

    Returns (target_child_id, is_new, ambiguous). Priority: an explicit message reference wins
    over the preset (the pin may be stale) -- so "for Ben" switches away from a UI-selected A.
    A bare reference that names no one falls back to the preset (the active child) rather than
    interrupting: the pinned/selected child disambiguates it.
      - matched(valid roster id) -> that child (may differ from preset == a switch)
      - new                       -> is_new (a non-roster child; don't force the preset)
      - ambiguous, preset pinned  -> the preset (the active child resolves the reference)
      - ambiguous, nothing pinned -> ask (no child to fall back to)
      - none / hallucinated id     -> fall back to preset
    """
    status = ref.get("status")
    ref_id = ref.get("child_id")
    if status == "matched" and ref_id in children:
        return ref_id, False, False
    if status == "new":
        return None, True, False
    if status == "ambiguous":
        # The message points at a child but not which one. If a child is already active,
        # attribute it there; only ask when there is no pin to fall back to.
        if preset in children:
            return preset, False, False
        return None, False, True
    return preset, False, False


def switch_signal(
    status: str | None,
    target_id: str | None,
    preset: str | None,
    children: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Describe a focus switch for the frontend, or {} if none happened this turn.

    Fires only on a confident, matched move to a *different* child than the pinned one (from
    is None when nothing was pinned yet -- a first-time selection the frontend still needs).
    Returns {} when the reference is unchanged, ambiguous, new, or absent.
    """
    if status != "matched" or not target_id or target_id == preset:
        return {}
    return {
        "from": preset,
        "to": target_id,
        "from_name": (children.get(preset) or {}).get("display_name")
        if preset
        else None,
        "to_name": (children.get(target_id) or {}).get("display_name"),
    }


def _roster(children: dict[str, dict[str, Any]]) -> str:
    lines = []
    for cid, prof in children.items():
        rp = prof.get("reading_profile") or {}
        bits = [
            prof.get("display_name"),
            f"age {prof['age']}" if prof.get("age") else None,
            rp.get("summary"),
        ]
        desc = ", ".join(str(b) for b in bits if b) or "(sparse profile)"
        lines.append(f"- id={cid}: {desc}")
    return "\n".join(lines) or "(no children on file)"


def understand(state: FlowState) -> dict[str, Any]:
    """Read the latest message into a structured Understanding and resolve the target child."""
    children = state.get("children") or {}
    system = SystemMessage(
        content=(
            "You read a parent's message about books for their child and return a structured "
            "understanding. Do NOT solve anything here.\n\n"
            "List EVERY intent that genuinely, independently applies to this message in "
            "`intents` (a single message may carry several -- e.g. recommend books AND ask for "
            "discussion questions AND request a social post). Order by prominence. Add an intent "
            "only for a real, distinct request; leave `intents` empty if nothing is actionable. "
            "Intents:\n"
            f"{intent_menu()}\n\n"
            "PROFILE/GOAL FACTS vs. TASKS: child_profile_update, parent_profile_update, and "
            "parent_goal_update are for messages whose POINT is to state/update a fact about the "
            "child, the parent, or a goal. When the message also asks for an actionable task "
            "(recommend / evaluate / compare / path / discussion / content), any profile or goal "
            "fact in it is just supporting context -- record it in user_signals and do NOT add a "
            "*_update intent for it. Emit a *_update intent ONLY when stating that fact is the "
            "whole message, with no other actionable request. Profile/goal facts always go into "
            "user_signals regardless of which intents you list.\n\n"
            "In child_ref, report which child THE MESSAGE points to -- evidence only. Do NOT "
            "consider any 'currently selected' child; report only what the message says. Use "
            "the roster below: status='matched' + child_id=<exact roster id> when the message "
            "clearly refers to a specific listed child; status='new' when it describes a child "
            "not on the roster; status='ambiguous' when it refers to a child but which one is "
            "unclear; status='none' when it does not single out any child. Also extract "
            "mentioned_books and user_signals (profile-relevant facts).\n\n"
            f"Children roster:\n{_roster(children)}"
        )
    )
    understanding = cast(
        Understanding, _structured.invoke([system, *state["messages"]])
    )

    # Deterministic reconciliation: explicit message reference wins over the pinned child;
    # otherwise fall back to the pin. (resolve_child owns this policy, not the LLM.)
    target_id, is_new, ambiguous = resolve_child(
        understanding.child_ref.model_dump(), children, state.get("target_child_id")
    )
    u = understanding.model_dump(mode="json")
    u["child_is_new"] = (
        is_new  # resolved outputs downstream (memory_policy/profile_update/clarify) read
    )
    u["child_ambiguous"] = ambiguous

    return {
        "understanding": u,
        "target_child_id": target_id,
        # Normalized so an unexpected LLM code (e.g. "zh-TW") maps to a supported value; the
        # downstream LLM nodes (clarify, respond) read this to reply in the parent's language.
        "reply_language": normalize_language(understanding.reply_language),
        # Always written (even when {}) so a switch from an earlier turn never lingers.
        "child_switch": switch_signal(
            understanding.child_ref.status,
            target_id,
            state.get("target_child_id"),
            children,
        ),
    }
