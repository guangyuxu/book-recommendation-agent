"""Conversation Understanding: natural language -> structured Understanding. No business logic.

The LLM reports only message evidence (including which child the MESSAGE points to, as
`child_ref`). Reconciling that evidence with the pinned/active child is a separate,
deterministic step -- `resolve_child` -- so the "which child" decision is testable and does not
silently let a stale pin override an explicit mention.
"""

from __future__ import annotations

from typing import cast

from langchain.messages import SystemMessage

from ..intents import intent_menu
from ..llm import model
from ..schemas import Understanding

_structured = model.with_structured_output(Understanding)


def resolve_child(
    ref: dict, children: dict[str, dict], preset: str | None
) -> tuple[str | None, bool, bool]:
    """Reconcile the message's child evidence with the pinned child.

    Returns (target_child_id, is_new, ambiguous). Priority: an explicit message reference wins
    over the preset (the pin may be stale) -- so "for Ben" switches away from a UI-selected A.
    Only when the message singles out no child do we fall back to the preset (the active child).
      - matched(valid roster id) -> that child (may differ from preset == a switch)
      - new                       -> is_new (a non-roster child; don't force the preset)
      - ambiguous                 -> ask (don't assume the preset)
      - none / hallucinated id     -> fall back to preset
    """
    status = ref.get("status")
    ref_id = ref.get("child_id")
    if status == "matched" and ref_id in children:
        return ref_id, False, False
    if status == "new":
        return None, True, False
    if status == "ambiguous":
        return None, False, True
    return preset, False, False


def _roster(children: dict[str, dict]) -> str:
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


def understand(state: dict) -> dict:
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
    understanding = cast(Understanding, _structured.invoke([system, *state["messages"]]))

    # Deterministic reconciliation: explicit message reference wins over the pinned child;
    # otherwise fall back to the pin. (resolve_child owns this policy, not the LLM.)
    target_id, is_new, ambiguous = resolve_child(
        understanding.child_ref.model_dump(), children, state.get("target_child_id")
    )
    u = understanding.model_dump(mode="json")
    u["child_is_new"] = is_new  # resolved outputs downstream (memory/profile_update/clarify) read
    u["child_ambiguous"] = ambiguous

    return {"understanding": u, "target_child_id": target_id}
