"""Conversation Understanding: natural language -> structured Understanding. No business logic.

Also resolves the single target child (the old resolve step, folded in): one child on file ->
use it; several -> the LLM picks from the roster; none -> mark child_is_new.
"""

from __future__ import annotations

from typing import cast

from langchain.messages import SystemMessage

from ..intents import intent_menu
from ..llm import model
from ..schemas import Understanding

_structured = model.with_structured_output(Understanding)


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
            "Pick the primary intent, and a secondary intent only if the message truly "
            "contains a second independent request (at most two total). Intents:\n"
            f"{intent_menu()}\n\n"
            "Resolve which child the message is about using the roster below: return that "
            "child's exact id in target_child_id. If the message describes a child not on the "
            "roster, set child_is_new. If a child is needed but you cannot tell which, set "
            "child_ambiguous. Also extract mentioned_books and user_signals "
            "(profile-relevant facts).\n\n"
            f"Children roster:\n{_roster(children)}"
        )
    )
    understanding = cast(Understanding, _structured.invoke([system, *state["messages"]]))

    preset = state.get("target_child_id")
    if preset:
        # load_context already pinned the child (explicit child_id or sole child): trust it.
        understanding.target_child_id = preset
        understanding.child_ambiguous = False
        understanding.child_is_new = False
    # Drop a hallucinated id that isn't on the roster.
    elif understanding.target_child_id and understanding.target_child_id not in children:
        understanding.target_child_id = None

    return {
        "understanding": understanding.model_dump(mode="json"),
        "target_child_id": understanding.target_child_id,
    }
