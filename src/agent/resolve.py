"""Semantic resolution of which child(ren) a turn is about.

A parent can have several children, and which one a message concerns is expressed in
natural language ("for my younger one", "what about him?") -- possibly switching across
turns. resolve_children reads the roster + full history and picks the target child(ren),
or flags the request as ambiguous / about a new child.
"""

from langchain.messages import SystemMessage
from pydantic import BaseModel, Field

from .llm import model


class ChildResolution(BaseModel):
    """Which existing children the latest message is about."""

    child_ids: list[str] = Field(
        default_factory=list,
        description="IDs (from the roster) of the children this message concerns. "
        "May be several. Empty if none can be determined or it is about a new child.",
    )
    is_new_child: bool = Field(
        default=False,
        description="True if the message describes a child not in the roster (a new child).",
    )
    ambiguous: bool = Field(
        default=False,
        description="True if a specific child is needed but cannot be determined from "
        "the conversation (and it is not clearly a new child).",
    )


_resolver = model.with_structured_output(ChildResolution)


def _roster_menu(children: dict[str, dict]) -> str:
    lines = []
    for cid, prof in children.items():
        traits = ", ".join(
            f"{k}={prof[k]}"
            for k in ("name", "reading_level", "recent_signal")
            if prof.get(k)
        )
        lines.append(f"- id={cid}: {traits or '(no details yet)'}")
    return "\n".join(lines)


def resolve_for(children: dict[str, dict], messages) -> ChildResolution:
    """Resolve target children for a message against a roster.

    Reused by the graph's resolve node and by the orchestrator (per subtask). With a single
    child there is no ambiguity, so we skip the LLM. With no children, anything specific is
    a new child.
    """
    children = children or {}
    if len(children) == 1:
        return ChildResolution(child_ids=[next(iter(children))])
    if not children:
        return ChildResolution(is_new_child=True)

    system = SystemMessage(
        content=(
            "Decide which of the parent's children the latest message is about. "
            "Use the whole conversation to resolve references like 'him' or 'the other "
            "one', including when the focus switches between children across turns. "
            "Return all that apply (a message may concern several children). If the "
            "message describes a child not in the roster, set is_new_child. If a child is "
            "needed but you cannot tell which, set ambiguous.\n\n"
            f"Children roster:\n{_roster_menu(children)}"
        )
    )
    return _resolver.invoke([system, *messages])


def resolve_children(state):
    """Graph node: write the resolved target children (+ flags) into state."""
    res = resolve_for(state.get("children") or {}, state["messages"])
    return {
        "target_child_ids": res.child_ids,
        "is_new_child": res.is_new_child,
        "ambiguous": res.ambiguous,
    }
