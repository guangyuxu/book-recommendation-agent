"""Graph state: shared channels with idempotent reducers, plus per-request context."""

from typing import Annotated, TypedDict

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages

# Accumulating channels shared with subgraphs must use idempotent reducers: a subgraph
# passes the whole value back to the parent, so a non-idempotent merge (e.g. operator.add)
# would double-count.


def merge_dict(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


def merge_children(
    left: dict[str, dict] | None, right: dict[str, dict] | None
) -> dict[str, dict]:
    """Merge the child roster per child_id (deep, one level): each child's profile is
    merge_dict-ed. Idempotent, like merge_dict -- a subgraph passing the whole roster back
    to the parent must not duplicate or drop children."""
    left = left or {}
    right = right or {}
    out = {cid: dict(prof) for cid, prof in left.items()}
    for cid, prof in right.items():
        out[cid] = merge_dict(out.get(cid), prof)
    return out


def merge_goals(left: list[str] | None, right: list[str] | None) -> list[str]:
    left = left or []
    right = right or []
    return left + [g for g in right if g not in left]


class FlowState(TypedDict):
    """Channels shared by the main graph and every subgraph.

    Same key + reducer on both sides, so profiles flow between parent and child graphs
    automatically.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    # children: the parent's full roster, keyed by str(child_id); each value is that
    # child's profile dict (includes its own "id"). Loaded once by load_context.
    children: Annotated[dict[str, dict], merge_children]
    parent_profile: Annotated[dict, merge_dict]
    parent_goals: Annotated[list[str], merge_goals]
    # target_child_ids: which child(ren) this turn is about, resolved semantically by the
    # resolve node. Lives in FlowState (not just the main graph) so child-specific subgraphs
    # receive it as input. Per-turn, last-write-wins -> no reducer.
    target_child_ids: list[str]


class MessagesState(FlowState):
    intent: str  # the Intent member's value; stored as str to stay serializable
    is_new_child: bool  # resolve: the message describes a not-yet-stored child
    ambiguous: bool  # resolve: a child is needed but which one can't be determined


def target_children(state) -> list[dict]:
    """The profiles of this turn's target children, in order. Skips ids missing from the roster.

    Shared by every child-specific flow so they iterate the same way over one or many children.
    """
    children = state.get("children") or {}
    return [children[cid] for cid in (state.get("target_child_ids") or []) if cid in children]


class AppContext(TypedDict):
    """Per-request runtime context (not in state, not checkpointed), via LangGraph's context channel.

    user_id is required -> marked required in the context schema, so Studio renders it as a
    required input; lifecycle.load_context validates it again at runtime.
    """

    user_id: str
