"""Graph state for the domain-driven pipeline.

The graph thinks in domains, never tables. Loaded context (family / members / children /
policies) and the per-stage products (understanding -> plan -> clarification ->
capability_results -> memory_operations) all live here as plain JSON-able dicts; only the
domain tools (see agent.domain) ever touch the database.
"""

from typing import Annotated, TypedDict

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_dict(left: dict | None, right: dict | None) -> dict:
    """Merge two dicts idempotently; right wins per key.

    Used for capability_results, where each capability writes its own key (at most two per turn).
    """
    return {**(left or {}), **(right or {})}


class FlowState(TypedDict, total=False):
    """The single graph's state. `total=False` because most channels are filled stage by stage.

    Channel groups:
    - messages: the conversation (only `respond` writes the user-facing reply).
    - loaded context: written once by lifecycle.load_context, read-only thereafter.
    - target_child_id: the one child this turn is about (MVP: one child per conversation).
    - pipeline products: one per stage, last-write-wins.
    """

    messages: Annotated[list[AnyMessage], add_messages]

    # Loaded context (lifecycle.load_context). family.id == AppContext.user_id.
    family: dict  # Family.to_dict()
    members: list[dict]  # FamilyMember.to_dict() rows
    children: dict[str, dict]  # str(child_id) -> child dict, with nested "reading_profile"
    policies: list[dict]  # active FamilyReadingPolicy rows for the turn

    # The single target child (resolved by the understand node).
    target_child_id: str | None

    # Per-stage products. Dicts are model_dump()s of the pydantic schemas in agent.schemas.
    understanding: dict
    plan: dict
    clarification: dict
    capability_results: Annotated[dict[str, dict], merge_dict]  # capability name -> result
    memory_operations: list[dict]


# Back-compat alias: callers may still import MessagesState; it is the same state now.
MessagesState = FlowState


class AppContext(TypedDict):
    """Per-request runtime context (not in state, not checkpointed), via LangGraph's context channel.

    user_id is the family's id (the household login identity). It is required: Studio renders
    it as a required input and lifecycle.load_context validates it again at runtime.
    """

    user_id: str
