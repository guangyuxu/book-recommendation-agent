"""Graph state for the domain-driven pipeline.

The graph thinks in domains, never tables. Loaded context (family / members / children /
policies) and the per-stage products (understanding -> plan -> clarification ->
capability_results -> memory_operations) all live here as plain JSON-able dicts; only the
domain tools (see agent.domain) ever touch the database.
"""

from typing import Annotated, Any, TypedDict
from uuid import UUID

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator


class FlowState(TypedDict, total=False):
    """The single graph's state. `total=False` because most channels are filled stage by stage.

    Channel groups:
    - messages: the conversation (only `respond` writes the user-facing reply).
    - loaded context: written once by lifecycle.load_context, read-only thereafter.
    - target_child_id: the one child this turn is about (MVP: one child per conversation).
    - pipeline products: one per stage, last-write-wins.
    """

    messages: Annotated[list[AnyMessage], add_messages]

    # Loaded context (lifecycle.load_context). family.id == AppContext.family_id.
    family: dict[str, Any]  # Family.to_dict()
    members: list[
        dict[str, Any]
    ]  # FamilyMember.to_dict() rows, each with nested "profile"
    children: dict[
        str, dict[str, Any]
    ]  # str(child_id) -> child dict, with nested "reading_profile"
    policies: list[dict[str, Any]]  # active FamilyReadingPolicy rows for the turn
    family_member_id: str  # AppContext.family_member_id -- who is asking

    # The single target child: pinned by load_context (explicit child_id or sole child) or
    # resolved by the understand node.
    target_child_id: str | None

    # A focus switch detected this turn (point 2): the message clearly pointed at a different
    # child than the pinned one, so target_child_id moved. {from, to, from_name, to_name} for
    # the frontend to swap the avatar (and offer an undo); {} when no switch happened this turn.
    # Always rewritten by understand so a stale switch never lingers across turns.
    child_switch: dict[str, Any]

    # Per-stage products. Dicts are model_dump()s of the pydantic schemas in agent.pipeline.schemas
    # (understanding/plan/clarification) and agent.memory.schemas (memory/confirmation).
    understanding: dict[str, Any]
    plan: dict[str, Any]
    clarification: dict[str, Any]
    # HITL confirmation gate (point 1/3), split across three nodes:
    #   confirmation_request  -- popup payload built by prepare_confirmation ({} => skip the gate)
    #   confirmation_decision -- the Accept/Reject resume value captured by request_confirmation
    #   confirmation          -- outcome written by apply_confirmation: {kind, status:
    #                            applied|rejected, operations}; profile_update downgrades an
    #                            applied status to "error" if the confirmed writes fail to persist.
    # All three are rewritten every turn by prepare_confirmation (to {}) so nothing goes stale.
    # While the gate is open the graph is paused on the interrupt() in request_confirmation.
    confirmation_request: dict[str, Any]
    confirmation_decision: dict[str, Any]
    confirmation: dict[str, Any]
    # Capability name -> result. Last-write-wins and rewritten in full every turn by `execute`
    # (even to {} when the turn runs no capabilities), so a prior turn's results never linger
    # into this turn's render/persist. `execute` is the sole writer.
    capability_results: dict[str, dict[str, Any]]
    memory_operations: list[dict[str, Any]]

    # LangGraph thread_id (from config["configurable"]["thread_id"]); None without checkpointer.
    # Stored in state for observability; the usage_tracker contextvar is the authoritative source.
    thread_id: str | None
    # Stable identifier for this invocation turn; generated in load_context.
    turn_id: str


class AppContext(BaseModel):
    """Per-request runtime input (not in state, not checkpointed), via LangGraph's context channel.

    A pydantic model, not a TypedDict, so LangGraph's _coerce_context validates it at the
    boundary: when a dict is passed to context=, missing/mistyped required fields raise a
    ValidationError before the graph runs. get_runtime(AppContext).context returns an instance.

    - family_id: the household identity. Required.
    - family_member_id: who is asking (a parent/caregiver). Required; recorded as the
      requester on recommendation turns.
    - child_id: which child this conversation is about. Optional -- if omitted and the family
      has exactly one child, that child is used by default; otherwise it is resolved from the
      conversation.
    """

    family_id: str = Field(
        description="UUID, The household identity. Required.",
        default="16555532-69b5-411e-8526-e0b321fbcfea",
    )
    family_member_id: str = Field(
        description="UUID, The identity of the family member asking.",
        default="659c1323-f47a-40eb-a0fe-5fb83f47c9c9",
    )
    child_id: str | None = Field(
        default="d63ae622-797b-4a1c-ae88-9c4309fb3b3a",
        description="UUID, The identity of the child being asked about.",
    )

    @field_validator("family_id", "family_member_id", "child_id")
    @classmethod
    def _valid_uuid(cls, v: str | None) -> str | None:
        """Validate that required fields are present and any present id is a parseable UUID string.

        We validate the UUID *format* here (at the boundary) but keep the field a str so the
        loaded context stays JSON-able. Existence in the DB is checked later in load_context.
        """
        if v is None:
            return v
        UUID(v)  # raises ValueError -> pydantic ValidationError on a malformed id
        return v
