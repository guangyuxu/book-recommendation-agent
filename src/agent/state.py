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


def merge_output_checks(
    existing: list[dict[str, Any]] | None, incoming: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Reducer for `output_checks`: append within a turn, reset when handed an empty list.

    The validation subgraph fans out to parallel check workers that each return a single-element
    list; those append. Its `select` node returns an empty list to RESET the accumulator at the
    start of a turn, so a prior turn's results never linger across a checkpointed thread. (A plain
    add-reducer could not reset; empty-list-means-reset is the one convention the workers never
    trigger, since a worker always emits exactly one result.)
    """
    if not incoming:
        return []
    return [*(existing or []), *incoming]


class FlowState(TypedDict, total=False):
    """The single graph's state. `total=False` because most channels are filled stage by stage.

    Channel groups:
    - messages: the conversation (only `respond` writes the user-facing reply).
    - loaded context: written once by lifecycle.load_context, read-only thereafter.
    - target_child_id: the one child this turn is about (MVP: one child per conversation).
    - pipeline products: one per stage, last-write-wins.
    """

    messages: Annotated[list[AnyMessage], add_messages]

    # Input safety verdict from the entry `guard` node (agent.guard): {blocked: bool,
    # score: float | None}. Rewritten every turn; when blocked, the turn short-circuits to END
    # with a canned refusal and never reaches load_context. score is None when the check was
    # skipped or could not run (fail-open).
    safety: dict[str, Any]

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

    # Reply language for this turn, inferred by `understand` from the parent's latest message
    # ("en" | "zh-Hans" | "zh-Hant"; default "en"). The downstream LLM nodes (clarify, respond)
    # read it via agent.language.reply_directive to answer in the parent's language. The guard
    # entry node localizes its own refusal independently (it runs before understand).
    reply_language: str

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

    # Output-validation channels (agent.validation subgraph, runs execute -> validate -> respond).
    # output_checks: transient per-turn accumulator the parallel check workers append to; reset to
    #   [] by the subgraph's `select` node each turn (see merge_output_checks above).
    # validation: the aggregated verdict {rating: ALLOW|WARNING|REWRITE|BLOCK, results: [...]},
    #   last-write-wins, written by `aggregate` and read by `respond`.
    output_checks: Annotated[list[dict[str, Any]], merge_output_checks]
    validation: dict[str, Any]

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
