"""Memory Policy: decide what is worth remembering. Emits domain operations; never writes DB.

The output is a list of domain-level operations (e.g. UpdateReadingInterest, RecordFinishedBook)
that the profile_update node executes via tools.
"""

from __future__ import annotations

from typing import cast

from langchain.messages import HumanMessage, SystemMessage

from ..domain import MEMORY_TOOLS_BY_NAME
from ..llm import model
from ..schemas import MemoryDecision

_structured = model.with_structured_output(MemoryDecision)

# The domain operations the profile_update agent can carry out. Derived from the single source
# of truth (agent.domain.MEMORY_TOOLS) so it can never drift from the executable tool names.
_AVAILABLE_OPERATIONS = ", ".join(MEMORY_TOOLS_BY_NAME)


def memory(state: dict) -> dict:
    """Decide which durable facts from this turn to persist, as domain operations."""
    u = state.get("understanding") or {}
    signals = u.get("user_signals") or []
    if not signals and not u.get("child_is_new"):
        return {"memory_operations": []}

    system = SystemMessage(
        content=(
            "You are a memory policy. Decide what from this turn is worth storing long-term "
            "about the child or family, and express each as a domain operation. Use the EXACT "
            "operation name from the available list, with plain domain arguments (never database "
            "ids). Skip transient or already-known facts.\n\n"
            "Argument names MUST match the tool's real parameters. In particular use "
            "`birth_date` (an ISO date; a bare year like '2023' is fine if only the year is "
            "known) -- never `age`, and never invent parameters. `gender` must be exactly "
            "'Male' or 'Female'.\n\n"
            "If the child being discussed is new (child_is_new=true) and worth remembering, emit "
            "create_child first (with whatever identity was given), then their facts. Creating a "
            "child and changing identity fields (gender, birth_date, name) are confirmed with the "
            "parent before they take effect -- still emit them normally; a later step handles the "
            "confirmation. Ordinary reading-profile facts are saved directly.\n\n"
            f"Available operations: {_AVAILABLE_OPERATIONS}"
        )
    )
    human = HumanMessage(
        content=(
            f"child_is_new={u.get('child_is_new')}\n"
            f"user_signals={signals}"
        )
    )
    result = cast(MemoryDecision, _structured.invoke([system, human, *state["messages"]]))
    return {"memory_operations": [op.model_dump() for op in result.operations]}
