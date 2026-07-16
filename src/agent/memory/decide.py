"""Memory Policy: decide what is worth remembering. Emits domain operations; never writes DB.

The output is a list of domain-level operations (e.g. UpdateReadingInterest, RecordFinishedBook)
that the profile_update node executes via tools.
"""

from __future__ import annotations

from typing import Any, cast

from .. import prompts
from ..domain import MEMORY_TOOLS_BY_NAME
from ..llm import FAST
from ..state import FlowState
from .schemas import MemoryDecision

_structured = FAST.structured(MemoryDecision)

# The domain operations the profile_update agent can carry out. Derived from the single source
# of truth (agent.domain.MEMORY_TOOLS) so it can never drift from the executable tool names.
_AVAILABLE_OPERATIONS = ", ".join(MEMORY_TOOLS_BY_NAME)


def memory_policy(state: FlowState) -> dict[str, Any]:
    """Decide which durable facts from this turn to persist, as domain operations."""
    u = state.get("understanding") or {}
    signals = u.get("user_signals") or []
    if not signals and not u.get("child_is_new"):
        return {"memory_operations": []}

    messages = prompts.render(
        "memory_policy.decide",
        available_operations=_AVAILABLE_OPERATIONS,
        child_is_new=u.get("child_is_new"),
        user_signals=signals,
    )
    result = cast(
        MemoryDecision,
        _structured.invoke(
            [*messages, *state["messages"]],
            config=prompts.config("memory_policy.decide"),
        ),
    )
    return {"memory_operations": [op.model_dump() for op in result.operations]}
