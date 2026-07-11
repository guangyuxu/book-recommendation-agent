"""Profile Update: the only node that persists. Executes memory operations via domain tools.

An LLM agent (bound to MEMORY_TOOLS) turns the domain operations from Memory Policy into tool
calls. The whole tool loop runs inside one domain_session, so every write shares one
transaction committed once. Intermediate tool/AI messages stay local -- only respond writes
to the conversation.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from langchain.messages import HumanMessage, SystemMessage, ToolMessage

from ..domain import MEMORY_TOOLS, MEMORY_TOOLS_BY_NAME, domain_session
from ..llm import STANDARD
from ..serialize import load_family_entities
from ..state import FlowState

logger = logging.getLogger(__name__)

_bound = STANDARD.tools(MEMORY_TOOLS)
_MAX_ITERATIONS = 5


def profile_update(state: FlowState) -> dict[str, Any]:
    """Apply the turn's memory operations by calling domain tools inside one transaction."""
    u = state.get("understanding") or {}
    operations = state.get("memory_operations") or []
    if not operations:
        return {}

    family = state.get("family") or {}
    family_id = family.get("id")
    if not family_id:
        logger.warning("profile_update: no family id in state, skipping persistence.")
        return {}

    target = state.get("target_child_id")
    ops_text = (
        "\n".join(
            f"- {o.get('operation')}: {o.get('arguments')} ({o.get('rationale')})"
            for o in operations
        )
        or "(none listed)"
    )

    with domain_session(
        family_id, target, requester_member_id=state.get("family_member_id")
    ) as ctx:
        system = SystemMessage(
            content=(
                "You persist what was decided this turn by calling the available domain tools. "
                "The family and target child are already set for you -- never pass ids. Apply "
                "each operation below using the matching tool; when an operation creates a child, "
                "call create_child first so the later operations target the new child. Then stop. "
                "Do NOT call any tool that is not one of the operations listed."
            )
        )
        messages = [
            system,
            HumanMessage(
                content=(
                    f"user_signals={u.get('user_signals')}\n\n"
                    f"Operations to apply:\n{ops_text}"
                )
            ),
        ]
        made_calls = False
        stopped_cleanly = False
        last_had_error = False
        for _ in range(_MAX_ITERATIONS):
            response = _bound.invoke(messages)
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                stopped_cleanly = True
                break
            made_calls = True
            last_had_error = False  # reflects only the FINAL batch of calls
            for call in tool_calls:
                tool = MEMORY_TOOLS_BY_NAME.get(call["name"])
                if tool is None:
                    last_had_error = True
                    content = f"Unknown tool {call['name']}."
                else:
                    try:
                        content = str(tool.invoke(call["args"]))
                    except (
                        Exception
                    ) as exc:  # surface the error to the agent, keep the turn alive
                        # Log only the exception type, not the message: the message may
                        # contain DB row data (child name, profile fields) which is PII.
                        logger.warning(
                            "profile_update tool %s failed: %s",
                            call["name"],
                            type(exc).__name__,
                        )
                        last_had_error = True
                        content = f"Error: {exc}"
                messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
        new_target = str(ctx.target_child_id) if ctx.target_child_id else None
        # Re-read inside the still-open session so the frontend syncs same-turn: the writes are
        # flushed (visible to this transaction) even though the commit happens on block exit.
        members, children = load_family_entities(ctx.session, UUID(str(family_id)))

    # We are confident the writes landed only if the agent ran at least one tool and then stopped
    # on its own with no error in the final batch. Anything else (an unrecovered tool error, or
    # exhausting the iteration budget mid-loop) means we must NOT tell the parent it was saved.
    writes_failed = made_calls and (last_had_error or not stopped_cleanly)

    out: dict[str, Any] = {"members": members, "children": children}
    if new_target and new_target != target:
        out["target_child_id"] = new_target

    # If a confirmed identity change failed to persist, downgrade the outcome so `respond` does
    # not falsely acknowledge it as saved. Untouched on the soft/skip path (status != "applied").
    confirmation = state.get("confirmation") or {}
    if writes_failed and confirmation.get("status") == "applied":
        logger.warning(
            "profile_update: confirmed writes failed to persist; reporting error."
        )
        out["confirmation"] = {**confirmation, "status": "error"}
    return out
