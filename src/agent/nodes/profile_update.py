"""Profile Update: the only node that persists. Executes memory operations via domain tools.

An LLM agent (bound to MEMORY_TOOLS) turns the domain operations from Memory Policy into tool
calls. The whole tool loop runs inside one domain_session, so every write shares one
transaction committed once. Intermediate tool/AI messages stay local -- only respond writes
to the conversation.
"""

from __future__ import annotations

import logging

from langchain.messages import HumanMessage, SystemMessage, ToolMessage

from ..domain import MEMORY_TOOLS, MEMORY_TOOLS_BY_NAME, domain_session
from ..llm import model

logger = logging.getLogger(__name__)

_bound = model.bind_tools(MEMORY_TOOLS)
_MAX_ITERATIONS = 5


def profile_update(state: dict) -> dict:
    """Apply the turn's memory operations by calling domain tools inside one transaction."""
    u = state.get("understanding") or {}
    operations = state.get("memory_operations") or []
    child_is_new = bool(u.get("child_is_new"))
    if not operations and not child_is_new:
        return {}

    family = state.get("family") or {}
    family_id = family.get("id")
    if not family_id:
        logger.warning("profile_update: no family id in state, skipping persistence.")
        return {}

    target = state.get("target_child_id")
    ops_text = "\n".join(
        f"- {o.get('operation')}: {o.get('arguments')} ({o.get('rationale')})"
        for o in operations
    ) or "(none listed)"

    with domain_session(
        family_id, target, requester_member_id=state.get("family_member_id")
    ) as ctx:
        system = SystemMessage(
            content=(
                "You persist what was decided this turn by calling the available domain tools. "
                "The family and target child are already set for you -- never pass ids. If the "
                "child is new, call create_child first; subsequent tools then target that child. "
                "Apply each operation below using the matching tool, then stop."
            )
        )
        messages = [
            system,
            HumanMessage(
                content=(
                    f"child_is_new={child_is_new}\n"
                    f"user_signals={u.get('user_signals')}\n\n"
                    f"Operations to apply:\n{ops_text}"
                )
            ),
        ]
        for _ in range(_MAX_ITERATIONS):
            response = _bound.invoke(messages)
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                break
            for call in tool_calls:
                tool = MEMORY_TOOLS_BY_NAME.get(call["name"])
                if tool is None:
                    content = f"Unknown tool {call['name']}."
                else:
                    try:
                        content = str(tool.invoke(call["args"]))
                    except Exception as exc:  # surface the error to the agent, keep the turn alive
                        logger.warning("profile_update tool %s failed: %s", call["name"], exc)
                        content = f"Error: {exc}"
                messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
        new_target = str(ctx.target_child_id) if ctx.target_child_id else None

    if new_target and new_target != target:
        return {"target_child_id": new_target}
    return {}
