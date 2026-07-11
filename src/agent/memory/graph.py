"""Memory subgraph builder: memory decision -> HITL confirmation gate -> single DB write.

The agent's long-term-memory write path, packaged as a LangGraph subgraph so the main graph can
run it in PARALLEL with `execute` (the answer-generation branch); both fan in at `respond`. It
shares the parent's FlowState schema, so every channel it touches (understanding,
memory_operations, confirmation*, members, children, target_child_id) flows straight through --
the subgraph reads and writes the same channels the main graph does, no input/output mapping
needed.

Named "memory" (not "persistence") to avoid colliding with LangGraph's persistence concept
(checkpointers / durable state); this subgraph is about *what to remember*, not *how state is
stored*.

    START -> memory_policy -> prepare_confirmation
        --skip------------------------------------------> profile_update -> END
        --confirm--> request_confirmation -> apply_confirmation -> profile_update -> END

The only interrupt() lives in request_confirmation. When this subgraph pauses there, the
interrupt propagates up to the parent graph (which owns the checkpointer). By then the sibling
`execute` branch has already completed and been checkpointed, so resuming re-runs only
request_confirmation -- never execute. Compiled WITHOUT a checkpointer on purpose: a nested
subgraph inherits the parent's.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..state import AppContext, FlowState
from ..usage_tracker import with_turn_context
from .confirm import (
    apply_confirmation,
    prepare_confirmation,
    request_confirmation,
    route_after_prepare,
)
from .decide import memory_policy
from .profile_update import profile_update

# These subgraph nodes run as their own LangGraph nodes (separate contexts from the
# parent "memory" node), so the LLM-invoking ones wrap themselves rather than relying
# on any wrapper applied to the "memory" node in the parent graph.
_builder = StateGraph(FlowState, context_schema=AppContext)
_builder.add_node("memory_policy", with_turn_context(memory_policy))
_builder.add_node("prepare_confirmation", prepare_confirmation)
_builder.add_node("request_confirmation", request_confirmation)
_builder.add_node("apply_confirmation", apply_confirmation)
_builder.add_node("profile_update", with_turn_context(profile_update))

_builder.add_edge(START, "memory_policy")
_builder.add_edge("memory_policy", "prepare_confirmation")
_builder.add_conditional_edges(
    "prepare_confirmation",
    route_after_prepare,
    {"confirm": "request_confirmation", "skip": "profile_update"},
)
_builder.add_edge("request_confirmation", "apply_confirmation")
_builder.add_edge("apply_confirmation", "profile_update")
_builder.add_edge("profile_update", END)

memory_graph = _builder.compile()
