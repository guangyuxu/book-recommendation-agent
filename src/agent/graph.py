"""Main graph: the domain-driven pipeline.

START -> load_context -> understand -> plan -> clarify
  clarify --ask_user--> END            (we asked the user; resume next turn)
  clarify --continue/best_effort--> execute -> memory -> prepare_confirmation
    prepare_confirmation --skip--> profile_update            (nothing high-stakes this turn)
    prepare_confirmation --confirm--> request_confirmation -> apply_confirmation -> profile_update
  profile_update -> respond -> END

The confirmation gate is three single-responsibility nodes (see nodes/confirm.py): prepare
builds the popup, request is the sole interrupt() pause for the parent's approval, apply turns
the Accept/Reject reply into the operations to persist. Only profile_update writes to the DB.
"""

from langgraph.graph import END, START, StateGraph

from .lifecycle import LOAD_CONTEXT_RETRY, load_context
from .nodes import (
    apply_confirmation,
    clarify,
    execute,
    memory,
    plan,
    prepare_confirmation,
    profile_update,
    request_confirmation,
    respond,
    route_after_clarify,
    route_after_prepare,
    understand,
)
from .state import AppContext, FlowState

builder = StateGraph(FlowState, context_schema=AppContext)  # type: ignore[arg-type]
builder.add_node("load_context", load_context, retry_policy=LOAD_CONTEXT_RETRY)  # type: ignore[arg-type]
builder.add_node("understand", understand)  # type: ignore[arg-type]
builder.add_node("plan", plan)  # type: ignore[arg-type]
builder.add_node("clarify", clarify)  # type: ignore[arg-type]
builder.add_node("execute", execute)  # type: ignore[arg-type]
builder.add_node("memory", memory)  # type: ignore[arg-type]
builder.add_node("prepare_confirmation", prepare_confirmation)  # type: ignore[arg-type]
builder.add_node("request_confirmation", request_confirmation)  # type: ignore[arg-type]
builder.add_node("apply_confirmation", apply_confirmation)  # type: ignore[arg-type]
builder.add_node("profile_update", profile_update)  # type: ignore[arg-type]
builder.add_node("respond", respond)  # type: ignore[arg-type]

builder.add_edge(START, "load_context")
builder.add_edge("load_context", "understand")
builder.add_edge("understand", "plan")
builder.add_edge("plan", "clarify")
builder.add_conditional_edges(
    "clarify", route_after_clarify, {"ask_user": END, "execute": "execute"}
)
builder.add_edge("execute", "memory")
builder.add_edge("memory", "prepare_confirmation")
builder.add_conditional_edges(
    "prepare_confirmation",
    route_after_prepare,
    {"confirm": "request_confirmation", "skip": "profile_update"},
)
builder.add_edge("request_confirmation", "apply_confirmation")
builder.add_edge("apply_confirmation", "profile_update")
builder.add_edge("profile_update", "respond")
builder.add_edge("respond", END)

graph = builder.compile()
