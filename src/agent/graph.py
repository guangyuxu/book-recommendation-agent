"""Main graph: the domain-driven pipeline.

START -> load_context -> understand -> plan -> clarify
  clarify --ask_user--> END            (we asked the user; resume next turn)
  clarify --continue/best_effort--> execute -> memory -> profile_update -> respond -> END
"""

from langgraph.graph import END, START, StateGraph

from .lifecycle import LOAD_CONTEXT_RETRY, load_context
from .nodes import (
    clarify,
    execute,
    memory,
    plan,
    profile_update,
    respond,
    route_after_clarify,
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
builder.add_edge("memory", "profile_update")
builder.add_edge("profile_update", "respond")
builder.add_edge("respond", END)

graph = builder.compile()
