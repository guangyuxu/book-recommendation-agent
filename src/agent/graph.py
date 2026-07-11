"""Main graph: the domain-driven pipeline.

START -> load_context -> understand -> plan -> clarify
  clarify --ask_user--> END                       (we asked the user; resume next turn)
  clarify --continue/best_effort--> {execute, memory}   (fan out to two parallel branches)
    execute  -> respond    (answer-generation branch: run the planned capabilities)
    memory   -> respond    (memory subgraph: decide -> confirm gate -> single DB write)
  respond -> END

`execute` and the `memory` subgraph run in PARALLEL and fan in at `respond`; they touch disjoint
state channels (capability_results vs memory_operations/confirmation*/members/children), so there
is no write conflict. The memory subgraph (see agent.memory) owns the HITL confirmation
gate and the only DB write; when it pauses on interrupt(), `execute` has already completed and is
checkpointed, so a resume re-runs only the paused gate node, never `execute`.
"""

from langgraph.graph import END, START, StateGraph

from .lifecycle import LOAD_CONTEXT_RETRY, load_context
from .memory import memory_graph
from .pipeline import (
    clarify,
    execute,
    plan,
    respond,
    route_after_clarify,
    understand,
)
from .state import AppContext, FlowState

builder = StateGraph(FlowState, context_schema=AppContext)
builder.add_node("load_context", load_context, retry_policy=LOAD_CONTEXT_RETRY)
builder.add_node("understand", understand)
builder.add_node("plan", plan)
builder.add_node("clarify", clarify)
builder.add_node("execute", execute)
builder.add_node("memory", memory_graph)  # the memory subgraph
builder.add_node("respond", respond)

builder.add_edge(START, "load_context")
builder.add_edge("load_context", "understand")
builder.add_edge("understand", "plan")
builder.add_edge("plan", "clarify")
# Fan out: proceed runs execute + the memory subgraph in parallel; ask_user ends the turn.
builder.add_conditional_edges(
    "clarify",
    route_after_clarify,
    {"ask_user": END, "execute": "execute", "memory": "memory"},
)
# Fan in: both branches join at respond, which composes the single user-facing reply.
builder.add_edge("execute", "respond")
builder.add_edge("memory", "respond")
builder.add_edge("respond", END)

graph = builder.compile()
