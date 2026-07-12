"""Main graph: the domain-driven pipeline.

START -> guard -> load_context -> understand -> plan -> clarify
  guard --blocked--> END                            (prompt-injection input; canned refusal)
  guard --ok-->      load_context
  clarify --ask_user--> END                       (we asked the user; resume next turn)
  clarify --continue/best_effort--> {execute, memory}   (fan out to two parallel branches)
    execute  -> validate -> respond    (answer branch: run capabilities, then gate the output)
    memory   -> respond                (memory subgraph: decide -> confirm gate -> single DB write)
  respond -> END

`execute -> validate` and the `memory` subgraph run in PARALLEL and fan in at `respond`; they
touch disjoint state channels (capability_results/output_checks/validation vs memory_operations/
confirmation*/members/children), so there is no write conflict. `validate` (see agent.validation)
is the OUTPUT gate: it runs policy/safety checks on the capability output and writes a rating
(ALLOW/WARNING/REWRITE/BLOCK) that `respond` acts on -- the output-side analogue of the input
`guard`. The memory subgraph (see agent.memory) owns the HITL confirmation gate and the only DB
write; when it pauses on interrupt(), the `execute -> validate` branch has already completed and is
checkpointed, so a resume re-runs only the paused gate node, never `execute`/`validate`.
"""

from langgraph.graph import END, START, StateGraph

from .guard import guard, route_after_guard
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
from .usage_tracker import with_turn_context
from .validation import validation_graph

builder = StateGraph(FlowState, context_schema=AppContext)
# load_context/plan make no LLM calls, so they are not wrapped. Every LLM-invoking node
# is wrapped so the billing ContextVar is live when its token-usage callback fires.
# guard makes a Groq (non-Anthropic) call and runs before turn_id is set, so it is not
# wrapped with with_turn_context (like load_context/plan, it is not on the Anthropic billing path).
builder.add_node("guard", guard)
builder.add_node("load_context", load_context, retry_policy=LOAD_CONTEXT_RETRY)
builder.add_node("understand", with_turn_context(understand))
builder.add_node("plan", plan)
builder.add_node("clarify", with_turn_context(clarify))
builder.add_node("execute", with_turn_context(execute))
builder.add_node(
    "validate", validation_graph
)  # the output-validation subgraph (its check nodes are stubs; no LLM today)
builder.add_node(
    "memory", memory_graph
)  # the memory subgraph (its LLM nodes wrap themselves)
builder.add_node("respond", with_turn_context(respond))

builder.add_edge(START, "guard")
# Entry gate: a prompt-injection input short-circuits to END with a canned refusal (written by
# guard); anything else proceeds to load_context. Placed before load_context so a blocked turn
# never touches the DB or an Anthropic model.
builder.add_conditional_edges(
    "guard",
    route_after_guard,
    {"blocked": END, "ok": "load_context"},
)
builder.add_edge("load_context", "understand")
builder.add_edge("understand", "plan")
builder.add_edge("plan", "clarify")
# Fan out: proceed runs execute + the memory subgraph in parallel; ask_user ends the turn.
builder.add_conditional_edges(
    "clarify",
    route_after_clarify,
    {"ask_user": END, "execute": "execute", "memory": "memory"},
)
# Answer branch: gate the capability output before it is composed. Memory branch is independent.
builder.add_edge("execute", "validate")
# Fan in: both branches join at respond, which composes the single user-facing reply.
builder.add_edge("validate", "respond")
builder.add_edge("memory", "respond")
builder.add_edge("respond", END)

graph = builder.compile()
