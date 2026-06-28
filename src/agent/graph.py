"""Main graph: START -> load_context -> classify -> (route by intent) -> flows -> finalize -> END."""

from langgraph.graph import END, START, StateGraph

from .classify import classify
from .flows import orchestrator
from .intents import CHILD_SPECIFIC, Intent
from .lifecycle import LOAD_CONTEXT_RETRY, finalize, load_context
from .registry import HANDLERS
from .resolve import resolve_children
from .state import AppContext, MessagesState

_CHILD_SPECIFIC_VALUES = [i.value for i in Intent if i in CHILD_SPECIFIC]
_NON_CHILD_VALUES = [i.value for i in Intent if i not in CHILD_SPECIFIC]


def route_after_classify(state: MessagesState):
    """Child-specific intents go through resolve first; the rest route straight to their node."""
    return "resolve" if Intent(state["intent"]) in CHILD_SPECIFIC else state["intent"]


def route_after_resolve(state: MessagesState):
    """Ambiguous target child (parent has several, can't tell which) -> ask. Else the intent's node."""
    return "clarify" if state.get("ambiguous") else state["intent"]


builder = StateGraph(MessagesState, context_schema=AppContext)  # type: ignore[arg-type]
builder.add_node("load_context", load_context, retry_policy=LOAD_CONTEXT_RETRY)  # type: ignore[arg-type]
builder.add_node("classify", classify)
builder.add_node("resolve", resolve_children)
builder.add_node("finalize", finalize)  # type: ignore[arg-type]

for intent, node in HANDLERS.items():
    builder.add_node(intent.value, node)
# The orchestrator is not in the registry (to avoid recursing into itself); mounted separately.
builder.add_node(Intent.MULTI_INTENT.value, orchestrator.graph)

builder.add_edge(START, "load_context")
builder.add_edge("load_context", "classify")
builder.add_conditional_edges("classify", route_after_classify, ["resolve", *_NON_CHILD_VALUES])
builder.add_conditional_edges("resolve", route_after_resolve, ["clarify", *_CHILD_SPECIFIC_VALUES])
for intent in Intent:
    builder.add_edge(intent.value, "finalize")
builder.add_edge("finalize", END)

graph = builder.compile()
