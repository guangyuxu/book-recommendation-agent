"""Output-validation subgraph builder: select -> (parallel checks) -> aggregate.

Packaged as a LangGraph subgraph so the main graph runs it as one node ("validate") on the
answer-generation branch, between `execute` and `respond` (see agent.graph). Like the memory
subgraph it shares the parent FlowState, so `capability_results` (its input) and `validation`
(its output) flow straight through with no input/output mapping.

    START -> select --route_checks--> [check_<name> ...] -> aggregate -> END
                     (no checks apply) -----------------> aggregate

The check nodes are GENERATED from CHECK_REGISTRY (the registry is the single source of truth), so
adding/removing a check is a registry edit -- the wiring below does not change. They fan out via a
conditional edge returning the list of applicable node names (the same fan-out idiom the main
graph uses for clarify -> {execute, memory}) and fan back in at `aggregate`, which runs once.

Compiled WITHOUT a checkpointer on purpose: a nested subgraph inherits the parent's. The check
stubs make no LLM calls, so no node is wrapped with with_turn_context; a future LLM-backed check
must wrap itself (same convention as the memory subgraph).
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from ..state import AppContext, FlowState
from .checks import CHECK_REGISTRY, node_name
from .nodes import aggregate, make_check_node, route_checks, select

_builder = StateGraph(FlowState, context_schema=AppContext)
_builder.add_node("select", select)
_builder.add_node("aggregate", aggregate)

# One worker node per registered check, generated from the registry. Cast because add_node's
# overloads match a plain `def` but not a factory-returned Callable variable (same behavior, only
# a typing quirk).
_check_nodes = [node_name(check) for check in CHECK_REGISTRY.values()]
for _check in CHECK_REGISTRY.values():
    _builder.add_node(node_name(_check), cast(Any, make_check_node(_check)))

_builder.add_edge(START, "select")
# Fan out to the applicable check nodes (or straight to aggregate when none apply).
_builder.add_conditional_edges(
    "select",
    route_checks,
    [*_check_nodes, "aggregate"],
)
# Fan in: every check node joins at aggregate, which runs once after they all complete.
for _node in _check_nodes:
    _builder.add_edge(_node, "aggregate")
_builder.add_edge("aggregate", END)

validation_graph = _builder.compile()
