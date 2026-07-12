"""Capability Execution subgraph: fan the planned capabilities out in parallel, then aggregate.

`plan` produces a flat, unordered list of capabilities (they are independent -- none consumes
another's output). This subgraph has one node per capability; `dispatch` routes -- via a plain
list-of-targets conditional edge, exactly like the main graph's clarify fan-out -- to the nodes
whose capability the plan asked for, they run in parallel, and `aggregate` merges their outputs
into `capability_results` (of which it is the sole writer, so the channel stays last-write-wins
for the rest of the pipeline exactly as before).

    START -> dispatch --(plan has recommend)--> recommend --+
                       --(plan has content)---> content ----+--> aggregate -> END
                       --(... one branch per capability ...)-+
                       --(empty plan)------------------------> aggregate

Mirrors the memory subgraph (agent.memory.graph): compiled WITHOUT a checkpointer (a nested
subgraph inherits the parent's), with each LLM-invoking capability node wrapping itself in
with_turn_context so its token usage is billed to the turn.
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langchain.messages import AnyMessage
from langgraph.graph import END, START, StateGraph

from ..capabilities import REGISTRY
from ..state import AppContext
from ..usage_tracker import with_turn_context

logger = logging.getLogger(__name__)


class ExecuteState(TypedDict, total=False):
    """Private state for the execute subgraph.

    The context channels (messages .. thread_id) and `plan` are mapped in from the parent
    FlowState by name; `capability_results` is mapped back out (aggregate is its sole writer).
    `results` is private -- not in FlowState -- so it never leaks to the parent or across turns:
    it is the fan-in scratch that the parallel capability nodes append to (operator.add), read
    only by aggregate.

    `messages` is deliberately a plain (last-write-wins) channel here, not the add_messages
    channel it is in FlowState: execute only ever reads the conversation, never appends to it.
    """

    # context the capabilities read (mapped in from the parent FlowState)
    messages: list[AnyMessage]
    target_child_id: str | None
    children: dict[str, dict[str, Any]]
    policies: list[dict[str, Any]]
    turn_id: str
    thread_id: str | None
    plan: dict[str, Any]
    # written back to the parent; aggregate is the sole writer
    capability_results: dict[str, dict[str, Any]]
    # private / ephemeral fan-in scratch
    results: Annotated[list[dict[str, dict[str, Any]]], operator.add]


def dispatch(state: ExecuteState) -> dict[str, Any]:
    """Anchor node for the fan-out; the plan is read on the outgoing edge (fan_out)."""
    return {}


def fan_out(state: ExecuteState) -> list[str]:
    """Return the capability node(s) the plan asked for, or ["aggregate"] when the plan is empty.

    A list of node names fans out to all of them in parallel -- the same mechanism the main
    graph's clarify uses to start execute + memory together.
    """
    steps = (state.get("plan") or {}).get("steps") or []
    caps = [s["capability"] for s in steps if s["capability"] in REGISTRY]
    return caps or ["aggregate"]


def _capability_node(name: str) -> Callable[[ExecuteState], dict[str, Any]]:
    """Build the node that runs one capability and appends its result to the fan-in scratch.

    A failing capability contributes nothing -- logged by exception type only, never PII -- so
    one broken capability never sinks the turn; the rest still aggregate.
    """

    def node(state: ExecuteState) -> dict[str, Any]:
        # graph-backed capabilities are wired as their own node, not here.
        run = REGISTRY[name].run
        if run is None:
            return {}
        try:
            result = run(dict(state))
        except Exception as exc:  # one capability failing must not sink the turn
            logger.warning(
                "execute: capability %s failed: %s", name, type(exc).__name__
            )
            return {}
        return {"results": [{name: result}]}

    node.__name__ = name
    return node


def aggregate(state: ExecuteState) -> dict[str, Any]:
    """Merge the parallel capabilities' results into capability_results (its sole writer).

    Always writes the channel in full (even {} for an empty plan) so a prior turn's results never
    linger into this turn's render/persist.
    """
    merged: dict[str, dict[str, Any]] = {}
    for item in state.get("results") or []:
        merged.update(item)
    return {"capability_results": merged}


_builder = StateGraph(ExecuteState, context_schema=AppContext)
_builder.add_node("dispatch", dispatch)
# One node per capability, dispatched the way the capability declares (registry.Capability):
#   - graph-backed (e.g. recommend's generate/validate self-critique subgraph): wired in directly
#     as a nested node. Its own LLM nodes wrap themselves, and it appends {name: result} to the
#     `results` fan-in channel itself.
#   - run-backed: wrapped in _capability_node, which wraps itself with with_turn_context (like the
#     memory subgraph's LLM nodes) so its transitive LLM call is billed to the turn.
# Either way the node lands its contribution in `results`, so aggregate treats them identically.
for _name, _cap in REGISTRY.items():
    if _cap.graph is not None:
        _builder.add_node(_name, _cap.graph)
    else:
        _builder.add_node(_name, with_turn_context(_capability_node(_name)))
_builder.add_node("aggregate", aggregate)

_builder.add_edge(START, "dispatch")
_builder.add_conditional_edges("dispatch", fan_out, [*REGISTRY, "aggregate"])
for _name in REGISTRY:
    _builder.add_edge(_name, "aggregate")
_builder.add_edge("aggregate", END)

execute_graph = _builder.compile()
