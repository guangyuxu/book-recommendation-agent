"""Planner Policy: turn the understanding's intents into an ordered, dependency-aware plan.

Deterministic DAG resolver -- no LLM. Each task intent maps to a capability (the turn's
"goals"); goals are then connected via the registry's produces/required_inputs vocabulary: if
goal B needs a resource that goal A produces, B depends_on A. Ambient resources (target_child,
reading_profile, policies, user-named books) come from state and are preconditions, not edges.
Profile-update / clarify intents map to no capability, so an empty plan is normal.

We only wire edges BETWEEN goals -- a producer is never pulled in just to satisfy an input
(that would fabricate work the user did not ask for). So e.g. recommend->discussion forms only
when both intents are present.
"""

from __future__ import annotations

import logging
from typing import Any

from ..capabilities import AMBIENT, REGISTRY, for_intent
from ..intents import to_intent
from ..state import FlowState
from .schemas import Plan, PlanStep

logger = logging.getLogger(__name__)


def _goals(intents: list[str]) -> list[str]:
    """Ordered, de-duplicated capability names for the turn's task intents."""
    seen: set[str] = set()
    goals: list[str] = []
    for value in intents:
        if not value:
            continue
        cap = for_intent(to_intent(value))
        if cap and cap.name not in seen:
            seen.add(cap.name)
            goals.append(cap.name)
    return goals


def ambient_satisfied(resource: str, state: FlowState) -> bool:
    """Whether an ambient resource can be met from state this turn (no capability needed).

    Shared with the clarify node so both reason about availability the same way.
    """
    if resource not in AMBIENT:
        return False
    u = state.get("understanding") or {}
    child_known = bool(state.get("target_child_id")) or bool(u.get("child_is_new"))
    if resource in ("target_child", "reading_profile"):
        # A profile may be sparse; capabilities degrade gracefully, so a resolved child suffices.
        return child_known
    if resource == "policies":
        return True  # may be empty; only ever an optional input
    if resource == "books":
        return bool(u.get("mentioned_books"))
    return False


def _resolve_edges(goals: list[str], state: FlowState) -> list[tuple[str, str]]:
    """Producer->consumer edges: a goal input satisfied by another goal's produces.

    A producing goal takes priority over ambient for that resource -- if we are recommending
    this turn, downstream book-consumers use the recommendations rather than any named book.
    """
    producer_of = {
        resource: cap for cap in goals for resource in REGISTRY[cap].produces
    }
    edges: list[tuple[str, str]] = []
    for cap in goals:
        spec = REGISTRY[cap]
        for resource in (*spec.required_inputs, *spec.optional_inputs):
            producer = producer_of.get(resource)
            if producer and producer != cap:
                edges.append((producer, cap))
    return edges


def _topo_order(goals: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Kahn topological sort; fall back to goal order if a cycle is detected."""
    indegree = {g: 0 for g in goals}
    succ: dict[str, list[str]] = {g: [] for g in goals}
    for a, b in edges:
        succ[a].append(b)
        indegree[b] += 1
    queue = [g for g in goals if indegree[g] == 0]
    order: list[str] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for m in succ[n]:
            indegree[m] -= 1
            if indegree[m] == 0:
                queue.append(m)
    if len(order) < len(goals):
        logger.warning("plan: dependency cycle among %s; using intent order", goals)
        return goals
    return order


def plan(state: FlowState) -> dict[str, Any]:
    """Resolve the understanding's intents into an ordered, dependency-aware capability plan."""
    u = state.get("understanding") or {}
    goals = _goals(u.get("intents") or [])
    if not goals:
        return {"plan": Plan(steps=[]).model_dump()}

    edges = _resolve_edges(goals, state)
    order = _topo_order(goals, edges)

    deps_of: dict[str, list[str]] = {g: [] for g in goals}
    for producer, consumer in edges:
        deps_of[consumer].append(producer)

    steps = [PlanStep(capability=cap, depends_on=deps_of[cap]) for cap in order]
    return {"plan": Plan(steps=steps).model_dump()}
