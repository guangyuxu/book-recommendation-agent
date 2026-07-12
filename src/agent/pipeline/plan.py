"""Planner Policy: turn the understanding's intents into a flat list of capabilities to run.

Deterministic -- no LLM. Each task intent maps to a capability (the turn's "goals"). Capabilities
are independent (none consumes another's output), so the plan is an unordered set with no edges;
Execute fans them out in parallel. Ambient resources (target_child, reading_profile, policies,
user-named books) come from state and are checked by clarify, not by the planner. Profile-update /
clarify intents map to no capability, so an empty plan is normal.
"""

from __future__ import annotations

from typing import Any

from ..capabilities import AMBIENT, for_intent
from ..intents import to_intent
from ..state import FlowState
from .schemas import Plan, PlanStep


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


def plan(state: FlowState) -> dict[str, Any]:
    """Resolve the understanding's intents into a flat list of capabilities to run."""
    u = state.get("understanding") or {}
    goals = _goals(u.get("intents") or [])
    steps = [PlanStep(capability=cap) for cap in goals]
    return {"plan": Plan(steps=steps).model_dump()}
