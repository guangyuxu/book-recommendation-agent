"""Planner Policy: decide which capabilities to run, in what order. Never touches the DB.

The LLM proposes a plan from the capability menu; we then validate every step against the
registry (dropping anything hallucinated) and fall back to a deterministic intent->capability
mapping if the LLM produced nothing usable. Profile-update / clarify intents map to no
capability, so an empty plan is normal.
"""

from __future__ import annotations

from typing import cast

from langchain.messages import HumanMessage, SystemMessage

from ..capabilities import REGISTRY, for_intent, menu
from ..intents import to_intent
from ..llm import model
from ..schemas import Plan, PlanStep

_structured = model.with_structured_output(Plan)


def _fallback_steps(intent_values: list[str | None]) -> list[PlanStep]:
    steps: list[PlanStep] = []
    seen: set[str] = set()
    for value in intent_values:
        if not value:
            continue
        cap = for_intent(to_intent(value))
        if cap and cap.name not in seen:
            steps.append(PlanStep(capability=cap.name, reason="intent maps to this capability"))
            seen.add(cap.name)
    return steps


def plan(state: dict) -> dict:
    """Turn the understanding into an ordered, validated capability plan."""
    u = state.get("understanding") or {}
    intent_values = [u.get("primary_intent"), u.get("secondary_intent")]

    system = SystemMessage(
        content=(
            "You are a planner. Given the understanding, choose which capabilities to run this "
            "turn and order them (set depends_on when one needs another's output first). Use "
            "ONLY these capabilities; if none apply, return an empty plan (the turn may be a "
            "pure profile update).\n\n"
            f"{menu()}"
        )
    )
    human = HumanMessage(
        content=(
            f"primary_intent={u.get('primary_intent')}, secondary_intent={u.get('secondary_intent')}\n"
            f"mentioned_books={u.get('mentioned_books')}\n"
            f"planner_hints={u.get('planner_hints')}"
        )
    )
    result = cast(Plan, _structured.invoke([system, human]))

    valid = [s for s in result.steps if s.capability in REGISTRY]
    if not valid:
        valid = _fallback_steps(intent_values)
    return {"plan": Plan(steps=valid).model_dump()}
