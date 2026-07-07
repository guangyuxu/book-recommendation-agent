"""Clarification Policy: continue, ask the user, or run best-effort.

Ambiguous-child is decided deterministically (always ask). Otherwise the LLM weighs the
planned capabilities' required inputs against what's known and chooses. On ask_user the node
appends the question and the graph routes to END; the next user turn re-enters understand.
"""

from __future__ import annotations

from typing import cast

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from ..capabilities import REGISTRY, required_inputs
from ..llm import model
from ..schemas import ClarificationDecision
from .plan import ambient_satisfied

_structured = model.with_structured_output(ClarificationDecision)


def _ask(decision: ClarificationDecision) -> dict:
    return {
        "clarification": decision.model_dump(),
        "messages": [AIMessage(content=decision.question or "Could you tell me a bit more?")],
    }


def clarify(state: dict) -> dict:
    """Decide whether to proceed, ask a question, or run with assumptions."""
    u = state.get("understanding") or {}
    plan = state.get("plan") or {}
    steps = plan.get("steps") or []

    # Deterministic: a needed-but-unknown child always requires asking.
    if u.get("child_ambiguous"):
        return _ask(
            ClarificationDecision(
                decision="ask_user",
                missing_inputs=["target_child"],
                question="Which child is this for?",
            )
        )

    # Only genuinely-unmet required inputs: drop those an in-plan step produces (a dependency
    # will supply them) and those already satisfiable from state (ambient).
    produced = {r for s in steps for r in REGISTRY[s["capability"]].produces}
    needs = sorted(
        {
            inp
            for s in steps
            for inp in required_inputs(s["capability"])
            if inp not in produced and not ambient_satisfied(inp, state)
        }
    )
    has_child = bool(state.get("target_child_id")) or bool(u.get("child_is_new"))

    system = SystemMessage(
        content=(
            "You are a clarification policy. Decide one of: continue (we can act now), "
            "ask_user (a required input is missing and we must ask), or best_effort (proceed "
            "with stated assumptions). Prefer continue or best_effort; only ask_user when a "
            "genuinely required input (e.g. a specific book title for evaluate/compare/"
            "discussion) is absent and cannot be reasonably assumed. If you ask_user, write a "
            "single concise question."
        )
    )
    human = HumanMessage(
        content=(
            f"planned_capabilities={[s['capability'] for s in steps]}\n"
            f"required_inputs={needs}\n"
            f"target_child_known={has_child}\n"
            f"mentioned_books={u.get('mentioned_books')}"
        )
    )
    result = cast(
        ClarificationDecision, _structured.invoke([system, human, *state["messages"]])
    )
    if result.decision == "ask_user":
        return _ask(result)
    return {"clarification": result.model_dump()}


def route_after_clarify(state: dict) -> str:
    """Route to END when we asked the user, otherwise on to capability execution."""
    decision = (state.get("clarification") or {}).get("decision")
    return "ask_user" if decision == "ask_user" else "execute"
