"""Clarification Policy: continue, ask the user, or run best-effort.

Ambiguous-child is decided deterministically (always ask). Otherwise the LLM weighs the
planned capabilities' required inputs against what's known and chooses. On ask_user the node
appends the question and the graph routes to END; the next user turn re-enters understand.
"""

from __future__ import annotations

from typing import Any, cast

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from ..capabilities import required_inputs
from ..language import Language, normalize_language, reply_directive
from ..llm import FAST
from ..state import FlowState
from .plan import ambient_satisfied
from .schemas import ClarificationDecision

_structured = FAST.structured(ClarificationDecision)

# Localized deterministic questions (the LLM-generated question is produced in-language via
# reply_directive; these two are hardcoded paths that never touch the LLM).
_ASK_WHICH_CHILD: dict[Language, str] = {
    "en": "Which child is this for?",
    "zh-Hans": "这是为哪个孩子问的呢？",
    "zh-Hant": "這是為哪個孩子問的呢？",
}
_ASK_MORE: dict[Language, str] = {
    "en": "Could you tell me a bit more?",
    "zh-Hans": "可以再多告诉我一些吗？",
    "zh-Hant": "可以再多告訴我一些嗎？",
}


def _ask(decision: ClarificationDecision, lang: Language) -> dict[str, Any]:
    return {
        "clarification": decision.model_dump(),
        "messages": [AIMessage(content=decision.question or _ASK_MORE[lang])],
    }


def clarify(state: FlowState) -> dict[str, Any]:
    """Decide whether to proceed, ask a question, or run with assumptions."""
    u = state.get("understanding") or {}
    plan = state.get("plan") or {}
    steps = plan.get("steps") or []
    lang = normalize_language(state.get("reply_language"))

    # Deterministic: a needed-but-unknown child always requires asking.
    if u.get("child_ambiguous"):
        return _ask(
            ClarificationDecision(
                decision="ask_user",
                missing_inputs=["target_child"],
                question=_ASK_WHICH_CHILD[lang],
            ),
            lang,
        )

    # Only genuinely-unmet required inputs: keep those not already satisfiable from state (ambient).
    needs = sorted(
        {
            inp
            for s in steps
            for inp in required_inputs(s["capability"])
            if not ambient_satisfied(inp, state)
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
            "single concise question." + reply_directive(lang)
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
        return _ask(result, lang)
    return {"clarification": result.model_dump()}


def route_after_clarify(state: FlowState) -> str | list[str]:
    """Route to END when we asked the user, otherwise fan out to both parallel branches.

    On proceed we return BOTH branch names so LangGraph starts `execute` (answer generation) and
    the `memory` subgraph (decide -> confirm -> persist) in the same superstep; they fan back in
    at respond.
    """
    decision = (state.get("clarification") or {}).get("decision")
    return "ask_user" if decision == "ask_user" else ["execute", "memory"]
