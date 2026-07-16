"""Clarification Policy: continue, ask the user, or run best-effort.

Ambiguous-child is decided deterministically (always ask), and so is the common case where
every required input is already satisfied (always continue -- no LLM). The LLM is consulted
only when a required input is genuinely unmet, to weigh asking vs. a reasonable assumption.
On ask_user the node appends the question and the graph routes to END; the next user turn
re-enters understand.
"""

from __future__ import annotations

from typing import Any, cast

from langchain.messages import AIMessage

from .. import prompts
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

    # No unmet required input -> nothing to adjudicate; proceed deterministically. Consulting the
    # LLM here is what caused over-asking: given only a boolean target_child_known (never the
    # profile contents), it would invent "missing" inputs like age/interests and ask_user, even
    # though recommend is designed to degrade gracefully on a sparse profile. The LLM only earns a
    # say when a required input is actually absent (needs non-empty) and might be assumable.
    if not needs:
        return {
            "clarification": ClarificationDecision(decision="continue").model_dump()
        }

    has_child = bool(state.get("target_child_id")) or bool(u.get("child_is_new"))

    messages = prompts.render(
        "clarify.decide",
        reply_directive=reply_directive(lang).lstrip("\n"),
        planned_capabilities=[s["capability"] for s in steps],
        required_inputs=needs,
        target_child_known=has_child,
        mentioned_books=u.get("mentioned_books"),
    )
    result = cast(
        ClarificationDecision,
        _structured.invoke(
            [*messages, *state["messages"]], config=prompts.config("clarify.decide")
        ),
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
