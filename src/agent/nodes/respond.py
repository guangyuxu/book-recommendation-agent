"""Response: compose the final reply and deterministically persist a recommendation turn.

Composition stitches the capability outputs into one coherent message. Persistence (writing a
recommendation_session + items) happens here, not via the LLM, and only for actual
recommendation turns (a recommend result with a booklist and a resolved child) -- flavor A.
"""

from __future__ import annotations

import logging

from langchain.messages import AIMessage, SystemMessage

from ..domain import (
    create_recommendation_session,
    domain_session,
    save_recommendation_items,
)
from ..llm import model

logger = logging.getLogger(__name__)


def _last_human_text(messages: list) -> str:
    for m in reversed(messages):
        if getattr(m, "type", None) == "human":
            return str(m.content)
    return ""


def _render_outputs(results: dict) -> str:
    parts: list[str] = []
    rec = results.get("recommend")
    if rec and rec.get("books"):
        lines = ["Recommended books:"]
        for i, b in enumerate(rec["books"], start=1):
            extras = []
            if b.get("fit_summary"):
                extras.append(f"fit: {b['fit_summary']}")
            if b.get("risk_notes"):
                extras.append(f"watch-outs: {', '.join(b['risk_notes'])}")
            suffix = f" ({'; '.join(extras)})" if extras else ""
            author = f" by {b['author']}" if b.get("author") else ""
            lines.append(f"{i}. {b['title']}{author} — {b.get('recommendation_reason', '')}{suffix}")
        if rec.get("note"):
            lines.append(rec["note"])
        parts.append("\n".join(lines))
    for name, result in results.items():
        if name == "recommend" or not isinstance(result, dict):
            continue
        # Prose capabilities carry a single string value under their produced-resource key.
        prose = next((v for v in result.values() if isinstance(v, str) and v.strip()), None)
        if prose:
            parts.append(prose)
    return "\n\n".join(parts)


def _switch_note(state: dict) -> str:
    """A hint telling the reply to acknowledge a focus switch to another child (point 2)."""
    switch = state.get("child_switch") or {}
    to_name = switch.get("to_name")
    if not to_name:
        return ""
    return (
        f"\n\nThe conversation focus just switched to {to_name}. Briefly and naturally "
        "acknowledge at the start of your reply whom you are now talking about."
    )


def _confirmation_note(state: dict) -> str:
    """A hint to acknowledge the outcome of a confirmation gate (point 1/3)."""
    status = (state.get("confirmation") or {}).get("status")
    if status == "applied":
        return (
            "\n\nThe parent just confirmed a profile change; briefly acknowledge that it is saved."
        )
    if status == "rejected":
        return (
            "\n\nThe parent declined the proposed profile change; briefly acknowledge you will "
            "not save it."
        )
    return ""


def _compose(state: dict, rendered: str) -> str:
    system = SystemMessage(
        content=(
            "You are the family's reading assistant. Using the prepared material below, write "
            "one warm, concise reply to the parent's latest message. Do not invent books or "
            "facts beyond the material; if there is no material, respond helpfully to the "
            "message itself.\n\n"
            f"Prepared material:\n{rendered or '(none)'}"
            f"{_switch_note(state)}"
            f"{_confirmation_note(state)}"
        )
    )
    reply = model.invoke([system, *state["messages"]])
    return str(reply.content)


def _persist_recommendation(state: dict, response_text: str, rec: dict) -> None:
    u = state.get("understanding") or {}
    items = [
        {
            "title": b["title"],
            "author": b.get("author"),
            "rank": i,
            "recommendation_reason": b.get("recommendation_reason"),
            "fit_summary": b.get("fit_summary"),
            "risk_notes": b.get("risk_notes") or [],
        }
        for i, b in enumerate(rec.get("books") or [], start=1)
    ]
    with domain_session(
        state["family"]["id"],
        state.get("target_child_id"),
        requester_member_id=state.get("family_member_id"),
    ):
        session_id = create_recommendation_session.invoke(
            {
                "intents": u.get("intents") or [],
                "user_message": _last_human_text(state["messages"]),
                "understanding": u,
                "plan": state.get("plan") or {},
                "capability_result": state.get("capability_results") or {},
                "memory_decision": {"operations": state.get("memory_operations") or []},
                "response_text": response_text,
                "requester_member_id": state.get("family_member_id"),
            }
        )
        save_recommendation_items.invoke({"session_id": session_id, "items": items})


def respond(state: dict) -> dict:
    """Compose the user-facing reply; persist the turn if it produced a recommendation."""
    results = state.get("capability_results") or {}
    rendered = _render_outputs(results)
    reply_text = _compose(state, rendered)

    rec = results.get("recommend")
    if rec and rec.get("books") and state.get("target_child_id"):
        try:
            _persist_recommendation(state, reply_text, rec)
        except Exception as exc:  # persistence must not break the user's reply
            logger.warning("respond: failed to persist recommendation session: %s", exc)

    return {"messages": [AIMessage(content=reply_text)]}
