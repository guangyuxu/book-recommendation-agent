"""Response: compose the final reply and deterministically persist a recommendation turn.

Composition stitches the capability outputs into one coherent message. Persistence (writing a
recommendation_session + items) happens here, not via the LLM, and only for actual
recommendation turns (a recommend result with a booklist and a resolved child) -- flavor A.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from langchain.messages import AIMessage

from .. import prompts
from ..domain import (
    create_recommendation_session,
    domain_session,
    save_recommendation_items,
)
from ..language import reply_directive
from ..llm import STANDARD
from ..state import FlowState

logger = logging.getLogger(__name__)


def _last_human_text(messages: list[Any]) -> str:
    for m in reversed(messages):
        if getattr(m, "type", None) == "human":
            return str(m.content)
    return ""


def _render_outputs(results: dict[str, Any]) -> str:
    parts: list[str] = []
    rec = results.get("recommend")
    # noinspection PyUnresolvedReferences
    if rec and rec.get("books"):
        lines = ["Recommended books:"]
        # noinspection PyUnresolvedReferences
        for i, b in enumerate(rec["books"], start=1):
            extras = []
            if b.get("fit_summary"):
                extras.append(f"fit: {b['fit_summary']}")
            if b.get("risk_notes"):
                extras.append(f"watch-outs: {', '.join(b['risk_notes'])}")
            suffix = f" ({'; '.join(extras)})" if extras else ""
            author = f" by {b['author']}" if b.get("author") else ""
            lines.append(
                f"{i}. {b['title']}{author} — {b.get('recommendation_reason', '')}{suffix}"
            )
        # noinspection PyUnresolvedReferences
        if rec.get("note"):
            # noinspection PyUnresolvedReferences
            lines.append(rec["note"])
        parts.append("\n".join(lines))
    for name, result in results.items():
        if name == "recommend" or not isinstance(result, dict):
            continue
        prose = _prose(name, result)
        if prose:
            parts.append(prose)
    return "\n\n".join(parts)


def _prose(name: str, result: dict[str, Any]) -> str | None:
    """Pull a prose capability's text: each returns a single `{key: text}`, so take that string.

    Every prose capability (evaluate/compare/discussion/path/content) returns exactly one
    string-valued key, so the first non-empty string in the result is that text.
    """
    return next((v for v in result.values() if isinstance(v, str) and v.strip()), None)


def _gather_text(chunks: Iterable[Any]) -> str:
    """Reduce a stream of message chunks into the final reply text.

    `respond` streams its reply (so `stream_mode="messages"` surfaces tokens to the frontend as
    they arrive) but still needs the whole string for state + persistence. AIMessageChunks add
    together into one message; an empty stream yields "".
    """
    gathered: Any = None
    for chunk in chunks:
        gathered = chunk if gathered is None else gathered + chunk
    return str(gathered.content) if gathered is not None else ""


def _compose(state: FlowState, rendered: str) -> str:
    # Python decides the facts; the respond.compose prompt (respond.prompts.yaml) owns the wording,
    # including the conditional focus-switch / confirmation-outcome notes. reply_directive already
    # carries a leading blank line for f-string concatenation; strip it since the template spaces
    # its own blocks.
    switch = state.get("child_switch") or {}
    messages = prompts.render(
        "respond.compose",
        material=rendered,
        switch_to_name=switch.get("to_name"),
        confirmation_status=(state.get("confirmation") or {}).get("status"),
        reply_directive=reply_directive(state.get("reply_language")).lstrip("\n"),
    )
    # Stream (not .invoke) so token chunks reach the frontend via stream_mode="messages";
    # _gather_text reassembles the full reply for the AIMessage we return to state.
    return _gather_text(
        STANDARD.stream_chain().stream(
            [*messages, *state["messages"]], config=prompts.config("respond.compose")
        )
    )


def _persist_recommendation(
    state: FlowState, response_text: str, rec: dict[str, Any]
) -> None:
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
        # noinspection PyUnresolvedReferences
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
        # noinspection PyUnresolvedReferences
        save_recommendation_items.invoke({"session_id": session_id, "items": items})


def respond(state: FlowState) -> dict[str, Any]:
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
