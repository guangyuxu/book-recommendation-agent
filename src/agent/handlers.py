"""Profile-update intent handlers (currently mock). Promote to a flows/ subgraph as logic grows."""

import logging

from langchain.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from .db import SessionLocal, repo
from .intents import Intent
from .lifecycle import get_context
from .state import MessagesState, target_children

logger = logging.getLogger(__name__)


def make_mock_handler(intent: Intent, message: str | None = None):
    """Build a handler node that returns a placeholder reply.

    Defaults to the intent label; pass `message` to customize the mock text.
    """
    text = message or f"(mock) {intent.label}"

    def handler(state: MessagesState):
        return {"messages": [AIMessage(content=text)]}

    handler.__name__ = intent.value  # show as the intent name in traces
    return handler


def child_profile_update(state: MessagesState, config: RunnableConfig | None = None):
    update = {"reading_level": "intermediate", "recent_signal": "leans binary"}  # TODO real: extract from message
    user_id, _ = get_context(config)
    targets = state.get("target_child_ids") or []
    written: dict[str, dict] = {}
    if user_id:
        with SessionLocal() as s:
            # Existing target(s) -> update each by id; none -> create a new child.
            for cid in targets or [None]:
                new_id = repo.upsert_child_profile(s, user_id, fields=update, child_id=cid)
                written[new_id] = {**update, "id": new_id}
    else:  # no persistence context (e.g. direct call) -- still reflect in state
        for cid in targets or ["new"]:
            written[cid] = {**update, "id": cid}
    who = ", ".join(written) or "child"
    return {
        "children": written,
        "messages": [AIMessage(content=f"Child profile updated ({who}): {update}")],
    }


def reading_discussion(state: MessagesState, config: RunnableConfig | None = None):
    """Mock: discussion questions, addressed to the resolved target child(ren)."""
    who = ", ".join(c.get("name", c["id"]) for c in target_children(state)) or "the child"
    return {
        "messages": [
            AIMessage(content=f"(mock reading discussion flow) discussion questions for {who}")
        ]
    }


def parent_goal_update(state: MessagesState, config: RunnableConfig | None = None):
    goals = ["understand complex human nature, not simple good vs. evil"]  # TODO real: extract from message
    user_id, _ = get_context(config)
    if user_id:
        with SessionLocal() as s:
            repo.upsert_parent_profile(s, user_id, goals=goals)
    return {
        "parent_goals": goals,
        "messages": [AIMessage(content=f"Parent goal recorded: {'; '.join(goals)}")],
    }


def parent_profile_update(state: MessagesState, config: RunnableConfig | None = None):
    update = {"available_time": "weekends only", "self_taste": "sci-fi"}  # TODO real: extract from message
    user_id, _ = get_context(config)
    if user_id:
        with SessionLocal() as s:
            repo.upsert_parent_profile(s, user_id, fields=update)
    return {
        "parent_profile": update,
        "messages": [AIMessage(content=f"Parent profile updated: {update}")],
    }
