"""Graph entry/exit machinery: resolve request context, load profiles, log chat."""

import logging

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import get_runtime
from langgraph.types import RetryPolicy

from .db import SessionLocal, repo
from .state import AppContext, MessagesState

logger = logging.getLogger(__name__)


class MissingUserIDError(ValueError):
    """Raised when a request is missing user_id. Subclasses ValueError, so LangGraph won't retry."""


def _never_retry(exc: Exception) -> bool:
    return False


# A missing user_id is a deterministic input error; retrying only spams the stack trace.
# Disable retry so the validation failure is logged once.
LOAD_CONTEXT_RETRY = RetryPolicy(retry_on=_never_retry)


def get_context(config: RunnableConfig | None) -> tuple[str | None, str | None]:
    """Return (user_id, thread_id).

    user_id comes from the context channel first (Studio / context=), falling back to
    configurable for older callers. get_runtime only works inside graph execution, so the
    try/except lets direct calls (tests / orchestrator) work too.
    """
    cfg = (config or {}).get("configurable", {})
    user_id = None
    try:
        user_id = get_runtime(AppContext).context.get("user_id")
    except Exception:
        user_id = None
    if not user_id:
        user_id = cfg.get("user_id")
    return user_id, cfg.get("thread_id")


def load_context(state: MessagesState, config: RunnableConfig | None = None):
    """Entry: load the profile into state by user_id and log this user message. user_id is required."""
    user_id, thread_id = get_context(config)
    if not user_id:
        logger.warning("Validation failed: request is missing user_id, refusing to run.")
        raise MissingUserIDError(
            "Missing required parameter user_id: every request must carry user_id. "
            "In Studio, set user_id in the run config (context); "
            'for SDK/API calls pass context={"user_id": ...}.'
        )
    with SessionLocal() as s:
        delta = repo.load_state_profiles(s, user_id)
        last = state["messages"][-1] if state["messages"] else None
        if thread_id and getattr(last, "type", None) == "human":
            repo.add_chat_message(s, user_id, thread_id, "human", str(last.content))
    return delta


def finalize(state: MessagesState, config: RunnableConfig | None = None):
    """Exit: write this turn's new AI replies to chat_history.

    Takes all AI messages after the last human message: multi_intent produces several
    replies, so we can't log only the last one, while prior turns' AI messages (before the
    last human message) are not re-logged.
    """
    user_id, thread_id = get_context(config)
    if not (user_id and thread_id):
        return {}
    messages = state["messages"]
    last_human = max(
        (i for i, m in enumerate(messages) if getattr(m, "type", None) == "human"),
        default=-1,
    )
    new_ai = [
        m for m in messages[last_human + 1 :] if getattr(m, "type", None) == "ai"
    ]
    if new_ai:
        with SessionLocal() as s:
            for m in new_ai:
                repo.add_chat_message(s, user_id, thread_id, "ai", str(m.content))
    return {}
