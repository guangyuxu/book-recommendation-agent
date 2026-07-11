"""Token usage tracking: contextvar-based billing context + async queue/consumer.

Design
------
1. Every LLM-invoking node is wrapped with with_turn_context(), which re-establishes
   the billing ContextVar at node entry from graph state (turn_id/thread_id, placed
   there once by load_context) and the AppContext runtime (family_id/member_id).
   A ContextVar set in one node does NOT propagate to other nodes -- each LangGraph
   node runs in its own copied context -- so the ContextVar must be set inside the
   same node whose LLM call we want to capture.

2. UsageCallbackHandler is a stateless LangChain callback attached to every
   Strategy chain.  on_llm_end fires inside the model call, reads the ContextVar,
   and puts a lightweight dict into the internal SimpleQueue.

3. A daemon consumer thread drains the queue and writes TokenUsageRecord rows to
   the DB.  The write is fully async from the caller's perspective -- any DB
   failure is logged but never surfaces to the user.

LangGraph automatically injects `metadata["langgraph_node"]` for every node's
runnable call, so we get the node name for free without modifying any node.
"""

from __future__ import annotations

import functools
import logging
import queue
import threading
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from typing import Any
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


# ── Turn billing context ───────────────────────────────────────────────────────


class TurnContext:
    """Immutable billing identifiers set once per graph turn by load_context."""

    __slots__ = ("family_id", "turn_id", "thread_id", "family_member_id")

    def __init__(
        self,
        family_id: str,
        turn_id: str,
        thread_id: str | None,
        family_member_id: str | None,
    ) -> None:
        self.family_id = family_id
        self.turn_id = turn_id
        self.thread_id = thread_id
        self.family_member_id = family_member_id


_ctx: ContextVar[TurnContext | None] = ContextVar("_usage_ctx", default=None)


def _node_from_meta(meta: dict[str, Any]) -> str:
    """Best-effort graph-node name from run metadata.

    LangGraph exposes the node either as metadata["langgraph_node"] (some versions) or
    encoded in metadata["checkpoint_ns"] as "<node>:<task_id>", with nested subgraphs
    joined by "|" ("memory:<id>|profile_update:<id>"). The node that made the call is the
    last segment; take the name before its ":".
    """
    node = meta.get("langgraph_node")
    if node:
        return str(node)
    ns = meta.get("checkpoint_ns") or ""
    if ns:
        return ns.split("|")[-1].split(":")[0]
    return ""


def set_turn_context(ctx: TurnContext) -> None:
    """Set the billing context for the current node's execution context."""
    _ctx.set(ctx)


def _establish_turn_context(state: Mapping[str, Any]) -> None:
    """(Re)establish the billing ContextVar for the current node from state + runtime.

    A ContextVar set in one LangGraph node is not visible in any other node (each runs
    in its own copied context), so it must be set inside the node whose LLM call we want
    to capture. Identity comes from get_runtime(AppContext); the per-turn id from
    state["turn_id"] (generated once in load_context). Idempotent and cheap; a no-op
    (leaving usage capture off) when turn_id or the runtime context is unavailable.
    """
    turn_id = state.get("turn_id")
    if not turn_id:
        return
    # Lazy imports: usage_tracker is imported by llm.py before state/graph exist.
    from langgraph.runtime import get_runtime

    from .state import AppContext

    try:
        ctx = get_runtime(AppContext).context
    except Exception:
        return
    if ctx is None:
        return
    set_turn_context(
        TurnContext(
            family_id=ctx.family_id,
            turn_id=turn_id,
            thread_id=state.get("thread_id"),
            family_member_id=ctx.family_member_id,
        )
    )


def with_turn_context(node: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a LangGraph node so the billing ContextVar is live during its LLM calls.

    Apply to every node that directly (understand, respond, ...) or transitively
    (execute -> capabilities) invokes an LLM. The wrapper sets the ContextVar from the
    node's `state` at entry, so UsageCallbackHandler -- firing on_llm_end within the
    same node context -- sees the turn identity. Nodes with no LLM call (load_context,
    plan, confirm gate) do not need wrapping.
    """

    @functools.wraps(node)
    def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
        _establish_turn_context(state)
        return node(state, *args, **kwargs)

    return wrapper


# ── Internal queue + daemon consumer ──────────────────────────────────────────


_queue: queue.SimpleQueue[dict[str, Any] | None] = queue.SimpleQueue()


def _consume() -> None:
    """Daemon thread: drain _queue and write records to the DB one at a time."""
    # Lazy import so usage_tracker can be imported before the DB is initialised.
    from .db import TokenUsageRecord, TokenUsageRepository, session_scope

    while True:
        item = _queue.get()
        if item is None:  # None is the shutdown sentinel
            break
        try:
            with session_scope() as s:
                TokenUsageRepository(session=s).add(
                    TokenUsageRecord(
                        id=item["id"],
                        family_id=item["family_id"],
                        turn_id=item["turn_id"],
                        thread_id=item["thread_id"],
                        family_member_id=item["family_member_id"],
                        target_child_id=None,  # resolved after understand; not available here
                        model_id=item["model_id"],
                        strategy=item["strategy"],
                        node=item["node"],
                        input_tokens=item["input_tokens"],
                        output_tokens=item["output_tokens"],
                    )
                )
        except Exception as exc:
            # Log only the type -- the record may contain model ids but no PII.
            logger.warning(
                "usage_tracker: failed to write token record: %s", type(exc).__name__
            )


_consumer = threading.Thread(target=_consume, daemon=True, name="usage-consumer")
_consumer.start()


# ── Callback handler ───────────────────────────────────────────────────────────


class UsageCallbackHandler(BaseCallbackHandler):
    """Intercepts every LLM call and enqueues a token-usage record.

    Thread-safe: the turn identity lives in the ContextVar; per-run metadata is buffered
    under run_id (a lock guards the buffer, which parallel branches write concurrently).
    Silently skips when no TurnContext is set (e.g. unit tests or standalone calls).

    Run metadata (Strategy's _strategy tag, LangGraph's langgraph_node) is delivered to
    on_chat_model_start -- NOT to on_llm_end -- so we stash it at start, keyed by run_id,
    and consume it at end. langgraph_node is injected by LangGraph; _strategy by
    Strategy.chain() / .structured() / .tools() via .with_config().
    """

    raise_error = False  # never propagate errors from this handler

    def __init__(self) -> None:
        super().__init__()
        self._meta_by_run: dict[UUID, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _stash(self, run_id: UUID, metadata: dict[str, Any] | None) -> None:
        with self._lock:
            self._meta_by_run[run_id] = metadata or {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: Any,
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._stash(run_id, metadata)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: Any,
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Non-chat models fire on_llm_start instead of on_chat_model_start.
        self._stash(run_id, metadata)

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        # Drop the buffered metadata for a failed run so the buffer can't grow unbounded.
        with self._lock:
            self._meta_by_run.pop(run_id, None)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            meta = self._meta_by_run.pop(run_id, {})

        ctx = _ctx.get()
        if ctx is None:
            return

        try:
            gen = response.generations[0][0]
            ai_msg = getattr(gen, "message", None)
            if ai_msg is None:
                return
            usage: dict[str, Any] = getattr(ai_msg, "usage_metadata", None) or {}
            inp = int(usage.get("input_tokens", 0))
            out = int(usage.get("output_tokens", 0))
            if inp == 0 and out == 0:
                return
            model_id: str = (getattr(ai_msg, "response_metadata", None) or {}).get(
                "model"
            ) or "unknown"
        except (IndexError, AttributeError, TypeError):
            return

        _queue.put(
            {
                "id": uuid4(),
                "family_id": UUID(ctx.family_id),
                "turn_id": ctx.turn_id,
                "thread_id": ctx.thread_id,
                "family_member_id": UUID(ctx.family_member_id)
                if ctx.family_member_id
                else None,
                "model_id": model_id,
                # Strategy injects _strategy; node name comes from LangGraph run metadata.
                "strategy": meta.get("_strategy", ""),
                "node": _node_from_meta(meta),
                "input_tokens": inp,
                "output_tokens": out,
            }
        )
