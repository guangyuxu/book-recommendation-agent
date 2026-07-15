"""Usage-tracking end-to-end: the billing ContextVar must be live when a node's LLM
callback fires, so a token-usage record is captured for every LLM call.

The regression these guard against: a ContextVar set inside one LangGraph node is NOT
visible in any other node (each node runs in its own copied context). Setting it once in
load_context therefore captured nothing -- `with_turn_context` must re-establish it at the
entry of every LLM-invoking node. These tests run a synthetic 2-node graph (mirroring
load_context -> LLM-node) through real LangGraph execution and assert the record lands.
"""

from __future__ import annotations

import queue
from typing import Any, Iterator, TypedDict
from uuid import UUID

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph

from agent import usage_tracker
from agent.state import AppContext
from agent.usage_tracker import (
    UsageCallbackHandler,
    _emit_usage_event,
    _node_from_meta,
    with_turn_context,
)

FAMILY_ID = "16555532-69b5-411e-8526-e0b321fbcfea"
MEMBER_ID = "659c1323-f47a-40eb-a0fe-5fb83f47c9c9"


class _FakeUsageModel(BaseChatModel):
    """Minimal chat model returning an AIMessage with usage_metadata, so the real
    UsageCallbackHandler.on_llm_end path runs exactly as it does against Anthropic."""

    @property
    def _llm_type(self) -> str:
        return "fake-usage"

    def _generate(
        self,
        messages: list[Any],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            response_metadata={"model": "claude-test-1"},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def isolated_queue() -> Iterator[queue.SimpleQueue]:
    """Swap the module queue for a fresh one the daemon consumer can't drain.

    The consumer thread is blocked in `.get()` on the ORIGINAL queue at swap time, so every
    record enqueued during the test lands in ours and we read it deterministically -- no DB,
    no race. Restored afterward.
    """
    fresh: queue.SimpleQueue[Any] = queue.SimpleQueue()
    original = usage_tracker._queue
    usage_tracker._queue = fresh
    try:
        yield fresh
    finally:
        usage_tracker._queue = original


def _drain(q: queue.SimpleQueue) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def _llm_node(state: dict[str, Any]) -> dict[str, Any]:
    """A node that invokes an LLM, exactly like understand/respond do (config/callbacks
    propagate from the node into the nested model.invoke via LangChain contextvars)."""
    model = _FakeUsageModel().with_config(
        callbacks=[UsageCallbackHandler()], metadata={"_strategy": "HEAVY"}
    )
    model.invoke([HumanMessage(content="hi")])
    return {}


def _context_node(state: dict[str, Any]) -> dict[str, Any]:
    """Mirrors load_context: places turn_id/thread_id in state (makes no LLM call)."""
    return {"turn_id": "turn-abc", "thread_id": "thread-xyz"}


class _GS(TypedDict, total=False):
    turn_id: str
    thread_id: str


def _build_graph(*, wrap: bool):
    b = StateGraph(_GS, context_schema=AppContext)
    b.add_node("ctx", _context_node)
    b.add_node("llm", with_turn_context(_llm_node) if wrap else _llm_node)
    b.add_edge(START, "ctx")
    b.add_edge("ctx", "llm")
    b.add_edge("llm", END)
    return b.compile()


def _invoke(graph) -> None:
    graph.invoke(
        {},
        context=AppContext(
            family_id=FAMILY_ID, family_member_id=MEMBER_ID, child_id=None
        ),
    )


def test_wrapped_llm_node_captures_usage_record(
    isolated_queue: queue.SimpleQueue,
) -> None:
    """A with_turn_context-wrapped LLM node must enqueue a fully-attributed usage record."""
    _invoke(_build_graph(wrap=True))
    records = _drain(isolated_queue)

    assert len(records) == 1, "exactly one usage record should be enqueued"
    rec = records[0]
    assert rec["family_id"] == UUID(FAMILY_ID)
    assert rec["family_member_id"] == UUID(MEMBER_ID)
    assert rec["turn_id"] == "turn-abc"
    assert rec["thread_id"] == "thread-xyz"
    assert rec["input_tokens"] == 11
    assert rec["output_tokens"] == 7
    assert rec["strategy"] == "HEAVY"
    # Node name derived from LangGraph run metadata (checkpoint_ns / langgraph_node).
    assert rec["node"] == "llm"
    assert rec["model_id"] == "claude-test-1"


def test_unwrapped_llm_node_captures_nothing(isolated_queue: queue.SimpleQueue) -> None:
    """Without with_turn_context the ContextVar is unset in the node, so nothing is recorded.

    This is the exact failure the fix repairs: a ContextVar set in load_context does not reach
    another node, so an un-wrapped LLM node silently drops its usage.
    """
    _invoke(_build_graph(wrap=False))
    assert _drain(isolated_queue) == [], (
        "unwrapped node must not capture usage (ContextVar unset)"
    )


# --- node-name extraction ---------------------------------------------------------------
# LangGraph exposes the node either as langgraph_node or encoded in checkpoint_ns. The memory
# subgraph produces a nested namespace ("memory:<id>|profile_update:<id>"); the call belongs to
# the inner node.


def test_node_from_meta_prefers_explicit_langgraph_node() -> None:
    assert (
        _node_from_meta({"langgraph_node": "understand", "checkpoint_ns": "x:1"})
        == "understand"
    )


def test_node_from_meta_parses_checkpoint_ns() -> None:
    assert _node_from_meta({"checkpoint_ns": "respond:63380455-abc"}) == "respond"


def test_node_from_meta_parses_nested_subgraph_ns() -> None:
    """A subgraph LLM call must attribute to the inner node, not the parent "memory" node."""
    ns = "memory:0a1b|profile_update:9f8e"
    assert _node_from_meta({"checkpoint_ns": ns}) == "profile_update"


def test_node_from_meta_empty_when_unknown() -> None:
    assert _node_from_meta({}) == ""


# --- live {node, tokens} custom stream events -------------------------------------------
# on_llm_end also emits a per-node usage event over stream_mode="custom" (the live half of the
# usage view). It is best-effort: safe outside a run, and a no-op inside a non-streaming run.


def test_streaming_run_emits_node_tokens_custom_event(
    isolated_queue: queue.SimpleQueue,
) -> None:
    """A streamed graph run surfaces one {node, tokens} custom event per LLM call."""
    graph = _build_graph(wrap=True)
    events = list(
        graph.stream(
            {},
            context=AppContext(
                family_id=FAMILY_ID, family_member_id=MEMBER_ID, child_id=None
            ),
            stream_mode="custom",
        )
    )
    # 11 input + 7 output tokens (see _FakeUsageModel), attributed to the "llm" node.
    assert {"node": "llm", "tokens": 18} in events


def test_non_streaming_invoke_still_records_usage(
    isolated_queue: queue.SimpleQueue,
) -> None:
    """Outside a stream the emit is a no-op (LangGraph's no-op writer) and must not disturb the
    DB usage path: the record is still enqueued exactly as before."""
    _invoke(_build_graph(wrap=True))
    records = _drain(isolated_queue)
    assert len(records) == 1
    assert records[0]["input_tokens"] == 11
    assert records[0]["output_tokens"] == 7


def test_emit_usage_event_safe_outside_run() -> None:
    """Called with no active graph run, get_stream_writer() raises; the emit swallows it."""
    _emit_usage_event("respond", 42)  # must not raise


def test_emit_usage_event_skips_empty_node_or_zero_tokens() -> None:
    # No node name or no tokens => nothing to emit; must be a harmless no-op.
    _emit_usage_event("", 42)
    _emit_usage_event("respond", 0)
