"""LLM models and three-level fallback strategies.

Three strategies map to three Claude capability tiers:
  HEAVY    – quality-first   (recommend, evaluate, compare):  opus → sonnet → haiku
  STANDARD – balanced        (understand, respond, memory,
                               discussion, path, content):    sonnet → haiku → opus
  FAST     – speed-first     (clarify, memory-policy):        haiku → sonnet → opus

Each strategy exposes four methods:
  .chain()            – plain invoke chain (for single-shot LLM calls)
  .stream_chain()     – plain chain on streaming-enabled models (token-by-token .stream())
  .structured(schema) – per-model with_structured_output() chain with fallbacks
  .tools(tools)       – per-model bind_tools() chain with fallbacks

Every chain returned by these methods automatically carries:
  - UsageCallbackHandler  (fires on_llm_end → enqueues a token-usage record)
  - metadata["_strategy"] (so the callback knows which strategy tier ran)

LangGraph adds metadata["langgraph_node"] at runtime, giving the callback the
node name without any per-node instrumentation.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .usage_tracker import UsageCallbackHandler

# ── Individual models ─────────────────────────────────────────────────────────
# max_retries=2: the Anthropic SDK retries 429/5xx/connection errors with exponential
# backoff. Strategy fallbacks activate when a model exhausts its own retries.
# disable_streaming: avoids "No generations found in stream." on structured-output calls,
# so the plain/structured/tools models keep it on. The reply chain (stream_chain, used by
# `respond`) needs the opposite -- streaming-enabled models -- to emit tokens as they arrive;
# ChatAnthropic keeps stream_usage=True by default, so on_llm_end still sees usage_metadata on
# a streamed call and token accounting is unaffected.

_COMMON: dict[str, Any] = dict(temperature=0, max_retries=2, timeout=60)
_NOSTREAM: dict[str, Any] = {**_COMMON, "disable_streaming": True}

_opus = init_chat_model("claude-opus-4-6", **_NOSTREAM)
_sonnet = init_chat_model("claude-sonnet-4-6", **_NOSTREAM)
_haiku = init_chat_model("claude-haiku-4-5-20251001", **_NOSTREAM)

# Streaming-enabled twins, used only by stream_chain() for token-by-token replies.
_opus_stream = init_chat_model("claude-opus-4-6", **_COMMON)
_sonnet_stream = init_chat_model("claude-sonnet-4-6", **_COMMON)
_haiku_stream = init_chat_model("claude-haiku-4-5-20251001", **_COMMON)

_handler = UsageCallbackHandler()


class Strategy:
    """An ordered triple of Claude models: primary → secondary → tertiary.

    The chain activates the next model only when the previous one exhausts its retries
    (rate limit, server error, timeout). Within a single call, one model handles the request.

    Every chain returned by chain() / structured() / tools() is wrapped with
    .with_config(callbacks=[_handler], metadata={"_strategy": self.name}) so that
    UsageCallbackHandler fires transparently on every LLM call -- no per-node work needed.

    Usage per node:
        # plain invoke (run_text and similar)
        reply = MY_STRATEGY.chain().invoke([system, *messages])

        # streamed reply (respond): tokens arrive as they are generated
        for chunk in MY_STRATEGY.stream_chain().stream([system, *messages]):
            ...

        # structured output
        _chain = MY_STRATEGY.structured(MySchema)
        result = cast(MySchema, _chain.invoke([...]))

        # tool binding
        _bound = MY_STRATEGY.tools(TOOLS)
        response = _bound.invoke(messages)
    """

    def __init__(
        self,
        name: str,
        models: tuple[BaseChatModel, BaseChatModel, BaseChatModel],
        stream_models: tuple[BaseChatModel, BaseChatModel, BaseChatModel],
    ) -> None:
        self.name = name
        self._primary, self._secondary, self._tertiary = models
        self._invoke_chain: Any = self._primary.with_fallbacks(
            [self._secondary, self._tertiary]
        )
        # Streaming twins share the same primary→secondary→tertiary fallback order.
        self._stream_chain: Any = stream_models[0].with_fallbacks(
            list(stream_models[1:])
        )

    def _models(self) -> tuple[BaseChatModel, BaseChatModel, BaseChatModel]:
        return (self._primary, self._secondary, self._tertiary)

    def _with_tracking(self, chain: Any) -> Any:
        """Attach the usage handler and strategy tag to any chain."""
        return chain.with_config(
            callbacks=[_handler],
            metadata={"_strategy": self.name},
        )

    def chain(self) -> Any:
        """Plain invoke chain with usage tracking."""
        return self._with_tracking(self._invoke_chain)

    def stream_chain(self) -> Any:
        """Streaming plain chain with usage tracking (token-by-token `.stream()`)."""
        return self._with_tracking(self._stream_chain)

    def structured(self, schema: type) -> Any:
        """Structured-output chain with usage tracking."""
        chains = [m.with_structured_output(schema) for m in self._models()]
        return self._with_tracking(chains[0].with_fallbacks(chains[1:]))

    def tools(self, tools: Sequence[Any]) -> Any:
        """Tool-bound chain with usage tracking."""
        tool_list = list(tools)
        bound = [m.bind_tools(tool_list) for m in self._models()]
        return self._with_tracking(bound[0].with_fallbacks(bound[1:]))


# ── Strategies ────────────────────────────────────────────────────────────────

HEAVY = Strategy(
    "HEAVY",
    (_opus, _sonnet, _haiku),
    (_opus_stream, _sonnet_stream, _haiku_stream),
)  # recommend, evaluate, compare
STANDARD = Strategy(
    "STANDARD",
    (_sonnet, _haiku, _opus),
    (_sonnet_stream, _haiku_stream, _opus_stream),
)  # understand, respond, memory, discussion, path, content
FAST = Strategy(
    "FAST",
    (_haiku, _sonnet, _opus),
    (_haiku_stream, _sonnet_stream, _opus_stream),
)  # clarify, memory-policy
