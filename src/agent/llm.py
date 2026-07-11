"""LLM models and three-level fallback strategies.

Three strategies map to three Claude capability tiers:
  HEAVY    – quality-first   (recommend, evaluate, compare):  opus → sonnet → haiku
  STANDARD – balanced        (understand, respond, memory,
                               discussion, path, content):    sonnet → haiku → opus
  FAST     – speed-first     (clarify, memory-policy):        haiku → sonnet → opus

Each strategy exposes three methods:
  .chain()            – plain invoke chain (for single-shot LLM calls)
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
# disable_streaming: avoids "No generations found in stream." on structured-output calls.

_COMMON: dict[str, Any] = dict(
    temperature=0, max_retries=2, timeout=60, disable_streaming=True
)

_opus = init_chat_model("claude-opus-4-6", **_COMMON)
_sonnet = init_chat_model("claude-sonnet-4-6", **_COMMON)
_haiku = init_chat_model("claude-haiku-4-5-20251001", **_COMMON)

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
        primary: BaseChatModel,
        secondary: BaseChatModel,
        tertiary: BaseChatModel,
    ) -> None:
        self.name = name
        self._primary = primary
        self._secondary = secondary
        self._tertiary = tertiary
        self._invoke_chain: Any = primary.with_fallbacks([secondary, tertiary])

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

HEAVY = Strategy("HEAVY", _opus, _sonnet, _haiku)  # recommend, evaluate, compare
STANDARD = Strategy(
    "STANDARD", _sonnet, _haiku, _opus
)  # understand, respond, memory, discussion, path, content
FAST = Strategy("FAST", _haiku, _sonnet, _opus)  # clarify, memory-policy
