"""LLM strategy invariants: structured/tools/plain chains stay non-streaming (so
with_structured_output doesn't hit "No generations found in stream."), while the reply
chain (stream_chain, used by `respond`) is streaming-enabled for token-by-token delivery.
"""

from __future__ import annotations

import pytest

from agent.llm import FAST, HEAVY, STANDARD, Strategy


def _disable_streaming(model: object) -> bool:
    return bool(getattr(model, "disable_streaming", False))


@pytest.mark.parametrize("strategy", [HEAVY, STANDARD, FAST])
def test_invoke_models_are_non_streaming(strategy: Strategy) -> None:
    """The invoke/structured/tools models keep disable_streaming=True."""
    assert all(_disable_streaming(m) for m in strategy._models())


@pytest.mark.parametrize("strategy", [HEAVY, STANDARD, FAST])
def test_stream_chain_models_are_streaming_enabled(strategy: Strategy) -> None:
    """The stream_chain twins must NOT disable streaming, or tokens never flow."""
    stream_chain = strategy._stream_chain
    # RunnableWithFallbacks -> runnable (primary) + fallbacks (secondary, tertiary).
    models = [stream_chain.runnable, *stream_chain.fallbacks]
    assert models, "stream chain should expose its models"
    assert not any(_disable_streaming(m) for m in models)


def test_stream_chain_returns_a_runnable() -> None:
    chain = STANDARD.stream_chain()
    assert hasattr(chain, "stream")
    assert hasattr(chain, "invoke")
