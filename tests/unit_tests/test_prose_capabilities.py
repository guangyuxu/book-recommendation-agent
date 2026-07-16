"""Unit tests for the run-based prose capability nodes and their shared LLM engine.

`compare`, `discussion`, `path`, and `content` are the execute fan-out's run-backed capability
nodes: each is a thin wrapper that calls the shared `run_text` engine and returns its prose under
one produced key. We pin each node's CONTRACT offline (no LLM): the exact produced key that
`respond._prose` reads, and the reasoning strategy it selects. `run_text` itself -- the real LLM
boundary all four share -- is tested against a fake strategy so the prompt assembly and
strategy-selection logic are covered without an API call.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain.messages import HumanMessage

from agent import prompts
from agent.capabilities import _shared, compare, content, discussion, path
from agent.llm import HEAVY, STANDARD

# --- run() contracts: produced key + strategy + delegation -------------------------------
# (capability module, expected produced key, expected strategy)
_CASES = [
    (compare, "comparison", HEAVY),
    (discussion, "questions", STANDARD),
    (path, "reading_path", STANDARD),
    (content, "draft", STANDARD),
]


@pytest.mark.parametrize(
    "module,key,strategy",
    _CASES,
    ids=[m.__name__.rsplit(".", 1)[-1] for m, _, _ in _CASES],
)
def test_run_capability_contract(
    monkeypatch: pytest.MonkeyPatch, module: Any, key: str, strategy: Any
) -> None:
    """Each capability returns {its_key: <run_text output>} with the right strategy."""
    captured: dict[str, Any] = {}

    def fake_run_text(
        state: dict[str, Any], prompt_id: str, *, strategy: Any = None
    ) -> str:
        captured["state"] = state
        captured["prompt_id"] = prompt_id
        captured["strategy"] = strategy
        return "PROSE"

    # Each capability calls run_text via its own module-level import.
    monkeypatch.setattr(module, "run_text", fake_run_text)

    state = {"messages": [HumanMessage("do the thing")]}
    out = module.run(state)

    # 1) exactly the produced key respond._prose / the registry expect, carrying the engine output
    assert out == {key: "PROSE"}
    # 2) the declared reasoning strategy (HEAVY for deep analysis, STANDARD otherwise)
    assert captured["strategy"] is strategy
    # 3) delegated the turn's state and a registry prompt id that resolves to a real prompt
    assert captured["state"] is state
    assert prompts.version(captured["prompt_id"]) >= 1


def test_produced_keys_are_distinct() -> None:
    """The four prose capabilities must not collide on one produced key (respond reads by key)."""
    keys = [key for _, key, _ in _CASES]
    assert len(set(keys)) == len(keys)


# --- run_text: the shared LLM engine -----------------------------------------------------


class _FakeChain:
    def __init__(self, sink: dict[str, Any]) -> None:
        self._sink = sink

    def invoke(self, messages: list[Any], **_kwargs: Any) -> Any:
        self._sink["messages"] = messages
        return SimpleNamespace(content="  the reply  ")


class _FakeStrategy:
    """Stand-in Strategy: records that chain() was built and returns a scripted reply."""

    def __init__(self) -> None:
        self.sink: dict[str, Any] = {}
        self.chain_calls = 0

    def chain(self) -> _FakeChain:
        self.chain_calls += 1
        return _FakeChain(self.sink)


def _state() -> dict[str, Any]:
    return {
        "messages": [HumanMessage("please help")],
        "target_child_id": "c1",
        "children": {"c1": {"display_name": "Alex", "age": 6}},
        "policies": [{"goals": ["build curiosity"]}],
    }


def test_run_text_builds_prompt_and_returns_reply_text() -> None:
    strategy = _FakeStrategy()
    # run_text now renders a registry prompt by id; the template folds in the child/policy briefs.
    out = _shared.run_text(_state(), "compare.analyze", strategy=strategy)

    assert out == "  the reply  "  # returns the reply content verbatim (str())
    assert strategy.chain_calls == 1
    system, *rest = strategy.sink["messages"]
    # The rendered system prompt carries the child/policy briefs (passed as serialized vars)...
    assert "Alex" in system.content  # child_brief folded in
    assert "build curiosity" in system.content  # policies_brief folded in
    # ...and the conversation messages are appended after it.
    assert rest and getattr(rest[-1], "type", None) == "human"


def test_run_text_defaults_to_standard_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no strategy passed, run_text uses the module STANDARD strategy."""
    calls: dict[str, int] = {"n": 0}

    def fake_chain() -> _FakeChain:
        calls["n"] += 1
        return _FakeChain({})

    monkeypatch.setattr(STANDARD, "chain", fake_chain)
    out = _shared.run_text(_state(), "path.plan")  # no strategy -> STANDARD
    assert out == "  the reply  "
    assert calls["n"] == 1
