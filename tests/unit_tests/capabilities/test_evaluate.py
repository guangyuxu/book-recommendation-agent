"""Behavioral tests for the evaluate prepare/evaluate/validate/emit subgraph.

These run offline (no DB, no Anthropic): the analyst chain (`_analyst`) and the critic chain
(`_critic`) are monkeypatched with scripted fakes, so we exercise the loop -- prepare's input
assembly, the self-critique gate, the revise-on-gaps retry, the attempt cap, and the post-LLM
emit -- purely on the graph's control flow.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain.messages import HumanMessage

from agent.capabilities import evaluate
from agent.capabilities.evaluate import Critique
from agent.pipeline import execute as execute_mod


class _FakeChain:
    """A stand-in chain: returns a scripted response per invoke() call.

    Once the script is exhausted it repeats the last response, so a routing regression still
    terminates deterministically rather than raising.
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        self.calls = 0

    def invoke(self, _messages: Any, **_kwargs: Any) -> Any:
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response


def _analysis(text: str) -> SimpleNamespace:
    """Mimic a chat reply object (only `.content` is read)."""
    return SimpleNamespace(content=text)


def _state() -> dict[str, Any]:
    return {
        "messages": [HumanMessage("is 'The Hobbit' right for my kid?")],
        "children": {},
        "target_child_id": None,
        "policies": [],
        "understanding": {"mentioned_books": [{"title": "The Hobbit"}]},
    }


def _patch(monkeypatch, analyst: _FakeChain, critic: _FakeChain) -> None:
    monkeypatch.setattr(evaluate, "_analyst", analyst)
    monkeypatch.setattr(evaluate, "_critic", critic)


def _run(state: dict[str, Any]) -> dict[str, Any]:
    """Drive the subgraph out-of-graph and read its evaluation output."""
    final = evaluate.evaluate_subgraph.invoke(state)
    return {"evaluation": final.get("evaluation") or ""}


def test_subgraph_shape() -> None:
    nodes = set(evaluate.evaluate_subgraph.get_graph().nodes)
    edges = {(e.source, e.target) for e in evaluate.evaluate_subgraph.get_graph().edges}
    assert {"prepare", "evaluate", "validate", "emit"} <= nodes
    assert ("__start__", "prepare") in edges
    assert ("prepare", "evaluate") in edges
    assert ("evaluate", "validate") in edges
    assert ("validate", "evaluate") in edges  # revise loop
    assert ("validate", "emit") in edges  # accept -> emit into the execute fan-in
    assert ("emit", "__end__") in edges


def test_accepts_a_passing_evaluation_without_revising(monkeypatch) -> None:
    analyst = _FakeChain([_analysis("A balanced, concrete assessment.")])
    critic = _FakeChain([Critique(ok=True)])
    _patch(monkeypatch, analyst, critic)

    out = _run(_state())

    assert out["evaluation"] == "A balanced, concrete assessment."
    assert analyst.calls == 1  # review passed, so no revise


def test_revises_on_gaps_then_accepts(monkeypatch) -> None:
    analyst = _FakeChain(
        [_analysis("thin first pass"), _analysis("thorough second pass")]
    )
    critic = _FakeChain(
        [
            Critique(ok=False, issues=["no cautions", "ignores reading level"]),
            Critique(ok=True),
        ]
    )
    _patch(monkeypatch, analyst, critic)

    out = _run(_state())

    assert out["evaluation"] == "thorough second pass"
    assert analyst.calls == 2  # revised once after the reviewer flagged gaps
    assert critic.calls == 2


def test_stops_after_max_attempts(monkeypatch) -> None:
    # The reviewer never passes -> the analyst runs at most MAX_ATTEMPTS times, ships best effort.
    analyst = _FakeChain([_analysis("still not great")])
    critic = _FakeChain([Critique(ok=False, issues=["still one-sided"])])
    _patch(monkeypatch, analyst, critic)

    out = _run(_state())

    assert out["evaluation"] == "still not great"
    assert analyst.calls == evaluate.MAX_ATTEMPTS == 3


def test_empty_evaluation_revises_up_to_cap(monkeypatch) -> None:
    # An empty analyst reply is treated as a gap (validate short-circuits) and revised, capped.
    analyst = _FakeChain([_analysis("")])
    critic = _FakeChain([Critique(ok=True)])  # never consulted -- text is empty
    _patch(monkeypatch, analyst, critic)

    out = _run(_state())

    assert out["evaluation"] == ""
    assert analyst.calls == evaluate.MAX_ATTEMPTS
    assert critic.calls == 0  # validate never calls the critic on empty text


def test_evaluate_runs_as_subgraph_node_in_execute(monkeypatch) -> None:
    # The execute fan-out drives the evaluate subgraph directly: its `emit` node appends
    # {"evaluate": {"evaluation": ...}} to the same `results` channel, so aggregate merges it
    # into capability_results with no special-casing.
    analyst = _FakeChain([_analysis("fits well, with minor cautions")])
    critic = _FakeChain([Critique(ok=True)])
    _patch(monkeypatch, analyst, critic)

    out = execute_mod.execute_graph.invoke(
        {
            "plan": {"steps": [{"capability": "evaluate"}]},
            "understanding": {"mentioned_books": [{"title": "The Hobbit"}]},
        }
    )

    assert out["capability_results"]["evaluate"] == {
        "evaluation": "fits well, with minor cautions"
    }
