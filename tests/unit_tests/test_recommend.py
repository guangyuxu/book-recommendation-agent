"""Behavioral tests for the recommend generate/validate subgraph.

These run offline (no DB, no Anthropic): the two structured LLM chains (`_generate`, `_screen`)
are monkeypatched with scripted fakes, so we exercise the loop -- self-critique screening,
regenerate-on-full-rejection, the attempt cap, partial keeps, and the post-LLM gate -- purely
on the graph's control flow.
"""

from __future__ import annotations

from typing import Any

from langchain.messages import HumanMessage

from agent.capabilities import recommend
from agent.capabilities.recommend import (
    Booklist,
    BookVerdict,
    RecommendedBook,
    Screening,
)
from agent.pipeline import execute as execute_mod


class _FakeChain:
    """A structured-output stand-in: returns a scripted response per invoke() call.

    Once the script is exhausted it repeats the last response, so an unbounded loop (should the
    routing regress) still terminates deterministically rather than raising.
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        self.calls = 0

    def invoke(self, _messages: Any) -> Any:
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response


def _book(title: str) -> RecommendedBook:
    return RecommendedBook(title=title, recommendation_reason=f"why {title}")


def _booklist(*titles: str) -> Booklist:
    return Booklist(books=[_book(t) for t in titles], note="a note")


def _screening(verdicts: dict[str, bool]) -> Screening:
    return Screening(
        verdicts=[
            BookVerdict(title=t, keep=keep, reason=f"reason for {t}")
            for t, keep in verdicts.items()
        ]
    )


def _state() -> dict[str, Any]:
    return {
        "messages": [HumanMessage("recommend something for my kid")],
        "children": {},
        "target_child_id": None,
        "policies": [],
    }


def _patch(monkeypatch, generate: _FakeChain, screen: _FakeChain) -> None:
    monkeypatch.setattr(recommend, "_generate", generate)
    monkeypatch.setattr(recommend, "_screen", screen)


def _run(state: dict[str, Any]) -> dict[str, Any]:
    """Drive the subgraph out-of-graph (as the eval does) and read its booklist output."""
    final = recommend.recommend_graph.invoke(state)
    return {"books": final.get("books") or [], "note": final.get("note")}


def test_subgraph_shape() -> None:
    nodes = set(recommend.recommend_graph.get_graph().nodes)
    edges = {(e.source, e.target) for e in recommend.recommend_graph.get_graph().edges}
    assert {"generate", "validate", "emit"} <= nodes
    assert ("__start__", "generate") in edges
    assert ("generate", "validate") in edges
    assert ("validate", "generate") in edges  # retry loop
    assert ("validate", "emit") in edges  # keep -> emit into the execute fan-in
    assert ("emit", "__end__") in edges


def test_screening_drops_unfit_books(monkeypatch) -> None:
    # The post-LLM gate: a book the generator proposed is dropped when screening rejects it.
    generate = _FakeChain([_booklist("Fit Book", "Unfit Book")])
    screen = _FakeChain([_screening({"Fit Book": True, "Unfit Book": False})])
    _patch(monkeypatch, generate, screen)

    out = _run(_state())

    titles = [b["title"] for b in out["books"]]
    assert titles == ["Fit Book"]  # the rejected book never reaches the parent
    assert generate.calls == 1  # a fit book survived, so no regenerate


def test_regenerates_when_all_rejected_then_succeeds(monkeypatch) -> None:
    generate = _FakeChain([_booklist("A", "B"), _booklist("C", "D")])
    screen = _FakeChain(
        [
            _screening({"A": False, "B": False}),  # round 1: reject everything
            _screening({"C": True, "D": True}),  # round 2: both fit
        ]
    )
    _patch(monkeypatch, generate, screen)

    out = _run(_state())

    assert [b["title"] for b in out["books"]] == ["C", "D"]
    assert generate.calls == 2  # regenerated once after the full rejection
    assert screen.calls == 2


def test_stops_after_max_attempts(monkeypatch) -> None:
    # Screening rejects every book on every round -> generate runs at most MAX_ATTEMPTS times.
    generate = _FakeChain([_booklist("X", "Y")])
    screen = _FakeChain([_screening({"X": False, "Y": False})])
    _patch(monkeypatch, generate, screen)

    out = _run(_state())

    assert out["books"] == []  # nothing fit survived
    assert generate.calls == recommend.MAX_ATTEMPTS == 3


def test_unmatched_verdict_keeps_conservatively(monkeypatch) -> None:
    # A verdict is returned only for "A" (reject); "B" has no verdict (LLM omission) and is kept.
    generate = _FakeChain([_booklist("A", "B")])
    screen = _FakeChain([_screening({"A": False})])
    _patch(monkeypatch, generate, screen)

    out = _run(_state())

    assert [b["title"] for b in out["books"]] == ["B"]
    assert generate.calls == 1


def test_empty_generation_regenerates_up_to_cap(monkeypatch) -> None:
    # A generator that returns nothing is treated as a full rejection and regenerated, capped.
    generate = _FakeChain([Booklist(books=[], note=None)])
    screen = _FakeChain([_screening({})])
    _patch(monkeypatch, generate, screen)

    out = _run(_state())

    assert out["books"] == []
    assert generate.calls == recommend.MAX_ATTEMPTS


def test_recommend_runs_as_subgraph_node_in_execute(monkeypatch) -> None:
    # The execute fan-out drives the recommend subgraph directly (not via _capability_node):
    # its `emit` node appends {"recommend": booklist} to the same `results` channel, so
    # aggregate merges it into capability_results with no special-casing.
    generate = _FakeChain([_booklist("Fit Book", "Unfit Book")])
    screen = _FakeChain([_screening({"Fit Book": True, "Unfit Book": False})])
    _patch(monkeypatch, generate, screen)

    out = execute_mod.execute_graph.invoke(
        {"plan": {"steps": [{"capability": "recommend"}]}}
    )

    rec = out["capability_results"]["recommend"]
    assert [b["title"] for b in rec["books"]] == [
        "Fit Book"
    ]  # screening applied end-to-end
