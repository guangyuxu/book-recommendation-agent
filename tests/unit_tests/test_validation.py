"""Unit tests for the output-validation subgraph (agent.validation).

Hermetic: the checks are PASS stubs (no LLM, no DB). Cover the code-controlled aggregation
("worst wins" -> rating), the capability -> check selection mapping, the compiled subgraph's
end-to-end verdict, and the per-turn reset of the parallel-check accumulator.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent.validation.checks import CHECK_REGISTRY, applicable_checks, node_name
from agent.validation.graph import _builder, validation_graph
from agent.validation.nodes import aggregate
from agent.validation.schemas import CheckOutcome, Rating

ALWAYS = {"child_safety", "privacy", "product_values"}


def _checks(*outcomes: str) -> list[dict]:
    return [{"check": f"c{i}", "outcome": o} for i, o in enumerate(outcomes)]


# --- aggregation: worst wins ------------------------------------------------------------


@pytest.mark.parametrize(
    "outcomes,expected",
    [
        ([], Rating.ALLOW),
        (["pass", "pass"], Rating.ALLOW),
        (["pass", "warn"], Rating.WARNING),
        (["pass", "warn", "rewrite"], Rating.REWRITE),
        (["rewrite", "block", "warn"], Rating.BLOCK),  # block beats rewrite
        (["block"], Rating.BLOCK),
    ],
)
def test_aggregate_takes_the_worst_outcome(outcomes: list[str], expected: Rating) -> None:
    out = aggregate({"output_checks": _checks(*outcomes)})
    assert out["validation"]["rating"] == expected.value
    assert len(out["validation"]["results"]) == len(outcomes)


def test_aggregate_ignores_malformed_entries() -> None:
    # A malformed accumulator entry must not crash aggregation; it is dropped.
    out = aggregate({"output_checks": [{"nonsense": True}, {"check": "x", "outcome": "warn"}]})
    assert out["validation"]["rating"] == Rating.WARNING.value
    assert len(out["validation"]["results"]) == 1


# --- selection: which checks apply to which capabilities --------------------------------


def _names(caps: set[str]) -> set[str]:
    return {c.name for c in applicable_checks(caps)}


def test_always_on_checks_run_regardless() -> None:
    assert ALWAYS <= _names(set())
    assert _names(set()) == ALWAYS  # nothing else applies with no capabilities


def test_recommend_selects_its_checks_not_discussion() -> None:
    names = _names({"recommend"})
    assert {"recommendation_policy", "factuality", "age_appropriateness"} <= names
    assert "discussion_policy" not in names


def test_discussion_selects_its_checks_not_recommendation() -> None:
    names = _names({"discussion"})
    assert "discussion_policy" in names
    assert "age_appropriateness" in names
    assert "recommendation_policy" not in names
    assert "factuality" not in names


def test_node_name_is_derived_from_registry_key() -> None:
    for key, check in CHECK_REGISTRY.items():
        assert node_name(check) == f"check_{key}"


# --- compiled subgraph: end-to-end verdict ----------------------------------------------


def test_subgraph_allows_when_all_checks_pass() -> None:
    out = validation_graph.invoke({"capability_results": {"recommend": {"books": []}}})
    v = out["validation"]
    assert v["rating"] == Rating.ALLOW.value
    assert {r["check"] for r in v["results"]} == _names({"recommend"})
    assert all(r["outcome"] == CheckOutcome.PASS.value for r in v["results"])


def test_subgraph_runs_only_always_on_checks_with_no_capabilities() -> None:
    out = validation_graph.invoke({"capability_results": {}})
    assert {r["check"] for r in out["validation"]["results"]} == ALWAYS


def test_subgraph_resets_accumulator_across_turns() -> None:
    """Two turns on one checkpointed thread must not accumulate check results."""
    graph = _builder.compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    state = {"capability_results": {"discussion": {"questions": "x"}}}

    first = graph.invoke(state, cfg)
    second = graph.invoke(state, cfg)

    assert len(second["validation"]["results"]) == len(first["validation"]["results"])
    # The transient accumulator was reset, not doubled.
    assert len(second["output_checks"]) == len(first["validation"]["results"])
