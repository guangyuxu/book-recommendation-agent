"""Unit tests for the plan node: the deterministic, dependency-aware DAG resolver.

Pure functions -- no LLM, no DB. They turn an understanding's intents into an ordered
capability plan, wiring producer->consumer edges only when BOTH goals are present, and
topologically sorting the result. `ambient_satisfied` is the shared precondition check the
clarify node reuses, so it is pinned here too.
"""

from __future__ import annotations

from agent.pipeline.plan import _goals, ambient_satisfied, plan


def _steps(state: dict) -> list[dict]:
    return plan(state)["plan"]["steps"]


def _understanding(intents: list[str]) -> dict:
    return {"understanding": {"intents": intents}}


# --- _goals: intent -> capability mapping ------------------------------------------------


def test_goals_maps_task_intents_to_capabilities() -> None:
    assert _goals(["book_recommendation"]) == ["recommend"]


def test_goals_dedups_and_preserves_order() -> None:
    # Repeated intents collapse to one goal; first-seen order is kept.
    assert _goals(["book_evaluation", "book_recommendation", "book_evaluation"]) == [
        "evaluate",
        "recommend",
    ]


def test_goals_drops_profile_and_clarify_intents() -> None:
    # These intents map to no capability (their work is pure memory/persistence).
    assert _goals(["child_profile_update", "parent_goal_update", "clarify"]) == []


def test_goals_ignores_empty_values() -> None:
    assert _goals(["", "book_recommendation", ""]) == ["recommend"]


# --- plan: empty plans -------------------------------------------------------------------


def test_plan_empty_when_no_intents() -> None:
    assert _steps({"understanding": {}}) == []
    assert _steps({}) == []


def test_plan_empty_for_profile_update_only_turn() -> None:
    assert _steps(_understanding(["child_profile_update"])) == []


# --- plan: single goal, no edges ---------------------------------------------------------


def test_plan_single_goal_has_no_dependencies() -> None:
    steps = _steps(_understanding(["book_recommendation"]))
    assert len(steps) == 1
    assert steps[0]["capability"] == "recommend"
    assert steps[0]["depends_on"] == []


# --- plan: producer -> consumer edges ----------------------------------------------------


def test_recommend_feeds_discussion() -> None:
    # discussion requires `books`; recommend produces `books` -> recommend depends before discuss.
    steps = _steps(_understanding(["reading_discussion", "book_recommendation"]))
    by_cap = {s["capability"]: s["depends_on"] for s in steps}
    assert by_cap["discussion"] == ["recommend"]
    assert by_cap["recommend"] == []
    # Topological order: the producer comes first.
    assert [s["capability"] for s in steps] == ["recommend", "discussion"]


def test_recommend_feeds_evaluate() -> None:
    # evaluate requires `books`; recommend produces them.
    steps = _steps(_understanding(["book_evaluation", "book_recommendation"]))
    by_cap = {s["capability"]: s["depends_on"] for s in steps}
    assert by_cap["evaluate"] == ["recommend"]
    assert [s["capability"] for s in steps] == ["recommend", "evaluate"]


def test_no_edge_when_only_the_consumer_is_present() -> None:
    # A producer is never pulled in to satisfy an input -- discussion alone has no `recommend`
    # producer this turn, so it carries no dependency (books come from an ambient/named source).
    steps = _steps(_understanding(["reading_discussion"]))
    assert len(steps) == 1
    assert steps[0]["capability"] == "discussion"
    assert steps[0]["depends_on"] == []


# --- ambient_satisfied (shared with clarify) ---------------------------------------------


def test_ambient_unknown_resource_is_never_satisfied() -> None:
    assert ambient_satisfied("not_a_resource", {}) is False


def test_ambient_target_child_needs_a_resolved_or_new_child() -> None:
    assert ambient_satisfied("target_child", {"target_child_id": "c1"}) is True
    assert (
        ambient_satisfied("target_child", {"understanding": {"child_is_new": True}})
        is True
    )
    assert ambient_satisfied("target_child", {}) is False


def test_ambient_reading_profile_follows_child_availability() -> None:
    assert ambient_satisfied("reading_profile", {"target_child_id": "c1"}) is True
    assert ambient_satisfied("reading_profile", {}) is False


def test_ambient_policies_always_satisfied() -> None:
    # Policies may be empty; they are only ever an optional input.
    assert ambient_satisfied("policies", {}) is True


def test_ambient_books_requires_mentioned_books() -> None:
    assert (
        ambient_satisfied("books", {"understanding": {"mentioned_books": ["Frog"]}})
        is True
    )
    assert (
        ambient_satisfied("books", {"understanding": {"mentioned_books": []}}) is False
    )
    assert ambient_satisfied("books", {}) is False
