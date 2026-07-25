"""Unit tests for the plan node: the deterministic intent -> capability resolver.

Pure functions -- no LLM, no DB. Capabilities are independent (none consumes another's output),
so `plan` emits a FLAT, unordered list of capabilities with no dependency edges; Execute fans
them out in parallel. `ambient_satisfied` is the shared precondition check the clarify node
reuses, so it is pinned here too.
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


# --- plan: flat, unordered capability list (no dependency edges) -------------------------


def test_plan_single_goal_is_one_step() -> None:
    steps = _steps(_understanding(["book_recommendation"]))
    assert len(steps) == 1
    assert steps[0]["capability"] == "recommend"
    # Capabilities are independent: a step carries no dependency channel.
    assert "depends_on" not in steps[0]


def test_plan_multiple_goals_are_a_flat_list_in_goal_order() -> None:
    # Both intents become independent steps; no producer->consumer edge is wired between them.
    steps = _steps(_understanding(["book_recommendation", "reading_discussion"]))
    assert [s["capability"] for s in steps] == ["recommend", "discussion"]
    assert all("depends_on" not in s for s in steps)


def test_plan_does_not_pull_in_an_unrequested_producer() -> None:
    # discussion alone stays a single step -- recommend is NOT added to "produce" its books.
    steps = _steps(_understanding(["reading_discussion"]))
    assert [s["capability"] for s in steps] == ["discussion"]


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
