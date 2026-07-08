"""Unit tests for the deterministic child-resolution helpers in the understand node.

Pure functions -- no LLM, no DB. They own the "which child / did we switch" policy that
resolve_child feeds from the LLM's message-evidence (child_ref).
"""

from __future__ import annotations

from agent.nodes.understand import resolve_child, switch_signal

CHILDREN = {
    "a": {"display_name": "Son"},
    "b": {"display_name": "小儿子"},
}


# --- resolve_child ----------------------------------------------------------------------


def test_matched_reference_wins_over_pin() -> None:
    # Message clearly points at b while a is pinned -> switch to b.
    assert resolve_child({"status": "matched", "child_id": "b"}, CHILDREN, "a") == (
        "b",
        False,
        False,
    )


def test_hallucinated_matched_id_falls_back_to_pin() -> None:
    # status matched but the id is not on the roster -> keep the pinned child.
    assert resolve_child({"status": "matched", "child_id": "zzz"}, CHILDREN, "a") == (
        "a",
        False,
        False,
    )


def test_new_child_is_not_pinned() -> None:
    assert resolve_child({"status": "new"}, CHILDREN, "a") == (None, True, False)


def test_ambiguous_falls_back_to_pin() -> None:
    # A bare child reference resolves to the active child rather than interrupting.
    assert resolve_child({"status": "ambiguous"}, CHILDREN, "a") == ("a", False, False)


def test_ambiguous_asks_when_nothing_pinned() -> None:
    # No pin to fall back to -> we must ask which child.
    assert resolve_child({"status": "ambiguous"}, CHILDREN, None) == (None, False, True)


def test_ambiguous_asks_when_pin_not_on_roster() -> None:
    # A stale/invalid pin can't disambiguate -> ask.
    assert resolve_child({"status": "ambiguous"}, CHILDREN, "zzz") == (None, False, True)


def test_none_reference_keeps_pin() -> None:
    assert resolve_child({"status": "none"}, CHILDREN, "a") == ("a", False, False)


# --- switch_signal ----------------------------------------------------------------------


def test_switch_signal_on_confident_move() -> None:
    assert switch_signal("matched", "b", "a", CHILDREN) == {
        "from": "a",
        "to": "b",
        "from_name": "Son",
        "to_name": "小儿子",
    }


def test_no_signal_when_same_child() -> None:
    assert switch_signal("matched", "a", "a", CHILDREN) == {}


def test_no_signal_for_non_matched_reference() -> None:
    assert switch_signal("ambiguous", None, "a", CHILDREN) == {}
    assert switch_signal("new", None, "a", CHILDREN) == {}
    assert switch_signal("none", "a", "a", CHILDREN) == {}


def test_first_time_selection_reports_from_none() -> None:
    # Nothing pinned yet, message names b -> a switch the frontend still needs, with from=None.
    assert switch_signal("matched", "b", None, CHILDREN) == {
        "from": None,
        "to": "b",
        "from_name": None,
        "to_name": "小儿子",
    }
