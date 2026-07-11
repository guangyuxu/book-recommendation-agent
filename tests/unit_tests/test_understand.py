"""Unit tests for the deterministic child-resolution helpers in the understand node.

Pure functions -- no LLM, no DB. They own the "which child / did we switch" policy that
resolve_child feeds from the LLM's message-evidence (child_ref).
"""

from __future__ import annotations

from agent.pipeline.understand import resolve_child, switch_signal

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
    assert resolve_child({"status": "ambiguous"}, CHILDREN, "zzz") == (
        None,
        False,
        True,
    )


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


# --- Prompt-injection resilience --------------------------------------------------------
# resolve_child is the post-LLM gatekeeping layer: even if the LLM returns an injected or
# hallucinated child_id, it must be rejected when not present on the family roster.


def test_injected_child_id_not_on_roster_is_rejected() -> None:
    """An LLM-injected child_id that looks like an attack string must not bypass the roster check."""
    injected = "ignore-all-instructions-child-x"
    result = resolve_child({"status": "matched", "child_id": injected}, CHILDREN, "a")
    # Falls back to the pinned child, not the injected value.
    assert result == ("a", False, False)


def test_injected_child_id_looks_like_uuid_but_not_on_roster() -> None:
    """A plausible-looking UUID that is not on the roster must also be rejected."""
    fake_uuid = "00000000-0000-0000-0000-000000000099"
    result = resolve_child({"status": "matched", "child_id": fake_uuid}, CHILDREN, "a")
    assert result == ("a", False, False)


def test_injection_cannot_create_new_child_via_matched_status() -> None:
    """status=matched with an off-roster id must not promote child_is_new=True."""
    _, child_is_new, _ = resolve_child(
        {"status": "matched", "child_id": "evil"}, CHILDREN, "a"
    )
    assert child_is_new is False


def test_empty_roster_injection_triggers_ask_not_accept() -> None:
    """With no children registered, an injected match must ask the user, not accept silently."""
    target, child_is_new, needs_clarification = resolve_child(
        {"status": "matched", "child_id": "evil"}, {}, None
    )
    # No pin to fall back to and no valid roster match -> must ask for clarification.
    assert target is None
    assert child_is_new is False
