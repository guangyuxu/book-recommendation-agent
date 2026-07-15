"""Unit tests for the load_context entry node.

Two things load_context owns deterministically: pinning the turn's target child (a pure policy,
extracted as `_pin_target_child`) and rejecting a request that carries no context. The DB read
itself is exercised by the integration suite; here we stay hermetic (no DB, no runtime) and pin
the decisions.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import lifecycle
from agent.lifecycle import MissingContextError, _pin_target_child, load_context

CHILDREN = {"a": {"display_name": "Son"}, "b": {"display_name": "Daughter"}}


# --- _pin_target_child -------------------------------------------------------------------


def test_explicit_in_family_child_wins() -> None:
    assert _pin_target_child("b", CHILDREN) == "b"


def test_explicit_off_roster_child_is_not_pinned() -> None:
    # A stale or cross-family id must not pin: with >1 child on file we cannot default either.
    assert _pin_target_child("zzz", CHILDREN) is None


def test_defaults_to_the_only_child() -> None:
    assert _pin_target_child(None, {"solo": {"display_name": "Only"}}) == "solo"


def test_off_roster_child_still_defaults_to_the_only_child() -> None:
    # An unusable explicit id falls through to the single-child default.
    assert _pin_target_child("zzz", {"solo": {}}) == "solo"


def test_no_pin_when_multiple_children_and_no_valid_hint() -> None:
    assert _pin_target_child(None, CHILDREN) is None


def test_no_pin_when_no_children() -> None:
    assert _pin_target_child(None, {}) is None
    assert _pin_target_child("a", {}) is None


# --- load_context: missing-context guard -------------------------------------------------


def test_missing_context_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request that carries no context must fail fast before touching the DB."""
    monkeypatch.setattr(
        lifecycle, "get_runtime", lambda _schema: SimpleNamespace(context=None)
    )
    with pytest.raises(MissingContextError):
        load_context({})
