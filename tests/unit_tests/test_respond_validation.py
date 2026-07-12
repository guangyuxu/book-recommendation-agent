"""Tests for how `respond` acts on the output-validation rating (agent.pipeline.respond).

Hermetic: the BLOCK path and the _validation_note helper are exercised without any LLM/DB. The
ALLOW compose path (LLM-backed) is covered by integration, not here.
"""

from __future__ import annotations

import importlib

from langchain.messages import AIMessage, HumanMessage

# import_module returns the real submodule; the package re-exports `respond` (the function) under
# the same name, which would otherwise shadow it (same pattern as the execute test).
respond_mod = importlib.import_module("agent.pipeline.respond")


def test_block_returns_localized_canned_reply_without_composing() -> None:
    # BLOCK must short-circuit: no LLM compose, no persistence -- just the canned reply. If it
    # tried to compose it would hit the network (no key) and fail; returning cleanly proves it did
    # not. The Simplified refusal is chosen from reply_language.
    state = {
        "messages": [HumanMessage(content="推荐一本书")],
        "reply_language": "zh-Hans",
        "validation": {"rating": "BLOCK", "results": [{"check": "child_safety", "outcome": "block"}]},
        "capability_results": {"recommend": {"books": [{"title": "X"}]}},
    }

    out = respond_mod.respond(state)

    assert isinstance(out["messages"][0], AIMessage)
    assert out["messages"][0].content == respond_mod._BLOCKED_REPLY["zh-Hans"]


def test_block_defaults_to_english_when_language_missing() -> None:
    out = respond_mod.respond(
        {"messages": [HumanMessage(content="hi")], "validation": {"rating": "BLOCK"}}
    )
    assert out["messages"][0].content == respond_mod._BLOCKED_REPLY["en"]


def test_validation_note_empty_for_allow_or_absent() -> None:
    assert respond_mod._validation_note({}) == ""
    assert respond_mod._validation_note({"validation": {"rating": "ALLOW"}}) == ""


def test_validation_note_rewrite_carries_flagged_reasons() -> None:
    note = respond_mod._validation_note(
        {
            "validation": {
                "rating": "REWRITE",
                "results": [
                    {"check": "privacy", "outcome": "rewrite", "reason": "leaks a name"},
                    {"check": "child_safety", "outcome": "pass", "reason": ""},
                ],
            }
        }
    )
    assert "REWRITE".lower() in note.lower() or "revise" in note.lower()
    assert "privacy: leaks a name" in note
    assert "child_safety" not in note  # PASS checks are not surfaced as concerns


def test_validation_note_warning_is_a_caveat() -> None:
    note = respond_mod._validation_note(
        {
            "validation": {
                "rating": "WARNING",
                "results": [{"check": "factuality", "outcome": "warn", "reason": "unverified"}],
            }
        }
    )
    assert "caveat" in note.lower()
    assert "factuality: unverified" in note
