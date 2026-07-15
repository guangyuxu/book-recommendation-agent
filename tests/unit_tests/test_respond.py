"""Unit tests for the respond node's deterministic composition helpers.

The final reply itself is LLM-composed (`_compose`) and exercised in Studio; here we pin the
pure pieces that decide WHAT material is handed to the model: how capability outputs render into
the booklist + prose block, how a prose capability's single string result is pulled out, and
the switch/confirmation hint lines that steer the reply. No LLM, no DB.
"""

from __future__ import annotations

from langchain.messages import AIMessage, AIMessageChunk, HumanMessage

from agent.pipeline.respond import (
    _confirmation_note,
    _gather_text,
    _last_human_text,
    _prose,
    _render_outputs,
    _switch_note,
)

# --- _render_outputs: recommend booklist -------------------------------------------------


def test_render_recommend_booklist_numbers_and_annotates() -> None:
    results = {
        "recommend": {
            "books": [
                {
                    "title": "Where the Wild Things Are",
                    "author": "Maurice Sendak",
                    "recommendation_reason": "imaginative",
                    "fit_summary": "matches her level",
                    "risk_notes": ["mild scary scene"],
                }
            ],
            "note": "Enjoy!",
        }
    }
    out = _render_outputs(results)
    assert "Recommended books:" in out
    assert "1. Where the Wild Things Are by Maurice Sendak — imaginative" in out
    assert "fit: matches her level" in out
    assert "watch-outs: mild scary scene" in out
    assert out.endswith("Enjoy!")


def test_render_recommend_without_author_or_extras() -> None:
    results = {"recommend": {"books": [{"title": "Frog and Toad"}]}}
    out = _render_outputs(results)
    assert "1. Frog and Toad" in out
    assert " by " not in out  # no author
    assert "(" not in out  # no fit/risk extras block


def test_render_empty_results_is_empty_string() -> None:
    assert _render_outputs({}) == ""


def test_render_includes_prose_capabilities_after_booklist() -> None:
    results = {
        "recommend": {"books": [{"title": "Corduroy"}]},
        "evaluate": {"evaluation": "A gentle, age-appropriate story."},
    }
    out = _render_outputs(results)
    assert "Recommended books:" in out
    assert "A gentle, age-appropriate story." in out
    # Two blocks joined by a blank line.
    assert "\n\n" in out


# --- _prose: single-string-key extraction ------------------------------------------------
# Every prose capability (evaluate/compare/discussion/path/content) returns exactly one
# string-valued key, so _prose takes the first non-empty string in the result.


def test_prose_returns_the_single_string_value() -> None:
    assert _prose("evaluate", {"evaluation": "This suits an early reader."}) == (
        "This suits an early reader."
    )


def test_prose_skips_blank_value() -> None:
    assert _prose("evaluate", {"evaluation": "   "}) is None


def test_prose_ignores_non_string_fields() -> None:
    # Non-string values are skipped; the first non-empty string wins.
    assert _prose("mystery", {"n": 3, "text": "hello"}) == "hello"


# --- _switch_note ------------------------------------------------------------------------


def test_switch_note_present_when_focus_changed() -> None:
    note = _switch_note({"child_switch": {"to_name": "小儿子"}})
    assert "小儿子" in note
    assert "switched" in note


def test_switch_note_empty_without_a_switch() -> None:
    assert _switch_note({}) == ""
    assert _switch_note({"child_switch": {}}) == ""


# --- _confirmation_note ------------------------------------------------------------------


def test_confirmation_note_applied_acknowledges_saved() -> None:
    note = _confirmation_note({"confirmation": {"status": "applied"}})
    assert "saved" in note


def test_confirmation_note_rejected_acknowledges_not_saved() -> None:
    note = _confirmation_note({"confirmation": {"status": "rejected"}})
    assert "not save" in note


def test_confirmation_note_error_must_not_claim_saved() -> None:
    note = _confirmation_note({"confirmation": {"status": "error"}})
    assert "wasn't saved" in note
    # The error hint explicitly forbids claiming success.
    assert "do NOT" in note


def test_confirmation_note_empty_when_no_gate_resolved() -> None:
    assert _confirmation_note({}) == ""
    assert _confirmation_note({"confirmation": {}}) == ""


# --- _last_human_text --------------------------------------------------------------------


def test_last_human_text_returns_most_recent_human_message() -> None:
    messages = [
        HumanMessage(content="first"),
        AIMessage(content="assistant reply"),
        HumanMessage(content="latest question"),
    ]
    assert _last_human_text(messages) == "latest question"


def test_last_human_text_empty_when_no_human_message() -> None:
    assert _last_human_text([AIMessage(content="only assistant")]) == ""


# --- _gather_text: reassemble a streamed reply ------------------------------------------
# respond streams its reply (stream_mode="messages") but still needs the whole string for the
# AIMessage it writes to state; _gather_text adds the chunks back into one message's content.


def test_gather_text_concatenates_streamed_chunks() -> None:
    chunks = [
        AIMessageChunk(content="Hello"),
        AIMessageChunk(content=", "),
        AIMessageChunk(content="reader!"),
    ]
    assert _gather_text(chunks) == "Hello, reader!"


def test_gather_text_empty_stream_is_empty_string() -> None:
    assert _gather_text([]) == ""
