"""Unit tests for pipeline schema coercion (deterministic -- no LLM, no DB).

Guards the fix for the model intermittently returning a list-valued field as a JSON *string*
via forced tool-calling, which otherwise raised a list_type error and crashed the whole turn.
"""

from __future__ import annotations

from agent.pipeline.schemas import Understanding


def test_stringified_user_signals_is_parsed() -> None:
    u = Understanding(
        intents=["book_recommendation"],
        user_signals='[{"about": "child", "kind": "preference", "detail": "loves dragons"}]',
    )
    assert len(u.user_signals) == 1
    assert u.user_signals[0].detail == "loves dragons"


def test_stringified_intents_and_books_are_parsed() -> None:
    u = Understanding(
        intents='["book_recommendation", "reading_discussion"]',
        mentioned_books='[{"title": "Charlotte\'s Web"}]',
    )
    assert {i.value for i in u.intents} == {"book_recommendation", "reading_discussion"}
    assert u.mentioned_books[0].title == "Charlotte's Web"


def test_malformed_user_signals_degrades_to_empty() -> None:
    # Unescaped inner quote -> invalid JSON. user_signals is non-critical, so we drop it to []
    # rather than crashing the whole turn.
    u = Understanding(
        intents=["book_comparison"],
        user_signals='[{"about": "child", "detail": "我女儿"，性别为女"}]',
    )
    assert u.user_signals == []
    assert [i.value for i in u.intents] == ["book_comparison"]


def test_native_lists_still_work() -> None:
    u = Understanding(
        intents=["book_evaluation"],
        user_signals=[{"about": "member", "kind": "attribute", "detail": "busy parent"}],
    )
    assert u.user_signals[0].about == "member"
    assert [i.value for i in u.intents] == ["book_evaluation"]
