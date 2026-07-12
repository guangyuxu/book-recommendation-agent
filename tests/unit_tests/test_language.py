"""Unit tests for reply-language support (agent.language) and its use in clarify.

Pure functions -- no LLM, no DB. Cover normalization of arbitrary/aliased codes, the
dependency-free script heuristic used by the pre-LLM guard node, the LLM directive, and the
localized deterministic clarify question (the one clarify branch that never calls the LLM).
"""

from __future__ import annotations

import pytest

from agent.language import (
    DEFAULT_LANGUAGE,
    detect_language,
    normalize_language,
    reply_directive,
)
from agent.pipeline.clarify import _ASK_WHICH_CHILD, clarify

# --- normalize_language -----------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("en", "en"),
        ("EN", "en"),
        ("en-US", "en"),
        ("zh-Hans", "zh-Hans"),
        ("zh", "zh-Hans"),
        ("zh-CN", "zh-Hans"),
        ("zh_SG", "zh-Hans"),
        ("zh-Hant", "zh-Hant"),
        ("zh-TW", "zh-Hant"),
        ("zh-HK", "zh-Hant"),
        ("  ZH-hant ", "zh-Hant"),
    ],
)
def test_normalize_language_accepts_codes_and_aliases(value: str, expected: str) -> None:
    assert normalize_language(value) == expected


@pytest.mark.parametrize("value", ["", "fr", "klingon", None, 42, {"x": 1}])
def test_normalize_language_falls_back_to_default(value: object) -> None:
    assert normalize_language(value) == DEFAULT_LANGUAGE == "en"


# --- detect_language (guard's pre-LLM heuristic) ----------------------------------------


def test_detect_english_when_no_han() -> None:
    assert detect_language("Recommend a picture book for my 5 year old") == "en"


def test_detect_empty_is_english() -> None:
    assert detect_language("") == "en"


def test_detect_simplified_via_distinctive_chars() -> None:
    # 这/说 are Simplified-only.
    assert detect_language("这本书讲什么，请说说") == "zh-Hans"


def test_detect_traditional_via_distinctive_chars() -> None:
    # 這/說 are Traditional-only.
    assert detect_language("這本書講什麼，請說說") == "zh-Hant"


def test_detect_han_without_distinctive_chars_defaults_to_simplified() -> None:
    # Common Han text sharing both standards -> default to Simplified.
    assert detect_language("你好") == "zh-Hans"


# --- reply_directive --------------------------------------------------------------------


def test_reply_directive_names_the_language() -> None:
    assert "Simplified Chinese" in reply_directive("zh-Hans")
    assert "Traditional Chinese" in reply_directive("zh-Hant")
    assert "English" in reply_directive("en")


def test_reply_directive_normalizes_raw_values() -> None:
    # Accepts a raw/aliased state value and an unknown one (-> default English).
    assert "Traditional Chinese" in reply_directive("zh-TW")
    assert "English" in reply_directive(None)


# --- clarify: localized deterministic question ------------------------------------------


@pytest.mark.parametrize("lang", ["en", "zh-Hans", "zh-Hant"])
def test_clarify_ambiguous_child_question_is_localized(lang: str) -> None:
    """The child-ambiguous branch is deterministic (no LLM) and must respect reply_language."""
    out = clarify({"understanding": {"child_ambiguous": True}, "reply_language": lang})
    assert out["messages"][0].content == _ASK_WHICH_CHILD[lang]
    assert out["clarification"]["decision"] == "ask_user"


def test_clarify_defaults_to_english_when_language_missing() -> None:
    out = clarify({"understanding": {"child_ambiguous": True}})
    assert out["messages"][0].content == _ASK_WHICH_CHILD["en"]
