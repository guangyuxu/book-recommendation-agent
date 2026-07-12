"""Reply-language support: the supported set, normalization, cheap detection, and an LLM directive.

Supported reply languages: English ("en"), Simplified Chinese ("zh-Hans"), Traditional Chinese
("zh-Hant"). The pipeline decides the reply language in ONE place -- the `understand` node asks
the LLM to report it (reliable, and it distinguishes Simplified from Traditional) -- and stores
it on the `reply_language` state channel. The downstream LLM nodes (clarify, respond) append
`reply_directive(...)` to their system prompts so the reply comes back in the parent's language.

The `guard` entry node runs BEFORE `understand` (and before any Anthropic model), so it cannot
use the LLM-detected value. It calls `detect_language` here -- a dependency-free heuristic over
the message's script -- purely to localize its static refusal. Getting the Simplified/Traditional
split slightly wrong on a refusal to a blocked (likely hostile) turn is harmless; the heuristic
defaults to Simplified for any Han text that carries no distinctive characters.

No PII concern: nothing here logs; `detect_language` only inspects character ranges/membership.
"""

from __future__ import annotations

from typing import Literal

Language = Literal["en", "zh-Hans", "zh-Hant"]
DEFAULT_LANGUAGE: Language = "en"
SUPPORTED_LANGUAGES: tuple[Language, ...] = ("en", "zh-Hans", "zh-Hant")

# Human-readable name injected into the LLM directive so the model is unambiguous about variant.
_DISPLAY_NAME: dict[Language, str] = {
    "en": "English",
    "zh-Hans": "Simplified Chinese (简体中文)",
    "zh-Hant": "Traditional Chinese (繁體中文)",
}


def normalize_language(value: object) -> Language:
    """Coerce an arbitrary value (an LLM-reported code, a state value) to a supported Language.

    Accepts our canonical codes plus common aliases, case-insensitively: en-* -> en; zh, zh-CN,
    zh-SG, zh-Hans -> zh-Hans; zh-TW, zh-HK, zh-MO, zh-Hant -> zh-Hant. Anything unrecognized
    (including None / non-str) falls back to DEFAULT_LANGUAGE.
    """
    if not isinstance(value, str):
        return DEFAULT_LANGUAGE
    code = value.strip().lower().replace("_", "-")
    if code == "zh-hant" or code in ("zh-tw", "zh-hk", "zh-mo"):
        return "zh-Hant"
    if code in ("zh", "zh-hans", "zh-cn", "zh-sg"):
        return "zh-Hans"
    if code == "en" or code.startswith("en-"):
        return "en"
    return DEFAULT_LANGUAGE


# Characters that exist ONLY in the Simplified or ONLY in the Traditional standard: the presence
# of one side is strong evidence of that variant. A small common-word sample -- enough to localize
# a refusal, not a full OpenCC-grade mapping.
_SIMPLIFIED_ONLY = frozenset("们这国说爱会应儿师问题书没让实现将样张对时观点线读软")
_TRADITIONAL_ONLY = frozenset("們這國說愛會應兒師問題書沒讓實現將樣張對時觀點線讀軟")


def _has_han(text: str) -> bool:
    """Return True if `text` contains any CJK Unified Ideograph (the common Han block)."""
    return any("一" <= ch <= "鿿" for ch in text)


def detect_language(text: str) -> Language:
    """Best-effort, dependency-free reply-language guess for the pre-LLM `guard` node.

    No Han ideographs -> English. Otherwise Chinese: pick Traditional only when the text has
    strictly more Traditional-only than Simplified-only characters, else default to Simplified.
    """
    if not text or not _has_han(text):
        return "en"
    simplified = sum(ch in _SIMPLIFIED_ONLY for ch in text)
    traditional = sum(ch in _TRADITIONAL_ONLY for ch in text)
    return "zh-Hant" if traditional > simplified else "zh-Hans"


def reply_directive(language: object) -> str:
    """Build a one-line instruction to append to an LLM system prompt, pinning the reply language.

    Normalizes first, so callers can pass a raw state value. The returned text begins with a
    blank line so it concatenates cleanly onto an existing prompt body.
    """
    lang = normalize_language(language)
    return f"\n\nWrite your reply to the parent in {_DISPLAY_NAME[lang]}."
