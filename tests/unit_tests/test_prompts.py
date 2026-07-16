"""Prompt registry: loader behavior and the extracted recommend/respond prompt contracts.

These run fully offline -- rendering is pure Jinja over the co-located *.prompts.yaml files, no
LLM, no DB. We pin (1) the loader's public surface (render/get/version, unknown-id, strict
undefined) and (2) the conditional WORDING that moved out of Python into the templates: the
respond focus-switch / confirmation-outcome notes and the recommend retry directive.
"""

from __future__ import annotations

import pytest
from jinja2 import UndefinedError
from langchain_core.messages import SystemMessage

from agent import prompts

# --- loader surface ----------------------------------------------------------------------


def test_render_returns_system_message_list() -> None:
    msgs = prompts.render(
        "recommend.validate", child_brief="CB", policies_brief="PB", candidates="1. X"
    )
    assert len(msgs) == 1
    assert isinstance(msgs[0], SystemMessage)
    assert "1. X" in msgs[0].content


def test_version_is_exposed_for_reproducibility() -> None:
    # Every extracted prompt declares an integer version (recorded per turn later).
    assert prompts.version("respond.compose") == 1
    assert prompts.version("recommend.generate") == 1


def test_unknown_prompt_id_raises() -> None:
    with pytest.raises(KeyError):
        prompts.get("recommend.does_not_exist")


def test_missing_variable_is_a_hard_error() -> None:
    # StrictUndefined: a template referencing an unset variable fails loudly, never renders blank.
    with pytest.raises(UndefinedError):
        prompts.render("recommend.validate", child_brief="CB", policies_brief="PB")


# --- respond.compose: conditional note wording -------------------------------------------


def _compose(**kw: object) -> str:
    base = {
        "material": "some material",
        "switch_to_name": None,
        "confirmation_status": None,
        "reply_directive": "Write your reply to the parent in English.",
    }
    base.update(kw)
    return prompts.render("respond.compose", **base)[0].content


def test_compose_switch_note_only_when_focus_changed() -> None:
    assert "focus just switched to 小儿子" in _compose(switch_to_name="小儿子")
    assert "focus just switched" not in _compose(switch_to_name=None)


def test_compose_confirmation_outcomes() -> None:
    assert "it is saved" in _compose(confirmation_status="applied")
    assert "will not save it" in _compose(confirmation_status="rejected")
    error = _compose(confirmation_status="error")
    assert "wasn't saved" in error
    assert "do NOT claim it was saved" in error  # must never claim success on error
    # No gate resolved -> none of the confirmation lines appear.
    assert "profile change" not in _compose(confirmation_status=None)


def test_compose_always_pins_reply_language() -> None:
    assert "Write your reply to the parent in English." in _compose()


def test_compose_material_placeholder_when_empty() -> None:
    assert "Prepared material:\n(none)" in _compose(material="")


# --- recommend.generate: retry directive -------------------------------------------------


def test_generate_no_retry_block_without_feedback() -> None:
    out = prompts.render(
        "recommend.generate", child_brief="CB", policies_brief="PB", feedback=[]
    )[0].content
    assert "previous suggestions" not in out
    assert out.rstrip().endswith("PB")  # ends right after the policies brief


def test_generate_folds_in_each_rejection_reason() -> None:
    out = prompts.render(
        "recommend.generate",
        child_brief="CB",
        policies_brief="PB",
        feedback=["off level", "too scary"],
    )[0].content
    assert "were ALL rejected in screening" in out
    assert "- off level" in out
    assert "- too scary" in out
    assert "Propose a completely fresh list" in out
