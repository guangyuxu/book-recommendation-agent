"""Prompt registry: loader behavior and every extracted prompt's contract.

These run fully offline -- rendering is pure Jinja over the co-located *.prompts.yaml files, no
LLM, no DB. We pin (1) the loader's public surface (render/get/version, unknown-id, strict
undefined) and (2) the conditional WORDING / role structure that moved out of Python into the
templates: respond's focus-switch / confirmation-outcome notes, the recommend and evaluate retry
directives, the prose-capability briefs, clarify's two-role system+human with its reply-directive
block, the memory_policy / profile_update prompts, and understand's menu + roster interpolation.
"""

from __future__ import annotations

import pytest
from jinja2 import UndefinedError
from langchain_core.messages import HumanMessage, SystemMessage

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


def test_config_carries_prompt_id_and_version_for_langsmith() -> None:
    # The call-site config tags the LLM run with which prompt (+ version) produced it.
    cfg = prompts.config("recommend.generate")
    assert cfg["metadata"] == {
        "prompt_id": "recommend.generate",
        "prompt_version": prompts.version("recommend.generate"),
    }


def test_config_rejects_unknown_id() -> None:
    with pytest.raises(KeyError):
        prompts.config("nope.nope")


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


# --- prose capabilities: single system message embedding the briefs ----------------------


@pytest.mark.parametrize(
    "pid", ["compare.analyze", "content.draft", "discussion.questions", "path.plan"]
)
def test_prose_prompt_embeds_briefs(pid: str) -> None:
    msgs = prompts.render(pid, child_brief="CHILD-BRIEF", policies_brief="POLICY-BRIEF")
    assert len(msgs) == 1 and isinstance(msgs[0], SystemMessage)
    assert "CHILD-BRIEF" in msgs[0].content
    assert "POLICY-BRIEF" in msgs[0].content


# --- evaluate.analyze: revise directive folds in each gap --------------------------------


def _analyze(**kw: object) -> str:
    base = {"books": "- The Hobbit", "child_brief": "CB", "policies_brief": "PB"}
    base.update(kw)
    return prompts.render("evaluate.analyze", **base)[0].content


def test_evaluate_analyze_no_revise_block_without_feedback() -> None:
    out = _analyze(feedback=[])
    assert "reviewer found these gaps" not in out
    assert out.rstrip().endswith("PB")  # ends right after the policies brief


def test_evaluate_analyze_folds_in_each_gap() -> None:
    out = _analyze(feedback=["no cautions", "ignores level"])
    assert "reviewer found these gaps" in out
    assert "- no cautions" in out
    assert "- ignores level" in out
    assert "Revise your evaluation to address every gap." in out


def test_evaluate_validate_embeds_the_prior_evaluation() -> None:
    out = prompts.render(
        "evaluate.validate",
        books="- The Hobbit",
        child_brief="CB",
        policies_brief="PB",
        evaluation="PRIOR-EVAL-TEXT",
    )[0].content
    assert "PRIOR-EVAL-TEXT" in out


# --- clarify.decide: two roles, reply-directive block, human facts -----------------------


def test_clarify_decide_renders_system_and_human() -> None:
    msgs = prompts.render(
        "clarify.decide",
        reply_directive="Write your reply to the parent in English.",
        planned_capabilities=["recommend"],
        required_inputs=[],
        target_child_known=True,
        mentioned_books=[],
    )
    assert len(msgs) == 2
    assert isinstance(msgs[0], SystemMessage) and isinstance(msgs[1], HumanMessage)
    assert "Write your reply to the parent in English." in msgs[0].content
    assert "planned_capabilities=['recommend']" in msgs[1].content
    assert "target_child_known=True" in msgs[1].content


def test_clarify_decide_omits_reply_directive_block_when_empty() -> None:
    msgs = prompts.render(
        "clarify.decide",
        reply_directive="",
        planned_capabilities=[],
        required_inputs=[],
        target_child_known=False,
        mentioned_books=[],
    )
    # No trailing directive line when reply_directive is empty (the {% if %} suppresses it).
    assert msgs[0].content.rstrip().endswith("write a single concise question.")


# --- memory_policy.decide / profile_update.apply -----------------------------------------


def test_memory_policy_decide_lists_available_operations() -> None:
    msgs = prompts.render(
        "memory_policy.decide",
        available_operations="create_child, update_child_basic_info",
        child_is_new=True,
        user_signals=[{"detail": "loves dinosaurs"}],
    )
    assert len(msgs) == 2
    assert (
        "Available operations: create_child, update_child_basic_info" in msgs[0].content
    )
    assert "child_is_new=True" in msgs[1].content


def test_profile_update_apply_carries_ops_text() -> None:
    msgs = prompts.render(
        "profile_update.apply",
        user_signals=[],
        ops_text="- update_child_basic_info: {'birth_date': '2016'}",
    )
    assert len(msgs) == 2
    assert "never pass ids" in msgs[0].content  # static system instruction
    assert "- update_child_basic_info: {'birth_date': '2016'}" in msgs[1].content


# --- understand.read: intent menu + roster interpolation ---------------------------------


def test_understand_read_interpolates_menu_and_roster() -> None:
    out = prompts.render(
        "understand.read",
        intent_menu="- book_recommendation: recommend books",
        roster="- id=c1: Alex, age 6",
    )[0].content
    assert "- book_recommendation: recommend books" in out
    assert out.rstrip().endswith("- id=c1: Alex, age 6")  # roster is the final block
