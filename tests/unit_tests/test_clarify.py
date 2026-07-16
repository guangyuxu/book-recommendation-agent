"""Unit tests for the clarify node's deterministic branches.

The LLM branch (assumable-vs-ask adjudication) is exercised only when a required input is
genuinely unmet. The two deterministic paths -- ambiguous child (always ask) and "nothing
missing" (always continue) -- must never touch the LLM; that short-circuit is what prevents the
over-asking bug where a book-recommendation for a known child was answered with a question about
the child's age and interests.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

# agent.pipeline.__init__ re-exports the `clarify` function, shadowing the submodule attribute,
# so `import agent.pipeline.clarify as ...` would bind the function. Fetch the real module.
clarify_mod = importlib.import_module("agent.pipeline.clarify")


@pytest.fixture(autouse=True)
def _forbid_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a deterministic path reaches the structured LLM."""

    class _Boom:
        def invoke(self, *_a: Any, **_k: Any) -> Any:
            raise AssertionError("clarify consulted the LLM on a deterministic path")

    monkeypatch.setattr(clarify_mod, "_structured", _Boom())


def test_continues_without_llm_when_no_required_input_missing() -> None:
    # recommend's only required input (target_child) is satisfied by the pinned child, so there is
    # nothing to adjudicate -> continue, no question, no LLM call.
    state: dict[str, Any] = {
        "understanding": {"child_ambiguous": False, "mentioned_books": []},
        "target_child_id": "d63ae622",
        "reply_language": "zh-Hans",
        "plan": {"steps": [{"capability": "recommend"}]},
        "messages": [],
    }
    out = clarify_mod.clarify(state)
    assert out["clarification"]["decision"] == "continue"
    assert "messages" not in out  # no question appended


def test_empty_plan_continues_without_llm() -> None:
    # A profile-update-only turn plans no capability -> no required inputs -> continue.
    state: dict[str, Any] = {
        "understanding": {"child_ambiguous": False, "mentioned_books": []},
        "target_child_id": "d63ae622",
        "reply_language": "en",
        "plan": {"steps": []},
        "messages": [],
    }
    out = clarify_mod.clarify(state)
    assert out["clarification"]["decision"] == "continue"


def test_ambiguous_child_asks_deterministically_in_language() -> None:
    # A needed-but-unresolved child always asks, in the parent's language, without the LLM.
    state: dict[str, Any] = {
        "understanding": {"child_ambiguous": True, "mentioned_books": []},
        "target_child_id": None,
        "reply_language": "zh-Hans",
        "plan": {"steps": [{"capability": "recommend"}]},
        "messages": [],
    }
    out = clarify_mod.clarify(state)
    assert out["clarification"]["decision"] == "ask_user"
    assert out["clarification"]["missing_inputs"] == ["target_child"]
    assert out["messages"][0].content == clarify_mod._ASK_WHICH_CHILD["zh-Hans"]
