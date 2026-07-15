"""Unit tests for the profile_update node's deterministic control flow.

profile_update is the only writer, driven by an LLM tool loop. We do not exercise a real model;
we stub the bound model + tools and pin the parts that must be right regardless of the LLM:
the no-op early returns, and -- critically for not misleading the parent -- the `writes_failed`
determination that downgrades a confirmed change to `status="error"` when the writes did not land.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from langchain.messages import AIMessage

from agent.memory import profile_update as pu_mod
from agent.memory.profile_update import profile_update

FAMILY_ID = str(uuid4())
IDENTITY_OP = {
    "operation": "update_child_basic_info",
    "arguments": {"birth_date": "2016-03-15"},
    "rationale": "parent-confirmed",
}


# --- early returns (no LLM path) ---------------------------------------------------------


def test_no_operations_is_a_noop() -> None:
    assert profile_update({"memory_operations": []}) == {}
    assert profile_update({}) == {}


def test_missing_family_id_skips_persistence() -> None:
    # Operations present but no family in state -> nothing to persist against.
    out = profile_update({"memory_operations": [IDENTITY_OP], "family": {}})
    assert out == {}


# --- LLM tool loop: stubbed model + tools ------------------------------------------------


class _StubModel:
    """Bound-model stand-in: returns each queued AIMessage in turn on .invoke()."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)

    def invoke(self, _messages: Any) -> AIMessage:
        return self._responses.pop(0)


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    responses: list[AIMessage],
    tool_ok: bool = True,
) -> None:
    """Wire the module's LLM + tools + session + re-read to deterministic stubs."""
    monkeypatch.setattr(pu_mod, "_bound", _StubModel(responses))

    def _tool_invoke(_args: Any) -> str:
        if not tool_ok:
            raise RuntimeError("write blew up")
        return "ok"

    monkeypatch.setattr(
        pu_mod,
        "MEMORY_TOOLS_BY_NAME",
        {"update_child_basic_info": SimpleNamespace(invoke=_tool_invoke)},
    )

    @contextlib.contextmanager
    def _fake_session(*_args: Any, **_kwargs: Any):
        yield SimpleNamespace(session=object(), target_child_id=None)

    monkeypatch.setattr(pu_mod, "domain_session", _fake_session)
    monkeypatch.setattr(pu_mod, "load_family_entities", lambda _s, _fid: ({}, {}))


def _tool_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "update_child_basic_info", "args": {}, "id": "call-1"}],
    )


def _stop() -> AIMessage:
    return AIMessage(content="done")  # no tool_calls -> loop stops cleanly


def test_clean_apply_keeps_confirmation_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Agent calls a tool, the tool succeeds, then the agent stops -> writes landed.
    _install(monkeypatch, responses=[_tool_call(), _stop()], tool_ok=True)
    out = profile_update(
        {
            "memory_operations": [IDENTITY_OP],
            "family": {"id": FAMILY_ID},
            "confirmation": {"status": "applied"},
        }
    )
    # A successful apply does not touch the confirmation channel (respond acknowledges "saved").
    assert "confirmation" not in out
    assert out["members"] == {} and out["children"] == {}


def test_failed_write_downgrades_confirmation_to_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The tool raises: the final batch had an error -> we must NOT report the change as saved.
    _install(monkeypatch, responses=[_tool_call(), _stop()], tool_ok=False)
    out = profile_update(
        {
            "memory_operations": [IDENTITY_OP],
            "family": {"id": FAMILY_ID},
            "confirmation": {"status": "applied"},
        }
    )
    assert out["confirmation"]["status"] == "error"


def test_failed_write_without_confirmation_gate_is_left_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On the soft/skip path (no "applied" gate), a write failure does not synthesize a status.
    _install(monkeypatch, responses=[_tool_call(), _stop()], tool_ok=False)
    out = profile_update(
        {"memory_operations": [IDENTITY_OP], "family": {"id": FAMILY_ID}}
    )
    assert "confirmation" not in out
