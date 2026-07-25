"""Unit tests for the confirm node's pure policy.

The interrupt()/graph paused-state behavior is exercised in Studio; here we pin the
deterministic pieces: which ops are gated (classify), how raw ops fold into a clean record
(_build_child + normalization), how a resume value maps to a decision (_read_decision, fail-safe),
and how an approved record rebuilds clean operations (_child_op).
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.memory.confirm import (
    apply_confirmation,
    prepare_confirmation,
    request_confirmation,
    route_after_prepare,
)
from agent.memory.confirm_policy import (
    _build_child,
    _child_op,
    _read_decision,
    classify,
)

SOFT = {
    "operation": "update_reading_interest",
    "arguments": {"add_interests": ["dragons"]},
}
IDENTITY = {
    "operation": "update_child_basic_info",
    "arguments": {"birth_date": "2016-03-15"},
}
MEMBER_ID = {"operation": "update_member_basic_info", "arguments": {"gender": "Female"}}
CREATE = {"operation": "create_child", "arguments": {"display_name": "Nephew"}}


# --- classify ---------------------------------------------------------------------------


def test_soft_ops_are_auto() -> None:
    auto, gated = classify([SOFT])
    assert auto == [SOFT] and gated == []


def test_identity_ops_are_gated() -> None:
    auto, gated = classify([SOFT, IDENTITY, MEMBER_ID])
    assert auto == [SOFT]
    assert gated == [IDENTITY, MEMBER_ID]


def test_create_bundles_everything() -> None:
    # A new child gates ALL ops together -- the facts can't apply without the create.
    auto, gated = classify([CREATE, SOFT])
    assert auto == []
    assert gated == [CREATE, SOFT]


def test_operation_name_normalization() -> None:
    # PascalCase from the LLM must still be recognized as the gated tool.
    auto, gated = classify([{"operation": "UpdateChildBasicInfo", "arguments": {}}])
    assert auto == [] and len(gated) == 1


# --- _build_child (fold + normalize) ----------------------------------------------------


def test_build_child_drops_unknown_args_and_canonicalizes_gender() -> None:
    # Hallucinated args (age, birthday_upcoming) are dropped; lowercase gender is canonicalized.
    ops = [
        {
            "operation": "create_child",
            "arguments": {"display_name": "小白兔", "age": 5},
        },
        {
            "operation": "update_child_basic_info",
            "arguments": {
                "gender": "female",
                "birthday_upcoming": True,
                "birth_date": "2023",
            },
        },
    ]
    record = _build_child(ops)
    assert record.display_name == "小白兔"
    assert record.gender == "Female"
    assert record.birth_date == "2023"  # year-only preserved as-is on the record
    assert not hasattr(record, "age")


def test_build_child_unrecognized_gender_becomes_none() -> None:
    record = _build_child(
        [{"operation": "create_child", "arguments": {"gender": "unknown"}}]
    )
    assert record.gender is None


# --- _read_decision (fail-safe) ---------------------------------------------------------


def test_approved_true_dict() -> None:
    decision = _read_decision({"approved": True})
    assert decision.approved is True


def test_approved_false_dict() -> None:
    assert _read_decision({"approved": False}).approved is False


def test_missing_flag_defaults_to_rejected() -> None:
    assert _read_decision({}).approved is False


def test_non_dict_resume_is_rejected() -> None:
    for value in (True, None, "approve", "yes", []):
        assert _read_decision(value).approved is False


def test_decision_carries_edited_child_record() -> None:
    decision = _read_decision(
        {"approved": True, "child": {"display_name": "Edited", "gender": "Male"}}
    )
    assert decision.approved is True
    assert decision.child is not None
    assert decision.child.display_name == "Edited"
    assert decision.child.gender == "Male"


# --- _child_op (rebuild clean operation) ------------------------------------------------


def test_child_op_builds_create_from_record() -> None:
    record = _build_child(
        [{"operation": "create_child", "arguments": {"display_name": "Nia"}}]
    )
    ops = _child_op(record, creating=True)
    assert ops == [
        {
            "operation": "create_child",
            "arguments": {"display_name": "Nia"},
            "rationale": "parent-confirmed",
        }
    ]


def test_child_op_builds_update_when_not_creating() -> None:
    record = _build_child(
        [
            {
                "operation": "update_child_basic_info",
                "arguments": {"birth_date": "2016-03-15"},
            }
        ]
    )
    ops = _child_op(record, creating=False)
    assert ops[0]["operation"] == "update_child_basic_info"
    assert ops[0]["arguments"] == {"birth_date": "2016-03-15"}


def test_child_op_create_falls_back_to_alias_when_name_missing() -> None:
    # create_child requires a display_name; a create confirmed without one must still persist,
    # falling back to an alias so the child (and its bundled facts) are not silently dropped.
    record = _build_child(
        [
            {
                "operation": "create_child",
                "arguments": {"aliases": ["Bud"], "gender": "Male"},
            }
        ]
    )
    ops = _child_op(record, creating=True)
    assert ops[0]["operation"] == "create_child"
    assert ops[0]["arguments"]["display_name"] == "Bud"


def test_child_op_create_uses_placeholder_when_no_name_or_alias() -> None:
    record = _build_child(
        [{"operation": "create_child", "arguments": {"gender": "Female"}}]
    )
    ops = _child_op(record, creating=True)
    assert ops[0]["operation"] == "create_child"
    assert ops[0]["arguments"]["display_name"] == "New child"


# --- nodes: prepare / route / apply (interrupt-free paths) ------------------------------


def test_prepare_skips_when_nothing_gated() -> None:
    out = prepare_confirmation({"memory_operations": [SOFT]})
    assert out["confirmation_request"] == {}
    assert out["confirmation"] == {} and out["confirmation_decision"] == {}
    assert route_after_prepare(out) == "skip"


def test_prepare_builds_request_when_gated() -> None:
    out = prepare_confirmation({"memory_operations": [IDENTITY]})
    assert out["confirmation_request"]["type"] == "confirm_profile_writes"
    assert out["confirmation_request"]["child"]["birth_date"] == "2016-03-15"
    assert route_after_prepare(out) == "confirm"


def test_apply_persists_on_approve() -> None:
    # apply recomputes the plan from memory_operations + reads the stashed decision.
    state = {
        "memory_operations": [IDENTITY],
        "confirmation_decision": {"approved": True},
    }
    out = apply_confirmation(state)
    assert out["confirmation"]["status"] == "applied"
    assert out["memory_operations"][0]["operation"] == "update_child_basic_info"


def test_apply_drops_gated_on_reject() -> None:
    state = {
        "memory_operations": [SOFT, IDENTITY],
        "confirmation_decision": {"approved": False},
    }
    out = apply_confirmation(state)
    assert out["confirmation"]["status"] == "rejected"
    assert out["memory_operations"] == [SOFT]


# --- request_confirmation: the ONLY interrupt (via a compiled-graph harness) ------------
# request_confirmation is a non-LLM node whose whole body is a single interrupt() call. We drive
# it through real LangGraph execution: the first invoke pauses on the interrupt (surfacing the
# request), and resuming with a value stashes exactly that value into confirmation_decision.


class _ConfirmState(TypedDict, total=False):
    confirmation_request: dict
    confirmation_decision: Any


def _single_node_graph():
    """Compile a one-node graph around request_confirmation with an in-memory checkpointer."""
    builder = StateGraph(_ConfirmState)
    builder.add_node("request_confirmation", request_confirmation)
    builder.add_edge(START, "request_confirmation")
    builder.add_edge("request_confirmation", END)
    return builder.compile(checkpointer=MemorySaver())


def test_request_confirmation_pauses_then_stashes_the_resume_value() -> None:
    graph = _single_node_graph()
    cfg = {"configurable": {"thread_id": "confirm-1"}}
    request = {"type": "confirm_profile_writes", "child": {"display_name": "Nia"}}

    paused = graph.invoke({"confirmation_request": request}, cfg)
    # It paused on the interrupt, surfacing the request payload to the caller.
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value == request

    resumed = graph.invoke(Command(resume={"approved": True}), cfg)
    # The parent's reply becomes the node's confirmation_decision verbatim.
    assert resumed["confirmation_decision"] == {"approved": True}
