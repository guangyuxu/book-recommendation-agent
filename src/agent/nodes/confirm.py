"""Confirmation gate: pause for parent approval before high-stakes profile writes.

Human-in-the-loop for point 1 (changing identity fields: gender / birth_date / name) and
point 3 (saving a child not yet on file). Soft reading-profile facts are NOT gated -- they
flow straight through to profile_update.

Split into three single-responsibility nodes, per the official LangGraph guidance
(https://docs.langchain.com/oss/python/langgraph/interrupts): keep the interrupt in a node that
does nothing else, and put side effects in separate nodes.

    memory -> prepare_confirmation --skip------------------------------> profile_update
                     |  --confirm-->  request_confirmation -> apply_confirmation -> profile_update

    prepare_confirmation  build the popup contract; route "skip" (nothing high-stakes) or
                          "confirm". Pure -- writes only `confirmation_request` to state.
    request_confirmation  the ONLY interrupt: pause, and capture the Accept/Reject reply. Its
                          body is a single interrupt() call, so LangGraph re-running the node
                          from the top on resume replays nothing of consequence.
    apply_confirmation    turn the reply into the operations to persist. Pure -- rewrites
                          `memory_operations` and records the outcome in `confirmation`.

No DB write lives in any of these nodes: the confirmed operations are handed downstream to
profile_update, the only writer, which runs exactly once after the gate resolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langgraph.types import interrupt

from ..schemas import (
    ChildRecord,
    ConfirmationDecision,
    ConfirmationRequest,
    MemberRecord,
)

# Operations whose writes are high-stakes and must be confirmed before applying.
_CONFIRM_TOOLS = {"create_child", "update_child_basic_info", "update_member_basic_info"}


def _norm(name: str) -> str:
    """Normalize an operation name so PascalCase / snake_case / spacing all compare equal."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


_CONFIRM_NORM = {_norm(t) for t in _CONFIRM_TOOLS}
_CREATE_NORM = _norm("create_child")
_CHILD_ID_NORM = {_norm("create_child"), _norm("update_child_basic_info")}
_MEMBER_ID_NORM = {_norm("update_member_basic_info")}

# Free-text gender the model might emit -> the canonical enum value the tools accept.
_GENDER = {
    "male": "Male",
    "m": "Male",
    "boy": "Male",
    "female": "Female",
    "f": "Female",
    "girl": "Female",
}


@dataclass
class _Pending:
    """Everything PHASE 1 computed that PHASE 2 needs -- the plan carried across the seam.

    Recomputed from scratch on the resume run (the node re-runs), so it is a pure function of
    `state`, never of the resume value.
    """

    kind: str
    request: ConfirmationRequest
    auto: list[dict]  # ops that apply regardless of the decision
    bundled_soft: list[dict]  # soft ops gated only because a create bundled them
    child: ChildRecord | None
    member: MemberRecord | None
    creating: bool


# --- shared helpers ---------------------------------------------------------------------


def _is_create(op: dict) -> bool:
    return _norm(op.get("operation", "")) == _CREATE_NORM


def _op_norm(op: dict) -> str:
    return _norm(op.get("operation", ""))


def classify(operations: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split ops into (auto, needs_confirmation).

    If the turn creates a new child, EVERY op is gated together: the child's facts can't be
    applied unless the create is approved (they'd have no target child). Otherwise only the
    identity-field operations are gated; soft facts auto-apply.
    """
    if any(_is_create(o) for o in operations):
        return [], list(operations)
    auto, gated = [], []
    for o in operations:
        (gated if _op_norm(o) in _CONFIRM_NORM else auto).append(o)
    return auto, gated


def _canon_gender(value: object) -> str | None:
    """Map whatever the model produced to 'Male'/'Female', or None if unrecognized."""
    if value is None:
        return None
    return _GENDER.get(str(value).strip().lower())


def _fold(ops: list[dict]) -> dict:
    """Merge the arguments of several identity ops into one dict (later ops win per key)."""
    merged: dict = {}
    for o in ops:
        merged.update(o.get("arguments") or {})
    if "gender" in merged:
        merged["gender"] = _canon_gender(merged.get("gender"))
    return merged


def _build_child(ops: list[dict]) -> ChildRecord:
    # Unknown keys (e.g. a hallucinated `age`) are dropped: ChildRecord ignores extras.
    return ChildRecord.model_validate(_fold(ops))


def _build_member(ops: list[dict]) -> MemberRecord:
    return MemberRecord.model_validate(_fold(ops))


def _child_op(record: ChildRecord, creating: bool) -> list[dict]:
    args = record.model_dump(exclude_none=True)
    if not args.get("aliases"):
        args.pop("aliases", None)
    if not args:
        return []
    return [
        {
            "operation": "create_child" if creating else "update_child_basic_info",
            "arguments": args,
            "rationale": "parent-confirmed",
        }
    ]


def _member_op(record: MemberRecord) -> list[dict]:
    args = record.model_dump(exclude_none=True)
    if not args:
        return []
    return [
        {
            "operation": "update_member_basic_info",
            "arguments": args,
            "rationale": "parent-confirmed",
        }
    ]


# --- PHASE 1: before the pause -- build the popup contract ------------------------------


def _build_request(state: dict) -> _Pending | None:
    """Pure. Decide whether a confirmation is needed and, if so, build the popup + the plan.

    Returns None when nothing high-stakes is pending (the caller then skips the interrupt
    entirely). Runs on BOTH the first and the resume execution, so it must not depend on the
    resume value -- only on `state`.
    """
    operations = state.get("memory_operations") or []
    if not operations:
        return None

    auto, gated = classify(operations)
    if not gated:
        return None  # nothing high-stakes; profile_update applies all ops as-is

    creating = any(_is_create(o) for o in gated)
    child_id_ops = [o for o in gated if _op_norm(o) in _CHILD_ID_NORM]
    member_id_ops = [o for o in gated if _op_norm(o) in _MEMBER_ID_NORM]
    # Soft ops only appear in `gated` when a create bundled them; re-apply them on approval.
    bundled_soft = [o for o in gated if _op_norm(o) not in _CONFIRM_NORM]

    child = _build_child(child_id_ops) if child_id_ops else None
    member = _build_member(member_id_ops) if member_id_ops else None

    kind = "save_child" if creating else "profile_update"
    request = ConfirmationRequest(
        kind=kind,
        question=(
            "Save this child to your family?" if creating else "Update these identity details?"
        ),
        target_child_id=state.get("target_child_id"),
        child=child,
        member=member,
    )
    return _Pending(
        kind=kind,
        request=request,
        auto=auto,
        bundled_soft=bundled_soft,
        child=child,
        member=member,
        creating=creating,
    )


# --- PHASE 2: after Accept/Reject -- turn the reply into writes -------------------------


def _read_decision(resume: object) -> ConfirmationDecision:
    """Parse the resume value into a ConfirmationDecision. Fail-safe: a non-dict value (or a
    dict without approved=True) is treated as a rejection."""
    if isinstance(resume, dict):
        try:
            return ConfirmationDecision.model_validate(resume)
        except Exception:  # malformed payload -> safest to treat as a rejection
            return ConfirmationDecision()
    return ConfirmationDecision()


def _apply_decision(decision: ConfirmationDecision, pending: _Pending) -> dict:
    """Pure. Fold the parent's reply into the operations profile_update should persist."""
    if not decision.approved:
        return {
            "memory_operations": pending.auto,
            "confirmation": {"kind": pending.kind, "status": "rejected", "operations": []},
        }

    # The frontend may have edited the record in the form; prefer what it sent back.
    final_child = decision.child or pending.child
    final_member = decision.member or pending.member
    applied: list[dict] = list(pending.bundled_soft)
    if final_child is not None:
        applied = _child_op(final_child, pending.creating) + applied
    if final_member is not None:
        applied += _member_op(final_member)

    return {
        "memory_operations": pending.auto + applied,
        "confirmation": {"kind": pending.kind, "status": "applied", "operations": applied},
    }


# --- the three nodes --------------------------------------------------------------------


def prepare_confirmation(state: dict) -> dict:
    """Build the popup contract and decide whether a pause is needed. Runs every turn.

    Resets the confirmation channels so a prior turn's outcome never lingers. `confirmation`
    is left {} here and only re-filled by apply_confirmation when the gate actually resolves.
    """
    pending = _build_request(state)
    request = pending.request.model_dump() if pending is not None else {}
    return {"confirmation_request": request, "confirmation_decision": {}, "confirmation": {}}


def route_after_prepare(state: dict) -> str:
    """Route to the interrupt node only when something high-stakes is pending."""
    return "confirm" if state.get("confirmation_request") else "skip"


def request_confirmation(state: dict) -> dict:
    """The ONLY interrupt: pause for the parent, capture their Accept/Reject reply.

    Deliberately does nothing else -- on resume LangGraph re-runs this node from the top, and a
    body that is a single interrupt() call replays harmlessly. The reply becomes the return
    value of interrupt() and is stashed for apply_confirmation.
    """
    return {"confirmation_decision": interrupt(state["confirmation_request"])}


def apply_confirmation(state: dict) -> dict:
    """Fold the parent's reply into the operations profile_update should persist. Pure.

    Recomputes the plan from state (memory_operations is unchanged since prepare, so this
    reproduces the same _Pending) rather than serializing records through the state channel.
    """
    pending = _build_request(state)
    if pending is None:  # defensive: this node only runs on the confirm branch
        return {"confirmation": {}}
    decision = _read_decision(state.get("confirmation_decision"))
    return _apply_decision(decision, pending)
