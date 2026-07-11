"""Confirmation gate nodes: pause for parent approval before high-stakes profile writes.

Human-in-the-loop for changing identity fields (gender / birth_date / name) and for saving a
child not yet on file. Soft reading-profile facts are NOT gated -- they flow straight through to
profile_update. All the deciding is pure and lives in confirm_policy; these are just the graph
shells that wire it in, following the official LangGraph guidance to keep the interrupt in a node
that does nothing else and put side effects in separate nodes
(https://docs.langchain.com/oss/python/langgraph/interrupts):

    memory_policy -> prepare_confirmation --skip----------------------> profile_update
                        |  --confirm-->  request_confirmation -> apply_confirmation -> profile_update

    prepare_confirmation  build the popup contract; route "skip" or "confirm". Writes only the
                          confirmation_* channels to state.
    request_confirmation  the ONLY interrupt: pause, and capture the Accept/Reject reply. Its
                          body is a single interrupt() call, so re-running on resume replays
                          nothing of consequence.
    apply_confirmation    turn the reply into the operations to persist; record the outcome.

No DB write lives in any of these: the confirmed operations are handed to profile_update, the
only writer, which runs exactly once after the gate resolves.
"""

from __future__ import annotations

from langgraph.types import interrupt

from ..state import FlowState
from .confirm_policy import _apply_decision, _build_request, _read_decision


def prepare_confirmation(state: FlowState) -> dict:
    """Build the popup contract and decide whether a pause is needed. Runs every turn.

    Resets the confirmation channels so a prior turn's outcome never lingers. `confirmation`
    is left {} here and only re-filled by apply_confirmation when the gate actually resolves.
    """
    pending = _build_request(state)
    request = pending.request.model_dump() if pending is not None else {}
    return {"confirmation_request": request, "confirmation_decision": {}, "confirmation": {}}


def route_after_prepare(state: FlowState) -> str:
    """Route to the interrupt node only when something high-stakes is pending."""
    return "confirm" if state.get("confirmation_request") else "skip"


def request_confirmation(state: FlowState) -> dict:
    """Pause for the parent and capture their Accept/Reject reply -- the graph's ONLY interrupt.

    Deliberately does nothing else -- on resume LangGraph re-runs this node from the top, and a
    body that is a single interrupt() call replays harmlessly. The reply becomes the return
    value of interrupt() and is stashed for apply_confirmation.
    """
    return {"confirmation_decision": interrupt(state["confirmation_request"])}


def apply_confirmation(state: FlowState) -> dict:
    """Fold the parent's reply into the operations profile_update should persist. Pure.

    Recomputes the plan from state (memory_operations is unchanged since prepare, so this
    reproduces the same _Pending) rather than serializing records through the state channel.
    """
    pending = _build_request(state)
    if pending is None:  # defensive: this node only runs on the confirm branch
        return {"confirmation": {}}
    decision = _read_decision(state.get("confirmation_decision"))
    return _apply_decision(decision, pending)
