"""Structured-output + HITL contracts for the memory subgraph.

`MemoryDecision` is what memory_policy returns (via model.with_structured_output). The record
models below are the human-in-the-loop confirmation contract, reused in BOTH directions: the
confirm node ships one to the frontend inside a ConfirmationRequest (the popup renders every
field, blanks included), and the frontend ships it back inside a ConfirmationDecision on Accept
(possibly edited). The confirmed record -- not the LLM's raw memory arguments -- drives the
actual DB write, so the fields here are named to match the domain tools 1:1 (display_name ->
"name" label in UI, birth_date -> "birthday" label), keeping op rebuild a direct pass-through.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --- memory policy ----------------------------------------------------------------------


class MemoryOperation(BaseModel):
    """A domain-level operation to persist (executed by the profile_update agent's tools)."""

    operation: str  # domain operation name, e.g. "UpdateReadingInterest"
    arguments: dict = Field(default_factory=dict)
    rationale: str = ""


class MemoryDecision(BaseModel):
    """What this turn decided is worth remembering long-term."""

    operations: list[MemoryOperation] = Field(default_factory=list)


# --- confirm (human-in-the-loop popup contract) -----------------------------------------


class ChildRecord(BaseModel):
    """A child's identity as shown in the confirm popup and returned on Accept.

    Every field is always present so the popup can render a full form (blank where unknown).
    `birth_date` is a tolerant string: '2023', '2023-05', or '2023-05-01' are all accepted --
    a missing month/day is filled with 01 when it is finally persisted.
    """

    display_name: str | None = None  # UI label: "name"
    aliases: list[str] = Field(default_factory=list)
    gender: Literal["Male", "Female"] | None = None
    birth_date: str | None = None  # UI label: "birthday"; year-only is allowed
    grade: str | None = None
    primary_language: str | None = None
    reading_language: str | None = None


class MemberRecord(BaseModel):
    """A parent/caregiver's identity, shown/returned when a member identity change is gated."""

    display_name: str | None = None
    role: str | None = None
    gender: Literal["Male", "Female"] | None = None
    birth_date: str | None = None
    language_preference: str | None = None


class ConfirmationRequest(BaseModel):
    """Interrupt payload the frontend renders as an Accept/Reject popup (backend -> UI)."""

    type: Literal["confirm_profile_writes"] = "confirm_profile_writes"
    kind: Literal["save_child", "profile_update"]
    question: str
    target_child_id: str | None = None
    child: ChildRecord | None = None
    member: MemberRecord | None = None


class ConfirmationDecision(BaseModel):
    """Accept/Reject reply, parsed from the resume value (UI -> backend).

    The same record models come back (possibly edited in the form). Fail-safe: only an explicit
    `approved=True` applies; anything else (missing flag, non-dict resume value) means skip.
    """

    approved: bool = False
    child: ChildRecord | None = None
    member: MemberRecord | None = None
