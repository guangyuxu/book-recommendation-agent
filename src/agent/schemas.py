"""Structured-output schemas for the pipeline nodes.

Each LLM-backed node (understand, plan, clarify, memory) returns one of these via
model.with_structured_output(...). They are the contract between stages; nodes store their
model_dump() into the matching state channel.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .intents import Intent

# --- understand -------------------------------------------------------------------------


class MentionedBook(BaseModel):
    """A book the user named in the message."""

    title: str
    author: str | None = None


class UserSignal(BaseModel):
    """One profile-relevant fact the user revealed (fuel for Memory Policy).

    Deliberately coarse: `about` + `kind` are only a rough subject/type tag. Mapping a signal to
    a specific domain operation/tool is Memory Policy's job -- it reads the tool menu and the
    free-text `detail`, so understand stays DB-agnostic and need not track the tool taxonomy.
    """

    about: Literal["child", "member", "family"] = Field(
        description=(
            "Whose fact this is. 'child' = the target child (reading level, tastes, book "
            "events). 'member' = the speaking parent/caregiver themselves (their own taste, "
            "parenting style, available time, concerns). 'family' = a household-level goal or "
            "constraint that applies regardless of person."
        )
    )
    kind: Literal["preference", "attribute", "activity", "goal", "constraint", "other"] = Field(
        description=(
            "Coarse, subject-independent type of the fact. 'preference' = likes/dislikes/tastes; "
            "'attribute' = a stable trait or fact (age, reading level, occupation, available "
            "time); 'activity' = something done or in progress (finished or currently reading a "
            "book); 'goal' = a desired outcome; 'constraint' = a limit or something to avoid; "
            "'other' = anything else. Keep it rough -- Memory Policy maps `detail` to the exact "
            "domain operation."
        )
    )
    detail: str


class ChildRef(BaseModel):
    """Which roster child (if any) the MESSAGE itself points to -- evidence only.

    Reconciliation with the pinned/active child is NOT done here; the understand node feeds this
    to a deterministic resolver (resolve_child). Do not consider any "currently selected" child
    when filling this -- report only what the message says.
    """

    status: Literal["matched", "new", "ambiguous", "none"] = Field(
        default="none",
        description=(
            "'matched' = the message clearly refers to a specific child on the roster (set "
            "child_id); 'new' = it describes a child NOT on the roster; 'ambiguous' = it refers "
            "to a child but which one is unclear; 'none' = it does not single out any child."
        ),
    )
    child_id: str | None = None  # roster id, set only when status == "matched"


class Understanding(BaseModel):
    """Structured reading of the latest message. No business logic, no DB."""

    intents: list[Intent] = Field(
        default_factory=list,
        description=(
            "Every intent that genuinely, independently applies to this message -- a run-on "
            "message may carry several (e.g. recommend + discuss + write a post). Order by "
            "prominence, most central first. Empty only if nothing actionable applies."
        ),
    )
    child_ref: ChildRef = Field(default_factory=ChildRef)
    mentioned_books: list[MentionedBook] = Field(default_factory=list)
    user_signals: list[UserSignal] = Field(default_factory=list)

    @model_validator(mode="after")
    def _dedupe_intents(self) -> Understanding:
        seen: set[Intent] = set()
        deduped: list[Intent] = []
        for intent in self.intents:
            if intent not in seen:
                seen.add(intent)
                deduped.append(intent)
        self.intents = deduped
        return self


# --- plan -------------------------------------------------------------------------------


class PlanStep(BaseModel):
    """One capability to run, with any capabilities it depends on first."""

    capability: str  # must be a name registered in capabilities.registry
    depends_on: list[str] = Field(default_factory=list)
    reason: str = ""


class Plan(BaseModel):
    """Ordered capabilities to execute this turn (MVP: at most two; may be empty)."""

    steps: list[PlanStep] = Field(default_factory=list)


# --- clarify ----------------------------------------------------------------------------


class ClarificationDecision(BaseModel):
    """Whether to proceed, ask the user something, or run with assumptions."""

    decision: Literal["continue", "ask_user", "best_effort"]
    missing_inputs: list[str] = Field(default_factory=list)
    question: str | None = None  # set iff decision == "ask_user"
    assumptions: list[str] = Field(default_factory=list)  # set iff decision == "best_effort"


# --- memory -----------------------------------------------------------------------------


class MemoryOperation(BaseModel):
    """A domain-level operation to persist (executed by the profile_update agent's tools)."""

    operation: str  # domain operation name, e.g. "UpdateReadingInterest"
    arguments: dict = Field(default_factory=dict)
    rationale: str = ""


class MemoryDecision(BaseModel):
    """What this turn decided is worth remembering long-term."""

    operations: list[MemoryOperation] = Field(default_factory=list)
