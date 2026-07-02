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
    """One profile-relevant fact the user revealed (fuel for Memory Policy)."""

    kind: Literal[
        "interest",
        "ability",
        "finished_book",
        "disliked_book",
        "current_reading",
        "goal",
        "constraint",
        "other",
    ]
    detail: str


class Understanding(BaseModel):
    """Structured reading of the latest message. No business logic, no DB."""

    primary_intent: Intent
    secondary_intent: Intent | None = None  # at most one; must differ from primary
    target_child_id: str | None = None  # resolved roster id, if determinable
    child_is_new: bool = False  # message describes a child not on the roster
    child_ambiguous: bool = False  # a child is needed but which one is unclear
    mentioned_books: list[MentionedBook] = Field(default_factory=list)
    user_signals: list[UserSignal] = Field(default_factory=list)

    @model_validator(mode="after")
    def _at_most_two_distinct_intents(self) -> Understanding:
        if self.secondary_intent == self.primary_intent:
            self.secondary_intent = None
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
