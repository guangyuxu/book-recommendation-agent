"""Recommend Books capability: LLM proposes a fitted booklist for the target child.

MVP is LLM-only -- no retrieval/ranking/vector search. Returns a structured booklist so the
respond node can both render it and persist it (recommendation_session + items).
"""

from __future__ import annotations

from typing import Any, cast

from langchain.messages import SystemMessage
from pydantic import BaseModel, Field

from ..llm import model
from ._shared import child_brief, policies_brief


class RecommendedBook(BaseModel):
    """One recommended book with the reasoning the parent will see."""

    title: str
    author: str | None = None
    recommendation_reason: str
    fit_summary: str | None = None
    risk_notes: list[str] = Field(default_factory=list)


class Booklist(BaseModel):
    """The recommendation result: a ranked list of books plus an optional framing note."""

    books: list[RecommendedBook] = Field(default_factory=list)
    note: str | None = None


_structured = model.with_structured_output(Booklist)


def run(state: dict[str, Any]) -> dict[str, Any]:
    """Produce a fitted, English booklist for the target child."""
    system = SystemMessage(
        content=(
            "You are a children's-book recommendation expert. Recommend English books that "
            "fit this child's reading level, interests, and the family's goals/constraints. "
            "Rank them best-first, give a concrete reason per book, and flag any content "
            "risks. Recommend 3-5 books.\n\n"
            f"Target child profile:\n{child_brief(state)}\n\n"
            f"Family reading policies:\n{policies_brief(state)}"
        )
    )
    result = cast(Booklist, _structured.invoke([system, *state["messages"]]))
    return result.model_dump()
