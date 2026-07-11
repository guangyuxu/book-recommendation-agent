"""Recommendation domain operations: persist a recommendation turn and its items/feedback.

Wraps RecommendationSessionRepository / RecommendationItemRepository /
RecommendationFeedbackRepository (and BookCacheRepository to link items to cached books).
These are invoked deterministically by the respond node, not by the memory agent, so that
persistence never depends on LLM discretion.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..db import (
    BookCache,
    BookCacheRepository,
    RecommendationFeedback,
    RecommendationFeedbackRepository,
    RecommendationItem,
    RecommendationItemRepository,
    RecommendationSession,
    RecommendationSessionRepository,
)
from .context import current, require_child_id


class RecItemInput(BaseModel):
    """One recommended book to persist within a recommendation session."""

    title: str
    author: str | None = None
    rank: int | None = None
    recommendation_reason: str | None = None
    fit_summary: str | None = None
    risk_notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict[str, Any])


def _as_uuid(value: str | UUID | None) -> UUID | None:
    if value is None:
        return None
    return value if isinstance(value, UUID) else UUID(str(value))


@tool
def create_recommendation_session(
    user_message: str,
    intents: list[str] | None = None,
    understanding: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    capability_result: dict[str, Any] | None = None,
    memory_decision: dict[str, Any] | None = None,
    response_text: str | None = None,
    requester_member_id: str | None = None,
) -> str:
    """Persist a recommendation turn (the whole request/understanding/plan/result). Returns its id."""
    ctx = current()
    session = RecommendationSession(
        id=uuid4(),
        family_id=ctx.family_id,
        requester_member_id=_as_uuid(requester_member_id),
        target_child_id=require_child_id(),
        intents=intents or [],
        user_message=user_message,
        understanding=understanding or {},
        plan=plan or {},
        capability_result=capability_result or {},
        memory_decision=memory_decision or {},
        response_text=response_text,
    )
    RecommendationSessionRepository(session=ctx.session).add(session)
    return str(session.id)


@tool
def save_recommendation_items(session_id: str, items: list[RecItemInput]) -> str:
    """Persist the recommended books for a session, linking each to a cached book row."""
    ctx = current()
    book_repo = BookCacheRepository(session=ctx.session)
    item_repo = RecommendationItemRepository(session=ctx.session)
    saved = 0
    for item in items:
        rec = (
            item
            if isinstance(item, RecItemInput)
            else RecItemInput.model_validate(item)
        )
        book = book_repo.get_by_title_author(rec.title, rec.author)
        if book is None:
            book = BookCache(id=uuid4(), title=rec.title, author=rec.author)
            book_repo.add(book)
        item_repo.add(
            RecommendationItem(
                id=uuid4(),
                session_id=_as_uuid(session_id),
                book_id=book.id,
                title=rec.title,
                author=rec.author,
                rank=rec.rank,
                recommendation_reason=rec.recommendation_reason,
                fit_summary=rec.fit_summary,
                risk_notes=rec.risk_notes,
                metadata_=rec.metadata,
            )
        )
        saved += 1
    return f"Saved {saved} recommendation item(s) for session {session_id}."


@tool
def record_recommendation_feedback(
    reaction: str,
    session_id: str | None = None,
    recommendation_item_id: str | None = None,
    parent_note: str | None = None,
    child_note: str | None = None,
    member_id: str | None = None,
) -> str:
    """Record the family's reaction to a recommended item or session for the target child."""
    ctx = current()
    feedback = RecommendationFeedback(
        id=uuid4(),
        session_id=_as_uuid(session_id),
        recommendation_item_id=_as_uuid(recommendation_item_id),
        family_id=ctx.family_id,
        child_id=require_child_id(),
        member_id=_as_uuid(member_id),
        reaction=reaction,
        parent_note=parent_note,
        child_note=child_note,
    )
    RecommendationFeedbackRepository(session=ctx.session).add(feedback)
    return f"Recorded recommendation feedback ({reaction})."
