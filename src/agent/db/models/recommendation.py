"""Recommendation domain: a turn's session, its recommended items, and later feedback.

Owns session -> item -> feedback. References the book cache by FK (a local table). The family /
child / member it refers to now live in the accounts service, so those are PLAIN UUID columns
(no cross-service foreign keys, no ORM relationships) -- integrity for them is enforced by the
accounts service, not the agent DB.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Index, Integer, Text, Uuid, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import Base, JSONType, TextArray
from ._columns import _created_at, _uuid_pk

if TYPE_CHECKING:
    from .book import BookCache


class RecommendationSession(Base):
    """One recommendation turn: the request, its understanding/plan, and the response."""

    __tablename__ = "recommendation_session"

    id: Mapped[uuid.UUID] = _uuid_pk()
    # Accounts-owned identities: plain UUIDs, no FK (the rows live in another service).
    family_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    requester_member_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    target_child_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    intents: Mapped[list[Any]] = mapped_column(JSONType, server_default=text("'[]'"))
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    understanding: Mapped[dict[str, Any]] = mapped_column(
        JSONType, server_default=text("'{}'")
    )
    plan: Mapped[dict[str, Any]] = mapped_column(JSONType, server_default=text("'{}'"))
    capability_result: Mapped[dict[str, Any]] = mapped_column(
        JSONType, server_default=text("'{}'")
    )
    memory_decision: Mapped[dict[str, Any]] = mapped_column(
        JSONType, server_default=text("'{}'")
    )
    response_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()

    items: Mapped[list[RecommendationItem]] = relationship(
        back_populates="session", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        Index("idx_recommendation_session_family_id", "family_id"),
        Index("idx_recommendation_session_target_child_id", "target_child_id"),
    )


class RecommendationItem(Base):
    """One recommended book within a session."""

    __tablename__ = "recommendation_item"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("recommendation_session.id"), nullable=False
    )
    book_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("book_cache.id"))
    title: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str | None] = mapped_column(Text)
    rank: Mapped[int | None] = mapped_column(Integer)
    recommendation_reason: Mapped[str | None] = mapped_column(Text)
    fit_summary: Mapped[str | None] = mapped_column(Text)
    risk_notes: Mapped[list[str]] = mapped_column(
        TextArray, server_default=text("'{}'")
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONType, server_default=text("'{}'")
    )
    created_at: Mapped[datetime] = _created_at()

    session: Mapped[RecommendationSession] = relationship(
        back_populates="items", lazy="selectin"
    )
    book: Mapped[BookCache | None] = relationship(lazy="selectin")

    __table_args__ = (Index("idx_recommendation_item_session_id", "session_id"),)


class RecommendationFeedback(Base):
    """Parent/child reaction to a recommended item or a whole session."""

    __tablename__ = "recommendation_feedback"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("recommendation_session.id")
    )
    recommendation_item_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("recommendation_item.id")
    )
    # Accounts-owned identities: plain UUIDs, no FK (the rows live in another service).
    family_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    child_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    member_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    reaction: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'unknown'")
    )
    parent_note: Mapped[str | None] = mapped_column(Text)
    child_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()

    session: Mapped[RecommendationSession | None] = relationship(lazy="selectin")
    item: Mapped[RecommendationItem | None] = relationship(lazy="selectin")

    __table_args__ = (
        Index("idx_recommendation_feedback_session_id", "session_id"),
        Index("idx_recommendation_feedback_child_id", "child_id"),
    )
