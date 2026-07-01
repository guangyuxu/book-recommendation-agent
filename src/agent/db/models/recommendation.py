"""Recommendation domain: a turn's session, its recommended items, and later feedback.

References the family, child, and book domains by FK; owns session -> item -> feedback.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, Integer, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import Base, JSONType, TextArray
from ._columns import _created_at, _uuid_pk

if TYPE_CHECKING:
    from .book import BookCache
    from .child import ChildProfile
    from .family import Family, FamilyMember


class RecommendationSession(Base):
    """One recommendation turn: the request, its understanding/plan, and the response."""

    __tablename__ = "recommendation_session"

    id: Mapped[uuid.UUID] = _uuid_pk()
    family_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("family.id"), nullable=False
    )
    requester_member_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("family_member.id")
    )
    target_child_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("child_profile.id"), nullable=False
    )
    primary_intent: Mapped[str] = mapped_column(Text, nullable=False)
    secondary_intent: Mapped[str | None] = mapped_column(Text)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    understanding: Mapped[dict] = mapped_column(JSONType, server_default=text("'{}'"))
    plan: Mapped[dict] = mapped_column(JSONType, server_default=text("'{}'"))
    capability_result: Mapped[dict] = mapped_column(
        JSONType, server_default=text("'{}'")
    )
    memory_decision: Mapped[dict] = mapped_column(JSONType, server_default=text("'{}'"))
    response_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()

    family: Mapped[Family] = relationship(lazy="selectin")
    requester: Mapped[FamilyMember | None] = relationship(lazy="selectin")
    target_child: Mapped[ChildProfile] = relationship(lazy="selectin")
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
    risk_notes: Mapped[list[str]] = mapped_column(TextArray, server_default=text("'{}'"))
    metadata_: Mapped[dict] = mapped_column(
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
    family_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("family.id"), nullable=False
    )
    child_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("child_profile.id"), nullable=False
    )
    member_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("family_member.id"))
    reaction: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'unknown'")
    )
    parent_note: Mapped[str | None] = mapped_column(Text)
    child_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()

    session: Mapped[RecommendationSession | None] = relationship(lazy="selectin")
    item: Mapped[RecommendationItem | None] = relationship(lazy="selectin")
    family: Mapped[Family] = relationship(lazy="selectin")
    child: Mapped[ChildProfile] = relationship(lazy="selectin")
    member: Mapped[FamilyMember | None] = relationship(lazy="selectin")

    __table_args__ = (
        Index("idx_recommendation_feedback_session_id", "session_id"),
        Index("idx_recommendation_feedback_child_id", "child_id"),
    )
