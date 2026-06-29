"""ORM models for the book_agent schema (10 tables), hand-written to mirror the live DB.

Domains:
- Identity:  family -> family_member, family -> child_profile
- Profiles:  child_reading_profile (1:1 child), family_reading_policy, reading_history
- Catalog:   book_cache
- Recommendation: recommendation_session -> recommendation_item, recommendation_feedback

Conventions shared by every table:
- UUID primary key, generated server-side via gen_random_uuid().
- created_at / updated_at TIMESTAMP, defaulted server-side to now(); updated_at also bumped
  ORM-side via onupdate so writes through the ORM keep it fresh.
- text[] columns use TextArray and default to an empty array; JSONB columns default to {}.

Tables already exist in the DB; create_all is only a local/dev fallback (see base.init_db).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, JSONType, TextArray

# Reusable column definitions ------------------------------------------------------------


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime, nullable=False, server_default=func.now())


def _updated_at() -> Mapped[datetime]:
    return mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


# Identity -------------------------------------------------------------------------------


class Family(Base):
    """A household -- the login identity (family_id). Owns members and children."""

    __tablename__ = "family"

    id: Mapped[uuid.UUID] = _uuid_pk()
    family_name: Mapped[str | None] = mapped_column(Text)
    default_language: Mapped[str | None] = mapped_column(
        Text, server_default=text("'zh-CN'")
    )
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    members: Mapped[list[FamilyMember]] = relationship(
        back_populates="family", cascade="all, delete-orphan", lazy="selectin"
    )
    children: Mapped[list[ChildProfile]] = relationship(
        back_populates="family", cascade="all, delete-orphan", lazy="selectin"
    )
    policies: Mapped[list[FamilyReadingPolicy]] = relationship(
        back_populates="family", cascade="all, delete-orphan", lazy="selectin"
    )


class FamilyMember(Base):
    """A person in the family (parent/caregiver). The request's family_member_id."""

    __tablename__ = "family_member"

    id: Mapped[uuid.UUID] = _uuid_pk()
    family_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("family.id"), nullable=False
    )
    display_name: Mapped[str | None] = mapped_column(Text)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    is_primary_user: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    occupation_background: Mapped[str | None] = mapped_column(Text)
    education_background: Mapped[str | None] = mapped_column(Text)
    communication_style: Mapped[str | None] = mapped_column(Text)
    concerns: Mapped[list[str]] = mapped_column(TextArray, server_default=text("'{}'"))
    language_preference: Mapped[str | None] = mapped_column(
        Text, server_default=text("'zh-CN'")
    )
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    family: Mapped[Family] = relationship(
        back_populates="members", lazy="selectin"
    )

    __table_args__ = (Index("idx_family_member_family_id", "family_id"),)


class ChildProfile(Base):
    """A child in the family. The request's current_child_id / target_child_id."""

    __tablename__ = "child_profile"

    id: Mapped[uuid.UUID] = _uuid_pk()
    family_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("family.id"), nullable=False
    )
    display_name: Mapped[str | None] = mapped_column(Text)
    aliases: Mapped[list[str]] = mapped_column(TextArray, server_default=text("'{}'"))
    birth_year: Mapped[int | None] = mapped_column(Integer)
    age: Mapped[int | None] = mapped_column(Integer)
    grade: Mapped[str | None] = mapped_column(Text)
    school_system: Mapped[str | None] = mapped_column(Text)
    country_or_curriculum: Mapped[str | None] = mapped_column(Text)
    primary_language: Mapped[str | None] = mapped_column(Text)
    reading_language: Mapped[str | None] = mapped_column(
        Text, server_default=text("'English'")
    )
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    family: Mapped[Family] = relationship(
        back_populates="children", lazy="selectin"
    )
    reading_profile: Mapped[ChildReadingProfile | None] = relationship(
        back_populates="child",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="selectin",
    )
    reading_history: Mapped[list[ReadingHistory]] = relationship(
        back_populates="child", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (Index("idx_child_profile_family_id", "family_id"),)


# Profiles -------------------------------------------------------------------------------


class ChildReadingProfile(Base):
    """One reading profile per child (UNIQUE child_id): level, interests, tastes."""

    __tablename__ = "child_reading_profile"

    id: Mapped[uuid.UUID] = _uuid_pk()
    child_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("child_profile.id"), nullable=False, unique=True
    )
    reading_level_note: Mapped[str | None] = mapped_column(Text)
    cefr_level: Mapped[str | None] = mapped_column(Text)
    lexile: Mapped[int | None] = mapped_column(Integer)
    ar_level: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    current_stage: Mapped[str | None] = mapped_column(Text)
    independent_reading: Mapped[bool | None] = mapped_column(Boolean)
    needs_dictionary: Mapped[bool | None] = mapped_column(Boolean)
    can_read_chapter_books: Mapped[bool | None] = mapped_column(Boolean)
    can_handle_old_language: Mapped[bool | None] = mapped_column(Boolean)
    interests: Mapped[list[str]] = mapped_column(TextArray, server_default=text("'{}'"))
    preferred_genres: Mapped[list[str]] = mapped_column(
        TextArray, server_default=text("'{}'")
    )
    disliked_genres: Mapped[list[str]] = mapped_column(
        TextArray, server_default=text("'{}'")
    )
    liked_themes: Mapped[list[str]] = mapped_column(
        TextArray, server_default=text("'{}'")
    )
    disliked_themes: Mapped[list[str]] = mapped_column(
        TextArray, server_default=text("'{}'")
    )
    preferred_tone: Mapped[list[str]] = mapped_column(
        TextArray, server_default=text("'{}'")
    )
    avoid_topics: Mapped[list[str]] = mapped_column(
        TextArray, server_default=text("'{}'")
    )
    summary: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(3, 2))
    source: Mapped[str | None] = mapped_column(
        Text, server_default=text("'parent_report'")
    )
    observed_at: Mapped[datetime | None] = mapped_column(
        DateTime, server_default=func.now()
    )
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    child: Mapped[ChildProfile] = relationship(
        back_populates="reading_profile", lazy="selectin"
    )


class FamilyReadingPolicy(Base):
    """Reading goals/constraints scoped to a family, optionally narrowed to one child."""

    __tablename__ = "family_reading_policy"

    id: Mapped[uuid.UUID] = _uuid_pk()
    family_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("family.id"), nullable=False
    )
    child_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("child_profile.id"))
    goals: Mapped[list[str]] = mapped_column(TextArray, server_default=text("'{}'"))
    constraints: Mapped[list[str]] = mapped_column(
        TextArray, server_default=text("'{}'")
    )
    avoid_topics: Mapped[list[str]] = mapped_column(
        TextArray, server_default=text("'{}'")
    )
    content_preferences: Mapped[dict] = mapped_column(
        JSONType, server_default=text("'{}'")
    )
    notes: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    family: Mapped[Family] = relationship(
        back_populates="policies", lazy="selectin"
    )
    child: Mapped[ChildProfile | None] = relationship(lazy="selectin")

    __table_args__ = (
        Index("idx_family_reading_policy_family_id", "family_id"),
        Index("idx_family_reading_policy_child_id", "child_id"),
    )


class ReadingHistory(Base):
    """A book a child has read / is reading, with the family's reaction to it."""

    __tablename__ = "reading_history"

    id: Mapped[uuid.UUID] = _uuid_pk()
    child_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("child_profile.id"), nullable=False
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str | None] = mapped_column(Text)
    series_name: Mapped[str | None] = mapped_column(Text)
    book_order: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    liked: Mapped[bool | None] = mapped_column(Boolean)
    reasons: Mapped[list[str]] = mapped_column(TextArray, server_default=text("'{}'"))
    parent_note: Mapped[str | None] = mapped_column(Text)
    child_note: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[date | None] = mapped_column(Date)
    finished_at: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    child: Mapped[ChildProfile] = relationship(
        back_populates="reading_history", lazy="selectin"
    )

    __table_args__ = (
        Index("idx_reading_history_child_id", "child_id"),
        Index("idx_reading_history_title", "title"),
    )


# Catalog --------------------------------------------------------------------------------


class BookCache(Base):
    """Cached book metadata (OpenLibrary/Google + LLM summary). Unique by (title, author)."""

    __tablename__ = "book_cache"

    id: Mapped[uuid.UUID] = _uuid_pk()
    title: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str | None] = mapped_column(Text)
    series_name: Mapped[str | None] = mapped_column(Text)
    book_order: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    categories: Mapped[list[str]] = mapped_column(
        TextArray, server_default=text("'{}'")
    )
    subjects: Mapped[list[str]] = mapped_column(TextArray, server_default=text("'{}'"))
    isbn_10: Mapped[str | None] = mapped_column(Text)
    isbn_13: Mapped[str | None] = mapped_column(Text)
    openlibrary_work_key: Mapped[str | None] = mapped_column(Text)
    google_volume_id: Mapped[str | None] = mapped_column(Text)
    cover_url: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    published_year: Mapped[int | None] = mapped_column(Integer)
    language: Mapped[str | None] = mapped_column(Text, server_default=text("'en'"))
    llm_summary: Mapped[dict] = mapped_column(JSONType, server_default=text("'{}'"))
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    __table_args__ = (
        Index("book_cache_title_author_key", "title", "author", unique=True),
        Index("idx_book_cache_openlibrary_work_key", "openlibrary_work_key"),
        Index("idx_book_cache_title", "title"),
    )


# Recommendation -------------------------------------------------------------------------


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
