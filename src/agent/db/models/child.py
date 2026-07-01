"""Child domain: a child's identity/facts (ChildProfile) and agent-curated reading profile.

ChildProfile holds the backend/form-owned skeleton (name, age, grade); ChildReadingProfile
holds the agent's conversation-extracted reading picture (level, interests, tastes), with
provenance -- the pattern the member domain mirrors.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import Base, TextArray
from ._columns import _created_at, _updated_at, _uuid_pk

if TYPE_CHECKING:
    from .family import Family
    from .reading import ReadingHistory


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
