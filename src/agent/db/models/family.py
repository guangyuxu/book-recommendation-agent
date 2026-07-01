"""Family domain: the household, its members (+ agent-curated profiles), and reading policy.

Ownership split mirrors the child domain: FamilyMember holds backend-owned identity, while
FamilyMemberProfile holds the agent's conversation-extracted background (with provenance).
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
    Numeric,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import Base, JSONType, TextArray
from ._columns import _created_at, _updated_at, _uuid_pk

if TYPE_CHECKING:
    from .child import ChildProfile


class Family(Base):
    """A household -- the login identity (family_id). Owns members and children."""

    __tablename__ = "family"

    id: Mapped[uuid.UUID] = _uuid_pk()
    family_name: Mapped[str | None] = mapped_column(Text)
    default_language: Mapped[str | None] = mapped_column(
        Text, server_default=text("'us-EN'")
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
    """A person in the family (parent/caregiver). The request's family_member_id.

    Identity / registration only -- owned by the backend (set at sign-up, including invite-code
    joins). Soft, conversation-extracted background (occupation, communication style, concerns)
    lives in the 1:1 FamilyMemberProfile, the agent's write space.
    """

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
    language_preference: Mapped[str | None] = mapped_column(
        Text, server_default=text("'zh-CN'")
    )
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    family: Mapped[Family] = relationship(
        back_populates="members", lazy="selectin"
    )
    profile: Mapped[FamilyMemberProfile | None] = relationship(
        back_populates="member",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="selectin",
    )

    __table_args__ = (Index("idx_family_member_family_id", "family_id"),)


class FamilyMemberProfile(Base):
    """Agent-curated background for one member (UNIQUE member_id): occupation, style, concerns.

    The member counterpart of ChildReadingProfile: filled and refined by the agent from
    conversation, with source/confidence/observed_at provenance. Never written at registration.
    """

    __tablename__ = "family_member_profile"

    id: Mapped[uuid.UUID] = _uuid_pk()
    member_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("family_member.id"), nullable=False, unique=True
    )
    occupation_background: Mapped[str | None] = mapped_column(Text)
    education_background: Mapped[str | None] = mapped_column(Text)
    communication_style: Mapped[str | None] = mapped_column(Text)
    concerns: Mapped[list[str]] = mapped_column(TextArray, server_default=text("'{}'"))
    source: Mapped[str | None] = mapped_column(
        Text, server_default=text("'parent_report'")
    )
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(3, 2))
    observed_at: Mapped[datetime | None] = mapped_column(
        DateTime, server_default=func.now()
    )
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    member: Mapped[FamilyMember] = relationship(
        back_populates="profile", lazy="selectin"
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
