"""ORM models for three tables: chat_history (chat log), parent_profiles, child_profiles."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, JSONType


class ChatMessage(Base):
    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    thread_id: Mapped[str] = mapped_column(String(64), index=True)  # LangGraph thread_id
    role: Mapped[str] = mapped_column(String(16))  # human / ai / system / tool
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Common query: fetch a thread's messages ordered by time.
    __table_args__ = (Index("ix_chat_thread_time", "thread_id", "created_at"),)


class ParentProfile(Base):
    """Long-term parent profile -- the source of truth for profile data."""

    __tablename__ = "parent_profiles"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    available_time: Mapped[str | None] = mapped_column(String(128))
    self_taste: Mapped[str | None] = mapped_column(String(256))
    parent_goals: Mapped[list] = mapped_column(JSONType, default=list)  # dedup-appended goals
    extra: Mapped[dict] = mapped_column(JSONType, default=dict)  # other flexible fields
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    children: Mapped[list[ChildProfile]] = relationship(
        back_populates="parent", cascade="all, delete-orphan"
    )


class ChildProfile(Base):
    """Long-term child profile; linked to the parent via parent_user_id."""

    __tablename__ = "child_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    parent_user_id: Mapped[str] = mapped_column(
        ForeignKey("parent_profiles.user_id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str | None] = mapped_column(String(64))
    reading_level: Mapped[str | None] = mapped_column(String(64))
    recent_signal: Mapped[str | None] = mapped_column(String(256))
    extra: Mapped[dict] = mapped_column(JSONType, default=dict)  # other flexible fields
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    parent: Mapped[ParentProfile] = relationship(back_populates="children")
