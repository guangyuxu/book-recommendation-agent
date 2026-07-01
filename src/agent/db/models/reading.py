"""Reading domain: a child's reading history (books read / in progress, with reactions)."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Date, ForeignKey, Index, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import Base, TextArray
from ._columns import _created_at, _updated_at, _uuid_pk

if TYPE_CHECKING:
    from .child import ChildProfile


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
