"""Catalog domain: cached book metadata (standalone -- no FK to any other domain)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Index, Integer, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base, JSONType, TextArray
from ._columns import _created_at, _updated_at, _uuid_pk


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
