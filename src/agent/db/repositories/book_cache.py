"""Repository for the `book_cache` table (cached book metadata)."""

from __future__ import annotations

from advanced_alchemy.repository import SQLAlchemySyncRepository

from ..models import BookCache


class BookCacheRepository(SQLAlchemySyncRepository[BookCache]):
    model_type = BookCache

    def get_by_title_author(self, title: str, author: str | None) -> BookCache | None:
        """Look up a cached book by its unique (title, author) key."""
        author_filter = (
            BookCache.author == author
            if author is not None
            else BookCache.author.is_(None)
        )
        return self.get_one_or_none(BookCache.title == title, author_filter)
