"""Book Knowledge domain operations: cache book metadata and LLM summaries.

Wraps BookCacheRepository. book_cache is a cache (not a complete catalog); rows are upserted
by their unique (title, author) key.
"""

from __future__ import annotations

from uuid import uuid4

from langchain_core.tools import tool

from ..db import BookCache, BookCacheRepository
from .context import current


def _get_or_create(repo: BookCacheRepository, title: str, author: str | None) -> tuple[BookCache, bool]:
    row = repo.get_by_title_author(title, author)
    if row is not None:
        return row, False
    return BookCache(id=uuid4(), title=title, author=author), True


@tool
def cache_book(
    title: str,
    author: str | None = None,
    series_name: str | None = None,
    book_order: str | None = None,
    description: str | None = None,
    categories: list[str] | None = None,
    subjects: list[str] | None = None,
    isbn_10: str | None = None,
    isbn_13: str | None = None,
    cover_url: str | None = None,
    page_count: int | None = None,
    published_year: int | None = None,
    language: str = "en",
) -> str:
    """Cache a book's metadata (upserts by title+author). Use for books worth remembering."""
    repo = BookCacheRepository(session=current().session)
    row, created = _get_or_create(repo, title, author)
    for field, value in (
        ("series_name", series_name),
        ("book_order", book_order),
        ("description", description),
        ("categories", categories),
        ("subjects", subjects),
        ("isbn_10", isbn_10),
        ("isbn_13", isbn_13),
        ("cover_url", cover_url),
        ("page_count", page_count),
        ("published_year", published_year),
        ("language", language),
    ):
        if value is not None:
            setattr(row, field, value)
    repo.add(row) if created else repo.update(row)
    return f"Cached book '{title}' ({row.id})."


@tool
def update_book_metadata(
    title: str,
    author: str | None = None,
    description: str | None = None,
    categories: list[str] | None = None,
    subjects: list[str] | None = None,
    page_count: int | None = None,
    published_year: int | None = None,
) -> str:
    """Update metadata of an already-cached book (by title+author). Only given fields change."""
    repo = BookCacheRepository(session=current().session)
    row = repo.get_by_title_author(title, author)
    if row is None:
        return f"No cached book '{title}' to update."
    for field, value in (
        ("description", description),
        ("categories", categories),
        ("subjects", subjects),
        ("page_count", page_count),
        ("published_year", published_year),
    ):
        if value is not None:
            setattr(row, field, value)
    repo.update(row)
    return f"Updated metadata for '{title}'."


@tool
def update_book_summary(title: str, llm_summary: dict, author: str | None = None) -> str:
    """Store an LLM-generated structured summary for a cached book (upserts the book if new)."""
    repo = BookCacheRepository(session=current().session)
    row, created = _get_or_create(repo, title, author)
    row.llm_summary = llm_summary or {}
    repo.add(row) if created else repo.update(row)
    return f"Updated summary for '{title}'."
