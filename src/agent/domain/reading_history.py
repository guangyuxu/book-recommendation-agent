"""Reading History domain operations: record finished / current / disliked books.

Wraps ReadingHistoryRepository. Each book is upserted by (target child, title) so repeated
mentions update the same row rather than duplicating it.
"""

from __future__ import annotations

from datetime import date
from uuid import uuid4

from langchain_core.tools import tool

from ..db import ReadingHistory, ReadingHistoryRepository
from .context import current, require_child_id
from .util import merge_unique


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _upsert(title: str, author: str | None, **fields) -> ReadingHistory:
    ctx = current()
    repo = ReadingHistoryRepository(session=ctx.session)
    child_id = require_child_id()
    row = repo.get_one_or_none(child_id=child_id, title=title)
    if row is None:
        row = ReadingHistory(id=uuid4(), child_id=child_id, title=title, author=author)
        for key, value in fields.items():
            if value is not None:
                setattr(row, key, value)
        repo.add(row)
    else:
        if author is not None:
            row.author = author
        for key, value in fields.items():
            if value is not None:
                setattr(row, key, value)
        repo.update(row)
    return row


@tool
def record_finished_book(
    title: str,
    author: str | None = None,
    series_name: str | None = None,
    book_order: str | None = None,
    liked: bool | None = None,
    reasons: list[str] | None = None,
    parent_note: str | None = None,
    child_note: str | None = None,
    finished_at: str | None = None,
) -> str:
    """Record that the target child finished a book (with whether they liked it and why).

    finished_at is an ISO date string (YYYY-MM-DD) if known.
    """
    row = _upsert(
        title,
        author,
        series_name=series_name,
        book_order=book_order,
        status="finished",
        liked=liked,
        reasons=reasons or [],
        parent_note=parent_note,
        child_note=child_note,
        finished_at=_parse_date(finished_at),
    )
    return f"Recorded finished book '{title}' for child {row.child_id}."


@tool
def record_current_reading(
    title: str,
    author: str | None = None,
    series_name: str | None = None,
    started_at: str | None = None,
) -> str:
    """Record that the target child is currently reading a book.

    started_at is an ISO date string (YYYY-MM-DD) if known.
    """
    row = _upsert(
        title,
        author,
        series_name=series_name,
        status="reading",
        started_at=_parse_date(started_at),
    )
    return f"Recorded current reading '{title}' for child {row.child_id}."


@tool
def record_disliked_book(
    title: str,
    author: str | None = None,
    reasons: list[str] | None = None,
    parent_note: str | None = None,
) -> str:
    """Record that the target child disliked / abandoned a book, with reasons if given."""
    row = _upsert(
        title,
        author,
        status="abandoned",
        liked=False,
        reasons=merge_unique(None, reasons),
        parent_note=parent_note,
    )
    return f"Recorded disliked book '{title}' for child {row.child_id}."
