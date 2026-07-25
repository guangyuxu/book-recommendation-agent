"""Reading History domain operations: record finished / current / disliked books.

Calls the accounts internal API (`ctx.client`). Each book is upserted by (target child, title):
the child's history is listed, matched by title, then created (POST) or updated (PATCH) so
repeated mentions update the same row rather than duplicating it.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ._util import iso_date, merge_unique
from .context import accounts, current, require_child_id


def _find_by_title(entries: list[dict[str, Any]], title: str) -> dict[str, Any] | None:
    for e in entries:
        if e.get("title") == title:
            return e
    return None


def _upsert(title: str, author: str | None, **fields: Any) -> object:
    ctx = current()
    child_id = require_child_id()
    entries = accounts().list_reading_history(ctx.family_id, child_id)
    existing = _find_by_title(entries, title)

    body: dict[str, Any] = {}
    if author is not None:
        body["author"] = author
    for key, value in fields.items():
        if value is not None:
            body[key] = value

    if existing is None:
        body["title"] = title
        accounts().create_reading_history(ctx.family_id, child_id, body)
    else:
        # Merge list fields against the existing row (reasons accumulate, deduped).
        if "reasons" in body:
            body["reasons"] = merge_unique(existing.get("reasons"), body["reasons"])
        accounts().update_reading_history(ctx.family_id, child_id, existing["id"], body)
    return child_id


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
    child_id = _upsert(
        title,
        author,
        series_name=series_name,
        book_order=book_order,
        status="finished",
        liked=liked,
        reasons=reasons or [],
        parent_note=parent_note,
        child_note=child_note,
        finished_at=iso_date(finished_at),
    )
    return f"Recorded finished book '{title}' for child {child_id}."


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
    child_id = _upsert(
        title,
        author,
        series_name=series_name,
        status="reading",
        started_at=iso_date(started_at),
    )
    return f"Recorded current reading '{title}' for child {child_id}."


@tool
def record_disliked_book(
    title: str,
    author: str | None = None,
    reasons: list[str] | None = None,
    parent_note: str | None = None,
) -> str:
    """Record that the target child disliked / abandoned a book, with reasons if given."""
    child_id = _upsert(
        title,
        author,
        status="abandoned",
        liked=False,
        reasons=merge_unique(None, reasons),
        parent_note=parent_note,
    )
    return f"Recorded disliked book '{title}' for child {child_id}."
