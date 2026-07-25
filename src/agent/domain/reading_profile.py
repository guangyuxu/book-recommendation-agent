"""Reading Profile domain operations: ability, interests, genre/theme tastes, summary.

All act on the target child's single reading profile via the accounts internal API. List edits
(interests, genres, themes, ...) are add/remove merges computed against the turn's cached profile,
then the full field is sent as a reading-profile upsert; the cache is refreshed from the response.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.tools import tool

from ._util import merge_unique, remove_all
from .context import accounts, cached_child, current, require_child_id


def _cached_profile(child_id: UUID | str) -> dict[str, Any]:
    """Return the target child's cached reading_profile (for list merges), or {}."""
    return cached_child(child_id).get("reading_profile") or {}


def _upsert(child_id: UUID | str, body: dict[str, Any]) -> None:
    """Send a reading-profile upsert and refresh the cached profile from the response."""
    updated = accounts().upsert_reading_profile(current().family_id, child_id, body)
    key = str(child_id)
    prior = current().children.get(key, {})
    current().children[key] = {**prior, "reading_profile": updated}


@tool
def update_reading_ability(
    reading_level_note: str | None = None,
    cefr_level: str | None = None,
    lexile: int | None = None,
    ar_level: float | None = None,
    current_stage: str | None = None,
    independent_reading: bool | None = None,
    needs_dictionary: bool | None = None,
    can_read_chapter_books: bool | None = None,
    can_handle_old_language: bool | None = None,
    confidence: float | None = None,
    source: str = "parent_report",
) -> str:
    """Update the target child's reading ability (level, CEFR/Lexile/AR, stage, capabilities)."""
    child_id = require_child_id()
    body: dict[str, Any] = {"source": source}
    for field_name, value in (
        ("reading_level_note", reading_level_note),
        ("cefr_level", cefr_level),
        ("lexile", lexile),
        ("ar_level", ar_level),
        ("current_stage", current_stage),
        ("independent_reading", independent_reading),
        ("needs_dictionary", needs_dictionary),
        ("can_read_chapter_books", can_read_chapter_books),
        ("can_handle_old_language", can_handle_old_language),
        ("confidence", confidence),
    ):
        if value is not None:
            body[field_name] = value
    _upsert(child_id, body)
    return f"Updated reading ability for child {child_id}."


@tool
def update_reading_interest(
    add_interests: list[str] | None = None,
    remove_interests: list[str] | None = None,
) -> str:
    """Add or remove topics the target child is interested in (e.g. dragons, space, sports)."""
    child_id = require_child_id()
    prof = _cached_profile(child_id)
    interests = remove_all(
        merge_unique(prof.get("interests"), add_interests), remove_interests
    )
    _upsert(child_id, {"interests": interests})
    return f"Updated interests for child {child_id}."


@tool
def update_genre_preference(
    add_preferred: list[str] | None = None,
    remove_preferred: list[str] | None = None,
    add_disliked: list[str] | None = None,
    remove_disliked: list[str] | None = None,
) -> str:
    """Update the target child's preferred and disliked genres."""
    child_id = require_child_id()
    prof = _cached_profile(child_id)
    body = {
        "preferred_genres": remove_all(
            merge_unique(prof.get("preferred_genres"), add_preferred), remove_preferred
        ),
        "disliked_genres": remove_all(
            merge_unique(prof.get("disliked_genres"), add_disliked), remove_disliked
        ),
    }
    _upsert(child_id, body)
    return f"Updated genre preferences for child {child_id}."


@tool
def update_theme_tone_preference(
    add_liked_themes: list[str] | None = None,
    add_disliked_themes: list[str] | None = None,
    add_preferred_tone: list[str] | None = None,
    add_avoid_topics: list[str] | None = None,
) -> str:
    """Add to the target child's liked/disliked themes, preferred tone, and topics to avoid."""
    child_id = require_child_id()
    prof = _cached_profile(child_id)
    body = {
        "liked_themes": merge_unique(prof.get("liked_themes"), add_liked_themes),
        "disliked_themes": merge_unique(
            prof.get("disliked_themes"), add_disliked_themes
        ),
        "preferred_tone": merge_unique(prof.get("preferred_tone"), add_preferred_tone),
        "avoid_topics": merge_unique(prof.get("avoid_topics"), add_avoid_topics),
    }
    _upsert(child_id, body)
    return f"Updated themes/tone for child {child_id}."


@tool
def update_reading_summary(summary: str) -> str:
    """Set a short natural-language summary of the target child's reading profile."""
    child_id = require_child_id()
    _upsert(child_id, {"summary": summary})
    return f"Updated reading summary for child {child_id}."
