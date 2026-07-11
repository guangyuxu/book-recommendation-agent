"""Reading Profile domain operations: ability, interests, genre/theme tastes, summary.

All act on the target child's single ChildReadingProfile row (created on demand if missing).
Wraps ChildReadingProfileRepository.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

from langchain_core.tools import tool

from ..db import ChildReadingProfile, ChildReadingProfileRepository
from ._util import merge_unique, remove_all
from .context import current, require_child_id


def _profile(ctx) -> tuple[ChildReadingProfile, ChildReadingProfileRepository, bool]:
    """Return (profile, repo, created) for the target child, creating an empty one if absent."""
    repo = ChildReadingProfileRepository(session=ctx.session)
    child_id = require_child_id()
    profile = repo.get_by_child(child_id)
    created = profile is None
    if profile is None:
        profile = ChildReadingProfile(id=uuid4(), child_id=child_id)
    return profile, repo, created


def _persist(profile: ChildReadingProfile, repo: ChildReadingProfileRepository, created: bool) -> None:
    repo.add(profile) if created else repo.update(profile)


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
    ctx = current()
    profile, repo, created = _profile(ctx)
    for field, value in (
        ("reading_level_note", reading_level_note),
        ("cefr_level", cefr_level),
        ("lexile", lexile),
        ("ar_level", None if ar_level is None else Decimal(str(ar_level))),
        ("current_stage", current_stage),
        ("independent_reading", independent_reading),
        ("needs_dictionary", needs_dictionary),
        ("can_read_chapter_books", can_read_chapter_books),
        ("can_handle_old_language", can_handle_old_language),
        ("confidence", None if confidence is None else Decimal(str(confidence))),
    ):
        if value is not None:
            setattr(profile, field, value)
    profile.source = source
    _persist(profile, repo, created)
    return f"Updated reading ability for child {profile.child_id}."


@tool
def update_reading_interest(
    add_interests: list[str] | None = None,
    remove_interests: list[str] | None = None,
) -> str:
    """Add or remove topics the target child is interested in (e.g. dragons, space, sports)."""
    ctx = current()
    profile, repo, created = _profile(ctx)
    profile.interests = remove_all(merge_unique(profile.interests, add_interests), remove_interests)
    _persist(profile, repo, created)
    return f"Updated interests for child {profile.child_id}."


@tool
def update_genre_preference(
    add_preferred: list[str] | None = None,
    remove_preferred: list[str] | None = None,
    add_disliked: list[str] | None = None,
    remove_disliked: list[str] | None = None,
) -> str:
    """Update the target child's preferred and disliked genres."""
    ctx = current()
    profile, repo, created = _profile(ctx)
    profile.preferred_genres = remove_all(
        merge_unique(profile.preferred_genres, add_preferred), remove_preferred
    )
    profile.disliked_genres = remove_all(
        merge_unique(profile.disliked_genres, add_disliked), remove_disliked
    )
    _persist(profile, repo, created)
    return f"Updated genre preferences for child {profile.child_id}."


@tool
def update_theme_tone_preference(
    add_liked_themes: list[str] | None = None,
    add_disliked_themes: list[str] | None = None,
    add_preferred_tone: list[str] | None = None,
    add_avoid_topics: list[str] | None = None,
) -> str:
    """Add to the target child's liked/disliked themes, preferred tone, and topics to avoid."""
    ctx = current()
    profile, repo, created = _profile(ctx)
    profile.liked_themes = merge_unique(profile.liked_themes, add_liked_themes)
    profile.disliked_themes = merge_unique(profile.disliked_themes, add_disliked_themes)
    profile.preferred_tone = merge_unique(profile.preferred_tone, add_preferred_tone)
    profile.avoid_topics = merge_unique(profile.avoid_topics, add_avoid_topics)
    _persist(profile, repo, created)
    return f"Updated themes/tone for child {profile.child_id}."


@tool
def update_reading_summary(summary: str) -> str:
    """Set a short natural-language summary of the target child's reading profile."""
    ctx = current()
    profile, repo, created = _profile(ctx)
    profile.summary = summary
    _persist(profile, repo, created)
    return f"Updated reading summary for child {profile.child_id}."
