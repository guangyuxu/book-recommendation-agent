"""Child Profile domain operations: create a child and update basic / school / notes fields.

Wraps ChildProfileRepository (and seeds the 1:1 ChildReadingProfile on create). Updates act
on the turn's target child; create_child sets the newly created child as the target so any
following reading-profile/history operations in the same turn apply to it.
"""

from __future__ import annotations

from uuid import uuid4

from langchain_core.tools import tool

from ..db import (
    ChildProfile,
    ChildProfileRepository,
    ChildReadingProfile,
    ChildReadingProfileRepository,
    Gender,
)
from ._util import parse_iso_date
from .context import current, require_child_id


@tool
def create_child(
    display_name: str,
    aliases: list[str] | None = None,
    gender: Gender | None = None,
    birth_date: str | None = None,
    grade: str | None = None,
    school_system: str | None = None,
    country_or_curriculum: str | None = None,
    primary_language: str | None = None,
    reading_language: str = "English",
    notes: str | None = None,
) -> str:
    """Create a child in the current family and make them this turn's target child.

    `birth_date` is an ISO date ('YYYY-MM-DD'); the child's age is derived from it at read
    time, never stored. Also seeds an empty reading profile so later reading-profile updates
    apply cleanly. Use when the conversation introduces a child not already on file.
    """
    ctx = current()
    child = ChildProfile(
        id=uuid4(),
        family_id=ctx.family_id,
        display_name=display_name,
        aliases=aliases or [],
        gender=gender,
        birth_date=parse_iso_date(birth_date),
        grade=grade,
        school_system=school_system,
        country_or_curriculum=country_or_curriculum,
        primary_language=primary_language,
        reading_language=reading_language,
        notes=notes,
    )
    ChildProfileRepository(session=ctx.session).add(child)
    ChildReadingProfileRepository(session=ctx.session).add(
        ChildReadingProfile(id=uuid4(), child_id=child.id)
    )
    ctx.target_child_id = child.id  # subsequent tools this turn target the new child
    return f"Created child {display_name} ({child.id})."


def _target_child(ctx) -> ChildProfile:
    repo = ChildProfileRepository(session=ctx.session)
    child = repo.get_one_or_none(id=require_child_id())
    if child is None:
        raise RuntimeError("Target child not found in the database.")
    return child


@tool
def update_child_basic_info(
    display_name: str | None = None,
    aliases: list[str] | None = None,
    gender: Gender | None = None,
    birth_date: str | None = None,
    grade: str | None = None,
    primary_language: str | None = None,
    reading_language: str | None = None,
) -> str:
    """Update the target child's basic identity fields. Only provided fields change.

    `birth_date` is an ISO date ('YYYY-MM-DD'); age is derived from it at read time.
    """
    ctx = current()
    child = _target_child(ctx)
    for field, value in (
        ("display_name", display_name),
        ("aliases", aliases),
        ("gender", gender),
        ("birth_date", parse_iso_date(birth_date)),
        ("grade", grade),
        ("primary_language", primary_language),
        ("reading_language", reading_language),
    ):
        if value is not None:
            setattr(child, field, value)
    ChildProfileRepository(session=ctx.session).update(child)
    return f"Updated basic info for child {child.id}."


@tool
def update_school_information(
    grade: str | None = None,
    school_system: str | None = None,
    country_or_curriculum: str | None = None,
) -> str:
    """Update the target child's school context (grade, school system, country/curriculum)."""
    ctx = current()
    child = _target_child(ctx)
    for field, value in (
        ("grade", grade),
        ("school_system", school_system),
        ("country_or_curriculum", country_or_curriculum),
    ):
        if value is not None:
            setattr(child, field, value)
    ChildProfileRepository(session=ctx.session).update(child)
    return f"Updated school information for child {child.id}."


@tool
def update_child_notes(notes: str, mode: str = "append") -> str:
    """Update the target child's free-text notes. mode is "append" (default) or "replace"."""
    ctx = current()
    child = _target_child(ctx)
    if mode == "append" and child.notes:
        child.notes = f"{child.notes}\n{notes}"
    else:
        child.notes = notes
    ChildProfileRepository(session=ctx.session).update(child)
    return f"Updated notes for child {child.id}."
