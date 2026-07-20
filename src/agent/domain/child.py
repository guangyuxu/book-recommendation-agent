"""Child Profile domain operations: create a child and update basic / school / notes fields.

Calls the accounts internal API (`ctx.client`). create_child also seeds the 1:1 reading profile
(done by accounts) and sets the newly created child as the turn's target, so any following
reading-profile/history operations in the same turn apply to it. Updates act on the target child.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ..db import Gender
from ._util import iso_date
from .context import _as_uuid, accounts, cached_child, current, require_child_id


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
    body: dict[str, Any] = {
        "display_name": display_name,
        "aliases": aliases or [],
        "gender": gender.value if gender else None,
        "birth_date": iso_date(birth_date),
        "grade": grade,
        "school_system": school_system,
        "country_or_curriculum": country_or_curriculum,
        "primary_language": primary_language,
        "reading_language": reading_language,
        "notes": notes,
    }
    child = accounts().create_child(ctx.family_id, body)
    child_id = child["id"]
    ctx.children[str(child_id)] = {**child, "reading_profile": {}}
    # subsequent tools this turn target the new child
    ctx.target_child_id = _as_uuid(child_id)
    return f"Created child {display_name} ({child_id})."


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
    child_id = require_child_id()
    body: dict[str, Any] = {}
    for field_name, value in (
        ("display_name", display_name),
        ("aliases", aliases),
        ("gender", gender.value if gender else None),
        ("birth_date", iso_date(birth_date)),
        ("grade", grade),
        ("primary_language", primary_language),
        ("reading_language", reading_language),
    ):
        if value is not None:
            body[field_name] = value
    updated = accounts().update_child(current().family_id, child_id, body)
    _refresh_child_cache(child_id, updated)
    return f"Updated basic info for child {child_id}."


@tool
def update_school_information(
    grade: str | None = None,
    school_system: str | None = None,
    country_or_curriculum: str | None = None,
) -> str:
    """Update the target child's school context (grade, school system, country/curriculum)."""
    child_id = require_child_id()
    body: dict[str, Any] = {}
    for field_name, value in (
        ("grade", grade),
        ("school_system", school_system),
        ("country_or_curriculum", country_or_curriculum),
    ):
        if value is not None:
            body[field_name] = value
    updated = accounts().update_child(current().family_id, child_id, body)
    _refresh_child_cache(child_id, updated)
    return f"Updated school information for child {child_id}."


@tool
def update_child_notes(notes: str, mode: str = "append") -> str:
    """Update the target child's free-text notes. mode is "append" (default) or "replace"."""
    child_id = require_child_id()
    existing = cached_child(child_id).get("notes")
    new_notes = f"{existing}\n{notes}" if mode == "append" and existing else notes
    updated = accounts().update_child(
        current().family_id, child_id, {"notes": new_notes}
    )
    _refresh_child_cache(child_id, updated)
    return f"Updated notes for child {child_id}."


def _refresh_child_cache(child_id: object, updated: dict[str, Any]) -> None:
    """Merge an updated child row back into the turn cache, preserving nested reading_profile."""
    key = str(child_id)
    prior = current().children.get(key, {})
    current().children[key] = {
        **prior,
        **updated,
        "reading_profile": prior.get("reading_profile", {}),
    }
