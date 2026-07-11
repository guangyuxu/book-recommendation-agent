"""Serialize a family's ORM rows into the JSON-able dicts the graph state carries.

The shared seam between the two places that load family context: `lifecycle.load_context`
(turn entry) and `memory.profile_update` (refresh after writes so the frontend syncs
same-turn). Both need the same member/child serialization with derived age, so it lives here
rather than in either caller. Must run inside an open session so selectin relationships
(reading_profile, member profile) resolve before serialization.
"""

from datetime import date
from uuid import UUID

from .db import ChildProfileRepository, FamilyMemberRepository


def _age(dob: date | None) -> int | None:
    """Whole years from a date of birth to today, or None if unknown."""
    if dob is None:
        return None
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def _serialize_member(m) -> dict:
    """One member row -> state dict, with its 1:1 profile and derived age (never stored)."""
    md = m.to_dict()
    md["profile"] = m.profile.to_dict() if m.profile else {}
    md["age"] = _age(m.birth_date)
    md["birth_date"] = m.birth_date.isoformat() if m.birth_date else None
    return md


def _serialize_child(c) -> dict:
    """One child row -> state dict, with its 1:1 reading_profile and derived age."""
    prof = c.to_dict()
    prof["reading_profile"] = c.reading_profile.to_dict() if c.reading_profile else {}
    prof["age"] = _age(c.birth_date)
    prof["birth_date"] = c.birth_date.isoformat() if c.birth_date else None
    return prof


def load_family_entities(session, fid: UUID) -> tuple[list[dict], dict[str, dict]]:
    """(members, children-by-id) for a family, serialized for state.

    Shared by load_context (turn entry) and profile_update (refresh after writes so the
    frontend syncs same-turn). Must run inside an open session so selectin relationships
    resolve before serialization.
    """
    members = [
        _serialize_member(m) for m in FamilyMemberRepository(session=session).list_by_family(fid)
    ]
    children = {
        str(c.id): _serialize_child(c)
        for c in ChildProfileRepository(session=session).list_by_family(fid)
    }
    return members, children
