"""Family domain operations: members and reading policy.

Tools wrap FamilyRepository / FamilyMemberRepository / FamilyReadingPolicyRepository. The
family is always the one bound for the turn (context.current().family_id) -- never an id the
caller passes.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import uuid4

from langchain_core.tools import tool

from ..db import (
    Family,
    FamilyMember,
    FamilyMemberProfile,
    FamilyMemberProfileRepository,
    FamilyMemberRepository,
    FamilyReadingPolicy,
    FamilyReadingPolicyRepository,
    FamilyRepository,
    Gender,
)
from ._util import merge_unique, parse_iso_date, remove_all
from .context import current, require_member_id


@tool
def create_family(family_name: str | None = None, default_language: str = "en") -> str:
    """Create a new family (household). For seeding/tests; not used in normal turns."""
    ctx = current()
    family = Family(
        id=uuid4(), family_name=family_name, default_language=default_language
    )
    FamilyRepository(session=ctx.session).add(family)
    return f"Created family {family.id}."


@tool
def add_family_member(
    role: str,
    display_name: str | None = None,
    gender: Gender | None = None,
    birth_date: str | None = None,
    is_primary_user: bool = False,
    language_preference: str | None = None,
) -> str:
    """Add a parent or caregiver to the current family.

    `role` describes the relationship (e.g. "mother", "father", "caregiver"). `birth_date` is an
    ISO date ('YYYY-MM-DD'); the member's age is derived from it at read time. This records only
    identity; conversation-extracted background goes to update_member_profile.
    """
    ctx = current()
    member = FamilyMember(
        id=uuid4(),
        family_id=ctx.family_id,
        role=role,
        display_name=display_name,
        gender=gender,
        birth_date=parse_iso_date(birth_date),
        is_primary_user=is_primary_user,
        language_preference=language_preference,
    )
    FamilyMemberRepository(session=ctx.session).add(member)
    return f"Added family member {display_name or role} ({member.id})."


@tool
def update_member_basic_info(
    display_name: str | None = None,
    gender: Gender | None = None,
    birth_date: str | None = None,
    role: str | None = None,
    language_preference: str | None = None,
) -> str:
    """Update the requesting member's identity fields (name, gender, birth date, role, language).

    Acts on the turn's requester (the parent/caregiver who is asking). `birth_date` is an ISO
    date ('YYYY-MM-DD'); age is derived from it at read time. Only provided fields change.
    Conversation-extracted background (occupation, style, concerns) goes to update_member_profile.
    """
    ctx = current()
    repo = FamilyMemberRepository(session=ctx.session)
    member = repo.get_one_or_none(id=require_member_id())
    if member is None:
        raise RuntimeError("Requesting member not found in the database.")
    for field, value in (
        ("display_name", display_name),
        ("gender", gender),
        ("birth_date", parse_iso_date(birth_date)),
        ("role", role),
        ("language_preference", language_preference),
    ):
        if value is not None:
            setattr(member, field, value)
    repo.update(member)
    return f"Updated basic info for member {member.id}."


@tool
def update_member_profile(
    occupation_background: str | None = None,
    education_background: str | None = None,
    communication_style: str | None = None,
    add_concerns: list[str] | None = None,
    remove_concerns: list[str] | None = None,
    confidence: float | None = None,
    source: str = "parent_report",
) -> str:
    """Update the requesting member's background profile (occupation, education, style, concerns).

    Acts on the turn's requester (the parent/caregiver who is asking); creates their profile on
    demand. Concerns are merged into the existing list, deduped.
    """
    ctx = current()
    repo = FamilyMemberProfileRepository(session=ctx.session)
    member_id = require_member_id()
    profile = repo.get_by_member(member_id)
    created = profile is None
    if profile is None:
        profile = FamilyMemberProfile(id=uuid4(), member_id=member_id)
    for field, value in (
        ("occupation_background", occupation_background),
        ("education_background", education_background),
        ("communication_style", communication_style),
        ("confidence", None if confidence is None else Decimal(str(confidence))),
    ):
        if value is not None:
            setattr(profile, field, value)
    profile.concerns = remove_all(
        merge_unique(profile.concerns, add_concerns), remove_concerns
    )
    profile.source = source
    repo.add(profile) if created else repo.update(profile)
    return f"Updated member profile for {member_id}."


@tool
def update_family_reading_policy(
    goals: list[str] | None = None,
    constraints: list[str] | None = None,
    avoid_topics: list[str] | None = None,
    content_preferences: dict[str, Any] | None = None,
    notes: str | None = None,
    child_scoped: bool = True,
) -> str:
    """Record the family's reading goals/constraints (e.g. educational goals, topics to avoid).

    With child_scoped=True the policy applies to the turn's target child; otherwise it is a
    family-wide policy. Array fields are merged into the existing active policy, deduped.
    """
    ctx = current()
    repo = FamilyReadingPolicyRepository(session=ctx.session)
    child_id = ctx.target_child_id if child_scoped else None
    policy = repo.get_one_or_none(
        family_id=ctx.family_id, child_id=child_id, is_active=True
    )
    if policy is None:
        policy = FamilyReadingPolicy(
            id=uuid4(),
            family_id=ctx.family_id,
            child_id=child_id,
            goals=goals or [],
            constraints=constraints or [],
            avoid_topics=avoid_topics or [],
            content_preferences=content_preferences or {},
            notes=notes,
        )
        repo.add(policy)
    else:
        policy.goals = merge_unique(policy.goals, goals)
        policy.constraints = merge_unique(policy.constraints, constraints)
        policy.avoid_topics = merge_unique(policy.avoid_topics, avoid_topics)
        if content_preferences:
            policy.content_preferences = {
                **(policy.content_preferences or {}),
                **content_preferences,
            }
        if notes is not None:
            policy.notes = notes
        repo.update(policy)
    scope = "child" if child_scoped else "family"
    return f"Updated {scope} reading policy ({policy.id})."
