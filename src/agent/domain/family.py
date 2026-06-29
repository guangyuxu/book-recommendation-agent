"""Family domain operations: members and reading policy.

Tools wrap FamilyRepository / FamilyMemberRepository / FamilyReadingPolicyRepository. The
family is always the one bound for the turn (context.current().family_id) -- never an id the
caller passes.
"""

from __future__ import annotations

from uuid import uuid4

from langchain_core.tools import tool

from ..db import (
    Family,
    FamilyMember,
    FamilyMemberRepository,
    FamilyReadingPolicy,
    FamilyReadingPolicyRepository,
    FamilyRepository,
)
from .context import current
from .util import merge_unique


@tool
def create_family(family_name: str | None = None, default_language: str = "en") -> str:
    """Create a new family (household). For seeding/tests; not used in normal turns."""
    ctx = current()
    family = Family(id=uuid4(), family_name=family_name, default_language=default_language)
    FamilyRepository(session=ctx.session).add(family)
    return f"Created family {family.id}."


@tool
def add_family_member(
    role: str,
    display_name: str | None = None,
    is_primary_user: bool = False,
    occupation_background: str | None = None,
    education_background: str | None = None,
    communication_style: str | None = None,
    concerns: list[str] | None = None,
    language_preference: str | None = None,
) -> str:
    """Add a parent or caregiver to the current family.

    `role` describes the relationship (e.g. "mother", "father", "caregiver").
    """
    ctx = current()
    member = FamilyMember(
        id=uuid4(),
        family_id=ctx.family_id,
        role=role,
        display_name=display_name,
        is_primary_user=is_primary_user,
        occupation_background=occupation_background,
        education_background=education_background,
        communication_style=communication_style,
        concerns=concerns or [],
        language_preference=language_preference,
    )
    FamilyMemberRepository(session=ctx.session).add(member)
    return f"Added family member {display_name or role} ({member.id})."


@tool
def update_family_reading_policy(
    goals: list[str] | None = None,
    constraints: list[str] | None = None,
    avoid_topics: list[str] | None = None,
    content_preferences: dict | None = None,
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
