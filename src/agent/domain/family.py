"""Family domain operations: members and reading policy.

Calls the accounts internal API (`ctx.client`). The family is always the one bound for the turn
(context.current().family_id) -- never an id the caller passes. Member/profile/policy list edits
are merged against the turn's cached bundle, then the full field is sent to the API.

Family creation is NOT here: families are created at signup by the accounts service, so there is
no internal create-family endpoint.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ..db import Gender
from ._util import iso_date, merge_unique, remove_all
from .context import accounts, cached_member, current, require_member_id


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
    body: dict[str, Any] = {"role": role}
    for field_name, value in (
        ("display_name", display_name),
        ("gender", gender.value if gender else None),
        ("birth_date", iso_date(birth_date)),
        ("language_preference", language_preference),
    ):
        if value is not None:
            body[field_name] = value
    # is_primary_user is set at signup, not by the agent; the internal create endpoint does not
    # accept it, so it is intentionally omitted from the request body.
    member = accounts().create_member(ctx.family_id, body)
    ctx.members.append({**member, "profile": {}})
    return f"Added family member {display_name or role} ({member['id']})."


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
    member_id = require_member_id()
    body: dict[str, Any] = {}
    for field_name, value in (
        ("display_name", display_name),
        ("gender", gender.value if gender else None),
        ("birth_date", iso_date(birth_date)),
        ("role", role),
        ("language_preference", language_preference),
    ):
        if value is not None:
            body[field_name] = value
    updated = accounts().update_member(current().family_id, member_id, body)
    _refresh_member_cache(member_id, updated)
    return f"Updated basic info for member {member_id}."


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
    member_id = require_member_id()
    prior_profile = cached_member(member_id).get("profile") or {}
    body: dict[str, Any] = {"source": source}
    for field_name, value in (
        ("occupation_background", occupation_background),
        ("education_background", education_background),
        ("communication_style", communication_style),
        ("confidence", confidence),
    ):
        if value is not None:
            body[field_name] = value
    body["concerns"] = remove_all(
        merge_unique(prior_profile.get("concerns"), add_concerns), remove_concerns
    )
    updated = accounts().upsert_member_profile(current().family_id, member_id, body)
    _set_member_profile_cache(member_id, updated)
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
    child_id = ctx.target_child_id if child_scoped else None
    existing = _active_policy(child_id)
    if existing is None:
        body: dict[str, Any] = {
            "child_id": str(child_id) if child_id else None,
            "goals": goals or [],
            "constraints": constraints or [],
            "avoid_topics": avoid_topics or [],
            "content_preferences": content_preferences or {},
            "notes": notes,
        }
        created = accounts().create_policy(ctx.family_id, body)
        ctx.policies.append(created)
        scope = "child" if child_scoped else "family"
        return f"Updated {scope} reading policy ({created['id']})."

    body = {
        "goals": merge_unique(existing.get("goals"), goals),
        "constraints": merge_unique(existing.get("constraints"), constraints),
        "avoid_topics": merge_unique(existing.get("avoid_topics"), avoid_topics),
    }
    if content_preferences:
        body["content_preferences"] = {
            **(existing.get("content_preferences") or {}),
            **content_preferences,
        }
    if notes is not None:
        body["notes"] = notes
    updated = accounts().update_policy(ctx.family_id, existing["id"], body)
    _replace_policy_cache(updated)
    scope = "child" if child_scoped else "family"
    return f"Updated {scope} reading policy ({updated['id']})."


def _active_policy(child_id: object) -> dict[str, Any] | None:
    """Return the turn's active policy for the scope (child id or family-wide None), from cache."""
    want = str(child_id) if child_id is not None else None
    for p in current().policies:
        pc = p.get("child_id")
        pc = str(pc) if pc is not None else None
        if p.get("is_active", True) and pc == want:
            return p
    return None


def _refresh_member_cache(member_id: object, updated: dict[str, Any]) -> None:
    key = str(member_id)
    for i, m in enumerate(current().members):
        if str(m.get("id")) == key:
            current().members[i] = {**m, **updated, "profile": m.get("profile", {})}
            return
    current().members.append({**updated, "profile": {}})


def _set_member_profile_cache(member_id: object, profile: dict[str, Any]) -> None:
    key = str(member_id)
    for m in current().members:
        if str(m.get("id")) == key:
            m["profile"] = profile
            return
    current().members.append({"id": key, "profile": profile})


def _replace_policy_cache(updated: dict[str, Any]) -> None:
    for i, p in enumerate(current().policies):
        if str(p.get("id")) == str(updated.get("id")):
            current().policies[i] = updated
            return
    current().policies.append(updated)
