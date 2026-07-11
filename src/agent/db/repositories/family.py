"""Repositories for the family domain: household, members, member profiles, reading policy."""

from __future__ import annotations

from uuid import UUID

from advanced_alchemy.repository import SQLAlchemySyncRepository
from sqlalchemy import or_

from ..models import (
    Family,
    FamilyMember,
    FamilyMemberProfile,
    FamilyReadingPolicy,
)


class FamilyRepository(SQLAlchemySyncRepository[Family]):
    model_type = Family


class FamilyMemberRepository(SQLAlchemySyncRepository[FamilyMember]):
    model_type = FamilyMember

    def list_by_family(self, family_id: UUID) -> list[FamilyMember]:
        return self.get_many(
            family_id=family_id, order_by=FamilyMember.created_at.asc()
        )

    def primary_user(self, family_id: UUID) -> FamilyMember | None:
        """Return the family's designated primary user, if one is flagged."""
        return self.get_one_or_none(family_id=family_id, is_primary_user=True)


class FamilyMemberProfileRepository(SQLAlchemySyncRepository[FamilyMemberProfile]):
    model_type = FamilyMemberProfile

    def get_by_member(self, member_id: UUID) -> FamilyMemberProfile | None:
        """Return the member's profile (member_id is UNIQUE)."""
        return self.get_one_or_none(member_id=member_id)


class FamilyReadingPolicyRepository(SQLAlchemySyncRepository[FamilyReadingPolicy]):
    model_type = FamilyReadingPolicy

    def list_active(
        self, family_id: UUID, child_id: UUID | None = None
    ) -> list[FamilyReadingPolicy]:
        """Return active policies for a family.

        With `child_id`, returns both the child-specific policies and the family-wide ones
        (child_id IS NULL), since family-wide policies also apply to that child.
        """
        filters = [
            FamilyReadingPolicy.family_id == family_id,
            FamilyReadingPolicy.is_active.is_(True),
        ]
        if child_id is not None:
            # noinspection PyUnresolvedReferences
            filters.append(
                or_(
                    FamilyReadingPolicy.child_id == child_id,
                    FamilyReadingPolicy.child_id.is_(None),
                )
            )
        return self.get_many(*filters, order_by=FamilyReadingPolicy.created_at.asc())
