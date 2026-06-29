"""Repository for the `family_member` table (parents/caregivers in a family)."""

from __future__ import annotations

from uuid import UUID

from advanced_alchemy.repository import SQLAlchemySyncRepository

from ..models import FamilyMember


class FamilyMemberRepository(SQLAlchemySyncRepository[FamilyMember]):
    model_type = FamilyMember

    def list_by_family(self, family_id: UUID) -> list[FamilyMember]:
        return self.list(family_id=family_id, order_by=FamilyMember.created_at.asc())

    def primary_user(self, family_id: UUID) -> FamilyMember | None:
        """Return the family's designated primary user, if one is flagged."""
        return self.get_one_or_none(family_id=family_id, is_primary_user=True)
