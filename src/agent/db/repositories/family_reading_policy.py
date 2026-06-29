"""Repository for the `family_reading_policy` table (goals/constraints per family/child)."""

from __future__ import annotations

from uuid import UUID

from advanced_alchemy.repository import SQLAlchemySyncRepository
from sqlalchemy import or_

from ..models import FamilyReadingPolicy


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
            filters.append(
                or_(
                    FamilyReadingPolicy.child_id == child_id,
                    FamilyReadingPolicy.child_id.is_(None),
                )
            )
        return self.list(*filters, order_by=FamilyReadingPolicy.created_at.asc())
