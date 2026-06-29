"""Repository for the `child_profile` table (the children in a family)."""

from __future__ import annotations

from uuid import UUID

from advanced_alchemy.repository import SQLAlchemySyncRepository

from ..models import ChildProfile


class ChildProfileRepository(SQLAlchemySyncRepository[ChildProfile]):
    model_type = ChildProfile

    def list_by_family(self, family_id: UUID) -> list[ChildProfile]:
        return self.list(family_id=family_id, order_by=ChildProfile.created_at.asc())
