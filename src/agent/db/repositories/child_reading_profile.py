"""Repository for the `child_reading_profile` table (one reading profile per child)."""

from __future__ import annotations

from uuid import UUID

from advanced_alchemy.repository import SQLAlchemySyncRepository

from ..models import ChildReadingProfile


class ChildReadingProfileRepository(SQLAlchemySyncRepository[ChildReadingProfile]):
    model_type = ChildReadingProfile

    def get_by_child(self, child_id: UUID) -> ChildReadingProfile | None:
        """Return the child's reading profile (child_id is UNIQUE)."""
        return self.get_one_or_none(child_id=child_id)
