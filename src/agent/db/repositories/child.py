"""Repositories for the child domain: child profiles and their reading profiles."""

from __future__ import annotations

from uuid import UUID

from advanced_alchemy.repository import SQLAlchemySyncRepository

from ..models import ChildProfile, ChildReadingProfile


class ChildProfileRepository(SQLAlchemySyncRepository[ChildProfile]):
    model_type = ChildProfile

    def list_by_family(self, family_id: UUID) -> list[ChildProfile]:
        return self.get_many(family_id=family_id, order_by=ChildProfile.created_at.asc())


class ChildReadingProfileRepository(SQLAlchemySyncRepository[ChildReadingProfile]):
    model_type = ChildReadingProfile

    def get_by_child(self, child_id: UUID) -> ChildReadingProfile | None:
        """Return the child's reading profile (child_id is UNIQUE)."""
        return self.get_one_or_none(child_id=child_id)
