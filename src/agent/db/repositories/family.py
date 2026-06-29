"""Repository for the `family` table (household / login identity)."""

from __future__ import annotations

from advanced_alchemy.repository import SQLAlchemySyncRepository

from ..models import Family


class FamilyRepository(SQLAlchemySyncRepository[Family]):
    model_type = Family
