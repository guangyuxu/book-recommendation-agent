"""Repository for the `recommendation_item` table (books within a session)."""

from __future__ import annotations

from uuid import UUID

from advanced_alchemy.repository import SQLAlchemySyncRepository

from ..models import RecommendationItem


class RecommendationItemRepository(SQLAlchemySyncRepository[RecommendationItem]):
    model_type = RecommendationItem

    def list_by_session(self, session_id: UUID) -> list[RecommendationItem]:
        return self.list(session_id=session_id, order_by=RecommendationItem.rank.asc())
