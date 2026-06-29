"""Repository for the `recommendation_session` table (one recommendation turn)."""

from __future__ import annotations

from uuid import UUID

from advanced_alchemy.filters import LimitOffset
from advanced_alchemy.repository import SQLAlchemySyncRepository

from ..models import RecommendationSession


class RecommendationSessionRepository(SQLAlchemySyncRepository[RecommendationSession]):
    model_type = RecommendationSession

    def list_by_family(
        self, family_id: UUID, *, limit: int | None = None
    ) -> list[RecommendationSession]:
        filters: list = [RecommendationSession.family_id == family_id]
        if limit is not None:
            filters.append(LimitOffset(limit, 0))
        return self.list(*filters, order_by=RecommendationSession.created_at.desc())

    def latest_for_child(self, child_id: UUID) -> RecommendationSession | None:
        rows = self.list(
            RecommendationSession.target_child_id == child_id,
            LimitOffset(1, 0),
            order_by=RecommendationSession.created_at.desc(),
        )
        return rows[0] if rows else None
