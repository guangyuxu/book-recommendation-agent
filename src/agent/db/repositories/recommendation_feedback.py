"""Repository for the `recommendation_feedback` table (reactions to items/sessions)."""

from __future__ import annotations

from uuid import UUID

from advanced_alchemy.repository import SQLAlchemySyncRepository

from ..models import RecommendationFeedback


class RecommendationFeedbackRepository(SQLAlchemySyncRepository[RecommendationFeedback]):
    model_type = RecommendationFeedback

    def list_by_session(self, session_id: UUID) -> list[RecommendationFeedback]:
        return self.list(
            session_id=session_id, order_by=RecommendationFeedback.created_at.asc()
        )

    def list_by_child(self, child_id: UUID) -> list[RecommendationFeedback]:
        return self.list(
            child_id=child_id, order_by=RecommendationFeedback.created_at.desc()
        )
