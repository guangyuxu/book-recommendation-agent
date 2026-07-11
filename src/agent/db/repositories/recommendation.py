"""Repositories for the recommendation domain: sessions, items, and feedback."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from advanced_alchemy.filters import LimitOffset
from advanced_alchemy.repository import SQLAlchemySyncRepository

from ..models import (
    RecommendationFeedback,
    RecommendationItem,
    RecommendationSession,
)


class RecommendationSessionRepository(SQLAlchemySyncRepository[RecommendationSession]):
    model_type = RecommendationSession

    def list_by_family(
        self, family_id: UUID, *, limit: int | None = None
    ) -> list[RecommendationSession]:
        filters: list[Any] = [RecommendationSession.family_id == family_id]
        if limit is not None:
            filters.append(LimitOffset(limit, 0))
        return self.get_many(*filters, order_by=RecommendationSession.created_at.desc())

    def latest_for_child(self, child_id: UUID) -> RecommendationSession | None:
        rows = self.get_many(
            RecommendationSession.target_child_id == child_id,
            LimitOffset(1, 0),
            order_by=RecommendationSession.created_at.desc(),
        )
        return rows[0] if rows else None


class RecommendationItemRepository(SQLAlchemySyncRepository[RecommendationItem]):
    model_type = RecommendationItem

    def list_by_session(self, session_id: UUID) -> list[RecommendationItem]:
        return self.get_many(session_id=session_id, order_by=RecommendationItem.rank.asc())


class RecommendationFeedbackRepository(
    SQLAlchemySyncRepository[RecommendationFeedback]
):
    model_type = RecommendationFeedback

    def list_by_session(self, session_id: UUID) -> list[RecommendationFeedback]:
        return self.get_many(
            session_id=session_id, order_by=RecommendationFeedback.created_at.asc()
        )

    def list_by_child(self, child_id: UUID) -> list[RecommendationFeedback]:
        return self.get_many(
            child_id=child_id, order_by=RecommendationFeedback.created_at.desc()
        )
