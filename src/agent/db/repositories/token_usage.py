"""Repository for token usage records."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from advanced_alchemy.filters import LimitOffset
from advanced_alchemy.repository import SQLAlchemySyncRepository

from ..models import TokenUsageRecord


class TokenUsageRepository(SQLAlchemySyncRepository[TokenUsageRecord]):
    model_type = TokenUsageRecord

    def list_by_family(
        self, family_id: UUID, *, limit: int | None = None
    ) -> list[TokenUsageRecord]:
        filters: list[Any] = [TokenUsageRecord.family_id == family_id]
        if limit is not None:
            filters.append(LimitOffset(limit, 0))
        return self.get_many(*filters, order_by=TokenUsageRecord.created_at.desc())

    def list_by_turn(self, turn_id: str) -> list[TokenUsageRecord]:
        return self.get_many(
            TokenUsageRecord.turn_id == turn_id,
            order_by=TokenUsageRecord.created_at.asc(),
        )
