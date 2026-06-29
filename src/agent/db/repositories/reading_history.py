"""Repository for the `reading_history` table (books a child has read)."""

from __future__ import annotations

from uuid import UUID

from advanced_alchemy.repository import SQLAlchemySyncRepository

from ..models import ReadingHistory


class ReadingHistoryRepository(SQLAlchemySyncRepository[ReadingHistory]):
    model_type = ReadingHistory

    def list_by_child(
        self, child_id: UUID, *, status: str | None = None
    ) -> list[ReadingHistory]:
        filters = [ReadingHistory.child_id == child_id]
        if status is not None:
            filters.append(ReadingHistory.status == status)
        return self.list(*filters, order_by=ReadingHistory.created_at.desc())
