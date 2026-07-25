"""Database layer: engine/session/config (from .env), ORM models, and per-domain repositories.

Scope is the agent-owned tables only (book cache, recommendation, token usage). The family /
child / reading tables moved to the accounts service; reach them via `agent.accounts_client`.
"""

from .base import (
    Base,
    JSONType,
    SessionLocal,
    TextArray,
    engine,
    init_db,
    session_scope,
)
from .models import (
    BookCache,
    Gender,
    RecommendationFeedback,
    RecommendationItem,
    RecommendationSession,
    TokenUsageRecord,
)
from .repositories import (
    BookCacheRepository,
    RecommendationFeedbackRepository,
    RecommendationItemRepository,
    RecommendationSessionRepository,
    TokenUsageRepository,
)

__all__ = [
    # infra
    "Base",
    "engine",
    "SessionLocal",
    "session_scope",
    "init_db",
    "JSONType",
    "TextArray",
    # shared value types
    "Gender",
    # models
    "BookCache",
    "RecommendationSession",
    "RecommendationItem",
    "RecommendationFeedback",
    "TokenUsageRecord",
    # repositories
    "BookCacheRepository",
    "RecommendationSessionRepository",
    "RecommendationItemRepository",
    "RecommendationFeedbackRepository",
    "TokenUsageRepository",
]
