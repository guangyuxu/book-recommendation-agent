"""Per-domain repositories: Advanced Alchemy SQLAlchemySyncRepository subclasses, one per table.

Only the agent-owned tables remain here (book cache, recommendation, token usage); the family /
child / reading tables moved to the accounts service and are reached over its internal API
(`agent.accounts_client`), not through repositories. Re-exported here so callers keep a flat import:

    from agent.db import session_scope
    from agent.db.repositories import RecommendationSessionRepository

Build them on a session (see db.base.session_scope), one per request -- note `session` is a
keyword-only argument.
"""

from .book import BookCacheRepository
from .recommendation import (
    RecommendationFeedbackRepository,
    RecommendationItemRepository,
    RecommendationSessionRepository,
)
from .token_usage import TokenUsageRepository

__all__ = [
    "BookCacheRepository",
    "RecommendationSessionRepository",
    "RecommendationItemRepository",
    "RecommendationFeedbackRepository",
    "TokenUsageRepository",
]
