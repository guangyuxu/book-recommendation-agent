"""Per-domain repositories: Advanced Alchemy SQLAlchemySyncRepository subclasses, one per table.

Grouped by domain (mirroring db.models), and re-exported here so callers keep a flat import:

    from agent.db import session_scope
    from agent.db.repositories import ChildProfileRepository

    with session_scope() as s:
        children = ChildProfileRepository(session=s).list_by_family(family_id)

Build them on a session (see db.base.session_scope), one per request -- note `session` is a
keyword-only argument.
"""

from .book import BookCacheRepository
from .child import ChildProfileRepository, ChildReadingProfileRepository
from .family import (
    FamilyMemberProfileRepository,
    FamilyMemberRepository,
    FamilyReadingPolicyRepository,
    FamilyRepository,
)
from .reading import ReadingHistoryRepository
from .recommendation import (
    RecommendationFeedbackRepository,
    RecommendationItemRepository,
    RecommendationSessionRepository,
)
from .token_usage import TokenUsageRepository

__all__ = [
    "FamilyRepository",
    "FamilyMemberRepository",
    "FamilyMemberProfileRepository",
    "ChildProfileRepository",
    "ChildReadingProfileRepository",
    "FamilyReadingPolicyRepository",
    "ReadingHistoryRepository",
    "BookCacheRepository",
    "RecommendationSessionRepository",
    "RecommendationItemRepository",
    "RecommendationFeedbackRepository",
    "TokenUsageRepository",
]
