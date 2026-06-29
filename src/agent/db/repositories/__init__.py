"""Per-table repositories: one Advanced Alchemy SQLAlchemySyncRepository subclass per table.

Build them on a session (see db.base.session_scope), one per request -- note `session` is a
keyword-only argument:

    from agent.db import session_scope
    from agent.db.repositories import ChildProfileRepository

    with session_scope() as s:
        children = ChildProfileRepository(session=s).list_by_family(family_id)
"""

from .book_cache import BookCacheRepository
from .child_profile import ChildProfileRepository
from .child_reading_profile import ChildReadingProfileRepository
from .family import FamilyRepository
from .family_member import FamilyMemberRepository
from .family_reading_policy import FamilyReadingPolicyRepository
from .reading_history import ReadingHistoryRepository
from .recommendation_feedback import RecommendationFeedbackRepository
from .recommendation_item import RecommendationItemRepository
from .recommendation_session import RecommendationSessionRepository

__all__ = [
    "FamilyRepository",
    "FamilyMemberRepository",
    "ChildProfileRepository",
    "ChildReadingProfileRepository",
    "FamilyReadingPolicyRepository",
    "ReadingHistoryRepository",
    "BookCacheRepository",
    "RecommendationSessionRepository",
    "RecommendationItemRepository",
    "RecommendationFeedbackRepository",
]
