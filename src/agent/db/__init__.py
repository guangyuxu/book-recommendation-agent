"""Database layer: engine/session/config (from .env), ORM models, and per-domain repositories."""

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
    ChildProfile,
    ChildReadingProfile,
    Family,
    FamilyMember,
    FamilyMemberProfile,
    FamilyReadingPolicy,
    Gender,
    ReadingHistory,
    RecommendationFeedback,
    RecommendationItem,
    RecommendationSession,
    TokenUsageRecord,
)
from .repositories import (
    BookCacheRepository,
    ChildProfileRepository,
    ChildReadingProfileRepository,
    FamilyMemberProfileRepository,
    FamilyMemberRepository,
    FamilyReadingPolicyRepository,
    FamilyRepository,
    ReadingHistoryRepository,
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
    "Family",
    "FamilyMember",
    "FamilyMemberProfile",
    "ChildProfile",
    "ChildReadingProfile",
    "FamilyReadingPolicy",
    "ReadingHistory",
    "BookCache",
    "RecommendationSession",
    "RecommendationItem",
    "RecommendationFeedback",
    "TokenUsageRecord",
    # repositories
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
