"""ORM models for the book_agent schema (11 tables), hand-written to mirror the live DB.

Split by domain, one module each; this package re-exports every model so existing
`from agent.db.models import X` (and `from ..models import X` in repositories) keep working,
and so importing the package registers all classes on Base.metadata.

Domains:
- family:          family -> family_member -> family_member_profile (1:1),
                   family_reading_policy
- child:           child_profile -> child_reading_profile (1:1)
- reading:         reading_history
- book (catalog):  book_cache
- recommendation:  recommendation_session -> recommendation_item, recommendation_feedback

Conventions shared by every table (see ._columns):
- UUID primary key, generated server-side via gen_random_uuid().
- created_at / updated_at TIMESTAMP, defaulted server-side to now(); updated_at also bumped
  ORM-side via onupdate so writes through the ORM keep it fresh.
- text[] columns use TextArray and default to an empty array; JSONB columns default to {}.

Tables already exist in the DB; create_all is only a local/dev fallback (see base.init_db).
"""

from __future__ import annotations

from .book import BookCache
from .child import ChildProfile, ChildReadingProfile
from .family import Family, FamilyMember, FamilyMemberProfile, FamilyReadingPolicy
from .reading import ReadingHistory
from .recommendation import (
    RecommendationFeedback,
    RecommendationItem,
    RecommendationSession,
)

__all__ = [
    # family
    "Family",
    "FamilyMember",
    "FamilyMemberProfile",
    "FamilyReadingPolicy",
    # child
    "ChildProfile",
    "ChildReadingProfile",
    # reading
    "ReadingHistory",
    # book
    "BookCache",
    # recommendation
    "RecommendationSession",
    "RecommendationItem",
    "RecommendationFeedback",
]
