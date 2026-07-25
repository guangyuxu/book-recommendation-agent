"""ORM models for the agent-owned tables in the book_agent schema, hand-written to mirror the DB.

The family / member / child / reading-profile / reading-history / policy tables moved to the
accounts service, which is now their single owner; the agent reaches them over the internal API
(see `agent.accounts_client`), not the ORM. What remains here is what the agent still owns
directly: the book cache, the recommendation turn (session -> item, feedback), and token billing.

Split by domain, one module each; this package re-exports every model so existing
`from agent.db.models import X` keep working, and importing the package registers all classes on
Base.metadata.

Domains:
- book (catalog):  book_cache
- recommendation:  recommendation_session -> recommendation_item, recommendation_feedback
- billing:         token_usage_record

`Gender` (a shared value type used by the domain tools' signatures) still lives in `._columns`.

Conventions shared by every table (see ._columns):
- UUID primary key, generated server-side via gen_random_uuid().
- created_at / updated_at TIMESTAMP, defaulted server-side to now(); updated_at also bumped
  ORM-side via onupdate so writes through the ORM keep it fresh.
- text[] columns use TextArray and default to an empty array; JSONB columns default to {}.

The recommendation tables carry family_id / child_id / member_id as PLAIN UUID columns (no FK):
the referenced rows live in the accounts service, so cross-service foreign keys are not possible.
"""

from __future__ import annotations

from ._columns import Gender
from .book import BookCache
from .recommendation import (
    RecommendationFeedback,
    RecommendationItem,
    RecommendationSession,
)
from .token_usage import TokenUsageRecord

__all__ = [
    # shared value types
    "Gender",
    # book
    "BookCache",
    # recommendation
    "RecommendationSession",
    "RecommendationItem",
    "RecommendationFeedback",
    # billing
    "TokenUsageRecord",
]
