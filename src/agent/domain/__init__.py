"""Domain Operations layer: the only code that touches the database.

The graph thinks in domains and operations; these LangChain tools translate operations into
repository calls (one operation may span several tables). A turn binds a session + identity
with `domain_session(...)`; tools read it via the contextvar in `context`.
"""

from .books import cache_book, update_book_metadata, update_book_summary
from .child import (
    create_child,
    update_child_basic_info,
    update_child_notes,
    update_school_information,
)
from .context import DomainContext, current, domain_session, require_child_id
from .family import (
    add_family_member,
    create_family,
    update_family_reading_policy,
    update_member_basic_info,
    update_member_profile,
)
from .reading_history import (
    record_current_reading,
    record_disliked_book,
    record_finished_book,
)
from .reading_profile import (
    update_genre_preference,
    update_reading_ability,
    update_reading_interest,
    update_reading_summary,
    update_theme_tone_preference,
)
from .recommendation import (
    RecItemInput,
    create_recommendation_session,
    record_recommendation_feedback,
    save_recommendation_items,
)
from .toolkit import MEMORY_TOOLS, MEMORY_TOOLS_BY_NAME

__all__ = [
    # context
    "DomainContext",
    "domain_session",
    "current",
    "require_child_id",
    # toolkit
    "MEMORY_TOOLS",
    "MEMORY_TOOLS_BY_NAME",
    # family
    "create_family",
    "add_family_member",
    "update_member_basic_info",
    "update_member_profile",
    "update_family_reading_policy",
    # child
    "create_child",
    "update_child_basic_info",
    "update_school_information",
    "update_child_notes",
    # reading profile
    "update_reading_ability",
    "update_reading_interest",
    "update_genre_preference",
    "update_theme_tone_preference",
    "update_reading_summary",
    # reading history
    "record_finished_book",
    "record_current_reading",
    "record_disliked_book",
    # books
    "cache_book",
    "update_book_metadata",
    "update_book_summary",
    # recommendation
    "RecItemInput",
    "create_recommendation_session",
    "save_recommendation_items",
    "record_recommendation_feedback",
]
