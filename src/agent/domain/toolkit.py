"""Curated tool lists for binding to LLM agents.

MEMORY_TOOLS are the domain operations the Profile Update agent may call to persist what
Memory Policy decided is worth remembering. Recommendation-write tools are deliberately excluded:
recommendation persistence is done deterministically by the respond node. (Families themselves
are created at signup by the accounts service, so there is no create-family tool at all.)
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from .child import (
    create_child,
    update_child_basic_info,
    update_child_notes,
    update_school_information,
)
from .family import (
    add_family_member,
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

MEMORY_TOOLS: list[BaseTool] = [
    # Child profile
    create_child,
    update_child_basic_info,
    update_school_information,
    update_child_notes,
    # Reading profile
    update_reading_ability,
    update_reading_interest,
    update_genre_preference,
    update_theme_tone_preference,
    update_reading_summary,
    # Reading history
    record_finished_book,
    record_current_reading,
    record_disliked_book,
    # Family
    add_family_member,
    update_member_basic_info,
    update_member_profile,
    update_family_reading_policy,
]

MEMORY_TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in MEMORY_TOOLS}
