"""Intent definitions (single source of truth).

Each member carries value (used as a stable key) / label / description (shown to the LLM in
the understand node). Up to two intents may apply to one turn (one primary + one secondary);
there is no multi_intent member -- the understand node returns primary + optional secondary.
"""

from enum import Enum


class Intent(Enum):
    """The set of user intents the understand node chooses from."""

    def __new__(cls, key: str, label: str, description: str) -> "Intent":
        """Set _value_ to key so .value is the plain key string, not the whole tuple."""
        obj = object.__new__(cls)
        obj._value_ = key
        obj.label = label
        obj.description = description
        return obj

    BOOK_RECOMMENDATION = (
        "book_recommendation",
        "Recommendation flow",
        "User explicitly asks for a booklist or what to read next. A bare "
        "description of the child or their tastes, with no such request, is NOT "
        "this intent (that is child_profile_update).",
    )
    BOOK_EVALUATION = (
        "book_evaluation",
        "Book analysis flow",
        "User asks to assess or discuss ONE specific book: whether it suits the "
        "child, its themes, values, reading difficulty, or tendencies.",
    )
    BOOK_COMPARISON = (
        "book_comparison",
        "Book comparison flow",
        "User asks to compare TWO OR MORE specific books against each other (which "
        "is better, more suitable, harder, etc.).",
    )
    CHILD_PROFILE_UPDATE = (
        "child_profile_update",
        "Child profile update flow",
        "User states or updates something about the child: a trait, preference, "
        "age, reading level, or behavior, whether a stable fact or a recent "
        "change, without asking for a separate task.",
    )
    READING_PATH_PLANNING = (
        "reading_path_planning",
        "Stage planning flow",
        "User wants a reading path or transition plan between reading levels "
        "or between books.",
    )
    PARENT_GOAL_UPDATE = (
        "parent_goal_update",
        "Parent goal update flow",
        "Parent states an educational goal or expectation for what they want "
        "the child to gain.",
    )
    PARENT_PROFILE_UPDATE = (
        "parent_profile_update",
        "Parent profile update flow",
        "Parent describes their own situation, parenting style, available time, "
        "or personal taste: about the parent themselves, not a goal for the child.",
    )
    READING_DISCUSSION = (
        "reading_discussion",
        "Discussion question generation flow",
        "User wants post-reading discussion or reflection questions to guide "
        "the child's thinking.",
    )
    CONTENT_CREATION = (
        "content_creation",
        "Content creation flow",
        "User asks to create content such as articles, copy, or social posts.",
    )
    CLARIFY = (
        "clarify",
        "Clarification flow",
        "The message has no clear actionable request and no useful profile or "
        "goal information: a greeting, chit-chat, or too vague or incomplete to "
        "act on. Use this as the fallback when no other intent clearly applies.",
    )

    # Let the type checker know these dynamic attributes exist.
    label: str
    description: str


def intent_menu() -> str:
    """Render the intents as a bulleted menu for LLM prompts (understand node)."""
    return "\n".join(f"- {i.value}: {i.description}" for i in Intent)


def to_intent(value: str) -> Intent:
    """Look up an Intent by its value key (the enum's custom __new__ confuses callers/mypy)."""
    return Intent(value)  # type: ignore[call-arg]
