"""Intent definitions (single source of truth).

Each member carries value(node name) / label / description(for the classifier).
"""

from enum import Enum


class Intent(Enum):
    """The set of user intents the classifier chooses from; each value names one flow."""

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
        "User asks to assess or discuss a specific book: whether it suits the "
        "child, its themes, values, reading difficulty, or tendencies.",
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
    MULTI_INTENT = (
        "multi_intent",
        "Split flow",
        "The message combines multiple distinct intents, such as a profile "
        "update together with a task, or several tasks at once, and must be "
        "split and handled separately.",
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


# Intents that act on a specific child: they need target_child_ids resolved, and trigger
# clarify when the child is ambiguous. Single source of truth for graph routing,
# the orchestrator, and resolve. (MULTI_INTENT is excluded -- it resolves per subtask.)
CHILD_SPECIFIC: frozenset["Intent"] = frozenset(
    {
        Intent.BOOK_RECOMMENDATION,
        Intent.BOOK_EVALUATION,
        Intent.READING_PATH_PLANNING,
        Intent.READING_DISCUSSION,
        Intent.CHILD_PROFILE_UPDATE,
    }
)
