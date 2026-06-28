"""Single source of truth mapping intent -> handler. New intent: add a member in intents.py, a row here."""

from .flows import (
    book_evaluation,
    book_recommendation,
    content_creation,
    reading_path_planning,
)
from .handlers import (
    child_profile_update,
    make_mock_handler,
    parent_goal_update,
    parent_profile_update,
    reading_discussion,
)
from .intents import Intent

# Values are compiled subgraphs or state->delta functions. No MULTI_INTENT here -- the
# orchestrator is mounted separately by graph.py.
HANDLERS = {
    Intent.BOOK_RECOMMENDATION: book_recommendation.graph,
    Intent.BOOK_EVALUATION: book_evaluation.graph,
    Intent.READING_PATH_PLANNING: reading_path_planning.graph,
    Intent.CONTENT_CREATION: content_creation.graph,
    Intent.CHILD_PROFILE_UPDATE: child_profile_update,
    Intent.PARENT_PROFILE_UPDATE: parent_profile_update,
    Intent.PARENT_GOAL_UPDATE: parent_goal_update,
    Intent.READING_DISCUSSION: reading_discussion,
    Intent.CLARIFY: make_mock_handler(Intent.CLARIFY),
}

# A missing mapping fails at import time, not silently as a mock at runtime.
_missing = set(Intent) - set(HANDLERS) - {Intent.MULTI_INTENT}
assert not _missing, f"Intents not registered in registry: {[i.value for i in _missing]}"


def run_handler(intent: Intent, state: dict) -> dict:
    """Subgraphs use .invoke(); plain node functions are called directly."""
    node = HANDLERS[intent]
    return node.invoke(state) if hasattr(node, "invoke") else node(state)
