"""Graph entry: resolve the request's family and load its context into state.

load_context is the only place the graph reads the database directly (everything else goes
through domain tools). It maps AppContext.user_id == family.id to the family, its members,
its children (each with their reading profile), and the family's active reading policies.
"""

import logging
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import get_runtime
from langgraph.types import RetryPolicy

from .db import (
    ChildProfileRepository,
    FamilyMemberRepository,
    FamilyReadingPolicyRepository,
    FamilyRepository,
    session_scope,
)
from .state import AppContext, MessagesState

logger = logging.getLogger(__name__)


class MissingUserIDError(ValueError):
    """Raised when a request is missing user_id. Subclasses ValueError, so LangGraph won't retry."""


class FamilyNotFoundError(ValueError):
    """Raised when user_id does not match any family. Families must be seeded before use."""


def _never_retry(exc: Exception) -> bool:
    return False


# A missing/unknown user_id is a deterministic input error; retrying only spams the stack
# trace. Disable retry so the validation failure is logged once.
LOAD_CONTEXT_RETRY = RetryPolicy(retry_on=_never_retry)


def get_context(config: RunnableConfig | None) -> tuple[str | None, str | None]:
    """Return (user_id, thread_id).

    user_id comes from the context channel first (Studio / context=), falling back to
    configurable for older callers. get_runtime only works inside graph execution, so the
    try/except lets direct calls (tests) work too.
    """
    cfg = (config or {}).get("configurable", {})
    user_id = None
    try:
        user_id = get_runtime(AppContext).context.get("user_id")
    except Exception:
        user_id = None
    if not user_id:
        user_id = cfg.get("user_id")
    return user_id, cfg.get("thread_id")


def load_context(state: MessagesState, config: RunnableConfig | None = None):
    """Entry node: load the family's context into state by user_id (== family.id).

    Serializes every ORM object to a dict inside the session scope so the selectin-loaded
    relationships (reading_profile, etc.) are resolved before the session closes.
    """
    user_id, _ = get_context(config)
    if not user_id:
        logger.warning("Validation failed: request is missing user_id, refusing to run.")
        raise MissingUserIDError(
            "Missing required parameter user_id: every request must carry user_id. "
            "In Studio, set user_id in the run config (context); "
            'for SDK/API calls pass context={"user_id": ...}.'
        )
    fid = UUID(str(user_id))
    with session_scope() as s:
        family = FamilyRepository(session=s).get_one_or_none(id=fid)
        if family is None:
            raise FamilyNotFoundError(
                f"No family found for user_id={user_id}. Families must be created before "
                "the agent can run for them."
            )
        members = [m.to_dict() for m in FamilyMemberRepository(session=s).list_by_family(fid)]
        children: dict[str, dict] = {}
        for c in ChildProfileRepository(session=s).list_by_family(fid):
            prof = c.to_dict()
            prof["reading_profile"] = (
                c.reading_profile.to_dict() if c.reading_profile else {}
            )
            children[str(c.id)] = prof
        policies = [
            p.to_dict() for p in FamilyReadingPolicyRepository(session=s).list_active(fid)
        ]
        family_dict = family.to_dict()
    return {
        "family": family_dict,
        "members": members,
        "children": children,
        "policies": policies,
    }
