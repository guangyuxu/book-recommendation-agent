"""Graph entry: resolve the request's family and load its context into state.

load_context is the only place the graph reads the database directly (everything else goes
through domain tools). It reads AppContext.family_id / family_member_id / child_id, loads the
family with its members, children (each with their reading profile), and active reading
policies, and pins the target child when known (explicit child_id, or the only child on file).
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


class MissingContextError(ValueError):
    """Raised when family_id / family_member_id is missing. ValueError, so LangGraph won't retry."""


class FamilyNotFoundError(ValueError):
    """Raised when family_id does not match any family. Families must be seeded before use."""


def _never_retry(exc: Exception) -> bool:
    return False


# Missing/unknown context is a deterministic input error; retrying only spams the stack
# trace. Disable retry so the validation failure is logged once.
LOAD_CONTEXT_RETRY = RetryPolicy(retry_on=_never_retry)


def get_context(config: RunnableConfig | None) -> dict[str, str | None]:
    """Return {family_id, family_member_id, child_id}.

    Values come from the context channel first (Studio / context=), falling back to
    configurable for older callers. get_runtime only works inside graph execution, so the
    try/except lets direct calls (tests) work too.
    """
    cfg = (config or {}).get("configurable", {})
    runtime_ctx: dict = {}
    try:
        runtime_ctx = dict(get_runtime(AppContext).context or {})
    except Exception:
        runtime_ctx = {}

    def pick(key: str) -> str | None:
        return runtime_ctx.get(key) or cfg.get(key)

    return {
        "family_id": pick("family_id"),
        "family_member_id": pick("family_member_id"),
        "child_id": pick("child_id"),
    }


def load_context(state: MessagesState, config: RunnableConfig | None = None):
    """Entry node: load the family's context into state and pin the target child if known.

    Serializes every ORM object to a dict inside the session scope so the selectin-loaded
    relationships (reading_profile, etc.) are resolved before the session closes.
    """
    ctx = get_context(config)
    family_id, member_id = ctx["family_id"], ctx["family_member_id"]
    if not family_id or not member_id:
        logger.warning("Validation failed: request is missing family_id/family_member_id.")
        raise MissingContextError(
            "Missing required context: every request must carry family_id and "
            'family_member_id. In Studio, set them in the run config (context); for SDK/API '
            'calls pass context={"family_id": ..., "family_member_id": ...}.'
        )
    fid = UUID(str(family_id))
    with session_scope() as s:
        family = FamilyRepository(session=s).get_one_or_none(id=fid)
        if family is None:
            raise FamilyNotFoundError(
                f"No family found for family_id={family_id}. Families must be created before "
                "the agent can run for them."
            )
        members = []
        for m in FamilyMemberRepository(session=s).list_by_family(fid):
            md = m.to_dict()
            md["profile"] = m.profile.to_dict() if m.profile else {}
            members.append(md)
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

    # Pin the target child: an explicit child_id (if it belongs to this family) wins;
    # otherwise default to the only child on file; otherwise leave it for understand to resolve.
    ctx_child = ctx.get("child_id")
    if ctx_child and str(ctx_child) in children:
        target_child_id: str | None = str(ctx_child)
    elif len(children) == 1:
        target_child_id = next(iter(children))
    else:
        target_child_id = None

    return {
        "family": family_dict,
        "members": members,
        "children": children,
        "policies": policies,
        "family_member_id": str(member_id),
        "target_child_id": target_child_id,
    }
