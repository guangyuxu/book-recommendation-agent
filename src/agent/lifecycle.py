"""Graph entry: resolve the request's family and load its context into state.

load_context is the only place the graph reads the database directly (everything else goes
through domain tools). It reads AppContext.family_id / family_member_id / child_id, loads the
family with its members, children (each with their reading profile), and active reading
policies, and pins the target child when known (explicit child_id, or the only child on file).
"""

import logging
from typing import Any
from uuid import UUID, uuid4

from langgraph.config import get_config
from langgraph.runtime import get_runtime
from langgraph.types import RetryPolicy

from .db import (
    FamilyReadingPolicyRepository,
    FamilyRepository,
    session_scope,
)
from .serialize import load_family_entities
from .state import AppContext, FlowState

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


def _pin_target_child(ctx_child: str | None, children: dict[str, Any]) -> str | None:
    """Pin the turn's target child deterministically.

    An explicit child_id wins, but only when it belongs to this family (guards against a stale
    or cross-family id); otherwise default to the only child on file; otherwise leave it None
    for understand to resolve from the message.
    """
    if ctx_child and ctx_child in children:
        return ctx_child
    if len(children) == 1:
        return next(iter(children))
    return None


def load_context(state: FlowState) -> dict[str, Any]:
    """Entry node: load the family's context into state and pin the target child if known.

    Serializes every ORM object to a dict inside the session scope so the selectin-loaded
    relationships (reading_profile, etc.) are resolved before the session closes.
    """
    runtime = get_runtime(AppContext)
    ctx: AppContext = runtime.context
    if ctx is None:
        logger.warning("Validation failed: request carried no context.")
        raise MissingContextError(
            "Missing required context: every request must carry family_id and "
            "family_member_id. In Studio, set them in the run config (context); for SDK/API "
            'calls pass context={"family_id": ..., "family_member_id": ...}.'
        )

    fid = UUID(ctx.family_id)
    with session_scope() as s:
        family = FamilyRepository(session=s).get_one_or_none(id=fid)
        if family is None:
            raise FamilyNotFoundError(
                f"No family found for family_id={ctx.family_id}. Families must be created before "
                "the agent can run for them."
            )
        members, children = load_family_entities(s, fid)
        policies = [
            p.to_dict()
            for p in FamilyReadingPolicyRepository(session=s).list_active(fid)
        ]
        family_dict = family.to_dict()

    # Pin the target child: an explicit child_id (if it belongs to this family) wins;
    # otherwise default to the only child on file; otherwise leave it for understand to resolve.
    target_child_id = _pin_target_child(ctx.child_id, children)

    # thread_id from LangGraph's run config; None when running without a checkpointer.
    # (Runtime has no `.config`; the config lives on the runnable, via get_config().)
    try:
        config = get_config()
    except Exception:
        config = {}
    thread_id: str | None = (config.get("configurable") or {}).get("thread_id")
    turn_id = str(uuid4())

    # turn_id/thread_id are placed in state here (once per turn); every LLM node
    # re-establishes the billing ContextVar from them via with_turn_context, because
    # a ContextVar set in this node would NOT be visible in any downstream node
    # (each LangGraph node runs in its own copied context).
    return {
        "family": family_dict,
        "members": members,
        "children": children,
        "policies": policies,
        "family_member_id": ctx.family_member_id,
        "target_child_id": target_child_id,
        # Read by usage_tracker.with_turn_context; also handy for observability/debugging.
        "turn_id": turn_id,
        "thread_id": thread_id,
    }
