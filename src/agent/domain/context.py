"""Per-turn binding of a DB session + identity for domain tools.

Domain tools are bound to an LLM agent once, as module-level singletons. The LLM supplies
only domain content as tool arguments -- never a Session and never the family_id / child_id
(which it could hallucinate). Those flow instead through a contextvar that the calling node
sets for the duration of the turn via `domain_session(...)`. Tools read it with `current()`.

Constraint: the contextvar is visible only within the synchronous call stack of the node
that set it (a raw thread does not inherit it). Run all tool calls inside the same node body
-- do not fan them across a ThreadPoolExecutor without copying the context.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from ..db import session_scope


@dataclass
class DomainContext:
    """The session + identity a domain tool operates against, for one turn."""

    session: Session
    family_id: UUID
    target_child_id: UUID | None = None
    requester_member_id: UUID | None = None


_ctx: contextvars.ContextVar[DomainContext | None] = contextvars.ContextVar(
    "domain_ctx", default=None
)


def _as_uuid(value: object) -> UUID | None:
    if value is None:
        return None
    return value if isinstance(value, UUID) else UUID(str(value))


def current() -> DomainContext:
    """Return the active DomainContext, or raise if called outside a `domain_session`."""
    ctx = _ctx.get()
    if ctx is None:
        raise RuntimeError("Domain tool called outside a domain_session() scope.")
    return ctx


def require_child_id() -> UUID:
    """Return the turn's target child id, or raise if none is bound (tool needs a child)."""
    ctx = current()
    if ctx.target_child_id is None:
        raise RuntimeError(
            "This operation needs a target child, but none is set for this turn."
        )
    return ctx.target_child_id


def require_member_id() -> UUID:
    """Return the turn's requester member id, or raise if none is bound (tool needs a member)."""
    ctx = current()
    if ctx.requester_member_id is None:
        raise RuntimeError(
            "This operation needs a requester member, but none is set for this turn."
        )
    return ctx.requester_member_id


@contextmanager
def domain_session(
    family_id: UUID | str,
    target_child_id: UUID | str | None = None,
    requester_member_id: UUID | str | None = None,
    *,
    session: Session | None = None,
) -> Iterator[DomainContext]:
    """Bind a session + identity for the enclosed block, so domain tools can run.

    With no `session`, opens one via db.session_scope (commits on clean exit, rolls back on
    error). Tests pass an explicit `session` and manage its transaction themselves.
    """
    fid = _as_uuid(family_id)
    assert fid is not None  # noqa: S101 -- family_id is always required
    cid = _as_uuid(target_child_id)
    mid = _as_uuid(requester_member_id)

    def _run(s: Session) -> Iterator[DomainContext]:
        ctx = DomainContext(s, fid, cid, mid)
        token = _ctx.set(ctx)
        try:
            yield ctx
        finally:
            _ctx.reset(token)

    if session is not None:
        yield from _run(session)
    else:
        with session_scope() as s:
            yield from _run(s)
