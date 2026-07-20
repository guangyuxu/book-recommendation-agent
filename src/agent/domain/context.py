"""Per-turn binding of identity (+ helpers) for domain tools.

Domain tools are bound to an LLM agent once, as module-level singletons. The LLM supplies only
domain content as tool arguments -- never the family_id / child_id (which it could hallucinate).
Those flow instead through a contextvar that the calling node sets for the duration of the turn
via `domain_session(...)`. Tools read it with `current()`.

Two write backends live behind this context:
- The **accounts internal API** (`ctx.client`) owns family / member / child / reading-profile /
  reading-history / policy. Those tools read the turn's context bundle from the cache
  (`ctx.children` / `ctx.members` / `ctx.policies`, seeded from state) so they can merge
  add/remove list edits locally, then send the full field to the API and refresh the cache.
- The local DB **session** (`ctx.session`) still owns the recommendation / book tables (the
  respond node persists a turn there).

Constraint: the contextvar is visible only within the synchronous call stack of the node that set
it (a raw thread does not inherit it). Run all tool calls inside the same node body -- do not fan
them across a ThreadPoolExecutor without copying the context.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from ..accounts_client import AccountsClient, get_client
from ..db import session_scope


@dataclass
class DomainContext:
    """The identity + backends a domain tool operates against, for one turn."""

    session: Session
    family_id: UUID
    target_child_id: UUID | None = None
    requester_member_id: UUID | None = None
    # Accounts internal API client (lazily resolved via `accounts()` when None).
    client: AccountsClient | None = None
    # Per-turn context cache (serialized state bundle) for read-modify-write list merges. Copied
    # from state so tool writes don't mutate the caller's structures in place.
    children: dict[str, dict[str, Any]] = field(default_factory=dict)
    members: list[dict[str, Any]] = field(default_factory=list)
    policies: list[dict[str, Any]] = field(default_factory=list)


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


def accounts() -> AccountsClient:
    """Return the accounts client for this turn (the bound one, or the process singleton)."""
    return current().client or get_client()


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


def cached_child(child_id: UUID | str) -> dict[str, Any]:
    """Return the cached child bundle (with nested `reading_profile`), or {} if unknown."""
    return current().children.get(str(child_id), {})


def cached_member(member_id: UUID | str) -> dict[str, Any]:
    """Return the cached member bundle (with nested `profile`), or {} if unknown."""
    for m in current().members:
        if str(m.get("id")) == str(member_id):
            return m
    return {}


@contextmanager
def domain_session(
    family_id: UUID | str,
    target_child_id: UUID | str | None = None,
    requester_member_id: UUID | str | None = None,
    *,
    session: Session | None = None,
    client: AccountsClient | None = None,
    children: dict[str, dict[str, Any]] | None = None,
    members: list[dict[str, Any]] | None = None,
    policies: list[dict[str, Any]] | None = None,
) -> Iterator[DomainContext]:
    """Bind identity + backends for the enclosed block, so domain tools can run.

    With no `session`, opens one via db.session_scope (commits on clean exit, rolls back on error)
    for the local recommendation/book writes. `client` defaults to the process accounts client
    (tests inject a fake). `children`/`members`/`policies` seed the read-modify-write cache from
    state; they are shallow-copied so tool writes never mutate the caller's structures.
    """
    fid = _as_uuid(family_id)
    assert fid is not None  # noqa: S101 -- family_id is always required
    cid = _as_uuid(target_child_id)
    mid = _as_uuid(requester_member_id)

    def _run(s: Session) -> Iterator[DomainContext]:
        ctx = DomainContext(
            s,
            fid,
            cid,
            mid,
            client=client,
            children=dict(children or {}),
            members=list(members or []),
            policies=list(policies or []),
        )
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
