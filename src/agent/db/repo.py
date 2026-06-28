"""Data-access layer. Contract: each function takes a Session; the caller opens/closes it."""

from sqlalchemy.orm import Session

from .models import ChatMessage, ChildProfile, ParentProfile

# First-class columns; any other key goes into extra (JSONB).
_PARENT_COLS = {"available_time", "self_taste"}
_CHILD_COLS = {"name", "reading_level", "recent_signal"}


def _child_to_dict(child: ChildProfile) -> dict:
    """Assemble one child's state dict, keeping its id so callers can target it later."""
    cp = {
        "id": str(child.id),
        "name": child.name,
        "reading_level": child.reading_level,
        "recent_signal": child.recent_signal,
    }
    cp.update(child.extra or {})
    return {k: v for k, v in cp.items() if v is not None}


def load_state_profiles(session: Session, user_id: str) -> dict:
    """Read profiles by user_id and assemble them into graph state fields."""
    parent = session.get(ParentProfile, user_id)
    if not parent:
        return {}

    out: dict = {}
    pp = {"available_time": parent.available_time, "self_taste": parent.self_taste}
    pp.update(parent.extra or {})
    out["parent_profile"] = {k: v for k, v in pp.items() if v is not None}
    out["parent_goals"] = list(parent.parent_goals or [])

    # Full roster, keyed by str(child_id). Supports a parent with multiple children.
    out["children"] = {str(c.id): _child_to_dict(c) for c in parent.children}
    return out


def _get_or_create_parent(session: Session, user_id: str) -> ParentProfile:
    parent = session.get(ParentProfile, user_id)
    if not parent:
        parent = ParentProfile(user_id=user_id, parent_goals=[], extra={})
        session.add(parent)
        session.flush()
    return parent


def upsert_parent_profile(
    session: Session,
    user_id: str,
    *,
    fields: dict | None = None,
    goals: list[str] | None = None,
) -> None:
    """Update parent profile: known columns set directly, unknown keys into extra; goals dedup-appended."""
    parent = _get_or_create_parent(session, user_id)
    for k, v in (fields or {}).items():
        if k in _PARENT_COLS:
            setattr(parent, k, v)
        else:
            parent.extra = {**(parent.extra or {}), k: v}
    if goals:
        existing = list(parent.parent_goals or [])
        parent.parent_goals = existing + [g for g in goals if g not in existing]
    session.commit()


def upsert_child_profile(
    session: Session, user_id: str, *, fields: dict, child_id: str | int | None = None
) -> str:
    """Update one child's profile and return its id (as str).

    child_id pins an existing child; None creates a new one. Known keys go to columns,
    the rest into extra (JSONB).
    """
    parent = _get_or_create_parent(session, user_id)
    child = None
    if child_id is not None:
        child = session.get(ChildProfile, int(child_id))
        if child is not None and child.parent_user_id != user_id:
            raise ValueError(f"child {child_id} does not belong to user {user_id}")
    if child is None:
        child = ChildProfile(parent_user_id=user_id, extra={})
        session.add(child)
    for k, v in fields.items():
        if k in _CHILD_COLS:
            setattr(child, k, v)
        else:
            child.extra = {**(child.extra or {}), k: v}
    session.commit()
    return str(child.id)


def add_chat_message(
    session: Session, user_id: str, thread_id: str, role: str, content: str
) -> None:
    session.add(
        ChatMessage(
            user_id=user_id, thread_id=thread_id, role=role, content=content
        )
    )
    session.commit()
