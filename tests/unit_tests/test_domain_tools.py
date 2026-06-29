"""Unit tests for the domain tools, against an isolated in-memory sqlite database.

The tools read their session + identity from the domain_session contextvar, so we pass an
explicit sqlite session and seed rows with explicit UUIDs (sqlite has no gen_random_uuid()).
No LLM and no Postgres are involved.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from agent.db import (
    ChildProfileRepository,
    ChildReadingProfileRepository,
    Family,
    FamilyReadingPolicyRepository,
    ReadingHistoryRepository,
)
from agent.db.base import Base
from agent.domain import (
    create_child,
    domain_session,
    record_finished_book,
    update_family_reading_policy,
    update_genre_preference,
    update_reading_interest,
)


@pytest.fixture
def session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:", future=True, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    s = factory()
    try:
        yield s
    finally:
        s.close()


def _seed_family(s: Session) -> uuid.UUID:
    fid = uuid.uuid4()
    s.add(Family(id=fid, family_name="Test Family"))
    s.flush()
    return fid


def test_create_child_sets_target_and_seeds_reading_profile(session: Session) -> None:
    fid = _seed_family(session)
    with domain_session(fid, None, session=session) as ctx:
        out = create_child.invoke({"display_name": "Mia", "age": 8})
        assert "Created child" in out
        child_id = ctx.target_child_id  # create_child promoted the new child to the target
        assert child_id is not None

    children = ChildProfileRepository(session=session).list_by_family(fid)
    assert [c.display_name for c in children] == ["Mia"]
    # create_child also seeds the 1:1 reading profile.
    assert ChildReadingProfileRepository(session=session).get_by_child(child_id) is not None


def test_update_reading_interest_and_genre_merge(session: Session) -> None:
    fid = _seed_family(session)
    with domain_session(fid, None, session=session) as ctx:
        create_child.invoke({"display_name": "Leo", "age": 10})
        child_id = ctx.target_child_id
        update_reading_interest.invoke({"add_interests": ["dragons", "space"]})
        update_reading_interest.invoke({"add_interests": ["space"], "remove_interests": ["dragons"]})
        update_genre_preference.invoke({"add_preferred": ["fantasy"], "add_disliked": ["horror"]})

    rp = ChildReadingProfileRepository(session=session).get_by_child(child_id)
    assert rp.interests == ["space"]  # dragons removed, space not duplicated
    assert rp.preferred_genres == ["fantasy"]
    assert rp.disliked_genres == ["horror"]


def test_record_finished_book_upserts_by_title(session: Session) -> None:
    fid = _seed_family(session)
    with domain_session(fid, None, session=session) as ctx:
        create_child.invoke({"display_name": "Ada", "age": 9})
        child_id = ctx.target_child_id
        record_finished_book.invoke(
            {"title": "Percy Jackson", "author": "Rick Riordan", "liked": True}
        )
        # Same title again -> updates the same row rather than duplicating.
        record_finished_book.invoke({"title": "Percy Jackson", "liked": False})

    rows = ReadingHistoryRepository(session=session).list_by_child(child_id)
    assert len(rows) == 1
    assert rows[0].status == "finished"
    assert rows[0].liked is False


def test_update_family_reading_policy_child_scoped(session: Session) -> None:
    fid = _seed_family(session)
    with domain_session(fid, None, session=session) as ctx:
        create_child.invoke({"display_name": "Sam", "age": 7})
        update_family_reading_policy.invoke({"goals": ["build empathy"], "avoid_topics": ["gore"]})
        update_family_reading_policy.invoke({"goals": ["build empathy", "grow vocabulary"]})
        child_id = ctx.target_child_id

    policies = FamilyReadingPolicyRepository(session=session).list_active(fid, child_id)
    assert len(policies) == 1
    assert policies[0].goals == ["build empathy", "grow vocabulary"]  # merged, deduped
    assert policies[0].avoid_topics == ["gore"]
