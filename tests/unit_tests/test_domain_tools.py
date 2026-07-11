"""Unit tests for the domain tools, against an isolated in-memory sqlite database.

The tools read their session + identity from the domain_session contextvar, so we pass an
explicit sqlite session and seed rows with explicit UUIDs (sqlite has no gen_random_uuid()).
No LLM and no Postgres are involved.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from agent.db import (
    ChildProfileRepository,
    ChildReadingProfileRepository,
    Family,
    FamilyReadingPolicyRepository,
    Gender,
    ReadingHistoryRepository,
)
from agent.db.base import Base
from agent.db.repositories.recommendation import RecommendationSessionRepository
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
        out = create_child.invoke(
            {"display_name": "Mia", "gender": "Female", "birth_date": "2016-05-01"}
        )
        assert "Created child" in out
        child_id = (
            ctx.target_child_id
        )  # create_child promoted the new child to the target
        assert child_id is not None

    children = ChildProfileRepository(session=session).list_by_family(fid)
    assert [c.display_name for c in children] == ["Mia"]
    # gender/birth_date persist; age is derived from birth_date, never stored.
    assert children[0].gender == Gender.FEMALE
    assert children[0].birth_date == date(2016, 5, 1)
    # create_child also seeds the 1:1 reading profile.
    assert (
        ChildReadingProfileRepository(session=session).get_by_child(child_id)
        is not None
    )


def test_update_reading_interest_and_genre_merge(session: Session) -> None:
    fid = _seed_family(session)
    with domain_session(fid, None, session=session) as ctx:
        create_child.invoke({"display_name": "Leo"})
        child_id = ctx.target_child_id
        update_reading_interest.invoke({"add_interests": ["dragons", "space"]})
        update_reading_interest.invoke(
            {"add_interests": ["space"], "remove_interests": ["dragons"]}
        )
        update_genre_preference.invoke(
            {"add_preferred": ["fantasy"], "add_disliked": ["horror"]}
        )

    rp = ChildReadingProfileRepository(session=session).get_by_child(child_id)
    assert rp.interests == ["space"]  # dragons removed, space not duplicated
    assert rp.preferred_genres == ["fantasy"]
    assert rp.disliked_genres == ["horror"]


def test_record_finished_book_upserts_by_title(session: Session) -> None:
    fid = _seed_family(session)
    with domain_session(fid, None, session=session) as ctx:
        create_child.invoke({"display_name": "Ada"})
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


######################
# Cross-family isolation
######################


def test_list_by_family_does_not_leak_across_families(session: Session) -> None:
    """Children created under family A must not appear in family B's list."""
    fid_a = _seed_family(session)
    fid_b = _seed_family(session)

    with domain_session(fid_a, None, session=session):
        create_child.invoke({"display_name": "Alice"})

    children_b = ChildProfileRepository(session=session).list_by_family(fid_b)
    assert children_b == [], "family B must not see family A's children"


def test_domain_session_scopes_writes_to_own_family(session: Session) -> None:
    """Writes inside a family B session must not affect family A's data."""
    fid_a = _seed_family(session)
    fid_b = _seed_family(session)

    with domain_session(fid_a, None, session=session) as ctx_a:
        create_child.invoke({"display_name": "A-child"})
        child_a_id = ctx_a.target_child_id
        update_reading_interest.invoke({"add_interests": ["space"]})

    with domain_session(fid_b, None, session=session):
        create_child.invoke({"display_name": "B-child"})
        update_reading_interest.invoke({"add_interests": ["dragons"]})

    # Family A's profile must be unchanged.
    rp_a = ChildReadingProfileRepository(session=session).get_by_child(child_a_id)
    assert rp_a.interests == ["space"]


def test_latest_for_child_requires_matching_family(session: Session) -> None:
    """latest_for_child with a wrong family_id must return None even if child_id matches."""
    from uuid import uuid4

    from agent.db.models.recommendation import RecommendationSession

    fid_a = _seed_family(session)
    fid_b = _seed_family(session)

    with domain_session(fid_a, None, session=session) as ctx:
        create_child.invoke({"display_name": "Mia"})
        child_id = ctx.target_child_id

    # Seed a session row directly (bypass domain tools to keep test focused on the repo).
    rec = RecommendationSession(
        id=uuid4(),
        family_id=fid_a,
        target_child_id=child_id,
        intents=[],
        user_message="test",
        understanding={},
        plan={},
        capability_result={},
        memory_decision={},
    )
    session.add(rec)
    session.flush()

    repo = RecommendationSessionRepository(session=session)
    # Correct family -> finds the row.
    assert repo.latest_for_child(child_id, fid_a) is not None
    # Wrong family -> must return None (cross-family isolation).
    assert repo.latest_for_child(child_id, fid_b) is None


def test_update_family_reading_policy_child_scoped(session: Session) -> None:
    fid = _seed_family(session)
    with domain_session(fid, None, session=session) as ctx:
        create_child.invoke({"display_name": "Sam"})
        update_family_reading_policy.invoke(
            {"goals": ["build empathy"], "avoid_topics": ["gore"]}
        )
        update_family_reading_policy.invoke(
            {"goals": ["build empathy", "grow vocabulary"]}
        )
        child_id = ctx.target_child_id

    policies = FamilyReadingPolicyRepository(session=session).list_active(fid, child_id)
    assert len(policies) == 1
    assert policies[0].goals == ["build empathy", "grow vocabulary"]  # merged, deduped
    assert policies[0].avoid_topics == ["gore"]
