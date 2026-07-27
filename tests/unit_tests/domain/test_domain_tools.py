"""Unit tests for the domain tools against an in-memory fake of the accounts internal API.

The family / child / reading / policy tools now write through the accounts service (the single
owner of those tables); the `fake_accounts` fixture stands in for it. Cross-family isolation is
enforced and tested inside the accounts service -- here we verify the agent-side behavior: the
right endpoint is called, scoped to the turn's family_id, with the correctly merged payload, and
that create_child promotes the new child to the turn's target.

The recommendation tables are still agent-owned (local DB); their repository test uses sqlite.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

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


def test_create_child_sets_target_and_seeds_reading_profile(fake_accounts: Any) -> None:
    fid = uuid.uuid4()
    with domain_session(fid, None) as ctx:
        out = create_child.invoke(
            {"display_name": "Mia", "gender": "Female", "birth_date": "2016-05-01"}
        )
        assert "Created child" in out
        child_id = (
            ctx.target_child_id
        )  # create_child promoted the new child to the target
        assert child_id is not None

    child = fake_accounts.children[str(child_id)]
    assert child["display_name"] == "Mia"
    assert child["gender"] == "Female"  # StrEnum serialized to its value
    assert child["birth_date"] == "2016-05-01"  # normalized ISO string for the API
    assert str(child_id) in fake_accounts.reading_profiles  # 1:1 profile seeded


def test_update_reading_interest_and_genre_merge(fake_accounts: Any) -> None:
    fid = uuid.uuid4()
    with domain_session(fid, None) as ctx:
        create_child.invoke({"display_name": "Leo"})
        child_id = ctx.target_child_id
        update_reading_interest.invoke({"add_interests": ["dragons", "space"]})
        update_reading_interest.invoke(
            {"add_interests": ["space"], "remove_interests": ["dragons"]}
        )
        update_genre_preference.invoke(
            {"add_preferred": ["fantasy"], "add_disliked": ["horror"]}
        )

    rp = fake_accounts.reading_profiles[str(child_id)]
    assert rp["interests"] == ["space"]  # dragons removed, space not duplicated
    assert rp["preferred_genres"] == ["fantasy"]
    assert rp["disliked_genres"] == ["horror"]


def test_record_finished_book_upserts_by_title(fake_accounts: Any) -> None:
    fid = uuid.uuid4()
    with domain_session(fid, None) as ctx:
        create_child.invoke({"display_name": "Ada"})
        child_id = ctx.target_child_id
        record_finished_book.invoke(
            {"title": "Percy Jackson", "author": "Rick Riordan", "liked": True}
        )
        # Same title again -> updates the same row rather than duplicating.
        record_finished_book.invoke({"title": "Percy Jackson", "liked": False})

    rows = fake_accounts.reading_history[str(child_id)]
    assert len(rows) == 1
    assert rows[0]["status"] == "finished"
    assert rows[0]["liked"] is False


def test_update_family_reading_policy_child_scoped(fake_accounts: Any) -> None:
    fid = uuid.uuid4()
    with domain_session(fid, None):
        create_child.invoke({"display_name": "Sam"})
        update_family_reading_policy.invoke(
            {"goals": ["build empathy"], "avoid_topics": ["gore"]}
        )
        # A second edit targets the same active policy and merges arrays, deduped.
        update_family_reading_policy.invoke(
            {"goals": ["build empathy", "grow vocabulary"]}
        )

    assert len(fake_accounts.policies) == 1
    policy = fake_accounts.policies[0]
    assert policy["goals"] == ["build empathy", "grow vocabulary"]
    assert policy["avoid_topics"] == ["gore"]


def test_domain_tools_scope_writes_to_bound_family(fake_accounts: Any) -> None:
    """Every accounts call carries the turn's bound family_id -- never a caller-supplied one."""
    fid = uuid.uuid4()
    with domain_session(fid, None):
        create_child.invoke({"display_name": "Kid"})
        update_reading_interest.invoke({"add_interests": ["space"]})

    assert fake_accounts.calls  # calls were made
    assert all(scoped == str(fid) for _op, scoped in fake_accounts.calls)


######################
# Recommendation repository (agent-owned local table)
######################


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


def test_latest_for_child_requires_matching_family(session: Session) -> None:
    """latest_for_child with a wrong family_id must return None even if child_id matches."""
    from agent.db.models.recommendation import RecommendationSession

    fid_a = uuid.uuid4()
    fid_b = uuid.uuid4()
    child_id = uuid.uuid4()

    rec = RecommendationSession(
        id=uuid.uuid4(),
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
    assert repo.latest_for_child(child_id, fid_a) is not None  # correct family
    assert repo.latest_for_child(child_id, fid_b) is None  # cross-family isolation


def test_create_child_outside_domain_session_raises() -> None:
    """Domain tools require a domain_session (identity binding) -- fail clearly without one."""
    with pytest.raises(RuntimeError):
        create_child.invoke({"display_name": "Nobody"})
