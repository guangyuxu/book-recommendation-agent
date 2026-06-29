"""End-to-end graph test.

Exercises the full pipeline against the configured database and the Anthropic API, so it is
gated behind RUN_INTEGRATION=1. It seeds a throwaway family + child, runs a recommendation
turn, and cleans up afterwards.
"""

import os
import uuid

import pytest
from langchain.messages import HumanMessage

pytestmark = pytest.mark.anyio

_RUN = os.getenv("RUN_INTEGRATION") == "1"


@pytest.mark.skipif(not _RUN, reason="set RUN_INTEGRATION=1 (needs Postgres + Anthropic)")
async def test_recommendation_turn_end_to_end() -> None:
    from agent.db import (
        ChildProfile,
        ChildProfileRepository,
        Family,
        FamilyRepository,
        RecommendationSessionRepository,
        session_scope,
    )
    from agent.graph import graph

    family_id = uuid.uuid4()
    child_id = uuid.uuid4()
    with session_scope() as s:
        FamilyRepository(session=s).add(Family(id=family_id, family_name="IT Family"))
        ChildProfileRepository(session=s).add(
            ChildProfile(id=child_id, family_id=family_id, display_name="Iris", age=8)
        )

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage("Recommend books for my 8-year-old who loves dragons")]},
            context={"user_id": str(family_id)},
        )
        assert result["messages"], "expected at least one reply"
        assert result["messages"][-1].type == "ai"

        # Flavor A: a recommendation turn should have persisted a session row.
        with session_scope() as s:
            sessions = RecommendationSessionRepository(session=s).list_by_family(family_id)
            assert sessions, "recommendation_session row was not persisted"
    finally:
        with session_scope() as s:
            # Children/sessions cascade from the family delete via ORM relationships.
            family = FamilyRepository(session=s).get_one_or_none(id=family_id)
            if family is not None:
                FamilyRepository(session=s).delete(family.id)
