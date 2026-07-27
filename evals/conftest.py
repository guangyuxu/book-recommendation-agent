"""Shared pytest config for the eval harness.

Evals call the Anthropic API (they cost money and are non-deterministic), so -- exactly like
`tests/integration_tests` -- they are opt-in: skipped unless `RUN_EVAL=1`. We load `.env` here
because S1 imports only the `understand` node, not the DB layer that normally calls
`load_dotenv()`, so `ANTHROPIC_API_KEY` would otherwise be missing.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

RUN_EVAL = os.getenv("RUN_EVAL") == "1"

# Applied by strategy test modules: `pytestmark = [requires_eval]`.
requires_eval = pytest.mark.skipif(
    not RUN_EVAL, reason="set RUN_EVAL=1 to run evals (calls the Anthropic API)"
)


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Pin anyio to asyncio for any async eval run (the unit suite has no async tests)."""
    return "asyncio"
