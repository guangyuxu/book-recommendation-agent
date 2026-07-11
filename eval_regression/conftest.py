"""Pytest config for the regression gate -- same opt-in contract as the evals tree.

`eval_regression/` is a sibling of `evals/`, so `evals/conftest.py` does not apply here; this
re-establishes the `RUN_EVAL=1` gate and loads `.env` so the discovered node evals find
`ANTHROPIC_API_KEY`.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

RUN_EVAL = os.getenv("RUN_EVAL") == "1"

requires_eval = pytest.mark.skipif(
    not RUN_EVAL, reason="set RUN_EVAL=1 to run evals (calls the Anthropic API)"
)
