"""Node functions for the output-validation subgraph.

`select` picks the applicable checks and resets the per-turn accumulator; the fan-out runs each
selected check in parallel (one node per check, built in graph.py); `aggregate` reduces every
per-check outcome to a single rating in code ("worst wins"). No LLM here today -- the checks are
stubs -- so nothing needs `with_turn_context`; a future LLM-backed check must wrap itself.

CLAUDE.md: log only check names / outcomes / the rating -- never message text or profile fields.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..state import FlowState
from .checks import OutputCheck, applicable_checks, node_name
from .schemas import (
    CheckOutcome,
    CheckResult,
    Rating,
    ValidationResult,
    outcome_to_rating,
    severity,
)

logger = logging.getLogger(__name__)


def _capabilities(state: FlowState) -> set[str]:
    """Return the capability names that produced output this turn."""
    return set((state.get("capability_results") or {}).keys())


def select(state: FlowState) -> dict[str, Any]:
    """Reset the accumulator for this turn. The fan-out target is chosen by `route_checks`.

    Returns an empty `output_checks`, which the channel's reducer (agent.state.merge_output_checks)
    treats as a reset -- so a prior turn's results never leak into this one under a checkpointer.
    """
    return {"output_checks": []}


def route_checks(state: FlowState) -> list[str]:
    """Conditional edge: fan out to the applicable check nodes, or straight to aggregate if none."""
    nodes = [node_name(c) for c in applicable_checks(_capabilities(state))]
    return nodes or ["aggregate"]


def make_check_node(check: OutputCheck) -> Callable[[FlowState], dict[str, Any]]:
    """Build the worker node for one check: run it and append its result to the accumulator.

    A failing check must not sink the turn; on error we record a BLOCK (fail-safe: an output we
    could not validate is not shown as-is) and log the exception TYPE only, never its message.
    """

    def _node(state: FlowState) -> dict[str, Any]:
        try:
            result = check.run(dict(state))
        except Exception as exc:  # noqa: BLE001 -- never crash the turn; fail safe to BLOCK
            logger.warning("check %s failed: %s", check.name, type(exc).__name__)
            result = CheckResult(
                check=check.name, outcome=CheckOutcome.BLOCK, reason="check errored"
            )
        return {"output_checks": [result.model_dump(mode="json")]}

    _node.__name__ = node_name(check)
    return _node


def aggregate(state: FlowState) -> dict[str, Any]:
    """Reduce all per-check outcomes to one rating in code: the worst (max-severity) outcome wins.

    No checks -> ALLOW (nothing objected). Malformed accumulator entries are ignored defensively.
    """
    results: list[CheckResult] = []
    for raw in state.get("output_checks") or []:
        try:
            results.append(CheckResult.model_validate(raw))
        except Exception:  # noqa: BLE001 -- a malformed entry must not crash aggregation
            continue

    if results:
        worst = max((r.outcome for r in results), key=severity)
        rating = outcome_to_rating(worst)
    else:
        rating = Rating.ALLOW

    logger.info(
        "validation: rating=%s checks=%s",
        rating.value,
        {r.check: r.outcome.value for r in results},
    )
    return {"validation": ValidationResult(rating=rating, results=results).model_dump(mode="json")}
