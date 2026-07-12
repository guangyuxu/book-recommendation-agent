"""Types for the output-validation subgraph: per-check outcomes and the aggregated rating.

`str`-based enums so they serialize cleanly into the JSON-able state channels (`model_dump(
mode="json")` yields the plain strings the rest of the pipeline and the frontend read).

The two scales are intentionally parallel and ordered by severity so aggregation is a trivial
"worst wins": one CheckOutcome maps 1:1 to the Rating it would force on its own.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class CheckOutcome(StrEnum):
    """One check's verdict on the prepared output, least to most severe."""

    PASS = "pass"
    WARN = "warn"
    REWRITE = "rewrite"
    BLOCK = "block"


class Rating(StrEnum):
    """The turn-level decision `respond` acts on, least to most severe."""

    ALLOW = "ALLOW"
    WARNING = "WARNING"
    REWRITE = "REWRITE"
    BLOCK = "BLOCK"


# Severity rank (higher = worse) drives the "worst wins" aggregation in nodes.aggregate.
_SEVERITY: dict[CheckOutcome, int] = {
    CheckOutcome.PASS: 0,
    CheckOutcome.WARN: 1,
    CheckOutcome.REWRITE: 2,
    CheckOutcome.BLOCK: 3,
}

# Each outcome maps to the rating it forces on its own; the aggregate is the max-severity one.
_OUTCOME_TO_RATING: dict[CheckOutcome, Rating] = {
    CheckOutcome.PASS: Rating.ALLOW,
    CheckOutcome.WARN: Rating.WARNING,
    CheckOutcome.REWRITE: Rating.REWRITE,
    CheckOutcome.BLOCK: Rating.BLOCK,
}


def severity(outcome: CheckOutcome) -> int:
    """Return the severity rank of an outcome (higher is worse)."""
    return _SEVERITY[outcome]


def outcome_to_rating(outcome: CheckOutcome) -> Rating:
    """Map a single check outcome to the turn-level rating it would force on its own."""
    return _OUTCOME_TO_RATING[outcome]


class CheckResult(BaseModel):
    """The result of running one output check."""

    check: str
    outcome: CheckOutcome
    reason: str = ""


class ValidationResult(BaseModel):
    """The aggregated verdict for the turn: one rating plus the checks that ran."""

    rating: Rating
    results: list[CheckResult] = Field(default_factory=list)
