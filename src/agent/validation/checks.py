"""Output-check registry: the unified check interface and the (example) set of checks.

Each `OutputCheck` is a declarative record -- a name, which capabilities' output it guards, and a
`run(state) -> CheckResult` implementation -- so the subgraph can select and dispatch checks
without importing any check's internals (the same shape as the capability REGISTRY).

    !!! EXAMPLE / STUB LIST -- WILL CHANGE !!!
    The seven checks below and their capability mapping are an illustrative starting set. Every
    `run` is a HARDCODED STUB that returns PASS; none inspects the output yet. Real logic (LLM
    graders, PII scans, policy rules, groundedness checks against a book corpus, ...) is meant to
    be dropped into each `run` later WITHOUT touching the subgraph wiring or the aggregation. The
    set of checks, their `applies_to` mapping, and their severity are all expected to evolve.

`applies_to = ()` means the check ALWAYS runs (a turn-independent guard like child-safety /
privacy / product-values); a non-empty tuple means the check runs only when at least one of those
capabilities produced output this turn.

CLAUDE.md: a check `run` receives the full state (it may need the child profile / policies to do
its real job) but must NEVER log message text or profile fields -- log only names/outcomes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .schemas import CheckOutcome, CheckResult


@dataclass(frozen=True)
class OutputCheck:
    """One output check and the capabilities whose output it guards."""

    name: str
    description: str
    run: Callable[[dict[str, Any]], CheckResult]
    # Capabilities this check guards; empty tuple = always applies (turn-independent guard).
    applies_to: tuple[str, ...] = ()


# --- STUB implementations (all PASS) --------------------------------------------------------
# Replace each body with real logic. Signature is fixed: (state) -> CheckResult. Keep the
# `check=` name in sync with the registry key.


def _passed(name: str) -> CheckResult:
    """Return a neutral PASS result (the placeholder every stub returns for now)."""
    return CheckResult(check=name, outcome=CheckOutcome.PASS, reason="stub: not yet implemented")


def run_child_safety(state: dict[str, Any]) -> CheckResult:
    """STUB: is the output free of content unsafe for a child? Always PASS for now."""
    return _passed("child_safety")


def run_privacy(state: dict[str, Any]) -> CheckResult:
    """STUB: does the output avoid leaking PII (names, birthdays, ...)? Always PASS for now."""
    return _passed("privacy")


def run_product_values(state: dict[str, Any]) -> CheckResult:
    """STUB: does the output align with the product's tone/values? Always PASS for now."""
    return _passed("product_values")


def run_age_appropriateness(state: dict[str, Any]) -> CheckResult:
    """STUB: do the recommended/described books fit the child's age? Always PASS for now."""
    return _passed("age_appropriateness")


def run_factuality(state: dict[str, Any]) -> CheckResult:
    """STUB: are book claims real/grounded (titles, authors, facts)? Always PASS for now."""
    return _passed("factuality")


def run_recommendation_policy(state: dict[str, Any]) -> CheckResult:
    """STUB: does the recommendation obey the family reading policy? Always PASS for now."""
    return _passed("recommendation_policy")


def run_discussion_policy(state: dict[str, Any]) -> CheckResult:
    """STUB: do discussion questions obey policy (safe, on-topic)? Always PASS for now."""
    return _passed("discussion_policy")


# --- Registry (example mapping) -------------------------------------------------------------

CHECK_REGISTRY: dict[str, OutputCheck] = {
    # Turn-independent guards -- run on every answer, whatever ran.
    "child_safety": OutputCheck(
        "child_safety", "Screen the output for content unsafe for a child.", run_child_safety
    ),
    "privacy": OutputCheck(
        "privacy", "Ensure the output does not leak personal data.", run_privacy
    ),
    "product_values": OutputCheck(
        "product_values", "Ensure the output aligns with product tone/values.", run_product_values
    ),
    # Capability-scoped checks.
    "age_appropriateness": OutputCheck(
        "age_appropriateness",
        "Check the books fit the child's age.",
        run_age_appropriateness,
        applies_to=("recommend", "evaluate", "path", "discussion"),
    ),
    "factuality": OutputCheck(
        "factuality",
        "Check book claims are real and grounded.",
        run_factuality,
        applies_to=("recommend", "evaluate", "compare"),
    ),
    "recommendation_policy": OutputCheck(
        "recommendation_policy",
        "Check the recommendation obeys the family reading policy.",
        run_recommendation_policy,
        applies_to=("recommend",),
    ),
    "discussion_policy": OutputCheck(
        "discussion_policy",
        "Check discussion questions obey policy.",
        run_discussion_policy,
        applies_to=("discussion",),
    ),
}


def node_name(check: OutputCheck) -> str:
    """LangGraph node name for a check (stable, derived from the registry key)."""
    return f"check_{check.name}"


def applicable_checks(capabilities: set[str]) -> list[OutputCheck]:
    """Return the checks that apply this turn: always-on guards plus capability-scoped matches.

    `capabilities` is the set of capability names that produced output this turn
    (state["capability_results"] keys). Order follows the registry for deterministic fan-out.
    """
    return [
        check
        for check in CHECK_REGISTRY.values()
        if not check.applies_to or capabilities.intersection(check.applies_to)
    ]
