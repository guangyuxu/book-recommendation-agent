"""The pytest gate: one parametrized test per discovered node eval, marked by strategy.

Discovery drives collection, so categories and single nodes select for free:

    RUN_EVAL=1 pytest eval_regression -m classify        # all classify nodes
    RUN_EVAL=1 pytest eval_regression -m judge           # all judge nodes
    RUN_EVAL=1 pytest eval_regression -k understand      # one node
    RUN_EVAL=1 pytest eval_regression                    # everything

Each param carries its strategy as a pytest marker (`classify` / `judge`, registered in
pyproject.toml). Opt-in via `RUN_EVAL=1` -- these call the Anthropic API.
"""

from __future__ import annotations

import pytest

from evals._harness import report as report_mod
from evals._harness import thresholds as thresholds_mod
from evals._harness.discovery import discover

from .conftest import requires_eval

pytestmark = [requires_eval]


def _params() -> list:
    """Build one marked pytest param per discovered node eval (id = 'node:strategy')."""
    params = []
    for ev in discover():
        mark = getattr(pytest.mark, ev.strategy)  # classify | judge | ...
        params.append(pytest.param(ev, marks=mark, id=ev.id))
    return params


@pytest.mark.parametrize("ev", _params())
def test_node_eval_meets_thresholds(ev) -> None:  # noqa: ANN001
    """Run one node eval and assert its summary clears its co-located thresholds."""
    report = ev.run_all()
    summary = report["summary"]

    written = report_mod.write_report(ev.id.replace(":", "_"), report)
    print(f"\n[{ev.id}] report -> {written}")  # noqa: T201

    failures = thresholds_mod.assert_thresholds(
        summary, thresholds_mod.load(ev.thresholds)
    )
    if failures:
        print(f"[{ev.id}] misses:")  # noqa: T201
        for row in report.get("cases", []):
            print(f"  {row}")  # noqa: T201
    assert not failures, f"{ev.id} thresholds not met:\n" + "\n".join(failures)
