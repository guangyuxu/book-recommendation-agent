"""Regression gate: run node evals, assert each against its co-located thresholds, aggregate.

This is the single entrypoint CI calls. It does NOT re-implement any eval -- it discovers node
evals under `evals/` (see `evals/_harness/discovery.py`), calls each module's `run_all()` (the
execution seam evals exposes), and gates the returned `summary` against that node's own
`<strategy>_thresholds.json` (thresholds are co-located, reviewable next to the dataset).

Run everything, or a category, or one node:

    python -m eval_regression.run                     # all node evals
    python -m eval_regression.run --strategy classify # only classify nodes
    python -m eval_regression.run --strategy judge     # only judge nodes
    python -m eval_regression.run --node understand    # one node (substring match)

Exits non-zero if any node misses a threshold, printing a per-node failure list.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from evals._harness import report as report_mod
from evals._harness import thresholds as thresholds_mod
from evals._harness.discovery import discover


def gate(
    strategy: str | None = None, node: str | None = None
) -> tuple[bool, list[dict[str, Any]]]:
    """Run the selected node evals and gate each. Returns (all_passed, per-node results)."""
    results: list[dict[str, Any]] = []
    ok_all = True
    for ev in discover(strategy=strategy, node=node):
        report = ev.run_all()
        summary = report.get("summary", {})
        cfg = thresholds_mod.load(ev.thresholds)
        failures = thresholds_mod.assert_thresholds(summary, cfg)
        report_mod.write_report(ev.id.replace(":", "_"), report)
        ok_all = ok_all and not failures
        results.append({"id": ev.id, "summary": summary, "failures": failures})
    return ok_all, results


def _print(results: list[dict[str, Any]]) -> None:
    for r in results:
        status = "PASS" if not r["failures"] else "FAIL"
        print(f"[{status}] {r['id']}")  # noqa: T201
        for k, v in r["summary"].items():
            line = (
                f"        {k}={v:.3f}" if isinstance(v, float) else f"        {k}={v}"
            )
            print(line)  # noqa: T201
        for f in r["failures"]:
            print(f"    !!  {f}")  # noqa: T201


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run node evals and gate on co-located thresholds."
    )
    ap.add_argument("--strategy", help="only this strategy (e.g. classify | judge)")
    ap.add_argument(
        "--node", help="only nodes whose dotted path contains this substring"
    )
    args = ap.parse_args(argv)

    selected = discover(strategy=args.strategy, node=args.node)
    if not selected:
        print("no node evals matched the selection", file=sys.stderr)  # noqa: T201
        return 2

    ok, results = gate(strategy=args.strategy, node=args.node)
    _print(results)
    verdict = "ALL PASSED" if ok else "FAILURES PRESENT"
    print(f"\n{verdict} ({len(results)} node eval(s))")  # noqa: T201
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
