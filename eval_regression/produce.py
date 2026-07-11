"""Threshold producer: derive suggested pass/fail floors from observed metrics and write them back.

The other half of the eval_regression contract. `run.py` GATES against thresholds; `produce.py`
GENERATES them. It runs each node eval (optionally several times, to sample judge variance), then
for every `"<metric>_min"` key already present in that node's `<strategy>_thresholds.json` it
proposes a floor of `min(observed across runs) - margin`, clamped to >= 0 and rounded.

    python -m eval_regression.produce --dry-run          # print the diff, write nothing
    python -m eval_regression.produce --margin 0.1        # looser floors
    python -m eval_regression.produce --repeats 3          # sample variance (recommended for judge)
    python -m eval_regression.produce --strategy judge     # only judge nodes

It only ever touches metrics that ALREADY have a `_min` key -- it tightens/loosens existing gates,
it never invents new ones (adding a gate stays a deliberate, reviewed edit to the json).
"""

from __future__ import annotations

import argparse
import json

from evals._harness import thresholds as thresholds_mod
from evals._harness.discovery import discover


def _observed_floor(values: list[float], margin: float) -> float:
    """Suggested floor: worst observed value minus the margin, clamped to [0, ...], rounded."""
    return round(max(0.0, min(values) - margin), 3)


def produce(
    strategy: str | None,
    node: str | None,
    margin: float,
    repeats: int,
    dry_run: bool,
) -> int:
    changed = 0
    for ev in discover(strategy=strategy, node=node):
        cfg = thresholds_mod.load(ev.thresholds)
        # Collect each gated metric's observed values across `repeats` runs.
        observed: dict[str, list[float]] = {}
        for _ in range(max(1, repeats)):
            summary = ev.run_all().get("summary", {})
            for key in cfg:
                metric = key[:-4] if key.endswith("_min") else key
                if metric in summary and isinstance(summary[metric], (int, float)):
                    observed.setdefault(key, []).append(float(summary[metric]))

        proposed = dict(cfg)
        for key, vals in observed.items():
            proposed[key] = _observed_floor(vals, margin)

        if proposed != cfg:
            changed += 1
            print(f"\n{ev.id}  ({ev.thresholds})")  # noqa: T201
            for key in sorted(proposed):
                if proposed.get(key) != cfg.get(key):
                    print(f"    {key}: {cfg.get(key)} -> {proposed[key]}")  # noqa: T201
            if not dry_run:
                ev.thresholds.write_text(
                    json.dumps(proposed, indent=2) + "\n", encoding="utf-8"
                )

    if dry_run:
        summary = f"\n(dry-run) {changed} threshold file(s) would change"
    else:
        summary = f"\n{changed} threshold file(s) updated"
    print(summary)  # noqa: T201
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Produce/refresh co-located eval thresholds."
    )
    ap.add_argument("--strategy", help="only this strategy")
    ap.add_argument(
        "--node", help="only nodes whose dotted path contains this substring"
    )
    ap.add_argument(
        "--margin", type=float, default=0.05, help="floor = worst observed - margin"
    )
    ap.add_argument(
        "--repeats", type=int, default=1, help="runs per node (sample variance)"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the diff, write nothing"
    )
    args = ap.parse_args(argv)
    return produce(args.strategy, args.node, args.margin, args.repeats, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
