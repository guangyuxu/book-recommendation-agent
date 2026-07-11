"""Threshold loading and gating.

Thresholds live next to each strategy as a small JSON file (e.g.
`s1_classification/thresholds.json`) so the pass/fail bar is version-controlled and reviewable in
a PR. `assert_thresholds` compares a flat metrics dict against a config of `{"<metric>_min":
value}` keys and returns human-readable failures (empty list == pass).
"""

from __future__ import annotations

import json
from pathlib import Path


def load(path: str | Path) -> dict[str, float]:
    """Load a thresholds JSON file into a `{key: number}` dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_thresholds(metrics: dict[str, float], cfg: dict[str, float]) -> list[str]:
    """Return a list of failure messages; empty means every configured minimum was met.

    Convention: each config key `"<metric>_min"` requires `metrics["<metric>"] >= value`. A
    config key with no matching metric is itself a failure (a typo shouldn't silently pass).
    """
    failures: list[str] = []
    for key, floor in cfg.items():
        if not key.endswith("_min"):
            failures.append(f"unknown threshold key {key!r} (expected a '*_min' key)")
            continue
        metric = key[: -len("_min")]
        if metric not in metrics:
            failures.append(f"threshold {key!r} references missing metric {metric!r}")
            continue
        actual = metrics[metric]
        if actual < floor:
            failures.append(f"{metric}: {actual:.3f} < required {floor:.3f}")
    return failures
