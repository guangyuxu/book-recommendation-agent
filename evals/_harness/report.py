"""Result output: write a run report to disk, and the (stubbed) LangSmith export seam.

This module is the hinge between the self-built engine and a future hybrid setup. Today every
run writes a timestamped JSON report under `evals/results/` (gitignored). When you later want a
dashboard and historical trend lines, implement `report_to_langsmith` -- the directory layout,
datasets, and scoring code do NOT change.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# evals/_harness/report.py -> evals/results
_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def write_report(name: str, payload: dict[str, Any]) -> Path:
    """Write `payload` as pretty JSON to `evals/results/{name}-{YYYYmmdd-HHMMSS}.json`.

    Returns the path written. The timestamp keeps every run's report so you can eyeball a trend
    by listing the directory, even before a real dashboard exists.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = _RESULTS_DIR / f"{name}-{stamp}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def report_to_langsmith(name: str, payload: dict[str, Any]) -> None:
    """Seam for pushing a run to LangSmith (upgrades the harness to the hybrid model).

    Intentionally a no-op stub for now: the self-built engine is the source of truth. When you
    want UI comparison / trend history, implement this against the LangSmith SDK (the project
    already ships LANGSMITH_* env vars). Nothing else in the harness needs to move.
    """
    msg = f"[report] langsmith export not enabled (name={name!r}); wrote local report only"
    print(msg)  # noqa: T201
