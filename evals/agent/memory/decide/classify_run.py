"""Classify eval for the `memory_policy` node (memory subgraph): chosen operation SET vs gold.

`memory_policy` reads a turn's `user_signals` (+ `child_is_new`) and decides which durable domain
operations to persist. We score the SET of operation NAMES it chose against a spec-derived gold
set drawn from the real tool menu (`agent.domain.MEMORY_TOOLS_BY_NAME`). This is the classify
strategy applied to a subgraph node -- the eval layout mirrors `src/agent` all the way down.

Exact-set-match is strict (memory decisions are genuinely fuzzy), so `macro_f1` is the primary
diagnostic and thresholds are conservative. One case exercises the deterministic skip path
(nothing to remember), which needs no API call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.messages import HumanMessage

from agent.domain import MEMORY_TOOLS_BY_NAME
from agent.memory.decide import memory_policy
from evals._harness import metrics
from evals._harness.cases import load_jsonl

# --- interface (required by evals/_harness/discovery.py) --------------------------------
STRATEGY = "classify"
NODE = "agent.memory.decide"
THRESHOLDS = Path(__file__).with_name("classify_thresholds.json")

_DATASET = Path(__file__).with_name("classify_datasets.jsonl")
_LABELS = list(MEMORY_TOOLS_BY_NAME)  # the operation-name label space
_VALID = set(_LABELS)


def load_cases() -> list[dict]:
    """Load the labeled dataset (each case carries its own `understanding` + message)."""
    return load_jsonl(_DATASET)


def _gold(case: dict) -> set[str]:
    """Gold operation-name set, validated against the real tool menu (typos fail loudly)."""
    ops = set(case.get("expected_operations") or [])
    unknown = ops - _VALID
    if unknown:
        raise ValueError(
            f"case {case.get('id')}: unknown expected_operations {sorted(unknown)}"
        )
    return ops


def predict(case: dict) -> dict[str, Any]:
    """Run memory_policy on the case and return the chosen operation-name set."""
    state = {
        "understanding": case.get("understanding") or {},
        "messages": [HumanMessage(case.get("message", ""))],
    }
    try:
        out = memory_policy(state)
    except Exception as exc:  # noqa: BLE001 -- survive any single-case failure
        return {"operations": set(), "error": f"{type(exc).__name__}: {exc}"}
    names = {op.get("operation") for op in out.get("memory_operations") or []}
    return {"operations": {n for n in names if n}, "error": None}


def run_all(cases: list[dict] | None = None) -> dict[str, Any]:
    """Predict every case, score set-match + per-label F1, return a JSON-able report."""
    cases = cases or load_cases()
    rows: list[dict[str, Any]] = []
    pred_sets: list[set[str]] = []
    gold_sets: list[set[str]] = []
    for case in cases:
        gold = _gold(case)
        pred = predict(case)
        pred_sets.append(pred["operations"])
        gold_sets.append(gold)
        row: dict[str, Any] = {
            "id": case.get("id"),
            "pred_operations": sorted(pred["operations"]),
            "gold_operations": sorted(gold),
            "ok": pred["operations"] == gold,
        }
        if pred.get("error"):
            row["error"] = pred["error"]
        rows.append(row)

    prf = metrics.label_prf(pred_sets, gold_sets, _LABELS)
    summary: dict[str, float] = {
        "n_cases": float(len(cases)),
        "n_errors": float(sum(1 for r in rows if "error" in r)),
        "exact_match": metrics.set_exact_match(pred_sets, gold_sets),
        "micro_f1": prf["micro_f1"],
        "macro_f1": prf["macro_f1"],
    }
    return {"summary": summary, "cases": rows}


if __name__ == "__main__":
    import sys

    from evals._harness import thresholds as thresholds_mod

    report = run_all()
    print("=== agent.memory.decide / classify ===")  # noqa: T201
    print(report["summary"])  # noqa: T201
    for row in report["cases"]:
        flag = "" if row["ok"] else "  <-- MISS"
        line = f"  [{row['id']}] pred={row['pred_operations']} gold={row['gold_operations']}{flag}"
        print(line)  # noqa: T201
    if "--gate" in sys.argv:
        failures = thresholds_mod.assert_thresholds(
            report["summary"], thresholds_mod.load(THRESHOLDS)
        )
        if failures:
            print("\nTHRESHOLDS NOT MET:\n" + "\n".join(failures))  # noqa: T201
            sys.exit(1)
