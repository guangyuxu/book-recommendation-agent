"""Classify eval for the `understand` node: intent set + resolved target child vs gold labels.

Runs the REAL `understand` node (LLM only, no DB) on each labeled case and scores the two
structured decisions with a deterministic ground truth:
  - `intents` -- a MULTI-LABEL set; the product bar is exact-set-match (no misses, no extras).
  - `target_child_id` -- the deterministic child resolution, scored on flagged cases.

Pure logic, no pytest, so it runs both from `eval_regression` and directly from the CLI
(`python -m evals.agent.pipeline.understand.classify_run`). Gold labels are spec-derived (from
`src/agent/intents.py`), never produced by the model under test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.messages import HumanMessage
from pydantic import BaseModel, Field

from agent.intents import Intent
from agent.pipeline.understand import understand
from evals._harness import metrics
from evals._harness.cases import load_jsonl

# --- interface (required by evals/_harness/discovery.py) --------------------------------
STRATEGY = "classify"
NODE = "agent.pipeline.understand"
THRESHOLDS = Path(__file__).with_name("classify_thresholds.json")

_DATASET = Path(__file__).with_name("classify_datasets.jsonl")
_LABELS = [i.value for i in Intent]
_VALID_INTENTS = set(_LABELS)


class UnderstandCase(BaseModel):
    """One labeled `understand` example -- self-contained so scoring needs no database."""

    id: str
    message: str
    children: dict[str, dict] = Field(default_factory=dict)
    target_child_id: str | None = None  # the pinned/active child going in (may be None)

    expected_intents: list[str] = Field(default_factory=list)
    expected_target_child_id: str | None = None
    score_child_resolution: bool = False

    def gold_intents(self) -> set[str]:
        """Gold intents as a set, validated against the Intent enum (typos fail loudly)."""
        unknown = set(self.expected_intents) - _VALID_INTENTS
        if unknown:
            raise ValueError(
                f"case {self.id}: unknown expected_intents {sorted(unknown)}"
            )
        return set(self.expected_intents)


def load_cases() -> list[UnderstandCase]:
    """Load and validate the dataset into `UnderstandCase` objects."""
    return [UnderstandCase(**row) for row in load_jsonl(_DATASET)]


def predict(case: UnderstandCase) -> dict[str, Any]:
    """Run one case through `understand`; return the scored fields (an `error` marker on failure)."""
    state = {
        "messages": [HumanMessage(case.message)],
        "children": case.children,
        "target_child_id": case.target_child_id,
    }
    try:
        out = understand(state)
    except Exception as exc:  # noqa: BLE001 -- an eval must survive any single-case failure
        return {
            "intents": set(),
            "target_child_id": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "intents": set(out["understanding"]["intents"]),
        "target_child_id": out["target_child_id"],
        "error": None,
    }


def run_all(cases: list[UnderstandCase] | None = None) -> dict[str, Any]:
    """Predict every case, score with harness metrics, and return a JSON-able report."""
    cases = cases or load_cases()
    rows: list[dict[str, Any]] = []
    pred_sets: list[set[str]] = []
    gold_sets: list[set[str]] = []
    pred_child: list[str | None] = []
    gold_child: list[str | None] = []

    for case in cases:
        gold = case.gold_intents()
        pred = predict(case)
        pred_sets.append(pred["intents"])
        gold_sets.append(gold)

        row: dict[str, Any] = {
            "id": case.id,
            "pred_intents": sorted(pred["intents"]),
            "gold_intents": sorted(gold),
            "intents_ok": pred["intents"] == gold,
        }
        if pred.get("error"):
            row["error"] = pred["error"]
        if case.score_child_resolution:
            pred_child.append(pred["target_child_id"])
            gold_child.append(case.expected_target_child_id)
            row["pred_child"] = pred["target_child_id"]
            row["gold_child"] = case.expected_target_child_id
            row["child_ok"] = pred["target_child_id"] == case.expected_target_child_id
        rows.append(row)

    prf = metrics.label_prf(pred_sets, gold_sets, _LABELS)
    summary: dict[str, float] = {
        "n_cases": float(len(cases)),
        "n_errors": float(sum(1 for r in rows if "error" in r)),
        "exact_match": metrics.set_exact_match(pred_sets, gold_sets),
        "micro_f1": prf["micro_f1"],
        "macro_f1": prf["macro_f1"],
        "n_child_scored": float(len(gold_child)),
        "child_resolution": metrics.resolution_accuracy(pred_child, gold_child),
    }
    return {"summary": summary, "per_label": prf["per_label"], "cases": rows}


def _print_report(report: dict[str, Any]) -> None:
    s = report["summary"]
    print("=== understand / classify ===")  # noqa: T201
    print(  # noqa: T201
        f"cases={int(s['n_cases'])} errors={int(s['n_errors'])} "
        f"exact_match={s['exact_match']:.3f} macro_f1={s['macro_f1']:.3f} "
        f"micro_f1={s['micro_f1']:.3f} child_resolution={s['child_resolution']:.3f} "
        f"(n={int(s['n_child_scored'])})"
    )
    for row in report["cases"]:
        flags = "" if row["intents_ok"] else "  <-- INTENT MISS"
        if row.get("child_ok") is False:
            flags += "  <-- CHILD MISS"
        if "error" in row:
            flags += f"  <-- ERROR {row['error']}"
        line = f"  [{row['id']}] pred={row['pred_intents']} gold={row['gold_intents']}{flags}"
        print(line)  # noqa: T201


if __name__ == "__main__":
    import sys

    from evals._harness import thresholds as thresholds_mod

    report = run_all()
    _print_report(report)
    if "--gate" in sys.argv:
        failures = thresholds_mod.assert_thresholds(
            report["summary"], thresholds_mod.load(THRESHOLDS)
        )
        if failures:
            print("\nTHRESHOLDS NOT MET:\n" + "\n".join(failures))  # noqa: T201
            sys.exit(1)
