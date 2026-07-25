"""Classify eval for the `clarify` node: the chosen decision vs a spec-derived gold label.

`clarify` weighs the planned capabilities' required inputs against what is known and returns one
of three decisions -- `continue`, `ask_user`, `best_effort`. We score that SINGLE label against
gold. Decisions are single-label, so exact-set-match over singleton sets is just accuracy, and
`label_prf` gives the per-decision diagnostic view.

To mirror production we build each case's `plan` with the REAL `plan` node from the case's
understanding, then run `clarify` (LLM only, no DB) -- exactly the `understand -> plan -> clarify`
sequence. One case exercises the deterministic child-ambiguous branch (always `ask_user`, no API
call). Gold labels are spec-derived (from the node's own decision contract), never model output.

Pure logic, no pytest: runs from `eval_regression` and directly
(`python -m evals.agent.pipeline.clarify.classify_run`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.messages import HumanMessage

from agent.pipeline.clarify import clarify
from agent.pipeline.plan import plan
from evals._harness import metrics
from evals._harness.cases import load_jsonl

# --- interface (required by evals/_harness/discovery.py) --------------------------------
STRATEGY = "classify"
NODE = "agent.pipeline.clarify"
THRESHOLDS = Path(__file__).with_name("classify_thresholds.json")

_DATASET = Path(__file__).with_name("classify_datasets.jsonl")
_LABELS = [
    "continue",
    "ask_user",
    "best_effort",
]  # the ClarificationDecision label space
_VALID = set(_LABELS)


def load_cases() -> list[dict]:
    """Load the labeled dataset (each case is self-contained: message + understanding + pin)."""
    return load_jsonl(_DATASET)


def _gold(case: dict) -> str:
    """Gold decision, validated against the decision contract (a typo fails loudly)."""
    decision = case.get("expected_decision")
    if decision not in _VALID:
        raise ValueError(
            f"case {case.get('id')}: unknown expected_decision {decision!r} "
            f"(expected one of {sorted(_VALID)})"
        )
    return decision


def predict(case: dict) -> dict[str, Any]:
    """Build state (understand->plan) and run `clarify`; return the chosen decision."""
    state: dict[str, Any] = {
        "messages": [HumanMessage(case.get("message", ""))],
        "understanding": case.get("understanding") or {},
        "target_child_id": case.get("target_child_id"),
        "reply_language": case.get("reply_language"),
    }
    try:
        state["plan"] = plan(state)["plan"]
        out = clarify(state)
    except Exception as exc:  # noqa: BLE001 -- an eval must survive any single-case failure
        return {"decision": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"decision": out["clarification"]["decision"], "error": None}


def run_all(cases: list[dict] | None = None) -> dict[str, Any]:
    """Predict every case, score decision accuracy + per-label F1, return a JSON-able report."""
    cases = cases or load_cases()
    rows: list[dict[str, Any]] = []
    pred_sets: list[set[str]] = []
    gold_sets: list[set[str]] = []
    for case in cases:
        gold = _gold(case)
        pred = predict(case)
        pred_decision = pred["decision"]
        pred_sets.append({pred_decision} if pred_decision else set())
        gold_sets.append({gold})
        row: dict[str, Any] = {
            "id": case.get("id"),
            "pred_decision": pred_decision,
            "gold_decision": gold,
            "ok": pred_decision == gold,
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
    return {"summary": summary, "per_label": prf["per_label"], "cases": rows}


if __name__ == "__main__":
    import sys

    from evals._harness import thresholds as thresholds_mod

    report = run_all()
    print("=== agent.pipeline.clarify / classify ===")  # noqa: T201
    print(report["summary"])  # noqa: T201
    for row in report["cases"]:
        flag = "" if row["ok"] else "  <-- MISS"
        if "error" in row:
            flag += f"  <-- ERROR {row['error']}"
        print(  # noqa: T201
            f"  [{row['id']}] pred={row['pred_decision']} gold={row['gold_decision']}{flag}"
        )
    if "--gate" in sys.argv:
        failures = thresholds_mod.assert_thresholds(
            report["summary"], thresholds_mod.load(THRESHOLDS)
        )
        if failures:
            print("\nTHRESHOLDS NOT MET:\n" + "\n".join(failures))  # noqa: T201
            sys.exit(1)
