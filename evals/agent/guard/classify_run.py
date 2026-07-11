"""Classify eval for the `guard` node: does the input safety gate block attacks and pass benign?

Runs the REAL `guard` node (which calls Meta's Llama Prompt Guard 2 via Groq) on each labeled
case and scores its binary verdict against human/spec-derived gold:
  - gold "attack"  -> the node MUST block  (safety.blocked is True)
  - gold "benign"  -> the node MUST allow  (safety.blocked is False)

This is a binary problem, so we reuse the multi-label harness with single-element label sets
({"attack"} / {"benign"}): `set_exact_match` is overall accuracy, and per-label recall gives the
two metrics that actually matter for a safety filter:
  - `attack_recall`    -- of attacks, the fraction blocked (catch rate; the security bar).
  - `benign_pass_rate` -- of benign turns, the fraction allowed (1 - false-positive rate; the
                          UX bar -- do not block real parents).

Gold is decided by a human from what a prompt-injection IS (instruction override, system-prompt
exfiltration, injected instructions in pasted text), never by asking the classifier. Pure logic,
no pytest -- runs from `eval_regression` and directly (`python -m evals.agent.guard.classify_run`).

Needs GROQ_API_KEY (this node's model runs on Groq, not Anthropic). Without it the guard fails
open (blocks nothing), so `attack_recall` collapses to 0 and the gate fails loudly -- which is
the correct signal that screening is not actually wired up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.messages import HumanMessage
from pydantic import BaseModel

from agent.guard import guard
from evals._harness import metrics
from evals._harness.cases import load_jsonl

# --- interface (required by evals/_harness/discovery.py) --------------------------------
STRATEGY = "classify"
NODE = "agent.guard"
THRESHOLDS = Path(__file__).with_name("classify_thresholds.json")

_DATASET = Path(__file__).with_name("classify_datasets.jsonl")
_LABELS = ["attack", "benign"]
_VALID_LABELS = set(_LABELS)


class GuardCase(BaseModel):
    """One labeled `guard` example -- self-contained; scoring touches no database."""

    id: str
    message: str
    expected_label: str  # "attack" | "benign"
    provenance: str = "synthetic"
    difficulty: str = "medium"

    def gold_label(self) -> str:
        """Gold label, validated against the label set (a typo fails loudly, not as a wrong score)."""
        if self.expected_label not in _VALID_LABELS:
            raise ValueError(
                f"case {self.id}: unknown expected_label {self.expected_label!r} "
                f"(want one of {sorted(_VALID_LABELS)})"
            )
        return self.expected_label


def load_cases() -> list[GuardCase]:
    """Load and validate the dataset into `GuardCase` objects."""
    return [GuardCase(**row) for row in load_jsonl(_DATASET)]


def predict(case: GuardCase) -> dict[str, Any]:
    """Run one case through `guard`; return its verdict ('attack'/'benign') and score.

    Wrapped in try/except so a single-case failure is scored as an error, never a batch-killer.
    """
    state = {"messages": [HumanMessage(case.message)]}
    try:
        out = guard(state)
    except Exception as exc:  # noqa: BLE001 -- an eval must survive any single-case failure
        return {"label": None, "score": None, "error": f"{type(exc).__name__}: {exc}"}
    safety = out.get("safety") or {}
    blocked = bool(safety.get("blocked"))
    return {
        "label": "attack" if blocked else "benign",
        "score": safety.get("score"),
        "error": None,
    }


def run_all(cases: list[GuardCase] | None = None) -> dict[str, Any]:
    """Predict every case, score with harness metrics, and return a JSON-able report."""
    cases = cases or load_cases()
    rows: list[dict[str, Any]] = []
    pred_sets: list[set[str]] = []
    gold_sets: list[set[str]] = []

    for case in cases:
        gold = case.gold_label()
        pred = predict(case)
        # None label (an error) becomes an empty set -> never equals gold -> scored as a miss.
        pred_sets.append({pred["label"]} if pred["label"] else set())
        gold_sets.append({gold})

        row: dict[str, Any] = {
            "id": case.id,
            "pred": pred["label"],
            "gold": gold,
            "score": pred["score"],
            "ok": pred["label"] == gold,
        }
        if pred.get("error"):
            row["error"] = pred["error"]
        rows.append(row)

    prf = metrics.label_prf(pred_sets, gold_sets, _LABELS)
    summary: dict[str, float] = {
        "n_cases": float(len(cases)),
        "n_errors": float(sum(1 for r in rows if "error" in r)),
        "accuracy": metrics.set_exact_match(pred_sets, gold_sets),
        # recall of "attack" = fraction of attacks we blocked; recall of "benign" = pass rate.
        "attack_recall": prf["per_label"]["attack"]["recall"],
        "benign_pass_rate": prf["per_label"]["benign"]["recall"],
    }
    return {"summary": summary, "per_label": prf["per_label"], "cases": rows}


def _print_report(report: dict[str, Any]) -> None:
    s = report["summary"]
    print("=== guard / classify ===")  # noqa: T201
    print(  # noqa: T201
        f"cases={int(s['n_cases'])} errors={int(s['n_errors'])} "
        f"accuracy={s['accuracy']:.3f} attack_recall={s['attack_recall']:.3f} "
        f"benign_pass_rate={s['benign_pass_rate']:.3f}"
    )
    for row in report["cases"]:
        flags = "" if row["ok"] else "  <-- MISS"
        if "error" in row:
            flags += f"  <-- ERROR {row['error']}"
        score = f"{row['score']:.4f}" if isinstance(row["score"], float) else "n/a"
        print(  # noqa: T201
            f"  [{row['id']}] pred={row['pred']} gold={row['gold']} score={score}{flags}"
        )


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
