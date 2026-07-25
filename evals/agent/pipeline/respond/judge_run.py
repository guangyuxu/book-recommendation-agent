"""Judge eval for the `respond` node: rubric-scored reply quality (1-5 dimensions).

`respond` composes the single user-facing reply from the prepared capability material via an LLM,
so there is no single right answer -- we score it with an LLM judge against `judge_rubric.md` on
three dimensions: **faithfulness** (uses only the prepared material; invents no books or facts),
**relevance** (answers the parent's latest message and reflects any switch / confirmation note),
and **language** (correct reply language, warm, concise, error-free).

The node also *persists* a recommendation turn -- but only when a recommend booklist AND a resolved
`target_child_id` are both present. Every case here deliberately leaves `target_child_id` unset (or
carries no booklist), so the deterministic persistence path is skipped and the eval stays LLM-only
(no DB), judging exactly the composed reply.

Pure logic, no pytest: runs from `eval_regression` and directly
(`python -m evals.agent.pipeline.respond.judge_run`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.messages import HumanMessage

from agent.pipeline.respond import respond
from evals._harness import judge as judge_mod
from evals._harness import metrics
from evals._harness.cases import load_jsonl

# --- interface (required by evals/_harness/discovery.py) --------------------------------
STRATEGY = "judge"
NODE = "agent.pipeline.respond"
THRESHOLDS = Path(__file__).with_name("judge_thresholds.json")

# Keep in sync with judge_rubric.md headers and judge_thresholds.json ("mean_<dim>_min").
DIMENSIONS: tuple[str, ...] = ("faithfulness", "relevance", "language")
_PASS_FLOOR = 3  # a case "passes" when EVERY dimension >= this

_DATASET = Path(__file__).with_name("judge_datasets.jsonl")
_RUBRIC = Path(__file__).with_name("judge_rubric.md")

_LANGUAGE_NAME = {
    "en": "English",
    "zh-Hans": "Simplified Chinese",
    "zh-Hant": "Traditional Chinese",
}


def load_cases() -> list[dict]:
    """Load the judge scenarios (self-contained: message + capability_results + reply hints)."""
    return load_jsonl(_DATASET)


def _state(case: dict) -> dict[str, Any]:
    """Build the minimal FlowState `respond` reads. No target_child_id -> no DB persistence."""
    return {
        "messages": [HumanMessage(case["message"])],
        "capability_results": case.get("capability_results") or {},
        "understanding": case.get("understanding") or {},
        "confirmation": case.get("confirmation") or {},
        "child_switch": case.get("child_switch") or {},
        "reply_language": case.get("reply_language"),
    }


def generate(case: dict) -> str:
    """Run the real respond node and return the composed reply text."""
    out = respond(_state(case))
    return str(out["messages"][0].content)


def _context(case: dict) -> str:
    """Build a compact brief telling the judge what material was prepared and in which language."""
    lang = case.get("reply_language") or "en"
    parts = [
        f"Parent's latest message: {case.get('message', '')}",
        f"Required reply language: {_LANGUAGE_NAME.get(lang, lang)}",
        "Prepared material (the reply must not go beyond this):",
        json.dumps(case.get("capability_results") or {}, ensure_ascii=False),
    ]
    if case.get("confirmation"):
        parts.append(f"Confirmation outcome to acknowledge: {case['confirmation']}")
    if case.get("child_switch"):
        parts.append(f"Focus switched to: {case['child_switch']}")
    return "\n".join(parts)


def run_all(cases: list[dict] | None = None) -> dict[str, Any]:
    """Generate + judge every scenario; aggregate mean-per-dimension + pass_rate."""
    cases = cases or load_cases()
    rubric = judge_mod.load_rubric(_RUBRIC)
    scores: list[dict[str, int]] = []
    rows: list[dict[str, Any]] = []
    for case in cases:
        try:
            reply = generate(case)
            verdict = judge_mod.judge(rubric, reply, _context(case), DIMENSIONS)
        except Exception as exc:  # noqa: BLE001 -- one bad case must not kill the batch
            rows.append({"id": case.get("id"), "error": f"{type(exc).__name__}: {exc}"})
            continue
        scores.append({d: verdict[d] for d in DIMENSIONS})
        rows.append({"id": case.get("id"), **verdict})

    summary: dict[str, float] = {
        "n_cases": float(len(cases)),
        "n_errors": float(sum(1 for r in rows if "error" in r)),
        **metrics.mean_by_dimension(scores, DIMENSIONS),
        "pass_rate": metrics.pass_rate(scores, DIMENSIONS, _PASS_FLOOR),
    }
    return {"summary": summary, "cases": rows}


if __name__ == "__main__":
    import sys

    from evals._harness import thresholds as thresholds_mod

    report = run_all()
    print("=== agent.pipeline.respond / judge ===")  # noqa: T201
    print(report["summary"])  # noqa: T201
    for row in report["cases"]:
        print(f"  {row}")  # noqa: T201
    if "--gate" in sys.argv:
        failures = thresholds_mod.assert_thresholds(
            report["summary"], thresholds_mod.load(THRESHOLDS)
        )
        if failures:
            print("\nTHRESHOLDS NOT MET:\n" + "\n".join(failures))  # noqa: T201
            sys.exit(1)
