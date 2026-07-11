"""Judge eval for the `recommend` capability: rubric-scored booklist quality (1-5 dimensions).

`recommend` has no single right answer, so we score it with an LLM judge against an explicit
rubric (`judge_rubric.md`) on three dimensions: **fit**, **age_fit**, **language**. For each
scenario we GENERATE the real booklist (via the capability registry), render it, and judge it at
temperature 0. Aggregated as `mean_<dimension>` plus a `pass_rate` (every dimension >= floor).

Pure logic, no pytest -- runs from `eval_regression` and directly
(`python -m evals.agent.capabilities.recommend.judge_run`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.messages import HumanMessage

from agent.capabilities._shared import child_brief, policies_brief
from agent.capabilities.registry import REGISTRY
from evals._harness import judge as judge_mod
from evals._harness import metrics
from evals._harness.cases import load_jsonl

# --- interface (required by evals/_harness/discovery.py) --------------------------------
STRATEGY = "judge"
NODE = "agent.capabilities.recommend"
THRESHOLDS = Path(__file__).with_name("judge_thresholds.json")

# Keep in sync with judge_rubric.md headers and judge_thresholds.json ("mean_<dim>_min").
DIMENSIONS: tuple[str, ...] = ("fit", "age_fit", "language")
_PASS_FLOOR = 3  # a case "passes" when EVERY dimension >= this

_DATASET = Path(__file__).with_name("judge_datasets.jsonl")
_RUBRIC = Path(__file__).with_name("judge_rubric.md")


def load_cases() -> list[dict]:
    """Load the judge scenarios (self-contained: message + roster + pin + policies)."""
    return load_jsonl(_DATASET)


def _state(case: dict) -> dict[str, Any]:
    """Build the minimal FlowState the recommend capability reads."""
    return {
        "messages": [HumanMessage(case["message"])],
        "children": case.get("children") or {},
        "target_child_id": case.get("target_child_id"),
        "policies": case.get("policies") or [],
    }


def generate(case: dict) -> dict[str, Any]:
    """Run the real recommend capability and return its Booklist dump."""
    return REGISTRY["recommend"].run(_state(case))


def render(output: dict[str, Any]) -> str:
    """Render a booklist as the text the judge scores."""
    lines: list[str] = []
    for i, b in enumerate(output.get("books") or [], start=1):
        head = b.get("title") or "(untitled)"
        if b.get("author"):
            head += f" by {b['author']}"
        lines.append(f"{i}. {head}")
        if b.get("recommendation_reason"):
            lines.append(f"   reason: {b['recommendation_reason']}")
        if b.get("fit_summary"):
            lines.append(f"   fit: {b['fit_summary']}")
        for risk in b.get("risk_notes") or []:
            lines.append(f"   risk: {risk}")
    if output.get("note"):
        lines.append(f"note: {output['note']}")
    return "\n".join(lines) or "(empty booklist)"


def run_all(cases: list[dict] | None = None) -> dict[str, Any]:
    """Generate + judge every scenario; aggregate mean-per-dimension + pass_rate."""
    cases = cases or load_cases()
    rubric = judge_mod.load_rubric(_RUBRIC)
    scores: list[dict[str, int]] = []
    rows: list[dict[str, Any]] = []
    for case in cases:
        state = _state(case)
        context = f"{child_brief(state)}\n\nPolicies:\n{policies_brief(state)}"
        try:
            output = generate(case)
            verdict = judge_mod.judge(rubric, render(output), context, DIMENSIONS)
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
    print("=== recommend / judge ===")  # noqa: T201
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
