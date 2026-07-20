"""Classify eval for the `profile_update` node: which domain TOOLS the LLM invokes vs gold.

`profile_update` (memory subgraph) is the only writer: it drives an LLM tool loop that turns the
turn's `memory_operations` into concrete domain-tool calls (and, for a new child, must call
`create_child` FIRST so later ops target it). Its LLM job is that translation. We score the SET of
tool NAMES it actually invokes against a spec-derived gold set drawn from the real tool menu
(`agent.domain.MEMORY_TOOLS_BY_NAME`).

Staying LLM-only (no DB): the real node writes through those tools inside a `domain_session`. Here
we run the REAL LLM (`_bound`, still bound to the real tool *schemas*, so the model sees genuine
tools) but replace the tool EXECUTION sink with recorders that log the name and return "ok", and
stub the session + re-read. So we measure tool-selection fidelity without a database. Stubs are
installed around one prediction and always restored.

Pure logic, no pytest: runs from `eval_regression` and directly
(`python -m evals.agent.memory.profile_update.classify_run`).
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from agent.domain import MEMORY_TOOLS_BY_NAME
from agent.memory import profile_update as pu_mod
from evals._harness import metrics
from evals._harness.cases import load_jsonl

# --- interface (required by evals/_harness/discovery.py) --------------------------------
STRATEGY = "classify"
NODE = "agent.memory.profile_update"
THRESHOLDS = Path(__file__).with_name("classify_thresholds.json")

_DATASET = Path(__file__).with_name("classify_datasets.jsonl")
_LABELS = list(MEMORY_TOOLS_BY_NAME)  # the tool-name label space
_VALID = set(_LABELS)
_EVAL_FAMILY_ID = str(
    uuid4()
)  # a syntactically valid id; never hits a real DB (stubbed)


def load_cases() -> list[dict]:
    """Load the labeled dataset (each case carries its own `memory_operations` + gold tools)."""
    return load_jsonl(_DATASET)


def _gold(case: dict) -> set[str]:
    """Gold tool-name set, validated against the real tool menu (a typo fails loudly)."""
    tools = set(case.get("expected_tools") or [])
    unknown = tools - _VALID
    if unknown:
        raise ValueError(
            f"case {case.get('id')}: unknown expected_tools {sorted(unknown)}"
        )
    return tools


@contextlib.contextmanager
def _recording_sinks(calls: list[str]):
    """Swap the node's tool sink + session + re-read for DB-free recorders; always restore."""
    original_tools = pu_mod.MEMORY_TOOLS_BY_NAME
    original_session = pu_mod.domain_session
    original_get_client = pu_mod.get_client

    def _make_recorder(name: str) -> SimpleNamespace:
        def _invoke(_args: Any) -> str:
            calls.append(name)
            return "ok"

        return SimpleNamespace(invoke=_invoke)

    @contextlib.contextmanager
    def _fake_session(*_args: Any, **_kwargs: Any):
        yield SimpleNamespace(session=object(), target_child_id=None)

    # The post-loop refresh re-reads the family context from the accounts API; stub it.
    _empty_ctx = {"family": {}, "members": [], "children": {}, "policies": []}
    pu_mod.MEMORY_TOOLS_BY_NAME = {name: _make_recorder(name) for name in _LABELS}
    pu_mod.domain_session = _fake_session
    pu_mod.get_client = lambda: SimpleNamespace(get_context=lambda _fid: _empty_ctx)
    try:
        yield
    finally:
        pu_mod.MEMORY_TOOLS_BY_NAME = original_tools
        pu_mod.domain_session = original_session
        pu_mod.get_client = original_get_client


def predict(case: dict) -> dict[str, Any]:
    """Run the real LLM tool loop with recording sinks; return the invoked tool-name set."""
    state = {
        "understanding": {"user_signals": case.get("user_signals") or []},
        "memory_operations": case.get("memory_operations") or [],
        "family": {"id": _EVAL_FAMILY_ID},
        "target_child_id": case.get("target_child_id"),
    }
    calls: list[str] = []
    try:
        with _recording_sinks(calls):
            pu_mod.profile_update(state)
    except Exception as exc:  # noqa: BLE001 -- survive any single-case failure
        return {"tools": set(), "error": f"{type(exc).__name__}: {exc}"}
    return {"tools": set(calls), "error": None}


def run_all(cases: list[dict] | None = None) -> dict[str, Any]:
    """Predict every case, score tool-set match + per-tool F1, return a JSON-able report."""
    cases = cases or load_cases()
    rows: list[dict[str, Any]] = []
    pred_sets: list[set[str]] = []
    gold_sets: list[set[str]] = []
    for case in cases:
        gold = _gold(case)
        pred = predict(case)
        pred_sets.append(pred["tools"])
        gold_sets.append(gold)
        row: dict[str, Any] = {
            "id": case.get("id"),
            "pred_tools": sorted(pred["tools"]),
            "gold_tools": sorted(gold),
            "ok": pred["tools"] == gold,
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
    print("=== agent.memory.profile_update / classify ===")  # noqa: T201
    print(report["summary"])  # noqa: T201
    for row in report["cases"]:
        flag = "" if row["ok"] else "  <-- MISS"
        if "error" in row:
            flag += f"  <-- ERROR {row['error']}"
        print(  # noqa: T201
            f"  [{row['id']}] pred={row['pred_tools']} gold={row['gold_tools']}{flag}"
        )
    if "--gate" in sys.argv:
        failures = thresholds_mod.assert_thresholds(
            report["summary"], thresholds_mod.load(THRESHOLDS)
        )
        if failures:
            print("\nTHRESHOLDS NOT MET:\n" + "\n".join(failures))  # noqa: T201
            sys.exit(1)
