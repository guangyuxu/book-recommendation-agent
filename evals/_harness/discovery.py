"""Discover node-eval modules by convention, so nothing has to be registered by hand.

`evals/agent/` is a FULL mirror of `src/agent/`, and strategy is a filename prefix:

    evals/agent/pipeline/understand/classify_run.py -> node "agent.pipeline.understand", "classify"
    evals/agent/capabilities/recommend/judge_run.py -> node "agent.capabilities.recommend", "judge"
    evals/agent/memory/decide/classify_run.py       -> node "agent.memory.decide", "classify"

`discover()` walks the whole `evals/agent/` mirror, imports every `<strategy>_run.py`, validates
the module interface, and returns `NodeEval` descriptors. Both the category runners and
`eval_regression` consume this -- adding a node dir is all it takes to be picked up (no central
list to edit, and no tree enumeration since the mirror root covers all of src/agent).

The module interface every `<strategy>_run.py` MUST expose:
    STRATEGY: str                     # "classify" | "judge" | ... (matches the filename prefix)
    NODE: str                         # dotted module path, e.g. "agent.pipeline.understand"
    THRESHOLDS: Path                  # co-located thresholds json
    load_cases() -> list
    run_all(cases=None) -> dict       # {"summary": {metric: float}, "cases": [...], ...}
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

# evals/_harness/discovery.py -> evals/
_EVALS_DIR = Path(__file__).resolve().parent.parent
# The mirror root: evals/agent/ mirrors src/agent/ in full. Walk it recursively for node evals.
_NODE_ROOT = _EVALS_DIR / "agent"
_REQUIRED = ("STRATEGY", "NODE", "THRESHOLDS", "load_cases", "run_all")


@dataclass(frozen=True)
class NodeEval:
    """One discovered node eval: its identity, its module, and its thresholds file."""

    strategy: str
    node: str
    module: ModuleType
    thresholds: Path

    @property
    def id(self) -> str:
        """Stable id used as the pytest param id and report key, e.g. 'agent.pipeline.understand:classify'."""
        return f"{self.node}:{self.strategy}"

    def load_cases(self) -> list:
        """Load the node's dataset (delegates to the module)."""
        return self.module.load_cases()

    def run_all(self, cases: Any = None) -> dict:
        """Run the eval and return its JSON-able report (delegates to the module)."""
        return (
            self.module.run_all(cases) if cases is not None else self.module.run_all()
        )


def _module_name(run_py: Path) -> str:
    """Map evals/agent/pipeline/understand/classify_run.py -> 'evals.agent.pipeline.understand.classify_run'."""
    rel = run_py.resolve().relative_to(_EVALS_DIR.parent).with_suffix("")
    return ".".join(rel.parts)


def _validate(module: ModuleType, path: Path) -> None:
    missing = [attr for attr in _REQUIRED if not hasattr(module, attr)]
    if missing:
        raise AttributeError(
            f"{path}: node eval is missing required interface members {missing} "
            f"(see evals/_harness/discovery.py for the contract)."
        )


def discover(strategy: str | None = None, node: str | None = None) -> list[NodeEval]:
    """Return all node evals, optionally filtered by `strategy` and/or `node` substring.

    `node` matches by substring so `discover(node="understand")` finds `agent.pipeline.understand`.
    Results are sorted by id for a stable run order.
    """
    found: list[NodeEval] = []
    if _NODE_ROOT.is_dir():
        for run_py in sorted(_NODE_ROOT.rglob("*_run.py")):
            if run_py.name.startswith("_"):
                continue
            module = importlib.import_module(_module_name(run_py))
            _validate(module, run_py)
            found.append(
                NodeEval(
                    strategy=module.STRATEGY,
                    node=module.NODE,
                    module=module,
                    thresholds=Path(module.THRESHOLDS),
                )
            )
    if strategy is not None:
        found = [e for e in found if e.strategy == strategy]
    if node is not None:
        found = [e for e in found if node in e.node]
    return sorted(found, key=lambda e: e.id)


def strategies() -> list[str]:
    """Return the distinct strategy names in the tree (for CLI help / category listing)."""
    return sorted({e.strategy for e in discover()})
