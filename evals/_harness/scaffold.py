"""Scaffold a new node eval from templates, so adding one is mechanical and consistent.

    python -m evals._harness.scaffold agent.pipeline.understand classify
    python -m evals._harness.scaffold agent.capabilities.recommend judge

The node arg is the node's full dotted import path (INCLUDING the `agent` package layer), so it
lands under the `evals/agent/` mirror. Creates `evals/<node.dotted.parts>/` with the strategy's
file set: `<strategy>_run.py`,
`<strategy>_datasets.jsonl`, `<strategy>_thresholds.json`, a readme, and (judge only) a rubric.
The run.py is a skeleton with the discovery interface pre-filled and node-specific logic left as
`NotImplementedError` -- fill it in following the worked examples. Existing files are never
overwritten (safe to re-run).

House rule: everything authored here is English-only, and datasets ship with <= 3 sample cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TEMPLATES = Path(__file__).resolve().parent / "templates"
_EVALS_DIR = Path(__file__).resolve().parent.parent


def _render(template: str, subs: dict[str, str]) -> str:
    text = (_TEMPLATES / template).read_text(encoding="utf-8")
    for key, val in subs.items():
        text = text.replace(f"{{{{{key}}}}}", val)
    return text


def scaffold(node: str, strategy: str) -> list[Path]:
    """Create the file set for `evals/<node>/` and return the paths written (skips existing)."""
    node_parts = node.split(".")
    subs = {
        "NODE_DOTTED": node,
        "NODE_PATH": "/".join(node_parts),
        "NODE_NAME": node_parts[-1],
        "STRATEGY": strategy,
    }
    node_dir = _EVALS_DIR.joinpath(*node_parts)
    node_dir.mkdir(parents=True, exist_ok=True)

    # __init__.py at every level so the dotted module import resolves.
    for i in range(1, len(node_parts) + 1):
        (_EVALS_DIR.joinpath(*node_parts[:i]) / "__init__.py").touch(exist_ok=True)

    plan = {
        f"{strategy}_run.py": f"{strategy}_run.py.tmpl",
        f"{strategy}_datasets.jsonl": f"{strategy}_datasets.jsonl.tmpl",
        f"{strategy}_thresholds.json": f"{strategy}_thresholds.json.tmpl",
        f"{strategy}_readme.md": "readme.md.tmpl",
    }
    if strategy == "judge":
        plan[f"{strategy}_rubric.md"] = "judge_rubric.md.tmpl"

    written: list[Path] = []
    for out_name, template in plan.items():
        dest = node_dir / out_name
        if dest.exists():
            continue
        dest.write_text(_render(template, subs), encoding="utf-8")
        written.append(dest)
    return written


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m evals._harness.scaffold <node.dotted> <classify|judge>")  # noqa: T201
        raise SystemExit(2)
    paths = scaffold(sys.argv[1], sys.argv[2])
    if paths:
        print("created:\n" + "\n".join(f"  {p}" for p in paths))  # noqa: T201
    else:
        print("nothing to do (all files already exist)")  # noqa: T201
