# Layout & interface

## The law: mirror `src/agent` in full

`evals/agent/` is a complete mirror of `src/agent/` — the `agent` package layer is included, so the
path maps one-to-one. This is non-negotiable; it's what makes "where's the eval for X?" answerable
by inspection.

**What's mirrored is the `agent` package root, not the `src/` container.** `src/` is just the
Python source root (src-layout); `agent` is the real import root the app runs as — it's the default
package of the LangGraph "New Project" starter and is load-bearing in `langgraph.json`
(`"agent": "agent.graph:graph"`) and `pyproject.toml`. So node ids are `agent.pipeline.understand`
(the actual import path), with `src`/`evals` stripped — the eval's identity tracks the same stable
root langgraph deploys. If the package is ever renamed, it's one deliberate change across
`pyproject.toml` + `langgraph.json` + `evals/agent/` together.

Why `evals` still prefixes the *import* path (`evals.agent.pipeline.understand.classify_run`) even
though `src` does not prefix `agent.pipeline.understand`: the mirror deliberately reuses the name
`agent`, so it MUST live under its own top package or it would shadow the real `agent` on `sys.path`
and break `from agent.pipeline.understand import understand`. The `evals.` prefix is that guard, not
redundancy — and it never appears in a node's id.

| Source module | Eval directory | Node id |
|---------------|----------------|---------|
| `src/agent/pipeline/understand.py` | `evals/agent/pipeline/understand/` | `agent.pipeline.understand` |
| `src/agent/capabilities/recommend.py` | `evals/agent/capabilities/recommend/` | `agent.capabilities.recommend` |
| `src/agent/memory/decide.py` (subgraph node) | `evals/agent/memory/decide/` | `agent.memory.decide` |

- One directory per node under test. The directory name is the node's file/module stem; the eval
  path under `evals/` equals the node's dotted import path (`agent.pipeline.understand`).
- Subgraphs keep full parity: a node inside `src/agent/memory/` maps to `evals/agent/memory/<node>/`.
- Shared infrastructure lives ONLY in `evals/_harness/`. Never put a node eval there, and never
  put shared code in a node dir.
- `discovery.py` walks the whole `evals/agent/` mirror (`_NODE_ROOT`) recursively — no tree list to
  maintain. A new area under `src/agent/` is picked up automatically once its eval dir exists.

## Strategy = filename prefix

A node can be evaluated several ways at once; each strategy is a **prefix** on its own file set, so
they never collide:

```
evals/agent/pipeline/understand/
  classify_run.py  classify_datasets.jsonl  classify_thresholds.json  classify_readme.md
evals/agent/capabilities/recommend/
  judge_run.py  judge_datasets.jsonl  judge_rubric.md  judge_thresholds.json  judge_readme.md
```

If `understand` later needed a judge pass over some free-text field, it would gain
`judge_run.py` + `judge_*` siblings in the same directory — no conflict with `classify_*`.

Built-in strategies: `classify` (structured output with a right answer) and `judge` (generative
output, rubric-scored). The taxonomy is **open** — add a new prefix with its own `<strategy>_run.py`
when a node needs a method these don't cover; discovery picks it up automatically.

## File set per strategy

- `<strategy>_run.py` — pure logic; the module interface below. **No pytest here.**
- `<strategy>_datasets.jsonl` — one self-contained case per line (English; ≤ 3 for samples).
- `<strategy>_thresholds.json` — co-located floors, `"<metric>_min"` keys.
- `<strategy>_readme.md` — what it measures, coverage, how to run, how to grow.
- `judge_rubric.md` — judge only: the 1/3/5-anchored rubric.

## Module interface (contract enforced by `discovery.py`)

```python
STRATEGY = "classify"                 # matches the filename prefix
NODE = "agent.pipeline.understand"          # full dotted import path (for report keys / ids)
THRESHOLDS = Path(__file__).with_name("classify_thresholds.json")

def load_cases() -> list: ...
def run_all(cases=None) -> dict:      # {"summary": {metric: float}, "cases": [...], ...}

if __name__ == "__main__":
    # print a human report; with `--gate`, assert thresholds and sys.exit(1) on failure.
```

`run_all()` is the single execution seam. `summary` is a flat `{metric: float}` dict — exactly the
metrics your `thresholds.json` gates with `"<metric>_min"`. `cases` is the per-case detail used for
debugging a failure. Keep it JSON-able (`report.write_report` dumps it).

## Run matrix

| Goal | Command |
|------|---------|
| one node, ad-hoc | `python -m evals.<node>.<strategy>_run` |
| one node, gated | `python -m evals.<node>.<strategy>_run --gate` |
| all of a strategy | `python -m eval_regression.run --strategy classify` / `make eval_classify` |
| one node via gate | `python -m eval_regression.run --node understand` / `make eval_node NODE=understand` |
| everything (CI) | `python -m eval_regression.run` / `make eval` |
| pytest selection | `RUN_EVAL=1 pytest eval_regression -m judge -k recommend` |
| produce thresholds | `python -m eval_regression.produce [--dry-run] [--repeats N] [--margin M]` |

Categories work because `eval_regression/test_regression.py` tags each discovered node eval with a
pytest marker equal to its strategy (`classify`/`judge`, registered in `pyproject.toml`).
