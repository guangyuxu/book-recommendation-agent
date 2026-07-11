# eval_regression — the gate + threshold producer

A **flat**, node-agnostic layer that consumes the node evals under `evals/`. It has **no per-node
subdirectory tree** — it discovers node evals dynamically (`evals/_harness/discovery.py`), so a new
`evals/<tree>/<node>/` dir is picked up with zero edits here. The only subdir is `baselines/`
(optional committed metric snapshots for trend history).

## What it does
- **Gate** (`run.py`, `test_regression.py`): run each node eval's `run_all()` and assert its
  `summary` against that node's own co-located `<strategy>_thresholds.json`. Thresholds live next
  to the dataset (reviewable in the same PR); this layer only reads and enforces them.
- **Produce** (`produce.py`): run the evals and derive suggested floors (worst observed − margin),
  writing them back into each node's thresholds file. This is how thresholds are (re)generated.

## The seam evals exposes
Every `<strategy>_run.py` exposes `run_all() -> {"summary": {metric: float}, "cases": [...]}` plus
`STRATEGY`, `NODE`, `THRESHOLDS`. That single module is both the ad-hoc CLI and the programmatic
entrypoint this layer calls. Nothing here reaches into a node eval's internals.

## Run
```bash
RUN_EVAL=1 python -m eval_regression.run                 # gate everything (CI entrypoint)
RUN_EVAL=1 python -m eval_regression.run --strategy judge # a category
RUN_EVAL=1 python -m eval_regression.run --node decide    # one node
RUN_EVAL=1 pytest eval_regression -m classify -v          # gate via pytest, classify only

python -m eval_regression.produce --dry-run               # suggest threshold changes, write nothing
python -m eval_regression.produce --repeats 3             # sample judge variance, then write
```
Opt-in via `RUN_EVAL=1` (calls the Anthropic API). Needs `ANTHROPIC_API_KEY`.

## CI
Run it as a scheduled (cron) workflow, off the per-commit path:
`RUN_EVAL=1 python -m eval_regression.run`. A red gate means a node fell below its committed floor;
a deliberate behavior change is recorded as a reviewed edit to that node's `*_thresholds.json`
(optionally regenerated with `produce.py`).
