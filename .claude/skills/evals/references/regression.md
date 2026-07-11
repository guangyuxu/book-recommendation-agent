# eval_regression — the gate + threshold producer

`eval_regression/` is a **flat**, node-agnostic layer that consumes the node evals. It deliberately
has **no per-node subdirectory tree** — duplicating the node structure would be a second thing to
keep in sync. It discovers node evals dynamically (`evals/_harness/discovery.py`), so a new
`evals/<tree>/<node>/` directory is picked up with zero edits here. The only subdir is
`baselines/` (optional committed metric snapshots for trend history).

## Two responsibilities

### Gate — `run.py` + `test_regression.py`
Runs each node eval's `run_all()` and asserts its `summary` against that node's co-located
`<strategy>_thresholds.json`. It never reaches into a node eval's internals — the `run_all()`
contract is the whole seam.

```bash
RUN_EVAL=1 python -m eval_regression.run                  # gate everything (CI)
RUN_EVAL=1 python -m eval_regression.run --strategy judge  # a category
RUN_EVAL=1 python -m eval_regression.run --node decide     # one node
RUN_EVAL=1 pytest eval_regression -m classify -v           # via pytest, classify only
```
Exits non-zero with a per-node failure list when any floor is missed. Each run also writes a
timestamped JSON report to `evals/results/` (gitignored).

### Produce — `produce.py`
Generates the floors the gate enforces, so thresholds are data-derived, not guessed. For each
`"<metric>_min"` key already present in a node's thresholds file, it proposes
`min(observed across --repeats runs) - margin`, clamped ≥ 0.

```bash
python -m eval_regression.produce --dry-run      # preview the diff, write nothing
python -m eval_regression.produce --repeats 3    # sample judge variance, then write back
python -m eval_regression.produce --margin 0.1   # looser floors
```

It only ever tightens/loosens **existing** `_min` keys — it never invents a gate. Adding a new
gated metric stays a deliberate, reviewed edit to the `thresholds.json`.

## Thresholds live with the node, not here

Floors are **co-located** (`evals/<node>/<strategy>_thresholds.json`) so the pass/fail bar is
reviewed in the same PR as the dataset and prompt it guards. `eval_regression` reads and writes
those files; it does not own a central threshold store.

## Discipline

- A red gate means a node fell below its committed floor. Investigate the regression — don't reflex
  the floor downward.
- A **deliberate** behavior change is recorded as a reviewed edit to the node's `thresholds.json`
  (optionally regenerated with `produce.py`). The diff is the audit trail that the change was intended.
- Grow datasets before raising floors; a floor is only as trustworthy as the sample under it.

## CI

Run it as a scheduled (cron) workflow: `RUN_EVAL=1 python -m eval_regression.run` with
`ANTHROPIC_API_KEY` (and `EVAL_JUDGE_MODEL`) in the environment. Keep it off the per-commit
path — it costs tokens and is non-deterministic.
