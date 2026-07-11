# Evals — node-first, strategy-prefixed

Measures the **non-deterministic LLM output** unit tests can't: is intent classification accurate?
are the generated booklists any good? did a prompt tweak quietly break a memory decision? The
governing rules and workflows (scaffold a node, generate a dataset, run, gate) live in the
`.claude` **evals skill** (`.claude/skills/evals/`). This file is the quick reference.

## The layout law

The tree **mirrors `src/agent`**. One directory per node under test; **strategy is a filename
prefix**, so one node can host several strategies without collision.

```
evals/
  _harness/                          # shared infra — DO NOT put node evals here
    cases.py metrics.py judge.py thresholds.py report.py discovery.py scaffold.py templates/
  agent/                             # FULL mirror of src/agent/
    pipeline/understand/             # mirrors src/agent/pipeline/understand.py
      classify_run.py  classify_datasets.jsonl  classify_thresholds.json  classify_readme.md
    capabilities/recommend/          # mirrors src/agent/capabilities/recommend.py
      judge_run.py  judge_datasets.jsonl  judge_rubric.md  judge_thresholds.json  judge_readme.md
    memory/decide/                   # mirrors src/agent/memory/decide.py (a subgraph node)
      classify_run.py  classify_datasets.jsonl  classify_thresholds.json  classify_readme.md
```

Subgraphs keep parity too: a node in `src/agent/memory/` gets `evals/agent/memory/<node>/`.

## Strategies

| Prefix     | For                                   | Method                                             |
|------------|---------------------------------------|----------------------------------------------------|
| `classify_`| structured output with a right answer | labeled dataset + deterministic set/label metrics  |
| `judge_`   | generative output, no single answer   | rubric + LLM-as-judge, 1-5 dimensions, temp 0      |

The taxonomy is open — add a new prefix (its own `<strategy>_run.py`) when a node needs a method
these two don't cover. See the skill's `classification.md` / `judge.md` for the best practices.

## The module interface (what makes a node eval discoverable)

Every `<strategy>_run.py` exposes: `STRATEGY`, `NODE` (full dotted import path, e.g.
`agent.pipeline.understand`), `THRESHOLDS` (co-located
json), `load_cases()`, and `run_all() -> {"summary": {metric: float}, "cases": [...]}`. That one
module is both the ad-hoc CLI and the entrypoint `eval_regression` calls. `evals/_harness/discovery.py`
finds them by convention — no central registry to edit.

## Running

```bash
python -m evals.agent.pipeline.understand.classify_run          # one node, ad-hoc (prints)
python -m evals.agent.pipeline.understand.classify_run --gate   # one node, gated (non-zero on fail)
make eval_classify        # all classify nodes        (RUN_EVAL=1 python -m eval_regression.run --strategy classify)
make eval_judge           # all judge nodes
make eval_node NODE=decide# one node by name
make eval                 # everything (CI gate)
RUN_EVAL=1 pytest eval_regression -m judge -k recommend   # pytest selection by strategy/node
```

Evals are **opt-in** (they call the Anthropic API): `RUN_EVAL=1`, exactly like
`tests/integration_tests`. Needs `ANTHROPIC_API_KEY` (loaded from `.env`). Set `EVAL_JUDGE_MODEL`
to a stronger, separate model for judge evals.

## Thresholds & regression

Thresholds are **co-located per node** (`<strategy>_thresholds.json`, `"<metric>_min"` keys) so the
pass/fail bar is reviewable next to the dataset. `eval_regression/` is the flat gate that runs the
evals and enforces those floors, and `eval_regression/produce.py` (re)generates them. See
`eval_regression/README.md`.

## Conventions

- **English only** — messages, profiles, rubrics, docs, code, comments.
- **Datasets are JSONL**, one self-contained case per line; sample datasets ship **≤ 3 cases**.
- **Gold is spec-derived**, never produced by the model under test; raw user input is a *source*,
  never a test case verbatim.
- **`run.py` is pytest-free** (pure logic); the gate is `eval_regression`.
- Reports land in `results/` (gitignored, timestamped).
