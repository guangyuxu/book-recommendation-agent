# agent.capabilities.recommend — judge

**Node under test:** `src/agent/capabilities/recommend.py` (`run`)
**Strategy:** judge (LLM-as-judge; no single right answer)

## What it measures
The quality of a generated booklist against an explicit rubric on three 1-5 dimensions — **fit**,
**age_fit**, **language**. For each scenario the real capability generates a booklist, then a judge
model (temperature 0) scores it against `judge_rubric.md`.

## Files
- `judge_datasets.jsonl` — English scenarios (message + child roster + `target_child_id` + policies).
- `judge_rubric.md` — the scoring rubric with 1/3/5 anchors per dimension (editing it is a
  reviewable prompt change).
- `judge_run.py` — generate → render → judge → aggregate `mean_<dim>` + `pass_rate`.
- `judge_thresholds.json` — `"mean_<dim>_min"` and `"pass_rate_min"` floors.

## Sample coverage (3 cases, the house cap)
`dinosaurs-6yo` (clear interests/stage), `reluctant-9yo` (motivation + constraint), `esl-early-picture`
(beginner English / CEFR pre-A1). Each stresses a different `fit`/`age_fit` challenge.

## Judge model
Defaults to the app model so it runs with the existing key; set `EVAL_JUDGE_MODEL` to a **stronger,
different** model for a real bar (avoids self-enhancement bias). See the `.claude` evals skill →
`judge.md` for calibration and bias guards.

## Run
```bash
python -m evals.agent.capabilities.recommend.judge_run          # ad-hoc, prints per-case scores
python -m evals.agent.capabilities.recommend.judge_run --gate   # assert thresholds
RUN_EVAL=1 pytest eval_regression -m judge                # all judge nodes
```
Needs `ANTHROPIC_API_KEY`. No Postgres.
