# agent.pipeline.understand — classify

**Node under test:** `src/agent/pipeline/understand.py` (`understand`)
**Strategy:** classify (deterministic ground truth → pass/fail gate)

## What it measures
Whether `understand` maps a parent's message to the right **intent set** and resolves the right
**target child**. Intents are multi-label, so the product-meaningful bar is exact-set-match; F1 is
the per-label diagnostic. Child resolution is scored only on cases where the correct child is
unambiguous from the case alone.

## Files
- `classify_datasets.jsonl` — self-contained English cases (message + roster + optional pin +
  gold). `expected_intents` must be valid `Intent` keys (`src/agent/intents.py`); a typo fails loudly.
- `classify_run.py` — builds the minimal `FlowState`, calls the real node, scores with `_harness/metrics.py`.
- `classify_thresholds.json` — `"<metric>_min"` floors (conservative starting baseline for a tiny set).

## Sample coverage (3 cases, the house cap for samples)
- `rec-matched-child` — single intent + matched-child resolution.
- `rec-plus-discussion` — a genuine multi-intent turn (recommend **and** discuss).
- `fact-plus-task-not-profile-update` — the profile-vs-task confusion: a stated fact plus a task
  must stay `book_recommendation` (the fact goes to `user_signals`, NOT a `child_profile_update`).

## Run
```bash
python -m evals.agent.pipeline.understand.classify_run          # ad-hoc, prints per-case pred-vs-gold
python -m evals.agent.pipeline.understand.classify_run --gate   # assert thresholds, non-zero on fail
RUN_EVAL=1 pytest eval_regression -k understand           # via the gate
```
Needs `ANTHROPIC_API_KEY`. No Postgres.

## Grow it
Add phrasings, real production misfires, more multi-intent turns, and each child-resolution state
(matched / new / ambiguous / none). Keep it balanced; raise thresholds as N grows. See the
`.claude` evals skill for the dataset-generation workflow.
