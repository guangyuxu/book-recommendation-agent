# agent.memory.decide — classify

**Node under test:** `src/agent/memory/decide.py` (`memory_policy`, in the memory subgraph)
**Strategy:** classify (operation-set match against the real tool menu)

## What it measures
Whether `memory_policy` selects the right durable **domain operations** from a turn's
`user_signals`. Gold is the set of operation **names** it should emit, validated against
`agent.domain.MEMORY_TOOLS_BY_NAME`. This node lives inside a subgraph — its eval mirrors that:
`evals/agent/memory/decide/` matches `src/agent/memory/decide.py`.

## Metrics
Memory decisions are genuinely fuzzy (the model may reasonably add a summary update), so
`exact_match` is strict and `macro_f1` is the primary diagnostic; thresholds are conservative.

## Sample coverage (3 cases, the house cap)
- `reading-interest` — an interest fact → `update_reading_interest`.
- `finished-book` — a reading-history event → `record_finished_book`.
- `nothing-to-remember` — no signals: exercises the deterministic **skip** path (no API call, gold `[]`).

## Run
```bash
python -m evals.agent.memory.decide.classify_run          # ad-hoc
python -m evals.agent.memory.decide.classify_run --gate   # assert thresholds
RUN_EVAL=1 pytest eval_regression -k decide         # via the gate
```
Needs `ANTHROPIC_API_KEY` (except the skip-path case).

## Grow it
Add one case per operation family (child basic info, school, reading ability/genre/theme/summary,
current/disliked book, member info, family policy) and the `child_is_new` → `create_child` bundle.
`expected_operations` must be valid tool names; a typo fails loudly.
