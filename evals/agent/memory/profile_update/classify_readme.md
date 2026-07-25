# agent.memory.profile_update — classify

**Node under test:** `src/agent/memory/profile_update.py` (`profile_update`, in the memory subgraph)
**Strategy:** classify (invoked tool-name set against the real tool menu)

## What it measures
Whether `profile_update`'s LLM tool loop translates the turn's `memory_operations` into the right
domain-**tool calls** — including the ordering rule that a new child's `create_child` must run
before the ops that target it. Gold is the set of tool **names** it should invoke, validated
against `agent.domain.MEMORY_TOOLS_BY_NAME`. This complements `agent.memory.decide` (which chooses
the operations); here we score the operations → tool-call execution fidelity.

## Staying LLM-only (no DB)
The real node writes through these tools inside a `domain_session`. The eval runs the **real** LLM
(still bound to the genuine tool schemas) but swaps the tool **execution** sink for recorders that
log the name and return `ok`, and stubs the session + re-read. So it measures tool selection
without a database. Stubs are installed around a single prediction and always restored.

## Metrics
`exact_match` (did it call exactly the right tool set?) is the product bar; `micro_f1` / `macro_f1`
are the diagnostics. Thresholds are conservative — the model may reasonably add a summary update.

## Sample coverage (3 cases, the house cap)
- `set-reading-interest` — one interest op → `update_reading_interest`.
- `record-finished-book` — one reading-history op → `record_finished_book`.
- `create-child-then-interest` — a new child bundle → `create_child` **and**
  `update_reading_interest` (exercises the create-first ordering).

## Run
```bash
python -m evals.agent.memory.profile_update.classify_run          # ad-hoc
python -m evals.agent.memory.profile_update.classify_run --gate   # assert thresholds
RUN_EVAL=1 pytest eval_regression -k profile_update         # via the gate
```
Needs `ANTHROPIC_API_KEY`.

## Grow it
Add one case per operation family and a multi-op bundle, plus a negative case (an operation that
maps to no tool must invoke nothing). `expected_tools` must be valid tool names; a typo fails loudly.
