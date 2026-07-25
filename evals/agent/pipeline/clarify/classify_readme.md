# agent.pipeline.clarify — classify

**Node under test:** `src/agent/pipeline/clarify.py` (`clarify`)
**Strategy:** classify (decision label against the node's decision contract)

## What it measures
Whether `clarify` makes the right call — `continue`, `ask_user`, or `best_effort` — given the
planned capabilities and what is known about the turn. Each case is run through the real
`understand → plan → clarify` shape: we build the `plan` with the actual `plan` node from the
case's `understanding`, then run `clarify` (LLM only, no DB). Gold is the single spec-derived
decision label.

## Metrics
Decisions are single-label, so `exact_match` (over singleton sets) is accuracy; `label_prf` gives
the per-decision precision/recall/F1 diagnostic. `continue` vs `best_effort` is genuinely fuzzy,
so cases are chosen where the decision is unambiguous and `exact_match` is the gated metric.

## Sample coverage (3 cases, the house cap)
- `continue-recommend-child-known` — actionable recommendation, child pinned → `continue`.
- `ask-user-evaluate-no-book` — `evaluate` needs a specific book, none named/ambient → `ask_user`.
- `ask-user-ambiguous-child-deterministic` — `child_ambiguous` → the deterministic `ask_user`
  branch (no API call).

## Run
```bash
python -m evals.agent.pipeline.clarify.classify_run          # ad-hoc
python -m evals.agent.pipeline.clarify.classify_run --gate   # assert thresholds
RUN_EVAL=1 pytest eval_regression -k clarify           # via the gate
```
Needs `ANTHROPIC_API_KEY` (except the deterministic ambiguous-child case).

## Grow it
Add a clear `best_effort` case (a task that can proceed on stated assumptions) and one per
required-input gap (`compare`/`discussion` with a missing book) so each `ask_user` trigger is
covered. `expected_decision` must be one of `continue` / `ask_user` / `best_effort`.
