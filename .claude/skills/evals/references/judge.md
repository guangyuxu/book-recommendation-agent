# Judge strategy — best practices

For **generative** nodes with no single right answer: `recommend` (booklist), `evaluate`,
`compare`, `discussion`, `path`, `content`, and the `respond` composer. You GENERATE the real
output, then score it against an explicit rubric with an LLM judge. Worked example:
`evals/agent/capabilities/recommend/`. Shared judge: `evals/_harness/judge.py`.

## The rubric (`judge_rubric.md`)

- 2–4 **independent dimensions**, each scored **1–5**, each with explicit **1 / 3 / 5 anchors**.
  Anchors are what make scores comparable across runs and reviewers — never ship a dimension
  without them.
- Dimensions must be **orthogonal** (e.g. `fit`, `age_fit`, `language`). Overlapping dimensions
  double-count and inflate.
- Keep `DIMENSIONS` in `judge_run.py`, the rubric headers, and the `mean_<dim>_min` threshold keys
  in sync — all three name the same set.
- Editing a rubric is a **reviewable prompt change**. Treat it like code.

## The judge model

- **Temperature 0**, forced structured output: one int per dimension + a one-sentence
  justification (built dynamically by `_harness/judge.py` so it always matches `DIMENSIONS`).
- **Use a stronger, SEPARATE model from the generator.** Judging with the same model that produced
  the output invites self-enhancement bias. Set `EVAL_JUDGE_MODEL` to a stronger model; the default
  falls back to the app model only so evals run out of the box.

## Bias guards (baked into the judge prompt; keep them)

- **Independence** — score each dimension on its own; don't let a strong dimension halo a weak one.
- **Verbosity** — judge substance and fit, not length or eloquence.
- **Anchor discipline** — tie every score to the rubric's 1/3/5 text, not a vibe.
- If you ever judge *pairwise* (A vs B), randomize order to defeat **position bias**.

## Aggregation & thresholds

- `metrics.mean_by_dimension` → `mean_<dim>` (the average quality per dimension).
- `metrics.pass_rate` → fraction of cases where **every** dimension ≥ a floor (default 3). This
  catches the "great on two dims, terrible on the third" output that a mean hides. Gate on both.
- One bad case (judge/gen error) is recorded in `n_errors` and dropped from the means, not scored 0.

## Calibration / meta-eval (do this before trusting a new rubric)

A judge you haven't calibrated is an opinion, not a measurement.

1. **Separation** — seed 1–2 deliberately good and 1–2 deliberately bad outputs; confirm the judge
   scores good ≫ bad. If it can't separate them, the rubric or judge model is too weak.
2. **Self-consistency** — run the same case 2–3 times (`produce.py --repeats N`); scores should be
   stable (±1 at most). High variance means the rubric is underspecified.
3. **Spot-check justifications** — read a few; if the reason doesn't match the score, tighten the
   anchors. Justifications are logged into the report for exactly this.

## Dataset

Scenarios (message + roster + pin + policies) that each stress a different quality challenge — for
`recommend`: clear-interest, reluctant-reader-with-constraint, beginner/ESL. ≤ 3 for samples.
