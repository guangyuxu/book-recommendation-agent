# Rubric — Recommend (booklist quality)

The judge scores one `recommend` output (a booklist) for one child on three dimensions, each
**1-5**. Score each dimension independently; anchor to the 1/3/5 descriptions; judge substance and
fit, not length. Return one integer plus a one-sentence justification per dimension.

## fit — how well the list matches interests, reading stage, and family goals/constraints
- **5** — Every book clearly targets the child's interests AND reading stage; each reason cites the
  specific interest/goal/constraint from the context.
- **3** — Roughly on-theme but generic; some books ignore the stated interest, stage, or constraint.
- **1** — Off-topic, wrong stage, or ignores the context entirely.

## age_fit — whether content, themes, and difficulty suit the child's age
- **5** — All titles age-appropriate in difficulty and theme; real risks are flagged where they exist.
- **3** — Mostly fine but 1-2 titles are a stretch (too easy/hard or a mildly off theme).
- **1** — Contains clearly age-inappropriate content or difficulty for this child.

## language — clarity, tone, and correctness of the parent-facing text
- **5** — Clear, warm, concrete, error-free; a parent could act on it directly.
- **3** — Understandable but vague, repetitive, or slightly awkward.
- **1** — Confusing, generic boilerplate, or contains errors.

## Threshold mapping
Aggregated as `mean_fit` / `mean_age_fit` / `mean_language` across the dataset, plus `pass_rate`
(fraction of cases where every dimension >= 3); gated by `judge_thresholds.json`.
