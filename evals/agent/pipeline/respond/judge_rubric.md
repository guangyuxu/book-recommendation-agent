# Rubric — Respond (reply quality)

The judge scores one `respond` output (the single parent-facing reply) on three dimensions, each
**1-5**. Score each dimension independently; anchor to the 1/3/5 descriptions; judge substance and
fit, not length. Return one integer plus a one-sentence justification per dimension.

## faithfulness — the reply stays within the prepared material, inventing nothing
- **5** — Every book, fact, and claim in the reply is present in the prepared material; nothing is
  fabricated, and no book is named that the material did not contain.
- **3** — Substantially grounded, but adds a minor unsupported embellishment (a soft claim or an
  extra detail not in the material).
- **1** — Invents books, authors, or facts not in the material, or contradicts it.

## relevance — the reply answers the parent's latest message and reflects any required note
- **5** — Directly addresses the message using the material, and cleanly acknowledges the
  confirmation outcome / focus switch when one is present.
- **3** — Partially on point: answers loosely, or omits a required acknowledgment.
- **1** — Off-topic, ignores the material, or misstates a confirmation outcome (e.g. claims a
  change was saved when it was not).

## language — clarity, tone, and correct reply language
- **5** — Written in the required reply language, warm, concise, concrete, and error-free.
- **3** — Understandable but partly in the wrong language, vague, repetitive, or slightly awkward.
- **1** — Wrong language, confusing, or contains errors.

## Threshold mapping
Aggregated as `mean_faithfulness` / `mean_relevance` / `mean_language` across the dataset, plus
`pass_rate` (fraction of cases where every dimension >= 3); gated by `judge_thresholds.json`.
