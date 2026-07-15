# agent.pipeline.respond — judge

**Node under test:** `src/agent/pipeline/respond.py` (`respond`)
**Strategy:** judge (LLM-as-judge, 1-5 on three dimensions)

## What it measures
Whether the composed parent-facing reply is **faithful** (stays within the prepared capability
material, invents no books/facts), **relevant** (answers the latest message and acknowledges any
confirmation outcome / focus switch), and correct in **language** (right reply language, warm,
concise). The judge scores the real reply produced by the node against `judge_rubric.md`.

## Staying LLM-only (no DB)
`respond` persists a recommendation turn only when a recommend booklist AND a resolved
`target_child_id` are both present. Every case leaves `target_child_id` unset, so that
deterministic persistence branch is skipped and the eval never touches Postgres — it judges only
the LLM-composed reply. (The persistence branch itself is covered by unit tests.)

## Sample coverage (3 cases, the house cap)
- `recommend-booklist-en` — a two-book list; the reply must present exactly those books and invent
  none (faithfulness).
- `confirmation-applied-en` — an applied profile change; the reply must acknowledge it as saved
  (relevance) without fabricating book content.
- `evaluate-prose-zh-hans` — English evaluation material with `reply_language=zh-Hans`; the reply
  must be in Simplified Chinese (language) while staying grounded.

## Run
```bash
python -m evals.agent.pipeline.respond.judge_run          # ad-hoc
python -m evals.agent.pipeline.respond.judge_run --gate   # assert thresholds
RUN_EVAL=1 pytest eval_regression -k respond        # via the gate
```
Needs `ANTHROPIC_API_KEY`; set `EVAL_JUDGE_MODEL` to a stronger, separate judge model to avoid
self-enhancement bias.

## Grow it
Add a `rejected` and an `error` confirmation case (the reply must NOT claim a change was saved), a
focus-switch case, and a Traditional-Chinese language case.
