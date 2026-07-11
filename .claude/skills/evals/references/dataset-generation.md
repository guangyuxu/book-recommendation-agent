# Dataset generation — raw material → eval cases

The user will hand you **raw material** — a few real parent utterances, a product scenario, a bug
they saw. That is a *source*, not a dataset. Your job is to turn it into well-formed, spec-labeled
cases. Never paste a raw utterance in as a row.

## The pipeline

### 1. Pick node + strategy
Decide which node the material exercises and which strategy fits (`classify` if there's a right
answer, `judge` if quality is subjective). If the node dir doesn't exist yet, scaffold it
(`python -m evals._harness.scaffold <node> <strategy>`).

### 2. Normalize into the node's self-contained case schema
Each case is one JSONL line that stands alone. Add whatever the node reads that the raw utterance
omits:
- a **message** — rewrite the raw utterance into clear, natural **English**; fix noise, keep intent.
- a **`children` roster** — realistic `{child_id: {display_name, age, reading_profile{...}}}`.
- a **`target_child_id`** pin when the flow assumes an active child (and `null` when testing "no pin").
- **`policies`** (goals / constraints / avoid_topics) when the node reads them.
- for a subgraph node like `memory_policy`, the upstream state it consumes (`understanding` with
  `user_signals` + `child_is_new`), since it doesn't re-derive them.

### 3. Attach gold (classify) or context+rubric (judge)
- **classify** — decide the label from the **spec** (`src/agent/intents.py`, the domain tool menu),
  as `expected_intents` / `expected_operations` / etc. Validate against the enum. Set a secondary
  gold (`expected_target_child_id`, `score_child_resolution: true`) only when unambiguous.
- **judge** — no gold answer; make sure the scenario is rich enough for the rubric to bite (the
  child's interests/stage and a goal/constraint the judge can check `fit`/`age_fit` against).

### 4. Add coverage around the raw seed
One raw utterance becomes a small, deliberate spread — not three paraphrases:
- the **happy path** the utterance implies;
- a **near-miss / adversarial** variant targeting the node's known confusion (e.g. add a task to a
  profile-fact message so it must stay a task; make a child reference ambiguous);
- a **negative / fallback** where relevant (vague → `clarify`; nothing to persist → empty set).

### 5. Balance, de-dup, tag
- Balance the label distribution; don't stack three cases on the same easy label.
- Drop rows that test the same thing as another row.
- Tag each case with `"provenance"` (`synthetic` | `production` | `derived`) and `"difficulty"`
  (`easy` | `medium` | `hard`) so coverage and failures are legible later.

### 6. Write & smoke-test
Write English JSONL into `evals/<node>/<strategy>_datasets.jsonl`. **Samples cap at 3 cases.** Then
`python -m evals.<node>.<strategy>_run` to confirm the cases load and score, and eyeball the
per-case output.

## Guardrails

- **English only**, every field.
- **≤ 3 cases** for any sample/scaffold dataset you generate on request.
- **Gold from spec, not from the model.** If you're unsure of the correct label, resolve it from
  the source of truth or ask — do not let the node under test decide its own answer.
- **Realistic, not real.** Synthesize plausible rosters/profiles; never copy a real family's PII in.
- Prefer cases that mirror **actual production misfires** — those are the highest-signal rows.
