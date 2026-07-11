# Classify strategy — best practices

For nodes whose structured output has a **right answer**: `understand` (intent set + child
resolution), `memory_policy` (chosen operation set), `plan` (chosen capabilities), `clarify`
(decision), etc. You assert the node's real output against spec-derived gold with deterministic
metrics. Worked examples: `evals/agent/pipeline/understand/`, `evals/agent/memory/decide/`.

## Gold labels

- **Spec-derived, always.** Read the source of truth (`src/agent/intents.py` for intents,
  `agent.domain.MEMORY_TOOLS_BY_NAME` for operations) and decide the label yourself. Never let the
  model under test define its own ground truth.
- **Validate gold against the enum/menu at load time** so a typo fails loudly, not silently as a
  wrong score. (See `UnderstandCase.gold_intents` / `memory/decide` `_gold`.)
- **Self-contained cases.** Each row carries everything scoring needs (message, `children` roster,
  optional pin, policies) so no database is touched.

## Metrics (multi-label is the common shape)

Most classify nodes are **multi-label** (a turn may carry several intents; a memory decision may
emit several ops). Report, via `evals/_harness/metrics.py`:

- **`exact_match`** — fraction of cases where the predicted SET equals gold exactly. This is the
  product-meaningful bar (no misses, no extras). Gate on it.
- **`macro_f1` / `micro_f1`** — per-label diagnostics. Macro surfaces weak *rare* labels; micro is
  dominated by frequent ones. Gate on `macro_f1` too; read `per_label[...].support` to tell a
  genuinely hard label from one with too few examples to trust.
- **Secondary structured fields** get their own accuracy (e.g. `child_resolution` for the resolved
  `target_child_id`), scored only on cases where the answer is unambiguous from the case alone.

## Binary & asymmetric-cost classifiers (e.g. `guard`)

A node with a two-way verdict (safe/unsafe, block/allow, on-topic/off-topic) is just multi-label
with one label per case — **reuse `label_prf` with single-element sets** (`{"attack"}` /
`{"benign"}`), no new metric code. Then `set_exact_match` is plain accuracy, and each class's
`per_label[...].recall` is the number you actually care about. Worked example:
`evals/agent/guard/`.

The point of a binary eval is usually that **the two errors cost differently**, so do NOT gate on
aggregate F1 — pull each direction into its own named `summary` metric and gate both:
- catch rate on the dangerous class (`attack_recall` = `per_label["attack"].recall`) — a miss is a
  safety hole;
- pass rate on the safe class (`benign_pass_rate` = `per_label["benign"].recall` = 1 − false-positive
  rate) — a false block is a UX failure.

Cover the false-positive trap explicitly: a benign case that *looks* dangerous (for `guard`, a
legit request containing "ignore the rules" as a book theme). Keep both classes supported — with
the ≤ 3 sample cap, 2 of one class + 1 of the other so neither recall is computed from zero.

## Coverage — what a good dataset spans

- One clean case per label (happy path).
- Genuine **multi-label** turns (e.g. recommend **and** discuss).
- **Negatives / fallback** (a vague or chit-chat turn → `clarify`; nothing to remember → empty set).
- **Near-misses / adversarial** cases that target this node's known confusions. For `understand`:
  the profile-vs-task rule (a stated fact + a task stays a task; the fact goes to `user_signals`,
  not a `*_update` intent), and each child-resolution state (matched / new / ambiguous / none).
- Keep the label distribution **balanced**; note rare-label support explicitly.

## Determinism & robustness

- The node under test already runs at temperature 0 (`agent/llm.py`); keep it that way.
- Wrap the node call in `try/except` in `predict()` and return an `error` marker — one malformed
  structured output is scored as a miss, never a batch-killer. Surface `n_errors` in the summary.
- **Fail loud when the node isn't actually wired up.** If a missing key/dependency makes the node
  degrade to a constant answer (e.g. `guard` fails open → always "benign"), the eval must *fail its
  gate*, not pass green — design the metric so the degenerate output scores 0 (a missing
  `GROQ_API_KEY` collapses `attack_recall` to 0). A green run must mean the real path ran.

## Thresholds

- Start **conservative** (a small set is noisy). Gate on `exact_match` + `micro_f1` (+ any
  secondary accuracy like `child_resolution`). **`macro_f1` is a diagnostic, not a small-set gate:**
  it averages over the *whole* label space, so a tiny dataset that only exercises 2 of 10 labels
  scores low no matter how correct it is. Add `macro_f1_min` only once the dataset covers every
  label with real support. Raise floors as the dataset grows and numbers stabilize.
- Where a decision is genuinely fuzzy (e.g. `memory_policy` may reasonably add a summary op), keep
  `exact_match` low and lean on `macro_f1`; say so in the readme.
- Regenerate floors with `python -m eval_regression.produce` rather than hand-tuning by feel.
- "Start conservative because a small set is noisy" is about *LLM* nodes that sample. A node whose
  verdict comes from a **deterministic, temperature-0 external classifier** (e.g. `guard`'s Prompt
  Guard call) is not noisy in that way: if the observed margin is large and stable, strict floors
  (`attack_recall_min: 1.0`) are appropriate — say so in the readme and revisit only when harder
  cases land.
