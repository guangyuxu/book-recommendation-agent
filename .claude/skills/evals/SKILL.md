---
name: evals
description: >-
  Rules and workflows for the agent's eval system under evals/ and eval_regression/. Use when
  adding or changing an eval, generating an eval dataset from raw user material, running or gating
  evals, or (re)producing thresholds. Covers the node-first layout, the classify/judge strategies,
  the module interface, and dataset-generation best practices. Accepts an optional argument: a
  node name to scaffold/target, or raw material to turn into a dataset; no arg = general guidance.
---

# Evals — rules & workflows

This skill governs how evals are structured, written, generated, run, and gated for this repo.
Follow it whenever you touch `evals/` or `eval_regression/`. Read the matching reference file
before doing the work — each is short and specific.

- `references/layout.md` — the directory/naming LAW, the module interface, the run matrix.
- `references/classification.md` — the `classify_` strategy: metrics, coverage, thresholds.
- `references/judge.md` — the `judge_` strategy: rubric, judge model, calibration, bias guards.
- `references/regression.md` — `eval_regression`: the gate + threshold production.
- `references/dataset-generation.md` — turning raw user material into eval cases.

## Hard rules (never violate)

1. **Node-first layout.** The tree mirrors `src/agent` exactly. One dir per node under test;
   **strategy is a filename prefix** (`classify_*`, `judge_*`). A node's eval for
   `src/agent/pipeline/understand.py` lives at `evals/agent/pipeline/understand/`. Subgraphs keep parity
   (`src/agent/memory/decide.py` → `evals/agent/memory/decide/`). Shared infra lives only in
   `evals/_harness/`.
2. **English only.** Every artifact — SKILL/reference docs, READMEs, code, comments, and all
   dataset content (messages, child profiles, policies, rubrics) — is written in English.
3. **Gold is spec-derived, never model-derived.** Labels for `classify` come from the product spec
   (`src/agent/intents.py`, the domain tool menu, etc.), decided by a human/spec — never by asking
   the model under test what it thinks the answer is.
4. **Raw input is a source, not a test case.** A parent's raw utterance is material to derive a
   case from (normalize, add roster/pins/policies, attach spec gold, add near-misses). Never paste
   it in verbatim as a dataset row.
5. **Sample datasets ship ≤ 3 cases.** Worked examples and any scaffold you generate cap at 3
   cases. Real datasets grow later; samples stay tiny and illustrative.
6. **`run.py` is pytest-free pure logic.** The gate is `eval_regression`. Each `<strategy>_run.py`
   is both an ad-hoc CLI and the programmatic entrypoint the gate calls.
7. **Thresholds are co-located** per node (`<strategy>_thresholds.json`, `"<metric>_min"` keys) and
   reviewed in the same PR as the dataset.

## The module interface (every `<strategy>_run.py`)

```python
STRATEGY = "classify"                 # or "judge" (matches the filename prefix)
NODE = "agent.pipeline.understand"          # full dotted import path (incl. the agent layer)
THRESHOLDS = Path(__file__).with_name("classify_thresholds.json")
def load_cases() -> list: ...
def run_all(cases=None) -> dict:      # {"summary": {metric: float}, "cases": [...]}
if __name__ == "__main__":            # print report; support --gate (assert, exit non-zero)
```
`evals/_harness/discovery.py` finds modules by this convention — nothing is registered by hand.

## Workflow 1 — scaffold a new node eval

```bash
python -m evals._harness.scaffold <node.dotted> <classify|judge>
# e.g. python -m evals._harness.scaffold agent.pipeline.clarify classify
```
The node arg is the full dotted import path INCLUDING the `agent` layer, so it lands under the
`evals/agent/` mirror. This writes `evals/agent/pipeline/clarify/` with the strategy's file set (run/dataset/thresholds/readme, plus a
rubric for judge) from `evals/_harness/templates/`, with the interface pre-filled and node-specific
logic left as `NotImplementedError`. Then fill in `predict()`/`generate()`+`render()` following the
worked examples (`evals/agent/pipeline/understand/`, `evals/agent/capabilities/recommend/`).

The scaffolded dataset row and `classify_thresholds.json` are **`understand`-shaped placeholders**,
not a generic skeleton. Adapt them to your node: drop case fields the node doesn't read (e.g. a
message-only node like `guard` needs no `children`/`target_child_id`), and **rename the threshold
keys to your node's real `summary` metrics** — the template ships `exact_match_min`/`micro_f1_min`,
and a key that names no emitted metric is a hard failure at gate time (`references missing metric`),
by design.

## Workflow 2 — generate a dataset from raw material

The user gives raw utterances/scenarios; you produce ≤ 3 well-formed, spec-labeled cases. Follow
`references/dataset-generation.md` end to end: pick the node + strategy, normalize each raw item
into that node's self-contained case schema, derive gold from the spec (classify) or build
context + rubric (judge), add edge/negative/near-miss coverage, balance and de-dup, tag
`provenance` + `difficulty`, and write English JSONL into the node's `<strategy>_datasets.jsonl`.

## Workflow 3 — run

```bash
python -m evals.agent.pipeline.understand.classify_run           # one node, ad-hoc (prints)
python -m evals.agent.pipeline.understand.classify_run --gate    # one node, gated
make eval_classify        # all classify   |  make eval_judge  # all judge
make eval_node NODE=decide# one node by name
make eval                 # everything (CI gate)
RUN_EVAL=1 pytest eval_regression -m judge -k recommend    # pytest by strategy/node
```
Evals are opt-in (`RUN_EVAL=1`; they call the API) and need **whatever API key the node under
test calls** — usually `ANTHROPIC_API_KEY`, but a node that uses another provider needs that
provider's key (e.g. `agent.guard` calls Groq → `GROQ_API_KEY`). State the required key in the
node's readme. For judge, also set `EVAL_JUDGE_MODEL` to a stronger, separate model.

## Workflow 4 — gate & produce thresholds

`eval_regression` is the flat gate. `make eval` runs every node eval and enforces its co-located
floors. To (re)generate floors from observed metrics:
```bash
python -m eval_regression.produce --dry-run     # preview suggested changes
python -m eval_regression.produce --repeats 3   # sample judge variance, then write
```
See `references/regression.md`.
