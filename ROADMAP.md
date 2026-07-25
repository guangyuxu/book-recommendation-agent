# Roadmap

The agent is MVP-complete: a LangGraph pipeline (`guard → load_context → understand → plan →
clarify → {execute ∥ memory} → respond`) served on LangGraph Platform, with a HITL confirmation
gate, PII-safe logging, `family_id`-scoped data access, token-usage accounting, and a
discovery-based eval harness. Every node has a unit test or an eval.

This document tracks post-MVP hardening and capability work. It is not a backlog of bugs — it is
the set of architectural investments that take the system from "well-built MVP" to "operated at
scale for real families."

_Last updated: 2026-07-15._

---

## Backend-owned (pass-through BFF)

These production concerns are owned by the **pass-through backend** that fronts this agent (the
BFF/gateway that authenticates users and proxies requests into the LangGraph run) — **not** by this
repo. They are consolidated here as one list so the boundary, and the backend's own to-do, are
explicit in a single place. The agent's job is to stay safe under these assumptions and to
*produce* the data they consume.

- **B1 — Authentication & authorization.** Authenticate the caller and **derive** `family_id` /
  `family_member_id` from a verified token, then pass them in the run context. The agent trusts
  that context (domain tools read identity from `current()`, never from tool args) — it must never
  be settable by an untrusted client.
- **B2 — Edge input validation & rate limiting.** Request size caps, encoding/schema validation,
  and per-family/member rate limiting, applied before a run starts. (`guard` remains the *semantic*
  prompt-injection check only; see its module docstring.)
- **B3 — Cost governance (enforcement).** Per-family budgets, quotas, and circuit-breaking,
  metering on the agent's cost projection (see #3 and #8). The agent produces the usage/cost data;
  the backend enforces the limits.
- **B4 — Frontend HTTP surface, incl. the usage endpoint.** Serve the per-turn usage summary to the
  frontend — the agent supplies the `GROUP BY node` aggregation keyed by `turn_id` (see #3), the
  backend exposes it over HTTP.

---

## P0 — highest priority

### ✅ 1. Prompt management
Prompts were inline string literals across the nodes/capabilities — the main quality-and-maintenance
risk as capabilities are strengthened. Now a versioned prompt registry (`agent.prompts`) so prompts
can be reviewed, versioned, A/B-tested, and rolled back independently of code.
- ✅ Extracted every node/capability prompt into a versioned store: co-located `*.prompts.yaml`
  files (id `<namespace>.<key>` + integer `version` + Jinja template), loaded by `agent.prompts`.
  The YAML/Python boundary ("Python decides what is true; the template decides how to say it") is
  codified as a rule in CLAUDE.md.
- ✅ Record the prompt version used per turn — via the **trace**: every LLM call passes
  `config=prompts.config(id)`, tagging the run metadata with `prompt_id` + `prompt_version`
  (visible in LangSmith). Persisting it onto the `token_usage_record` row is a small follow-up
  folded into **#3 Observability** (which owns that store), not yet wired.
- A/B + rollback without a code deploy: delivered by **L2 — Assistants + `configurable`** (the
  registry is the prerequisite, now in place); tracked there.

### LangGraph advanced features — trial / adoption practice
A deliberate hands-on practice to exercise LangGraph's advanced surface and de-risk the roadmap.
Placed in P0 as learning/trial work; several directly accelerate later items (cross-refs noted).
Labelled **L1–L8** to stay distinct from the hardening items #1–#8.

- ✅ **L1 — Streaming** (`stream_mode="messages" / "updates" / "custom"`,
  `get_stream_writer`). `respond` now streams its reply via `STANDARD.stream_chain()` (streaming-
  enabled model twins in `llm.py`) so tokens reach the frontend as they arrive; `UsageCallbackHandler`
  emits per-node `{node, tokens}` custom events on every LLM call. → accelerates the live half of
  **#3 Observability** + latency UX.
- **L2 — Assistants + `configurable` / `config_schema`.** Versioned, named configs of the same
  graph (prompt version / model / knobs) selectable per request without a redeploy. → the
  platform-native backbone for **#1 Prompt management** (A/B + rollback).
- ✅ **L3 — `Send` API (dynamic map-reduce).** `execute` fans out to a *static* set of capability
  nodes; `Send` fans out dynamically to N per-item subtasks — e.g. evaluate/compare across each
  named book, or screen N candidates concurrently in `recommend.validate`. Reduce side already fits
  the `operator.add` `results` channel. → scales **#2 capabilities**.
  _Decision: item-level concurrency lives **inside** the capability node (`.batch()` with
  `max_concurrency` + `return_exceptions`, which preserves the usage-tracking contextvar), not as a
  top-level `Send` fan-out — keeps the graph topology legible; per-item checkpoint/HITL/retry aren't
  needed here._
- 🚫 **L4 — Node caching (`CachePolicy` + TTL).** Cache deterministic/expensive nodes (e.g.
  `load_context` per thread, identical capability calls). → feeds **#8 cost governance**.
  _N/A: `load_context` reads mutable DB rows that the `memory` branch writes within the same turn,
  so a per-thread/TTL cache serves a stale profile on the next turn (right after a HITL-confirmed
  change) — and it only reads a few rows anyway, so caching it saves ms, not tokens. Correct
  invalidation would need a version-in-key (an extra DB read that negates the win) or write-time
  `clear()` on a shared cache (InMemoryCache is per-pod; we run k8s multi-replica). The only sound
  target — caching LLM capability calls keyed on the profile/policy content snapshot — is left to
  #8 if/when repeat-cost data justifies it._
- **L5 — `Command` (state update + `goto`, `Command.PARENT`).** Let a node update state and pick
  the next node in one return, and navigate subgraph→parent; unifies some router/edge pairs and is
  the modern handoff idiom.
- **L6 — Time travel (`get_state_history` / `update_state`).** Checkpoint replay + state editing:
  debugging, richer HITL ("edit then resume"), and snapshotting real runs into eval cases. →
  supports **#4 Feedback loop** / **#5 Evals**.
- 🚫 **L7 — Scheduled / cron runs (Platform).** Host the feedback-harvest pipeline (**#4**) and the
  scheduled eval gate (**#5**) natively rather than via external cron.
  _N/A: this is a Platform/ops scheduling concern, not agent code, and it presupposes #4/#5 which
  aren't built yet; scheduling can just as well live in external cron / the backend (cf. the
  Backend-owned section), so it isn't this repo's work._
- **L8 — Broader `RetryPolicy` + durability modes.** Apply `RetryPolicy` to LLM nodes for transient
  429/5xx (complements `llm.py` model-level retries); be explicit about checkpoint `durability` if
  self-hosting persistence. → supports **#7 Provider resilience**.

---

## P1 — near-term

### 2. Grounded book knowledge layer + retrieval tools
_(merges former Tier-1 #2 "output grounding" and Tier-4 #11 "retrieval + tool interface" — they are
the requirement and the mechanism for the same thing.)_

Today all capabilities are LLM-only (`run_text` over a prompt), so recommendations can name books
that don't exist or misstate reading level. The fix is a grounded knowledge layer capabilities
**call** instead of inventing:
- A book catalog / knowledge base: structured metadata (title, author, ISBN), reading-level
  signals (Lexile/AR/CEFR), themes, and content flags; vector + structured search.
- A tool interface so `recommend` / `evaluate` / `compare` / `path` retrieve candidates and
  verify facts rather than hallucinate.
- A post-LLM grounding gate: every recommended title must resolve to a real catalog entry
  (the `recommend.validate` step screens *fit* today; add *existence*).

This is the core capability-strengthening epic the rest of the product builds on.

### 3. Observability: per-task token usage to the frontend (+ tracing)
_In design._ One source of truth — `token_usage_record` (per-node, per-turn, already carries
`node` / `input_tokens` / `output_tokens` / `model_id` / `strategy`) — projected two different
ways that must stay **separate**:

- **User-facing = usage.** Per-task token counts + a grand total; **no model split**. A simple
  transparency view (`GROUP BY node`, `SUM(input+output)`) → `{ per_node: [{node, tokens}], total }`.
  Surfaced to the frontend via a post-turn query keyed by `turn_id` (and/or live custom stream
  events). This is the number the parent sees; later it maps naturally to a credits/quota concept.
- **Internal = cost.** Actual dollars, which is **per-model by necessity** (HEAVY/STANDARD/FAST and
  input-vs-output price differently). A separate projection applying a per-`model_id` price table.
  Feeds billing / margin / cost governance (#8) — **never shown to the user**. This is why every
  row keeps `model_id` / `strategy` even though the user view ignores them.

Also:
- Wire real tracing (LangSmith `report_to_langsmith` is currently a stub) and basic metrics
  (latency, error rate, per-node cost).
- Durability: the usage queue is in-memory (`SimpleQueue`) — records in flight are lost on crash.
  Consider a durable sink / flush-on-shutdown.

### 4. Feedback loop (capture now, automate later)
Recommendation sessions are persisted but outcomes are not. Build **instrumentation now** (cheap),
**harvesting later**:
- Now: capture explicit signals (thumbs up/down, accepted / finished) into a table.
- Dev phase: human-curated review of signals into eval datasets (manual labeling).
- Stable phase: an automated pipeline turns signals into (a) eval cases and (b) a ranking signal —
  keeping a human QA gate on what enters the golden eval set (guard against feedback poisoning).

---

## P2 — planned

### 5. Evals as a CI gate
The eval harness exists but is opt-in (`RUN_EVAL=1`) and not run in CI (to save API tokens). Add a
**scheduled** (nightly / pre-release) job that runs `eval_regression` under the co-located
thresholds, so prompt/model changes can't silently regress quality. Version the datasets.

### 6. Integration / e2e tests
`tests/integration_tests/` is currently an empty package. Add a full-pipeline test
(`guard → respond`) with faked LLMs and a test DB, plus a HITL interrupt/resume test across a real
checkpointer.

### 7. Provider resilience
Each LLM strategy is single-provider. Add a fallback model/provider on the main Anthropic path so a
provider outage degrades gracefully rather than failing the turn. (`guard` already fails open on a
Groq outage, which is acceptable.)

### 8. Cost governance (agent-side inputs)
Enforcement lives in the pass-through backend (see Backend-owned, B3). This repo's part: emit clean,
queryable usage (Observability, #3) and consider recommendation caching to cut repeat cost. The
**cost** projection (dollars, per-`model_id` price table) is the internal counterpart of the
user-facing usage view in #3 — same store, different projection — and feeds the backend's budgets
/ quotas / circuit-breaking.

---

## Explicitly out of scope

- **DB migration tooling (Alembic)** — decided not needed at this stage; schema is managed via
  `init_db()` + the scripts in `scripts/`.
