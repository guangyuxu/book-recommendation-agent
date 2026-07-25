# Project Rules for Claude Code

## PII & Security

This project stores and processes children's personal data (name, birthday, gender, reading
level). Treat all child/family data as high-sensitivity PII.

### Logging rules

- **Never log PII values in `logger.*` calls.** This includes: child names, birth dates,
  genders, reading interests, goals, user messages, family member names, and any field from
  `ChildProfile`, `FamilyMember`, `ChildReadingProfile`, `FamilyReadingPolicy`.
- When logging exceptions that may have touched DB rows or user input, log only the exception
  **type** (`type(exc).__name__`), never the full exception object or message.
  ```python
  # WRONG
  logger.warning("failed: %s", exc)
  # RIGHT
  logger.warning("failed: %s", type(exc).__name__)
  ```
- Safe to log: IDs (UUIDs), capability names, intent names, operation names, row counts,
  boolean flags.

### Prompt / LLM rules

- Do not add `logger.*` calls that print the raw `state["messages"]` or any user-supplied
  string without explicitly stripping PII first.
- When adding new LLM nodes, never pass raw DB rows into a prompt; serialize only the fields
  the node needs.

### Authorization rules

- Every repository read method that takes a `child_id` or `member_id` **must also filter by
  `family_id`**. A query scoped only to `child_id` is a cross-family data leak.
- Domain tools must always read identity from `current()` (the contextvar), never from
  user-supplied tool arguments.

## Prompt authoring

Every model prompt lives in the versioned registry (`agent.prompts`), never as an inline
`SystemMessage`/`HumanMessage` string in a node or capability. This is what lets prompts be
reviewed, versioned, A/B-tested, and rolled back independently of code (ROADMAP #1).

- **One co-located `<module>.prompts.yaml`** next to the module that uses it (e.g.
  `pipeline/respond.prompts.yaml` beside `respond.py`). Each file has a top-level `namespace:`
  and a `prompts:` map. Each entry declares an integer `version:` and either a `system:`
  shorthand (one system message) or a `messages:` list of `{role, template}` for multi-role
  prompts. The stable id is `<namespace>.<key>` and must be globally unique.
- **Call it, don't inline it.** `system = prompts.render("<id>", **vars)` then
  `llm.invoke([*system, *state["messages"]])` (or `.stream(...)`). See `capabilities/recommend.py`
  for the reference call site.
- **Record the version on every call.** Pass `config=prompts.config("<id>")` to the
  `.invoke`/`.stream` so the run's metadata carries `prompt_id` + `prompt_version` (visible in
  LangSmith / the trace). This is what ties a turn back to the exact prompt version that produced
  it — do not omit it when adding a new LLM call. The metadata merges with the chain's own
  metadata (the strategy tag, the LangGraph node); it never replaces it.
- **The YAML/Python boundary — "Python decides what is true; the template decides how to say
  it":**
  - *In YAML:* all instruction wording, and all conditional **phrasing** (`{% if %}`, `{% for %}`,
    inline `if/else`) — retry directives, focus-switch notes, confirmation outcomes, feedback
    folding, where each dynamic block appears, and the role structure.
  - *In Python:* which facts are true (flags, status values, feedback lists) passed as render
    vars; **all DB-row/PII serialization** (`child_brief`, `policies_brief`, roster, book/candidate
    rendering, `ops_text`) done by helpers and passed as pre-serialized **string** vars — never a
    raw row into a template; menus derived from a code source-of-truth (`intent_menu()`, the
    available-operations list) computed in Python and passed as vars; control flow, graph wiring,
    and post-LLM gating.
- **PII:** the registry renders only caller-supplied vars and logs nothing. Pass serialized
  briefs, never raw `ChildProfile`/`FamilyReadingPolicy` rows (this is the same "never pass raw DB
  rows into a prompt" rule as above).
- **Jinja defaults are fixed:** `StrictUndefined` (a missing var is a hard render error, never a
  silently blank prompt) and `autoescape=False` (prompt text, not HTML). A template must be
  passed every var it references.
- **Testing:** every prompt with conditional wording gets a `tests/unit_tests/test_prompts.py`
  case pinning each branch (present/absent). Node behavior stays covered by the node's own
  tests/evals.

## Testing rules

- New repository methods that read data must have a cross-family isolation test: seed data
  under family A, query with family B's id, assert empty result.
- Prompt-injection scenarios for any new LLM node that takes user input: verify that an
  off-roster or malformed LLM output is rejected by the post-LLM gating logic.

## Build & verification

The Makefile `CHECKS` section is the single source of truth for verification. Nothing restates
those commands: GitHub Actions (`.github/workflows/ci.yml`) runs `make ci` verbatim, and the
pre-commit hooks (`.pre-commit-config.yaml`) run `make check` on commit and `make ci` on push. So
local and CI cannot drift. Evals are excluded (opt-in, cost API tokens).

After every code change, run the everyday gate and make sure it is green before treating the work
as done. Do NOT report a task as complete while any check fails.

```bash
make check   # lint (ruff check + ruff format --diff + mypy + codespell) + test — fast, offline
```

Before pushing, run the full CI mirror (lint + tests under coverage, with a `fail_under` floor):

```bash
make ci      # what GitHub Actions runs verbatim: lint + coverage (offline)
```

If `make check` reports formatting diffs, run `make format` to auto-fix them. Optional: install
the local hooks once with `uv run pre-commit install` (runs `make check` + gitleaks on commit,
`make ci` on push). Focused subsets while iterating: `make lint`, `make test`, `make spell_check`.

Security tooling (does not block the code gate): ruff's `S` (flake8-bandit) rules run inside
`lint`; `make audit` (pip-audit) runs on a schedule (`.github/workflows/audit.yml`); gitleaks
scans for secrets in pre-commit and in CI; Dependabot opens dependency-update PRs.
