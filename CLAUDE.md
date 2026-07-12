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
