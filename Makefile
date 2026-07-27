# ─────────────────────────────────────────────────────────────────────────────────────
# VERIFICATION MAP — this Makefile is the single source of truth. ci.yml and
# .pre-commit-config.yaml only CALL these targets (never restate commands), so no drift.
#
#   ci    = lint + coverage            ← GitHub Actions (verbatim) + pre-push hook
#   check = lint + test                ← everyday local + pre-commit hook
#   lint  = lint_ruff + lint_format + typecheck + spell_check
#
#           lint_ruff .... ruff check          spell_check .. codespell
#           lint_format .. ruff format --diff   test ......... pytest
#           typecheck .... mypy                 coverage ..... pytest + coverage report
#
#   coverage RUNS the full test suite (so `ci` does not skip tests); `check` is fully offline.
#   audit (pip-audit) is NOT in ci/check — it needs the network, so it runs on a schedule
#     (.github/workflows/audit.yml); run it by hand with `make audit`.
#   fixers (manual):  format = fix formatting + imports    spell_fix = fix spelling
#   Evals are separate (opt-in, cost API tokens) — see the EVALS section.
#   (k8s / deploy targets are intentionally omitted here — infra is handled separately, in the
#    book-recommendation-deploy repo, which owns the manifests for the WHOLE platform.)
# ─────────────────────────────────────────────────────────────────────────────────────

.PHONY: all \
	lint_ruff lint_format typecheck spell_check audit test coverage integration \
	lint check ci format spell_fix \
	eval eval_classify eval_judge eval_node eval_produce init-db graph \
	help

# Default target executed when no arguments are given to make.
all: help

######################
# CHECKS
######################
# Single source of truth for verification. Nothing else restates these commands:
#   - GitHub Actions (.github/workflows/ci.yml) runs `make ci` verbatim.
#   - pre-commit (.pre-commit-config.yaml) runs `make check` on commit and `make ci` on push.
# So local == CI by construction. Evals are excluded (opt-in, cost API tokens -- see EVALS below).
#
# Everyday use:  `make check`  (fast, offline: lint + test; lint = ruff + format + mypy + codespell)
# Before push:   `make ci`     (what GitHub Actions runs verbatim: lint + coverage; fully offline)
# Tests use sqlite:///:memory: by default; for Postgres set BOOK_AGENT_DATABASE_URL + `make init-db`.
#
# TEST LAYOUT (the same law in accounts / agent / service -- see tests/__init__.py):
#   tests/unit_tests/         fast + offline, tree MIRRORS src/agent/ -- the blocking gate
#                             (`test`/`coverage` scope HERE, so nothing slow can sneak into `ci`).
#   tests/integration_tests/  end-to-end journeys, organized by FLOW -- `make integration`, opt-in.
# LLM output quality is neither: it lives in evals/ (see the EVALS section).

CHECK_PATHS = src/ evals/ eval_regression/ tests/

# -- atomic checks: each is the ONE definition of that check --
lint_ruff:               ## ruff lint rules (import sorting included via [tool.ruff] lint.select)
	uv run ruff check $(CHECK_PATHS)

lint_format:             ## fail if any file is unformatted (does NOT modify files; run `make format` to fix)
	uv run ruff format --diff $(CHECK_PATHS)

typecheck:               ## mypy -- config-driven ([tool.mypy]: strict, files = src/agent)
	uv run mypy

spell_check:             ## codespell over the repo
	uv run codespell --skip ./.git --ignore-words .codespellignore .

audit:                   ## dependency vulnerability scan (hits the network)
	uv run pip-audit

test:                    ## fast unit suite (hermetic; the offline gate)
	uv run pytest tests/unit_tests

coverage:                ## runs the unit suite under coverage + report (this is how `make ci` runs tests)
	uv run coverage run -m pytest tests/unit_tests
	uv run coverage report

# pytest exits 5 ("no tests ran") on an empty suite; this one is still a placeholder, so treat 5 as
# a pass. Identical target text in accounts / service -- see the TEST LAYOUT note above.
integration:             ## end-to-end journeys vs real infrastructure (opt-in; empty for now)
	@uv run pytest tests/integration_tests; s=$$?; [ $$s -eq 5 ] && exit 0 || exit $$s

# -- composites --
lint: lint_ruff lint_format typecheck spell_check  ## all static checks: ruff + format + mypy + codespell (fast, offline)
check: lint test                                   ## everyday gate after code changes: lint + tests (offline)
ci: lint coverage                                  ## code gate CI runs verbatim: lint + tests(coverage); coverage RUNS the suite
# `audit` is intentionally NOT in `ci`: it needs the network, so it runs on a schedule
# (.github/workflows/audit.yml), not on the per-push/PR blocking path. Run it locally with `make audit`.

######################
# EVALS (LLM output quality; opt-in, calls the Anthropic API)
######################

# Node evals live under evals/<tree>/<node>/ (mirroring src/agent); eval_regression/ is the gate.
# All gated on RUN_EVAL=1 so a normal `make test` never spends API tokens. See evals/README.md.
eval:                    ## Gate ALL node evals against their thresholds (CI entrypoint)
	RUN_EVAL=1 uv run python -m eval_regression.run

eval_classify:           ## Gate only the classify-strategy nodes
	RUN_EVAL=1 uv run python -m eval_regression.run --strategy classify

eval_judge:              ## Gate only the judge-strategy nodes
	RUN_EVAL=1 uv run python -m eval_regression.run --strategy judge

eval_node:               ## Gate one node: make eval_node NODE=understand
	RUN_EVAL=1 uv run python -m eval_regression.run --node $(NODE)

eval_produce:            ## Regenerate co-located thresholds (add ARGS='--dry-run' to preview)
	uv run python -m eval_regression.produce $(ARGS)


######################
# AUTO-FIXERS  (the read-only checks live in the CHECKS section above)
######################

format:                  ## auto-fix formatting + import order (the fixer for lint_format)
	uv run ruff format $(CHECK_PATHS)
	uv run ruff check --select I --fix $(CHECK_PATHS)

spell_fix:               ## auto-fix spelling across the repo
	uv run codespell --skip ./.git --ignore-words .codespellignore -w .

######################
# DATABASE
######################

init-db:              ## Create schema + tables (idempotent; requires BOOK_AGENT_DATABASE_URL or .env)
	uv run python scripts/create_tables.py

graph:                ## Print Mermaid diagram for the main graph (update the mermaid block in README)
	BOOK_AGENT_DATABASE_URL=sqlite:///:memory: uv run python -c \
		"from agent.graph import graph; print(graph.get_graph(xray=1).draw_mermaid())"

######################
# HELP
######################

help:
	@echo '--- checks (local == CI; see .github/workflows/ci.yml) ---'
	@echo 'check                        - everyday gate after code changes: lint + test (offline)'
	@echo 'ci                           - faithful GitHub CI mirror: lint + coverage (offline)'
	@echo 'lint                         - static checks: ruff check + ruff format --diff + mypy + codespell'
	@echo 'format                       - auto-fix formatting + import order'
	@echo 'test                         - run the fast unit suite (tests/unit_tests, hermetic)'
	@echo 'integration                  - run end-to-end journeys (tests/integration_tests; empty for now)'
	@echo 'coverage                     - run tests with a coverage report'
	@echo 'spell_check                  - check spelling across the repo'
	@echo 'spell_fix                    - auto-fix spelling across the repo'
	@echo 'audit                        - dependency vulnerability scan (pip-audit; needs network)'
	@echo 'init-db                      - create schema + tables (idempotent; dev/CI setup)'
	@echo 'eval                         - gate all node evals (RUN_EVAL=1, needs API key)'
	@echo 'eval_classify                - gate only classify-strategy nodes'
	@echo 'eval_judge                   - gate only judge-strategy nodes'
	@echo 'eval_node NODE=understand    - gate one node by name'
	@echo 'eval_produce ARGS=--dry-run  - (re)generate co-located thresholds'
	@echo '--- deploy ---'
	@echo 'deployment lives in the book-recommendation-deploy repo (compose + k8s for the platform)'
