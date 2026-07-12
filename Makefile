# ─────────────────────────────────────────────────────────────────────────────────────
# VERIFICATION MAP — this Makefile is the single source of truth. ci.yml and
# .pre-commit-config.yaml only CALL these targets (never restate commands), so no drift.
#
#   ci    = lint + audit + coverage    ← GitHub Actions (verbatim) + pre-push hook
#   check = lint + test                ← everyday local + pre-commit hook
#   lint  = lint_ruff + lint_format + typecheck + spell_check
#
#           lint_ruff .... ruff check          spell_check .. codespell
#           lint_format .. ruff format --diff   test ......... pytest
#           typecheck .... mypy                 coverage ..... pytest + coverage report
#                                               audit ........ pip-audit  (needs network)
#
#   fixers (manual):  format = fix formatting + imports    spell_fix = fix spelling
#
#   coverage RUNS the full test suite (so `ci` does not skip tests); `check` is fully offline.
#   Evals are separate (opt-in, cost API tokens) — see the EVALS section.
# ─────────────────────────────────────────────────────────────────────────────────────

.PHONY: all \
	lint_ruff lint_format typecheck spell_check audit test coverage \
	lint check ci format spell_fix \
	eval eval_classify eval_judge eval_node eval_produce init-db graph \
	mk-start docker-build mk-load k8s-secret k8s-apply deploy redeploy \
	k8s-status k8s-logs k8s-pf k8s-down help

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
# Before push:   `make ci`     (faithful CI gate: also runs pip-audit + coverage; audit needs net)
# Tests use sqlite:///:memory: by default; for Postgres set BOOK_AGENT_DATABASE_URL + `make init-db`.

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

test:                    ## pytest suite
	uv run pytest tests/

coverage:                ## runs the FULL test suite under coverage + report (this is how `make ci` runs tests)
	uv run coverage run -m pytest tests/
	uv run coverage report

# -- composites --
lint: lint_ruff lint_format typecheck spell_check  ## all static checks: ruff + format + mypy + codespell (fast, offline)
check: lint test                                   ## everyday gate after code changes: lint + tests (offline)
ci: lint audit coverage                            ## full CI gate: lint + audit + tests(coverage); coverage RUNS the suite

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
# DEPLOY (local minikube)
######################

IMAGE     ?= book-recommendation-agent
DEPLOY    ?= book-recommendation-agent
CONTAINER ?= agent
K8S_NS    ?= book-agent

# Use a unique tag (timestamp) per build to avoid the "same tag won't update" trap.
# To pin a fixed tag: make redeploy TAG=0.1.0
DATE := $(shell date +%Y%m%d-%H%M%S)
TAG  ?= dev-$(DATE)

mk-start:                ## Start minikube (skip if already running)
	minikube status >/dev/null 2>&1 || minikube start --driver=docker

docker-build:            ## Build the image $(IMAGE):$(TAG)
	docker build -t $(IMAGE):$(TAG) .

mk-load: docker-build    ## Load the image into minikube (local image, no registry)
	minikube image load $(IMAGE):$(TAG)

k8s-secret:              ## Create/update the agent-env Secret from .env
	@test -f .env || { echo "ERROR: .env not found"; exit 1; }
	kubectl apply -f k8s/namespace.yaml
	kubectl create secret generic agent-env --from-env-file=.env -n $(K8S_NS) \
		--dry-run=client -o yaml | kubectl apply -f -

k8s-apply:               ## Apply the namespace / service / deployment manifests
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/service.yaml -f k8s/deployment.yaml

# First-time deploy: create the Secret, apply manifests, build & load the image, roll to the new tag, wait for rollout.
deploy: mk-start k8s-secret k8s-apply mk-load
	kubectl set image deploy/$(DEPLOY) $(CONTAINER)=$(IMAGE):$(TAG) -n $(K8S_NS)
	kubectl rollout status deploy/$(DEPLOY) -n $(K8S_NS) --timeout=120s

# Redeploy after code changes: build -> load -> roll to the new tag -> wait for rollout (most common).
redeploy: mk-load
	kubectl set image deploy/$(DEPLOY) $(CONTAINER)=$(IMAGE):$(TAG) -n $(K8S_NS)
	kubectl rollout status deploy/$(DEPLOY) -n $(K8S_NS) --timeout=120s

k8s-status:              ## Show pod / service status
	kubectl get pods,svc -n $(K8S_NS) -o wide

k8s-logs:                ## Follow logs from all pods
	kubectl logs -n $(K8S_NS) -l app=$(DEPLOY) --tail=80 -f

k8s-pf:                  ## Port-forward to local 8000 (http://localhost:8000/docs)
	kubectl port-forward -n $(K8S_NS) svc/$(DEPLOY) 8000:8000

k8s-down:                ## Delete the entire book-agent namespace (Secret/Deployment/Service)
	kubectl delete namespace $(K8S_NS) --ignore-not-found

######################
# HELP
######################

help:
	@echo '--- checks (local == CI; see .github/workflows/ci.yml) ---'
	@echo 'check                        - everyday gate after code changes: lint + test (offline)'
	@echo 'ci                           - faithful GitHub CI mirror: lint + audit + coverage'
	@echo 'lint                         - static checks: ruff check + ruff format --diff + mypy + codespell'
	@echo 'format                       - auto-fix formatting + import order'
	@echo 'test                         - run all tests under tests/'
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
	@echo '--- deploy (minikube) ---'
	@echo 'deploy                       - first-time deploy: secret + manifests + build/load + rollout'
	@echo 'redeploy                     - redeploy after code changes: build/load + new tag + rollout (most common)'
	@echo 'k8s-secret                   - update the Secret from .env (run after editing .env)'
	@echo 'k8s-status                   - show pod / service status'
	@echo 'k8s-logs                     - follow pod logs'
	@echo 'k8s-pf                       - port-forward to localhost:8000'
	@echo 'k8s-down                     - delete the entire namespace'
	@echo '  override tag: make redeploy TAG=0.1.0'
