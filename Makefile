.PHONY: all format format_diff lint lint_diff \
	test coverage ci \
	eval eval_classify eval_judge eval_node eval_produce \
	spell_check spell_fix init-db \
	mk-start docker-build mk-load k8s-secret k8s-apply deploy redeploy \
	k8s-status k8s-logs k8s-pf k8s-down help

# Default target executed when no arguments are given to make.
all: help

test:
	uv run pytest tests/

coverage:
	uv run coverage run -m pytest tests/
	uv run coverage report

# Faithful mirror of .github/workflows/ci.yml -- run this before pushing to catch what CI catches.
# Same commands, same order: ruff check -> mypy (config-driven) -> codespell -> unit tests.
# Tests run against sqlite:///:memory: by default (CI uses a Postgres service container).
# To test against Postgres locally: set BOOK_AGENT_DATABASE_URL, run `make init-db`, then `make ci`.
ci:
	uv run ruff check .
	uv run mypy
	uv run codespell --skip ./.git --ignore-words .codespellignore README.md
	uv run codespell --skip ./.git --ignore-words .codespellignore src/
	uv run pytest tests/

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
# LINTING AND FORMATTING
######################

LINT_PATHS = src/ evals/ eval_regression/ tests/
lint_diff format_diff: LINT_PATHS=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')

lint lint_diff:
	uv run ruff check $(LINT_PATHS)
	[ "$(LINT_PATHS)" = "" ] || uv run ruff format $(LINT_PATHS) --diff
	[ "$(LINT_PATHS)" = "" ] || uv run ruff check --select I $(LINT_PATHS)
	# Config-driven: reads [tool.mypy] (strict, files = src/agent).
	uv run mypy

format format_diff:
	uv run ruff format $(LINT_PATHS)
	uv run ruff check --select I --fix $(LINT_PATHS)

spell_check:
	uv run codespell --skip ./.git --ignore-words .codespellignore .

spell_fix:
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
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters on src/ evals/ eval_regression/ tests/ (ruff + mypy)'
	@echo 'test                         - run all tests under tests/'
	@echo 'coverage                     - run tests with a coverage report'
	@echo 'ci                           - mirror the full GitHub CI pipeline locally (run before pushing)'
	@echo 'spell_check                  - check spelling in README.md and src/ (same as CI)'
	@echo 'spell_fix                    - auto-fix spelling in README.md and src/'
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
