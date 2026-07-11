.PHONY: all format lint test tests test_watch coverage ci help extended_tests \
	mk-start docker-build mk-load k8s-secret k8s-apply deploy redeploy \
	k8s-status k8s-logs k8s-pf k8s-down

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

test:
	python -m pytest $(TEST_FILE)

tests: test   # alias so `make tests` works too

# No integration_tests target: tests/integration_tests/ is an empty placeholder. When you add
# end-to-end tests there, run them with `RUN_INTEGRATION=1 python -m pytest tests/integration_tests`.

test_watch:
	python -m ptw --snapshot-update --now . -- -vv tests/unit_tests

test_profile:
	python -m pytest -vv tests/unit_tests/ --profile-svg

coverage:
	python -m coverage run -m pytest tests/unit_tests/
	python -m coverage report

# Faithful mirror of .github/workflows/ci.yml -- run this before pushing to catch what CI catches.
# Same commands, same order: ruff check -> mypy (strict, config-driven) -> codespell -> unit tests.
# Uses `uv run` (like CI) so it works from a fresh `uv sync` without activating the venv first.
# (`make lint` is a stricter superset for day-to-day dev: it also diffs formatting and import order.)
ci:
	uv run ruff check .
	uv run mypy
	uv run codespell --skip ./.git --ignore-words .codespellignore README.md
	uv run codespell --skip ./.git --ignore-words .codespellignore src/
	uv run pytest tests/unit_tests

extended_tests:
	python -m pytest --only-extended $(TEST_FILE)

######################
# EVALS (LLM output quality; opt-in, calls the Anthropic API)
######################

# Node evals live under evals/<tree>/<node>/ (mirroring src/agent); eval_regression/ is the gate.
# All gated on RUN_EVAL=1 so a normal `make test` never spends API tokens. See evals/README.md.
eval:                    ## Gate ALL node evals against their thresholds (CI entrypoint)
	RUN_EVAL=1 python -m eval_regression.run

eval_classify:           ## Gate only the classify-strategy nodes
	RUN_EVAL=1 python -m eval_regression.run --strategy classify

eval_judge:              ## Gate only the judge-strategy nodes
	RUN_EVAL=1 python -m eval_regression.run --strategy judge

eval_node:               ## Gate one node: make eval_node NODE=understand
	RUN_EVAL=1 python -m eval_regression.run --node $(NODE)

eval_produce:            ## Regenerate co-located thresholds (add ARGS='--dry-run' to preview)
	python -m eval_regression.produce $(ARGS)


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files (used to scope ruff format/import checks).
PYTHON_FILES=src/
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=src
lint_tests: PYTHON_FILES=tests

lint lint_diff lint_package lint_tests:
	python -m ruff check .
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff check --select I $(PYTHON_FILES)
	# Config-driven: reads [tool.mypy] (strict, files = src/agent) -- the same single standard
	# CI runs. No --strict flag or path args here, so no entry point can drift.
	python -m mypy

format format_diff:
	ruff format $(PYTHON_FILES)
	ruff check --select I --fix $(PYTHON_FILES)

# Same scope/config as CI: .codespellignore + the README.md and src/ paths (not the whole tree).
spell_check:
	codespell --skip ./.git --ignore-words .codespellignore README.md src/

spell_fix:
	codespell --skip ./.git --ignore-words .codespellignore -w README.md src/

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
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'coverage                     - run unit tests with a coverage report'
	@echo 'ci                           - mirror the full GitHub CI pipeline locally (run before pushing)'
	@echo 'spell_check                  - check spelling in README.md and src/ (same as CI)'
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

