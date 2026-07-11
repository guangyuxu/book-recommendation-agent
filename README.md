# Book Recommendation Agent

[![CI](https://github.com/guangyuxu/book-recommendation-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/guangyuxu/book-recommendation-agent/actions/workflows/ci.yml)

An AI-powered book recommendation assistant for families, built on [LangGraph](https://github.com/langchain-ai/langgraph) and Claude (Anthropic). The agent understands reading preferences, child profiles, and family context to make personalized book recommendations, generate discussion questions, plan reading paths, and track reading history.

## Architecture

The agent is a `StateGraph` pipeline with parallel branches and a human-in-the-loop (HITL) confirmation gate:

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	load_context(load_context)
	understand(understand)
	plan(plan)
	clarify(clarify)
	execute(execute)
	respond(respond)
	__end__([<p>__end__</p>]):::last
	__start__ --> load_context;
	clarify -. &nbsp;ask_user&nbsp; .-> __end__;
	clarify -.-> execute;
	clarify -.-> memory\3amemory_policy;
	execute --> respond;
	load_context --> understand;
	memory\3aprofile_update --> respond;
	plan --> clarify;
	understand --> plan;
	respond --> __end__;
	subgraph memory
	memory\3amemory_policy(memory_policy)
	memory\3aprepare_confirmation(prepare_confirmation)
	memory\3arequest_confirmation(request_confirmation)
	memory\3aapply_confirmation(apply_confirmation)
	memory\3aprofile_update(profile_update)
	memory\3aapply_confirmation --> memory\3aprofile_update;
	memory\3amemory_policy --> memory\3aprepare_confirmation;
	memory\3aprepare_confirmation -. &nbsp;skip&nbsp; .-> memory\3aprofile_update;
	memory\3aprepare_confirmation -. &nbsp;confirm&nbsp; .-> memory\3arequest_confirmation;
	memory\3arequest_confirmation --> memory\3aapply_confirmation;
	end
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

| Node | Role |
|---|---|
| `load_context` | Loads family, members, children, and policies from DB into state |
| `understand` | NLU: extracts intents, child references, and user signals from the message |
| `plan` | Deterministic planner: maps intents → capability DAG (no LLM, fully testable) |
| `clarify` | Checks plan feasibility; routes to `ask_user` or proceeds |
| `execute` | Runs capabilities in dependency order; exceptions are isolated per capability |
| `memory` | Memory subgraph: extracts profile updates, optionally pauses for HITL confirmation |
| `respond` | Composes the final reply from capability results and memory outcome |

**Capabilities**: `recommend`, `evaluate`, `compare`, `discussion`, `path`, `content`

The `memory` subgraph runs in parallel with `execute`. When HITL confirmation is required, `interrupt()` fires in the memory branch only — the `execute` branch has already checkpointed, so it does not re-run on resume.

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- PostgreSQL (for the app's `BOOK_AGENT_DATABASE_URL` and LangGraph Server's `DATABASE_URI`)
- Redis (for LangGraph Server's checkpointer: `REDIS_URI`)
- Anthropic API key

## Setup

**1. Clone and install dependencies**

```bash
git clone <repo-url>
cd book-recommendation-agent
uv sync
```

**2. Create `.env`**

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# App database (SQLAlchemy psycopg driver, search_path pinned to book_agent schema)
BOOK_AGENT_DATABASE_URL=postgresql+psycopg://user:password@host:5432/dbname?options=-csearch_path%3Dbook_agent

# LangGraph Server (postgres:// scheme, not postgresql+psycopg://)
DATABASE_URI=postgres://user:password@host:5432/dbname?sslmode=disable
REDIS_URI=redis://:password@host:6379

# Optional: LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=book-recommendation-agent
```

**3. Create the database schema and tables**

```bash
make init-db
```

This runs `CREATE SCHEMA IF NOT EXISTS book_agent` and `Base.metadata.create_all` (idempotent — existing tables are never dropped).

**4. Start the development server**

```bash
langgraph dev
```

The LangGraph API starts at `http://localhost:2024`. Open [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) to inspect graph state, edit checkpoints, and replay nodes.

### Runtime context

Each API call requires an `AppContext` configuration:

```json
{
  "family_id": "<uuid>",
  "family_member_id": "<uuid>",
  "child_id": "<uuid>"
}
```

`child_id` is optional. When omitted, the agent infers the target child from the conversation or asks for clarification if the family has more than one child.

## Development

```bash
uv sync        # install dev tooling (ruff, mypy, pytest, codespell, coverage)
make ci        # run the full CI pipeline locally before pushing
```

`make ci` runs these steps in the same order as the GitHub Actions workflow:

| Step | Command | What it checks |
|---|---|---|
| 1. Lint | `ruff check .` | style and lint rules |
| 2. Types | `mypy` | strict type checking (`[tool.mypy]` in pyproject.toml) |
| 3. Spelling | `codespell README.md src/` | typos in README and source |
| 4. Tests | `pytest tests/unit_tests` | unit tests (sqlite in-memory by default) |

CI runs tests against a real Postgres service container. To replicate that locally:

```bash
export BOOK_AGENT_DATABASE_URL=postgresql+psycopg://...
make init-db
make test
```

### Common targets

```bash
make test              # unit tests
make test_watch        # watch mode
make coverage          # unit tests with coverage report
make lint              # ruff + format diff + mypy
make format            # auto-format with ruff
make spell_check       # codespell on README.md and src/
make spell_fix         # auto-fix spelling
make init-db           # create/update DB schema and tables
```

### Evals (LLM quality gates)

Node-level evals under `evals/` measure output quality against stored thresholds. They call the Anthropic API and are opt-in via `RUN_EVAL=1`:

```bash
make eval                       # gate all nodes
make eval_node NODE=understand  # gate one node
make eval_produce               # regenerate thresholds
```

See [`evals/README.md`](evals/README.md) for the full workflow.

## Deployment (minikube)

```bash
make deploy     # first-time: build image, apply k8s manifests, create Secret from .env, roll out
make redeploy   # after code changes: rebuild + rollout (most common)
make k8s-logs   # follow pod logs
make k8s-pf     # port-forward to localhost:8000
make k8s-down   # tear down the entire namespace
```

Kubernetes manifests live in `k8s/`. Secrets are populated from `.env` via `kubectl create secret`. Use `make k8s-secret` after editing `.env` without redeploying the image.

To pin a release tag:

```bash
make redeploy TAG=1.0.0
```
