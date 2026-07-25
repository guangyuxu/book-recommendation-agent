"""Custom HTTP routes mounted alongside the built-in LangGraph API.

Wired in via ``langgraph.json`` -> ``http.app: "./src/agent/webapp.py:app"``.

Two things live here:

1. ``/internal/assistant-ids`` -- resolves a graph key to the assistant_id
   LangGraph derives for it (``uuid5(NAMESPACE_GRAPH, graph_key)``), so callers
   do not have to hand-compute the hash to build A2A / API URLs.
2. A curated A2A agent card that *shadows* LangGraph's auto-generated one.
   User routes are mounted before the built-in A2A mount, so a same-path route
   here wins the match (Starlette is first-match-wins). LangGraph's default card
   is a hardcoded template that also dumps every graph state field name into
   ``skills[].metadata.inputSchema.properties`` -- an information leak we do not
   want on a public ``.well-known`` endpoint for a children's-PII service.

   The card's ``skills`` are DERIVED from the capability ``REGISTRY`` (the code
   source of truth the Planner already uses), so adding/removing a capability
   updates the card automatically -- no hand-maintained parallel list. Python
   supplies the structural facts (which capabilities exist + their description);
   the small ``_SKILL_TITLES`` / ``_SKILL_EXAMPLES`` maps below supply only
   phrasing. Capability names are product features, not PII, so exposing them is
   safe (unlike the state schema the default card leaked).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import Any
from uuid import uuid5

from langgraph_api.graph import GRAPHS, NAMESPACE_GRAPH  # type: ignore[import-untyped]
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .capabilities.registry import REGISTRY

# --------------------------------------------------------------------------- #
# Top-level card identity -- EDIT to describe the agent as a whole.
# --------------------------------------------------------------------------- #
AGENT_NAME = "Book Recommendation Agent"
AGENT_DESCRIPTION = (
    "Helps families choose and reason about children's books -- recommending, "
    "evaluating, comparing, and planning reading -- tailored to each child's "
    "level and interests and the family's reading policies."
)

_DEFAULT_MODES = ["application/json", "text/plain"]

# --------------------------------------------------------------------------- #
# Auth advertisement. The endpoint is currently OPEN (no auth enforced), so we
# DECLARE the intended scheme for discoverability but leave ``security`` empty.
# ``security`` is machine-readable and enforced by clients regardless of the
# human-readable description -- a non-empty value here would tell callers to send
# a token we do not yet verify. When real auth lands (auth.path handler), move
# the scheme to ``SECURITY`` so the card matches what the server enforces.
# --------------------------------------------------------------------------- #
SECURITY_SCHEMES: dict[str, Any] = {
    "bearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": (
            "Unimplemented: planned company-IdP JWT, resolved to family_id "
            "server-side. Not yet enforced -- the endpoint is currently open."
        ),
    }
}
# Empty = no auth currently required. Set to [{"bearerAuth": []}] once enforced.
SECURITY: list[dict[str, Any]] = []

# --------------------------------------------------------------------------- #
# Phrasing layer for the derived skills. Keys are capability names from REGISTRY;
# a capability with no entry still appears, just without a prettier title /
# examples. Descriptions come from REGISTRY, not here, to avoid drift.
# --------------------------------------------------------------------------- #
_SKILL_TITLES: dict[str, str] = {
    "recommend": "Book recommendations",
    "evaluate": "Book fit evaluation",
    "compare": "Book comparison",
    "discussion": "Reading discussion questions",
    "path": "Reading path planning",
    "content": "Content drafting",
}
_SKILL_EXAMPLES: dict[str, list[str]] = {
    "recommend": [
        "Recommend a booklist for my 6-year-old who loves space.",
        "What should my kindergartener read next?",
    ],
    "evaluate": ["Is 'Charlotte's Web' a good fit for my 7-year-old?"],
    "compare": ["Compare 'Frog and Toad' and 'Elephant & Piggie' for my child."],
    "discussion": ["Give me discussion questions after reading 'The Gruffalo'."],
    "path": ["Plan a reading path from picture books to early chapter books."],
    "content": ["Draft a short newsletter blurb recommending summer reads for kids."],
}

try:
    AGENT_VERSION = pkg_version("agent")
except PackageNotFoundError:  # pragma: no cover - package always installed in prod
    AGENT_VERSION = "0.0.0"


def _skills() -> list[dict[str, Any]]:
    """Build A2A skills from the capability registry (structure) + phrasing maps."""
    skills: list[dict[str, Any]] = []
    for cap in REGISTRY.values():
        skill: dict[str, Any] = {
            "id": cap.name,
            "name": _SKILL_TITLES.get(cap.name, cap.name),
            "description": cap.description,
            "tags": ["books", "children", cap.name],
            "inputModes": _DEFAULT_MODES,
            "outputModes": _DEFAULT_MODES,
        }
        examples = _SKILL_EXAMPLES.get(cap.name)
        skill["examples"] = examples if examples else []
        skills.append(skill)
    return skills


# Registry is static at import time, so compute once.
AGENT_SKILLS = _skills()


def _assistant_id(graph_key: str) -> str:
    return str(uuid5(NAMESPACE_GRAPH, graph_key))


def _default_assistant_id() -> str:
    """Assistant id of the (single) configured graph, for the query-less card."""
    keys = list(GRAPHS.keys())
    return _assistant_id(keys[0]) if keys else ""


def _base_url(request: Request, assistant_id: str) -> str:
    """Reconstruct the externally-visible base URL (mirrors LangGraph's logic).

    Strips the well-known / a2a suffixes so the same handler serves both the
    domain-root card and the per-assistant path card, and honours a reverse
    proxy via ``x-forwarded-proto`` plus any ``mount_prefix`` left in the path.
    """
    scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.url.hostname or "localhost"
    port = request.url.port
    path = (
        request.url.path.removesuffix("/.well-known/agent-card.json")
        .removesuffix("/.well-known/agent.json")
        .removesuffix(f"/a2a/{assistant_id}")
    )
    if port and (
        (scheme == "http" and port != 80) or (scheme == "https" and port != 443)
    ):
        return f"{scheme}://{host}:{port}{path}"
    return f"{scheme}://{host}{path}"


async def assistant_ids(request: Request) -> JSONResponse:
    """Return assistant_id(s) derived from graph key(s)."""
    graph = request.query_params.get("graph")
    keys = [graph] if graph else list(GRAPHS.keys())
    return JSONResponse(
        {
            "assistants": [
                {"graph_id": k, "assistant_id": _assistant_id(k)} for k in keys
            ]
        }
    )


async def agent_card(request: Request) -> JSONResponse:
    """Serve the curated A2A agent card (shadows LangGraph's default)."""
    assistant_id = (
        request.path_params.get("assistant_id")
        or request.query_params.get("assistant_id")
        or _default_assistant_id()
    )
    agent_url = f"{_base_url(request, assistant_id)}/a2a/{assistant_id}"
    card = {
        "name": AGENT_NAME,
        "description": AGENT_DESCRIPTION,
        "url": agent_url,
        "supportedInterfaces": [
            {
                "url": agent_url,
                "protocolBinding": "jsonrpc",
                "protocolVersion": "1.0",
            },
        ],
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "securitySchemes": SECURITY_SCHEMES,
        "security": SECURITY,
        "defaultInputModes": _DEFAULT_MODES,
        "defaultOutputModes": _DEFAULT_MODES,
        "skills": AGENT_SKILLS,
        "version": AGENT_VERSION,
    }
    return JSONResponse(card)


app = Starlette(
    routes=[
        Route("/internal/assistant-ids", assistant_ids, methods=["GET"]),
        # Shadow the built-in A2A cards (domain-root + per-assistant + legacy alias).
        Route("/.well-known/agent-card.json", agent_card, methods=["GET"]),
        Route(
            "/a2a/{assistant_id}/.well-known/agent-card.json",
            agent_card,
            methods=["GET"],
        ),
        Route(
            "/a2a/{assistant_id}/.well-known/agent.json",
            agent_card,
            methods=["GET"],
        ),
    ]
)
