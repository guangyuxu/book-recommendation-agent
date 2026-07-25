"""Book recommendation agent: the compiled LangGraph graph is the package entry point."""

# Import the DB package before the graph so agent.db is fully initialized up front. The graph's
# import chain reaches it lazily (graph -> memory -> domain -> books -> `from ..db import ...`);
# importing it here first avoids a first-touch deep in that chain (which trips CPython's per-module
# import deadlock detector). load_context used to import agent.db early via FamilyRepository; now
# that it reads context over the accounts API instead, this makes the ordering explicit.
from . import db  # noqa: F401
from .graph import graph

__all__ = ["graph"]
