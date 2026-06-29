"""Capability Execution: run the planned capabilities in dependency order.

Capabilities run in-node (not as subgraphs), sequentially, with earlier results visible to
later steps via capability_results. LLM-only in MVP -- no retrieval/ranking.
"""

from __future__ import annotations

from ..capabilities import REGISTRY


def execute(state: dict) -> dict:
    """Execute plan.steps, honoring depends_on, and collect each capability's result."""
    steps = list((state.get("plan") or {}).get("steps") or [])
    if not steps:
        return {}

    results: dict[str, dict] = {}
    done: set[str] = set()
    progressed = True
    while steps and progressed:
        progressed = False
        for step in list(steps):
            if not all(dep in done for dep in step.get("depends_on") or []):
                continue
            cap = REGISTRY.get(step["capability"])
            if cap is not None:
                # Expose results so far so a dependent capability can read them.
                view = {**state, "capability_results": {**(state.get("capability_results") or {}), **results}}
                results[cap.name] = cap.run(view)
            done.add(step["capability"])
            steps.remove(step)
            progressed = True

    return {"capability_results": results} if results else {}
