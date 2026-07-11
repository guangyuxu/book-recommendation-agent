"""Capability Execution: run the planned capabilities in dependency order.

Capabilities run in-node (not as subgraphs), sequentially, with earlier results visible to
later steps via capability_results. LLM-only in MVP -- no retrieval/ranking.
"""

from __future__ import annotations

import logging

from ..capabilities import REGISTRY
from ..state import FlowState

logger = logging.getLogger(__name__)


def execute(state: FlowState) -> dict:
    """Execute plan.steps, honoring depends_on, and collect each capability's result.

    Capabilities are independent: a failing one is logged and skipped (it simply produces no
    result), and the rest of the plan still runs -- so one broken capability never sinks the turn.
    """
    steps = list((state.get("plan") or {}).get("steps") or [])
    if not steps:
        # Always overwrite the channel (last-write-wins) so a prior turn's results don't linger.
        return {"capability_results": {}}

    results: dict[str, dict] = {}
    done: set[str] = set()
    progressed = True
    while steps and progressed:
        progressed = False
        for step in list(steps):
            if not all(dep in done for dep in step.get("depends_on") or []):
                continue
            # Step is attempted now regardless of outcome, so drop it from the queue.
            steps.remove(step)
            progressed = True
            cap = REGISTRY.get(step["capability"])
            if cap is None:
                logger.warning("execute: unknown capability %s; skipping.", step["capability"])
                continue
            # Expose only THIS turn's results so far (never a prior turn's stale state) so
            # the capability reads exactly what its upstream steps produced this turn.
            view = {
                **state,
                "capability_results": dict(results),
                "_current_step": step,
            }
            try:
                results[cap.name] = cap.run(view)
            except Exception as exc:  # one capability failing must not sink the turn
                logger.warning("execute: capability %s failed: %s", cap.name, exc)
                continue
            # Mark satisfied ONLY on success, so dependents of a failed/unknown producer do not
            # run as if their upstream input exists -- they are skipped instead.
            done.add(cap.name)

    return {"capability_results": results}
