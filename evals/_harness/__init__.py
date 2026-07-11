"""Shared eval infrastructure used by all three strategies.

Kept deliberately small and dependency-light: dataset loading (`cases`), scoring (`metrics`),
result output (`report`), and threshold gating (`thresholds`). Nothing here imports the agent's
DB layer, so strategies that only touch the LLM (e.g. S1) run without Postgres.
"""
