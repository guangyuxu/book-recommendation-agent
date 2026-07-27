"""End-to-end journeys against real infrastructure. Opt-in: `make integration`.

EMPTY ON PURPOSE -- see ROADMAP: a full-pipeline turn against a real Postgres (and a real Anthropic
call) belongs here. Unlike `tests/unit_tests` (which mirrors `src/agent/`), this suite is organized
by FLOW -- one file per journey -- because a journey crosses many modules by definition.

It is deliberately kept out of the blocking `make ci` gate: it needs a real DB and hits the network.
`make integration` is the entrypoint (it treats an empty suite as a pass while this stays a
placeholder). Also gate anything that costs API tokens behind `RUN_INTEGRATION=1`, the way `evals/`
gates on `RUN_EVAL=1`, so a stray `pytest tests/` can never spend money.
"""
