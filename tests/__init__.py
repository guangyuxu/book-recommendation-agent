"""Test suite. Two suites, one layout law -- identical in the sibling repos (accounts, service).

    unit_tests/         fast + offline; the tree MIRRORS `src/agent/` (one dir per subpackage).
                        Run by the blocking gate: `make test` / `make coverage` / `make ci`.
    integration_tests/  end-to-end journeys vs real infrastructure; organized by FLOW.
                        Opt-in: `make integration` (kept out of `make ci`).

Each suite's `__init__.py` states its own rules. LLM OUTPUT QUALITY is not tested here at all -- it
lives in `evals/`, which mirrors `src/agent` the same way (see `evals/README.md`).
"""
