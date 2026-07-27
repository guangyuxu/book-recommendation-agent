"""Fast, offline unit tests. This tree MIRRORS `src/agent/`.

One directory per source subpackage (`pipeline/`, `memory/`, `capabilities/`, `domain/`), so a
test's location tells you what it covers and a source package with no mirror directory is a visible
coverage gap. Filenames stay descriptive rather than strictly 1:1 with module names -- one file may
cover a family of sibling modules (`capabilities/test_prose_capabilities.py` covers compare /
discussion / path / content plus the `_shared` engine they all call).

Tests for `src/agent/*.py` top-level modules sit at this root (`test_graph_structure.py`,
`test_guard.py`, `test_language.py`, `test_lifecycle.py`, `test_llm.py`, `test_prompts.py`,
`test_usage_tracker.py`).

Everything here is hermetic: no DB, no network, no Anthropic call. LLM OUTPUT QUALITY is measured in
`evals/` instead, which mirrors `src/agent` the same way; end-to-end journeys go in
`tests/integration_tests/`.
"""
