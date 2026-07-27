"""Structural + behavioral tests for the parallel-branch graph.

These run in CI (no DB, no Anthropic): they assert the wiring of the main graph and the
memory subgraph, the clarify fan-out router, execute's per-capability resilience, and --
deterministically, on a synthetic graph mirroring our topology -- the key invariant that a
branch running in parallel with an interrupting branch executes exactly once across a
resume (so `execute` is never re-run when the confirmation gate pauses the turn).
"""

from __future__ import annotations

import importlib
import operator
from types import SimpleNamespace
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from langgraph.types import Command, interrupt

from agent.graph import graph
from agent.memory import memory_graph
from agent.pipeline import route_after_clarify

# import_module returns the real submodule (the package re-exports `execute` the function under
# the same name, which would otherwise shadow it), so monkeypatching its REGISTRY works.
execute_mod = importlib.import_module("agent.pipeline.execute")


def _edges(compiled) -> set[tuple[str, str]]:
    return {(e.source, e.target) for e in compiled.get_graph().edges}


def _nodes(compiled) -> set[str]:
    return set(compiled.get_graph().nodes)


def test_graph_is_a_compiled_pregel() -> None:
    """The module-level export LangGraph Server serves must be a compiled graph, not a builder."""
    assert isinstance(graph, Pregel)


# --- main graph shape: fan-out + join ---------------------------------------------------


def test_clarify_fans_out_to_both_branches() -> None:
    edges = _edges(graph)
    assert ("clarify", "execute") in edges
    assert ("clarify", "memory") in edges


def test_both_branches_join_at_respond() -> None:
    edges = _edges(graph)
    # Both parallel branches join directly at respond, which composes the single reply.
    assert ("execute", "respond") in edges
    assert ("memory", "respond") in edges
    assert ("respond", "__end__") in edges
    assert "validate" not in _nodes(graph)


def test_old_serial_confirm_nodes_are_gone_from_main_graph() -> None:
    # The confirm gate + profile_update moved into the memory subgraph; the main graph no longer
    # holds them (only the "memory" subgraph node), and the old serial execute -> memory edge is
    # gone -- the two branches now run in parallel off clarify.
    nodes = _nodes(graph)
    for gone in (
        "prepare_confirmation",
        "request_confirmation",
        "apply_confirmation",
        "profile_update",
    ):
        assert gone not in nodes
    assert ("execute", "memory") not in _edges(graph)


# --- memory subgraph shape --------------------------------------------------------------


def test_memory_subgraph_shape() -> None:
    edges = _edges(memory_graph)
    assert ("__start__", "memory_policy") in edges
    assert ("memory_policy", "prepare_confirmation") in edges
    assert ("prepare_confirmation", "request_confirmation") in edges  # confirm branch
    assert ("prepare_confirmation", "profile_update") in edges  # skip branch
    assert ("request_confirmation", "apply_confirmation") in edges
    assert ("apply_confirmation", "profile_update") in edges
    assert ("profile_update", "__end__") in edges


# --- clarify router ---------------------------------------------------------------------


def test_route_after_clarify_fans_out_on_proceed() -> None:
    assert route_after_clarify({"clarification": {"decision": "continue"}}) == [
        "execute",
        "memory",
    ]
    assert route_after_clarify({"clarification": {"decision": "best_effort"}}) == [
        "execute",
        "memory",
    ]


def test_route_after_clarify_ends_on_ask_user() -> None:
    assert (
        route_after_clarify({"clarification": {"decision": "ask_user"}}) == "ask_user"
    )


# --- execute subgraph: shape, parallel fan-out, aggregation, resilience ------------------


def _run_execute(state: dict) -> dict:
    """Invoke the compiled execute subgraph on its own (no checkpointer for a single turn)."""
    return execute_mod.execute_graph.invoke(state)


def test_execute_subgraph_shape() -> None:
    edges = _edges(execute_mod.execute_graph)
    nodes = _nodes(execute_mod.execute_graph)
    # One node per capability, plus the dispatch anchor and the aggregate fan-in.
    for cap in ("recommend", "evaluate", "compare", "discussion", "path", "content"):
        assert cap in nodes
        assert ("dispatch", cap) in edges  # fan-out branch (conditional)
        assert (cap, "aggregate") in edges  # fan-in
    assert ("__start__", "dispatch") in edges
    assert ("dispatch", "aggregate") in edges  # empty-plan short-circuit
    assert ("aggregate", "__end__") in edges


def test_execute_skips_failing_capability_and_keeps_the_rest(monkeypatch) -> None:
    # `recommend` and `evaluate` run as their own subgraph nodes (not _capability_node), so this
    # generic resilience test uses two run-backed capabilities instead.
    calls: list[str] = []

    def good_run(view: dict) -> dict:
        calls.append("content")
        return {"draft": "a lovely post"}

    def bad_run(view: dict) -> dict:
        raise RuntimeError("boom")

    fake_registry = {
        "content": SimpleNamespace(name="content", run=good_run),
        "compare": SimpleNamespace(name="compare", run=bad_run),
    }
    monkeypatch.setattr(execute_mod, "REGISTRY", fake_registry)

    out = _run_execute(
        {"plan": {"steps": [{"capability": "content"}, {"capability": "compare"}]}}
    )

    assert out["capability_results"]["content"]["draft"] == "a lovely post"
    assert "compare" not in out["capability_results"]  # failed step produced no result
    assert calls == ["content"]


def test_execute_runs_independent_capabilities_in_parallel(monkeypatch) -> None:
    # Two independent capabilities both run and both land in capability_results -- no ordering,
    # no dependency (the decoupling this refactor introduced).
    def compare_run(view: dict) -> dict:
        return {"comparison": "book A edges out book B"}

    def content_run(view: dict) -> dict:
        return {"draft": "a lovely post"}

    monkeypatch.setattr(
        execute_mod,
        "REGISTRY",
        {
            "compare": SimpleNamespace(name="compare", run=compare_run),
            "content": SimpleNamespace(name="content", run=content_run),
        },
    )
    out = _run_execute(
        {"plan": {"steps": [{"capability": "compare"}, {"capability": "content"}]}}
    )
    assert set(out["capability_results"]) == {"compare", "content"}
    assert out["capability_results"]["content"]["draft"] == "a lovely post"


def test_execute_always_returns_capability_results_channel() -> None:
    # No steps -> aggregate still writes the channel as {} so a prior turn's results never linger.
    out = _run_execute({"plan": {"steps": []}})
    assert out["capability_results"] == {}


def test_execute_scratch_does_not_leak_across_turns(monkeypatch) -> None:
    # Run the subgraph inside a minimal parent (as it runs in the real graph) twice on one thread.
    # The private `results` scratch must reset each turn: turn 2's capability_results must hold
    # ONLY turn 2's capability, not turn 1's leaked in.
    monkeypatch.setattr(
        execute_mod,
        "REGISTRY",
        {
            "compare": SimpleNamespace(
                name="compare", run=lambda v: {"comparison": ""}
            ),
            "content": SimpleNamespace(name="content", run=lambda v: {"draft": "x"}),
        },
    )

    class P(TypedDict, total=False):
        plan: dict
        capability_results: dict

    pb = StateGraph(P)
    pb.add_node("execute", execute_mod.execute_graph)
    pb.add_edge(START, "execute")
    pb.add_edge("execute", END)
    parent = pb.compile(checkpointer=MemorySaver())

    cfg = {"configurable": {"thread_id": "t-exec"}}
    first = parent.invoke({"plan": {"steps": [{"capability": "compare"}]}}, cfg)
    second = parent.invoke({"plan": {"steps": [{"capability": "content"}]}}, cfg)

    assert set(first["capability_results"]) == {"compare"}
    assert set(second["capability_results"]) == {"content"}  # turn 1 did not leak in


def test_execute_does_not_clobber_pass_through_channels(monkeypatch) -> None:
    """execute must write back ONLY capability_results, never its read-only input channels.

    Regression for InvalidUpdateError at 'target_child_id': execute maps target_child_id/children/
    policies/plan in for the capabilities to read, but must not re-emit them -- otherwise, at the
    main graph's fan-in, they collide with the parallel memory branch writing target_child_id
    (profile_update), since those are single-value (LastValue) channels. Mirrors the real
    execute ∥ memory topology with a sibling node that writes target_child_id.
    """
    monkeypatch.setattr(
        execute_mod,
        "REGISTRY",
        {"content": SimpleNamespace(name="content", run=lambda v: {"draft": "x"})},
    )

    class P(TypedDict, total=False):
        plan: dict
        target_child_id: str | None
        capability_results: dict

    pb = StateGraph(P)
    pb.add_node("execute", execute_mod.execute_graph)
    pb.add_node("sibling", lambda s: {"target_child_id": "new-child"})  # like memory
    pb.add_node("join", lambda s: {})
    pb.add_edge(START, "execute")
    pb.add_edge(START, "sibling")
    pb.add_edge("execute", "join")
    pb.add_edge("sibling", "join")
    pb.add_edge("join", END)
    parent = pb.compile()

    # Must not raise InvalidUpdateError; the sibling's target_child_id write wins uncontested.
    out = parent.invoke(
        {"plan": {"steps": [{"capability": "content"}]}, "target_child_id": "old-child"}
    )
    assert out["target_child_id"] == "new-child"
    assert set(out["capability_results"]) == {"content"}


# --- planner: intents map to a flat, independent capability list -------------------------


def test_plan_produces_flat_independent_steps() -> None:
    from agent.pipeline import plan

    out = plan(
        {"understanding": {"intents": ["book_recommendation", "book_evaluation"]}}
    )
    steps = out["plan"]["steps"]
    assert [s["capability"] for s in steps] == ["recommend", "evaluate"]
    # Decoupled: steps carry no inter-capability dependency (the field no longer exists).
    assert all("depends_on" not in s for s in steps)


# --- core invariant: interrupt in a parallel branch does not re-run the sibling ----------


def test_parallel_sibling_runs_once_across_interrupt_resume() -> None:
    """Mirror our topology: START fans out to a counting branch and an interrupting branch,
    which join. Prove the counting branch runs exactly once even though the other branch
    pauses on interrupt() and the graph is resumed -- i.e. execute is never re-run when the
    confirmation gate pauses the turn."""

    class S(TypedDict, total=False):
        runs: Annotated[list[str], operator.add]
        decision: dict

    def execute_branch(state: S) -> dict:
        return {"runs": ["execute"]}

    def prepare(state: S) -> dict:
        return {}

    def gate(state: S) -> dict:
        return {"decision": interrupt("confirm?")}

    def apply(state: S) -> dict:
        return {}

    def join(state: S) -> dict:
        return {"runs": ["respond"]}

    b = StateGraph(S)
    b.add_node("execute", execute_branch)
    b.add_node("prepare", prepare)
    b.add_node("gate", gate)
    b.add_node("apply", apply)
    b.add_node("respond", join)
    b.add_edge(START, "execute")
    b.add_edge(START, "prepare")
    b.add_edge("prepare", "gate")
    b.add_edge("gate", "apply")
    b.add_edge("execute", "respond")
    b.add_edge("apply", "respond")
    b.add_edge("respond", END)
    compiled = b.compile(checkpointer=MemorySaver())

    cfg = {"configurable": {"thread_id": "t1"}}
    first = compiled.invoke({}, cfg)
    assert "__interrupt__" in first  # paused on the gate

    final = compiled.invoke(Command(resume={"approved": True}), cfg)
    assert final["runs"].count("execute") == 1  # NOT re-run on resume
    assert "respond" in final["runs"]  # join ran after both branches
