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
from langgraph.types import Command, interrupt

from agent.graph import graph
from agent.memory import memory_graph
from agent.pipeline import route_after_clarify
from agent.validation import validation_graph
from agent.validation.checks import CHECK_REGISTRY, node_name

# import_module returns the real submodule (the package re-exports `execute` the function under
# the same name, which would otherwise shadow it), so monkeypatching its REGISTRY works.
execute_mod = importlib.import_module("agent.pipeline.execute")


def _edges(compiled) -> set[tuple[str, str]]:
    return {(e.source, e.target) for e in compiled.get_graph().edges}


def _nodes(compiled) -> set[str]:
    return set(compiled.get_graph().nodes)


# --- main graph shape: fan-out + join ---------------------------------------------------


def test_clarify_fans_out_to_both_branches() -> None:
    edges = _edges(graph)
    assert ("clarify", "execute") in edges
    assert ("clarify", "memory") in edges


def test_both_branches_join_at_respond() -> None:
    edges = _edges(graph)
    # The answer branch now gates the output: execute -> validate -> respond (not execute ->
    # respond directly). The memory branch still joins respond in parallel.
    assert ("execute", "validate") in edges
    assert ("validate", "respond") in edges
    assert ("memory", "respond") in edges
    assert ("respond", "__end__") in edges
    assert ("execute", "respond") not in edges  # replaced by the validate hop
    assert "validate" in _nodes(graph)


# --- validation subgraph shape ----------------------------------------------------------


def test_validation_subgraph_fans_out_to_checks_and_aggregates() -> None:
    edges = _edges(validation_graph)
    nodes = _nodes(validation_graph)
    assert ("__start__", "select") in edges
    assert ("aggregate", "__end__") in edges
    # Every registered check is its own node, fanned out from select and joined at aggregate.
    for check in CHECK_REGISTRY.values():
        name = node_name(check)
        assert name in nodes
        assert ("select", name) in edges
        assert (name, "aggregate") in edges


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


# --- execute resilience (independent-branch fault tolerance) -----------------------------


def test_execute_skips_failing_capability_and_keeps_the_rest(monkeypatch) -> None:
    calls: list[str] = []

    def good_run(view: dict) -> dict:
        calls.append("recommend")
        return {"books": [{"title": "Where the Wild Things Are"}]}

    def bad_run(view: dict) -> dict:
        raise RuntimeError("boom")

    fake_registry = {
        "recommend": SimpleNamespace(name="recommend", run=good_run),
        "evaluate": SimpleNamespace(name="evaluate", run=bad_run),
    }
    monkeypatch.setattr(execute_mod, "REGISTRY", fake_registry)

    state = {
        "plan": {
            "steps": [
                {"capability": "recommend", "depends_on": []},
                {"capability": "evaluate", "depends_on": []},
            ]
        }
    }
    out = execute_mod.execute(state)  # must not raise

    assert out["capability_results"]["recommend"]["books"][0]["title"].startswith(
        "Where"
    )
    assert "evaluate" not in out["capability_results"]  # failed step produced no result
    assert calls == ["recommend"]


def test_execute_skips_dependent_when_its_producer_fails(monkeypatch) -> None:
    # A capability whose upstream producer failed must NOT run as if the input existed -- the
    # failed producer is not marked "done", so its dependent is skipped.
    calls: list[str] = []

    def bad_run(view: dict) -> dict:
        calls.append("recommend")
        raise RuntimeError("boom")

    def dependent_run(view: dict) -> dict:
        calls.append("discuss")
        return {"discussion": "questions"}

    fake_registry = {
        "recommend": SimpleNamespace(name="recommend", run=bad_run),
        "discuss": SimpleNamespace(name="discuss", run=dependent_run),
    }
    monkeypatch.setattr(execute_mod, "REGISTRY", fake_registry)

    state = {
        "plan": {
            "steps": [
                {"capability": "recommend", "depends_on": []},
                {"capability": "discuss", "depends_on": ["recommend"]},
            ]
        }
    }
    out = execute_mod.execute(state)  # must not raise, must not run the dependent

    assert calls == ["recommend"]  # discuss was skipped
    assert out["capability_results"] == {}


def test_execute_always_returns_capability_results_channel() -> None:
    # No steps -> still overwrite the channel to {} so a prior turn's results never linger.
    assert execute_mod.execute({"plan": {"steps": []}}) == {"capability_results": {}}


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
