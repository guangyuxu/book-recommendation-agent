"""Recommendation flow: per target child, read profile -> retrieve candidates -> rank -> compose. All mock."""

from langchain.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from ..state import FlowState, target_children


class RecommendationState(FlowState):
    levels: dict  # child_id -> reading level
    candidates: dict  # child_id -> candidate titles
    ranked: dict  # child_id -> ranked titles


def load_profiles(state: RecommendationState):
    return {
        "levels": {
            c["id"]: c.get("reading_level", "unknown") for c in target_children(state)
        }
    }


def retrieve_candidates(state: RecommendationState):
    # Mock: same candidate pool per child; real retrieval would key off each child's level.
    return {
        "candidates": {
            cid: ["Wings of Fire", "Percy Jackson", "Redwall"] for cid in state["levels"]
        }
    }


def rank(state: RecommendationState):
    return {"ranked": {cid: list(c) for cid, c in state["candidates"].items()}}


def compose_booklist(state: RecommendationState):
    names = {c["id"]: c.get("name", f"child {c['id']}") for c in target_children(state)}
    if not state["ranked"]:
        return {"messages": [AIMessage(content="(mock recommendation flow) no target child resolved.")]}
    lines = [
        f"{names.get(cid, cid)} (level {state['levels'][cid]}): {', '.join(books)}"
        for cid, books in state["ranked"].items()
    ]
    return {"messages": [AIMessage(content="(mock recommendation flow)\n" + "\n".join(lines))]}


def _build():
    b = StateGraph(RecommendationState)  # type: ignore[arg-type]
    b.add_node("load_profiles", load_profiles)  # type: ignore[arg-type]
    b.add_node("retrieve_candidates", retrieve_candidates)  # type: ignore[arg-type]
    b.add_node("rank", rank)  # type: ignore[arg-type]
    b.add_node("compose_booklist", compose_booklist)  # type: ignore[arg-type]
    b.add_edge(START, "load_profiles")
    b.add_edge("load_profiles", "retrieve_candidates")
    b.add_edge("retrieve_candidates", "rank")
    b.add_edge("rank", "compose_booklist")
    b.add_edge("compose_booklist", END)
    return b.compile()


graph = _build()
