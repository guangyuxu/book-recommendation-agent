"""Book analysis flow: fetch info -> analyze dimensions -> verdict tied to parent goals. All mock."""

from langchain.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from ..state import FlowState, target_children


class EvaluationState(FlowState):
    book_info: dict
    analysis: dict


def fetch_book_info(state: EvaluationState):
    return {"book_info": {"title": "Warriors", "age": "8-12"}}


def analyze_dimensions(state: EvaluationState):
    return {
        "analysis": {
            "binary_thinking": "some good-vs-evil opposition present",
            "reading_difficulty": "moderate",
        }
    }


def compose_verdict(state: EvaluationState):
    points = "; ".join(f"{k}: {v}" for k, v in state["analysis"].items())
    goals = state.get("parent_goals") or []
    goal_note = f" given goals ({'; '.join(goals)})" if goals else " (no parent goals yet)"
    targets = target_children(state)
    if targets:
        verdicts = [
            f"{c.get('name', c['id'])}: moderate fit{goal_note}." for c in targets
        ]
    else:
        verdicts = ["(no target child resolved)"]
    return {
        "messages": [
            AIMessage(
                content=f"(mock book analysis flow) analysis -- {points}.\n"
                + "\n".join(verdicts)
            )
        ]
    }


def _build():
    b = StateGraph(EvaluationState)
    b.add_node("fetch_book_info", fetch_book_info)
    b.add_node("analyze_dimensions", analyze_dimensions)
    b.add_node("compose_verdict", compose_verdict)
    b.add_edge(START, "fetch_book_info")
    b.add_edge("fetch_book_info", "analyze_dimensions")
    b.add_edge("analyze_dimensions", "compose_verdict")
    b.add_edge("compose_verdict", END)
    return b.compile()


graph = _build()
