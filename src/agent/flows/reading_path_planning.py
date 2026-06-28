"""Stage planning flow: per target child, assess level -> plan stages -> pick books -> compose path. All mock."""

from langchain.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from ..state import FlowState, target_children


class PathState(FlowState):
    levels: dict  # child_id -> level
    stages: list[str]
    picks: dict  # stage -> pick


def assess_level(state: PathState):
    return {
        "levels": {
            c["id"]: c.get("reading_level", "intermediate") for c in target_children(state)
        }
    }


def plan_stages(state: PathState):
    return {"stages": ["transition", "advancing", "target"]}


def pick_books(state: PathState):
    return {"picks": {stage: f"{stage} pick" for stage in state["stages"]}}


def compose_path(state: PathState):
    path = " -> ".join(f"{s} ({state['picks'][s]})" for s in state["stages"])
    names = {c["id"]: c.get("name", f"child {c['id']}") for c in target_children(state)}
    if not state["levels"]:
        return {"messages": [AIMessage(content="(mock stage planning flow) no target child resolved.")]}
    lines = [
        f"{names.get(cid, cid)} (current level {level}): {path}"
        for cid, level in state["levels"].items()
    ]
    return {"messages": [AIMessage(content="(mock stage planning flow)\n" + "\n".join(lines))]}


def _build():
    b = StateGraph(PathState)
    b.add_node("assess_level", assess_level)
    b.add_node("plan_stages", plan_stages)
    b.add_node("pick_books", pick_books)
    b.add_node("compose_path", compose_path)
    b.add_edge(START, "assess_level")
    b.add_edge("assess_level", "plan_stages")
    b.add_edge("plan_stages", "pick_books")
    b.add_edge("pick_books", "compose_path")
    b.add_edge("compose_path", END)
    return b.compile()


graph = _build()
