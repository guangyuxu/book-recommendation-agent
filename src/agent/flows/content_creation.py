"""Content creation flow: pick topic -> make outline -> write draft. All mock."""

from langchain.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from ..state import FlowState


class ContentState(FlowState):
    topic: str
    outline: list[str]
    draft: str


def pick_topic(state: ContentState):
    return {"topic": "how to read out complex human nature with your child"}


def make_outline(state: ContentState):
    return {"outline": ["opening hook", "three points", "closing question"]}


def write_draft(state: ContentState):
    sections = "/".join(state["outline"])
    return {"draft": f"draft on <{state['topic']}> ({sections})"}


def compose_draft(state: ContentState):
    return {
        "messages": [AIMessage(content=f"(mock content creation flow) {state['draft']}")]
    }


def _build():
    b = StateGraph(ContentState)
    b.add_node("pick_topic", pick_topic)
    b.add_node("make_outline", make_outline)
    b.add_node("write_draft", write_draft)
    b.add_node("compose_draft", compose_draft)
    b.add_edge(START, "pick_topic")
    b.add_edge("pick_topic", "make_outline")
    b.add_edge("make_outline", "write_draft")
    b.add_edge("write_draft", "compose_draft")
    b.add_edge("compose_draft", END)
    return b.compile()


graph = _build()
