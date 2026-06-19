from langgraph.graph import END, START, StateGraph

from .nodes import llm_call, tool_node
from .state import MessagesState


def should_continue(state: MessagesState):
    """路由逻辑"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


# 构建图
builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")

# 编译导出
graph = builder.compile()
