from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langchain_core.messages import ToolMessage

from .state import MessagesState
from .tools import tools, tools_by_name

# 初始化模型
model = init_chat_model("claude-sonnet-4-6", temperature=0)
model_with_tools = model.bind_tools(tools)


def llm_call(state: MessagesState):
    """LLM 节点：决定调用工具还是直接回复"""
    response = model_with_tools.invoke(
        [SystemMessage(content="You are a helpful assistant.")] + state["messages"]
    )
    return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}


def tool_node(state: MessagesState):
    """工具节点：执行工具调用"""
    results = []
    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        tool_func = tools_by_name[tool_call["name"]]
        observation = tool_func.invoke(tool_call["args"])
        results.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )

    return {"messages": results}
