from langchain.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b


tools = [add, multiply]
tools_by_name = {tool.name: tool for tool in tools}
