import operator
from typing import Annotated

from langchain.messages import AnyMessage
from typing_extensions import TypedDict


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
