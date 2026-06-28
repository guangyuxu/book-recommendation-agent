"""Routing: classify a single message (classify) or split it into subtasks (split_intents)."""

from langchain.messages import SystemMessage
from pydantic import BaseModel, Field

from .intents import Intent
from .llm import model


class IntentResult(BaseModel):
    """The single intent that best matches the user's latest message."""

    intent: Intent


class SubTask(BaseModel):
    """One independently handleable subtask extracted from a mixed-intent message."""

    intent: Intent = Field(description="Must not be multi_intent")
    query: str = Field(description="The subtask rewritten so it stands on its own")


class IntentSplit(BaseModel):
    """A mixed-intent message split into separate subtasks, each tagged with its own intent."""

    tasks: list[SubTask]


classifier = model.with_structured_output(IntentResult)
splitter = model.with_structured_output(IntentSplit)


def _intent_menu() -> str:
    return "\n".join(f"- {i.value}: {i.description}" for i in Intent)


def classify(state):
    system = SystemMessage(
        content=(
            "You are an intent classifier. Based on the user's latest message, "
            "pick the single best-matching intent below. Return multi_intent ONLY "
            "when the message contains two or more independent requests that each "
            "need separate handling. Context that merely describes the child or "
            "parent in support of a single request (for example, giving the "
            "child's age and tastes before asking what to read) is part of that "
            "one request, not a separate intent. Do not force a business intent "
            "onto a message that does not request one: if there is no clear "
            "actionable request, prefer clarify (or child/parent profile updates "
            "for bare descriptions).\n\n"
            f"{_intent_menu()}"
        )
    )
    result: IntentResult = classifier.invoke([system, *state["messages"]])
    return {"intent": result.intent.value}


def split_intents(messages) -> list[SubTask]:
    system = SystemMessage(
        content=(
            "You are an intent splitter. Split the user's message into "
            "self-contained subtasks, each tagged with its intent. "
            "Do not use multi_intent.\n\n"
            f"{_intent_menu()}"
        )
    )
    result: IntentSplit = splitter.invoke([system, *messages])
    return result.tasks
