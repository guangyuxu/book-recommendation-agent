"""Multi-intent orchestration: one fixed graph for "any N updates + any M business tasks".

It does not grow with the combination. The trick is to leave all three concerns to graph
primitives instead of hardcoding combinations:
- which to run: plan writes them into `pending`; each node has a guard (no-op if not in pending) -> O(n+m) nodes;
- merging: handled by the channel reducers automatically, no merge node;
- ordering: all updates -> gate (fan-in barrier) -> all business, so business reads the merged profile.

    START -> plan -+-> child_profile_update  -+
                   +-> parent_profile_update -+-> gate -+-> book_recommendation  -+
                   +-> parent_goal_update    -+         +-> book_evaluation       -+
                                                        +-> reading_path_planning -+-> END
                                                        +-> content_creation      -+
                                                        +-> reading_discussion    -+
"""

from langchain.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from ..classify import split_intents
from ..intents import CHILD_SPECIFIC, Intent
from ..registry import run_handler
from ..resolve import resolve_for
from ..state import FlowState


class MultiState(FlowState):
    pending: list[str]  # intents to run this turn; written by plan, read by each guard
    child_targets: dict  # intent value -> [child_id]; per-subtask targets, written by plan


_UPDATE_INTENTS = [
    Intent.CHILD_PROFILE_UPDATE,
    Intent.PARENT_PROFILE_UPDATE,
    Intent.PARENT_GOAL_UPDATE,
]
_BUSINESS_INTENTS = [
    Intent.BOOK_RECOMMENDATION,
    Intent.BOOK_EVALUATION,
    Intent.READING_PATH_PLANNING,
    Intent.CONTENT_CREATION,
    Intent.READING_DISCUSSION,
]

# A business subgraph's .invoke() returns internal fields (level/candidates/...); write back only
# the whitelisted channels to avoid writing channels this graph didn't declare and to avoid
# parallel business nodes overwriting each other's profile.
_UPDATE_OUT = ("messages", "children", "parent_profile", "parent_goals")
_BUSINESS_OUT = ("messages",)


def plan(state: MultiState):
    if state.get("pending"):  # already injected by caller/test -> respect it, don't call the LLM
        return {}
    pending: list[str] = []
    child_targets: dict[str, list[str]] = {}
    children = state.get("children") or {}
    for task in split_intents(state["messages"]):
        if task.intent == Intent.MULTI_INTENT or task.intent.value in pending:
            continue
        pending.append(task.intent.value)
        if task.intent in CHILD_SPECIFIC:
            # Resolve this subtask's child(ren): full history for cross-turn references plus
            # the rewritten standalone query for which child this subtask is about.
            res = resolve_for(children, [*state["messages"], HumanMessage(content=task.query)])
            child_targets[task.intent.value] = res.child_ids
    return {"pending": pending, "child_targets": child_targets}


def _guard(intent: Intent, out_keys: tuple[str, ...]):
    """Wrap a handler into a run-on-demand node: no-op if not in pending, else write back only out_keys."""

    def node(state: MultiState, config=None):
        if intent.value not in (state.get("pending") or []):
            return {}
        local = dict(state)
        if intent in CHILD_SPECIFIC:
            # Per-subtask target injected via subgraph input (NOT the shared channel), so
            # parallel child-specific subtasks targeting different children don't race.
            local["target_child_ids"] = (state.get("child_targets") or {}).get(intent.value, [])
        result = run_handler(intent, local)
        return {k: result[k] for k in out_keys if k in result}

    node.__name__ = intent.value
    return node


def gate(state: MultiState):
    """Barrier: fan-in all updates, then fan out to business, separating writes from reads."""
    return {}


def _build():
    b = StateGraph(MultiState)
    b.add_node("plan", plan)
    b.add_node("gate", gate)
    b.add_edge(START, "plan")

    for intent in _UPDATE_INTENTS:
        b.add_node(intent.value, _guard(intent, _UPDATE_OUT))
        b.add_edge("plan", intent.value)
        b.add_edge(intent.value, "gate")

    for intent in _BUSINESS_INTENTS:
        b.add_node(intent.value, _guard(intent, _BUSINESS_OUT))
        b.add_edge("gate", intent.value)
        b.add_edge(intent.value, END)

    return b.compile()


graph = _build()
