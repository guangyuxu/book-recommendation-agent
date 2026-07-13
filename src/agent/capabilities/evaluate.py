"""Evaluate Book capability, built as a self-critique subgraph wired into execute.

Demo scope: LLM-only (no retrieval / DB / book-metadata lookup yet). The capability is a small
LangGraph subgraph plugged directly into the execute fan-out (agent.pipeline.execute) as the
"evaluate" node -- a sibling of the other capability nodes:

    START -> prepare -> evaluate -> validate
        --revise (reviewer found gaps, attempts < MAX)---> evaluate
        --accept (review passed, or attempt budget spent)-> emit -> END

`prepare` (no LLM) assembles the inputs: the book(s) the user named this turn. `evaluate` is the
LLM analyst that assesses the book's themes, values, reading difficulty, and content cautions for
the child (carrying reviewer feedback on a revise). `validate` is the post-LLM gate: an
independent LLM pass that checks the evaluation is specific, balanced, and tied to the child +
policies, listing concrete gaps if not. If gaps remain, `evaluate` runs again -- at most
MAX_ATTEMPTS (3) analyst passes. `emit` appends the accepted evaluation to the execute subgraph's
`results` fan-in channel under the "evaluate" key, the same contribution shape every capability
node returns.

Like the recommend/memory subgraph nodes, the LLM-invoking nodes wrap themselves in
with_turn_context so their token usage is billed to the turn, and the graph is compiled WITHOUT a
checkpointer (a nested subgraph inherits the parent's). A degraded LLM call falls back rather than
raising, preserving execute's "one broken capability never sinks the turn" invariant.

The registry declares this capability with `graph=evaluate_subgraph` (no `run`). Out-of-graph
callers (unit tests, evals) invoke `evaluate_subgraph` and read its `evaluation` output channel.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict, cast

from langchain.messages import AnyMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ..llm import HEAVY
from ..state import AppContext
from ..usage_tracker import with_turn_context
from ._shared import child_brief, mentioned_books, policies_brief

logger = logging.getLogger(__name__)

# the analyst may run at most this many times per turn (the self-critique revise budget).
MAX_ATTEMPTS = 3


class Critique(BaseModel):
    """The validator's verdict on one evaluation: does it pass review, and if not, why."""

    ok: bool
    issues: list[str] = Field(default_factory=list)


_analyst = HEAVY.chain()
_critic = HEAVY.structured(Critique)


class EvaluateState(TypedDict, total=False):
    """Execute-subgraph state for evaluate.

    The context channels (messages / target_child_id / children / policies) and billing ids
    (turn_id / thread_id) are shared BY NAME with the parent ExecuteState, so they map straight
    in. `results` is the parent's fan-in channel (same operator.add reducer): `emit` appends this
    capability's contribution and it maps straight back out to aggregate. `books` / `evaluation` /
    `feedback` / `attempts` are private working channels -- they exist only here, so they never
    leak to the parent or across turns.
    """

    messages: list[AnyMessage]
    target_child_id: str | None
    children: dict[str, dict[str, Any]]
    policies: list[dict[str, Any]]
    turn_id: str
    thread_id: str | None
    results: Annotated[list[dict[str, dict[str, Any]]], operator.add]
    books: list[dict[str, Any]]
    evaluation: str
    feedback: list[str]
    attempts: int


def _render_books(books: list[dict[str, Any]]) -> str:
    """Render the named book(s) as a compact block for the prompts."""
    if not books:
        return "(the user did not name a specific book)"
    lines: list[str] = []
    for b in books:
        author = f" by {b['author']}" if b.get("author") else ""
        lines.append(f"- {b.get('title') or '(untitled)'}{author}")
    return "\n".join(lines)


def _retry_directive(feedback: list[str]) -> str:
    """Fold the reviewer's gaps into a revise instruction for the next analyst pass."""
    if not feedback:
        return ""
    joined = "\n".join(f"- {issue}" for issue in feedback)
    return (
        "\n\nA reviewer found these gaps in your previous evaluation:\n"
        f"{joined}\n"
        "Revise your evaluation to address every gap."
    )


def prepare(state: EvaluateState) -> dict[str, Any]:
    """Assemble the inputs (no LLM): the book(s) the user named this turn, and reset the loop."""
    books = mentioned_books(cast("dict[str, Any]", state))
    return {"books": books, "feedback": [], "attempts": 0}


def evaluate(state: EvaluateState) -> dict[str, Any]:
    """LLM analyst: assess the named book's fit for the child (carrying any reviewer feedback)."""
    ctx = cast("dict[str, Any]", state)
    system = SystemMessage(
        content=(
            "You are a children's-book analyst. The user named a book. Evaluate whether it suits "
            "this child: its themes, values, reading difficulty, and any content to be aware of. "
            "Be concrete and balanced -- name both strengths and cautions, and tie the judgment "
            "to the child's reading level, interests, and the family's policies.\n\n"
            f"Book(s) under evaluation:\n{_render_books(state.get('books') or [])}\n\n"
            f"Target child profile:\n{child_brief(ctx)}\n\n"
            f"Family reading policies:\n{policies_brief(ctx)}"
            f"{_retry_directive(state.get('feedback') or [])}"
        )
    )
    try:
        reply = _analyst.invoke([system, *(state.get("messages") or [])])
        text = str(reply.content)
    except Exception as exc:  # degrade rather than sink the turn
        logger.warning("evaluate.evaluate failed: %s", type(exc).__name__)
        text = ""
    return {"evaluation": text, "attempts": state.get("attempts", 0) + 1}


def validate(state: EvaluateState) -> dict[str, Any]:
    """Post-LLM gate: review the evaluation and record concrete gaps if it is not up to standard.

    An empty `feedback` means the review passed; a non-empty `feedback` lists the gaps the analyst
    must address on the next revise pass.
    """
    text = state.get("evaluation") or ""
    if not text.strip():
        return {"feedback": ["The evaluation was empty."]}

    ctx = cast("dict[str, Any]", state)
    system = SystemMessage(
        content=(
            "You are a strict editor reviewing a children's-book evaluation BEFORE the parent "
            "sees it. Judge whether it: names the specific book; covers themes, values, and "
            "reading difficulty; flags any content to be aware of; stays balanced (not one-"
            "sided); and ties its judgment to this child's level/interests and the family's "
            "policies. If any of these is missing or unsupported it is NOT ok -- list the "
            "concrete gaps.\n\n"
            f"Book(s) under evaluation:\n{_render_books(state.get('books') or [])}\n\n"
            f"Target child profile:\n{child_brief(ctx)}\n\n"
            f"Family reading policies:\n{policies_brief(ctx)}\n\n"
            f"Evaluation to review:\n{text}"
        )
    )
    try:
        critique = cast(
            Critique, _critic.invoke([system, *(state.get("messages") or [])])
        )
    except Exception as exc:  # review unavailable -> accept the evaluation as-is
        logger.warning("evaluate.validate failed: %s", type(exc).__name__)
        return {"feedback": []}

    logger.info(
        "evaluate.validate: attempt=%d ok=%s issues=%d",
        state.get("attempts", 0),
        critique.ok,
        len(critique.issues),
    )
    if critique.ok:
        return {"feedback": []}
    return {"feedback": critique.issues or ["The evaluation needs revision."]}


def route_after_validate(state: EvaluateState) -> str:
    """Accept the evaluation if review passed or the budget is spent; else revise."""
    if not (state.get("feedback") or []):
        return "accept"  # review passed
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "accept"  # budget spent -- ship the best effort rather than loop forever
    return "revise"


def emit(state: EvaluateState) -> dict[str, Any]:
    """Append the accepted evaluation to execute's fan-in channel under the "evaluate" key."""
    return {"results": [{"evaluate": {"evaluation": state.get("evaluation") or ""}}]}


_builder = StateGraph(EvaluateState, context_schema=AppContext)
# prepare/emit make no LLM call; the analyst/critic nodes wrap themselves (like the recommend
# subgraph nodes) so the billing ContextVar is live when their token-usage callback fires.
_builder.add_node("prepare", prepare)
_builder.add_node("evaluate", with_turn_context(evaluate))
_builder.add_node("validate", with_turn_context(validate))
_builder.add_node("emit", emit)

_builder.add_edge(START, "prepare")
_builder.add_edge("prepare", "evaluate")
_builder.add_edge("evaluate", "validate")
_builder.add_conditional_edges(
    "validate", route_after_validate, {"revise": "evaluate", "accept": "emit"}
)
_builder.add_edge("emit", END)

evaluate_subgraph = _builder.compile()
