"""Recommend Books capability, built as a self-critique subgraph wired into execute.

MVP is LLM-only -- no retrieval/ranking/vector search. The capability is a small LangGraph
subgraph plugged DIRECTLY into the execute fan-out (agent.pipeline.execute) as the "recommend"
node -- a sibling of the other capability nodes, not a function hidden behind the generic runner:

    START -> generate -> validate
        --retry (screening rejected every book, attempts < MAX)----> generate
        --keep  (fit books remain, or the attempt budget is spent)-> emit -> END

`generate` proposes a fresh ranked booklist from scratch (carrying prior rejection feedback on a
retry). `validate` is the post-LLM gate: an independent LLM pass that judges each proposed book
against the child + policies and DROPS any that do not fit, recording why. If every book is
rejected, `generate` runs again with those reasons -- at most MAX_ATTEMPTS (3) generate passes.
`emit` appends the surviving booklist to the execute subgraph's `results` fan-in channel under
the "recommend" key -- exactly the contribution shape every other capability node returns, so
aggregate merges it with no special-casing.

Like the execute/memory subgraph nodes, the LLM-invoking nodes wrap themselves in
with_turn_context so their token usage is billed to the turn, and the graph is compiled WITHOUT a
checkpointer (a nested subgraph inherits the parent's). A generate/validate LLM failure degrades
to an empty/unscreened list rather than raising, preserving execute's "one broken capability
never sinks the turn" invariant now that recommend runs as a subgraph rather than under
_capability_node's try/except.

The registry declares this capability with `graph=recommend_graph` (no `run`), so execute wires
the subgraph in directly. Out-of-graph callers (the eval harness, unit tests) invoke
`recommend_graph` and read its `books`/`note` output channels -- the booklist the parent sees.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict, cast

from langchain.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .. import prompts
from ..llm import HEAVY
from ..state import AppContext
from ..usage_tracker import with_turn_context
from ._shared import child_brief, policies_brief

logger = logging.getLogger(__name__)

# generate may run at most this many times per turn (the self-critique retry budget).
MAX_ATTEMPTS = 3


class RecommendedBook(BaseModel):
    """One recommended book with the reasoning the parent will see."""

    title: str
    author: str | None = None
    recommendation_reason: str
    fit_summary: str | None = None
    risk_notes: list[str] = Field(default_factory=list)


class Booklist(BaseModel):
    """The recommendation result: a ranked list of books plus an optional framing note."""

    books: list[RecommendedBook] = Field(default_factory=list)
    note: str | None = None


class BookVerdict(BaseModel):
    """One screening verdict: whether a proposed book stays on the list, and why."""

    title: str
    keep: bool
    reason: str


class Screening(BaseModel):
    """The validator's per-book keep/reject verdicts for the proposed booklist."""

    verdicts: list[BookVerdict] = Field(default_factory=list)


_generate = HEAVY.structured(Booklist)
_screen = HEAVY.structured(Screening)


class RecommendState(TypedDict, total=False):
    """Execute-subgraph state for recommend.

    The context channels (messages / target_child_id / children / policies) and the billing ids
    (turn_id / thread_id) are shared BY NAME with the parent ExecuteState, so they map straight
    in. `results` is the parent's fan-in channel (same operator.add reducer): `emit` appends this
    capability's contribution and it maps straight back out to aggregate. `books` / `note` /
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
    note: str | None
    feedback: list[str]
    attempts: int


class RecommendOutput(TypedDict, total=False):
    """What recommend writes back: `results` (execute's fan-in) plus the standalone booklist.

    An explicit output schema stops the subgraph from re-emitting its mapped input channels
    (target_child_id / children / policies), which would collide at execute's fan-in with a
    sibling capability subgraph that also re-emits them. `books`/`note` stay in the output so
    out-of-graph callers (evals, unit tests) that invoke recommend_graph directly still read them.
    """

    results: Annotated[list[dict[str, dict[str, Any]]], operator.add]
    books: list[dict[str, Any]]
    note: str | None


def generate(state: RecommendState) -> dict[str, Any]:
    """Propose a fresh ranked booklist (incorporating any prior rejection feedback)."""
    ctx = cast("dict[str, Any]", state)
    system = prompts.render(
        "recommend.generate",
        child_brief=child_brief(ctx),
        policies_brief=policies_brief(ctx),
        feedback=state.get("feedback") or [],
    )
    try:
        result = cast(
            Booklist,
            _generate.invoke(
                [*system, *(state.get("messages") or [])],
                config=prompts.config("recommend.generate"),
            ),
        )
        dump = result.model_dump()
    except Exception as exc:  # degrade rather than sink the turn
        logger.warning("recommend.generate failed: %s", type(exc).__name__)
        dump = {"books": [], "note": None}
    return {
        "books": dump.get("books") or [],
        "note": dump.get("note"),
        "attempts": state.get("attempts", 0) + 1,
    }


def _render_candidates(books: list[dict[str, Any]]) -> str:
    """Render the proposed books as a compact numbered list for the screening prompt."""
    lines: list[str] = []
    for i, b in enumerate(books, start=1):
        author = f" by {b['author']}" if b.get("author") else ""
        title = b.get("title") or "(untitled)"
        lines.append(f"{i}. {title}{author} -- {b.get('recommendation_reason', '')}")
    return "\n".join(lines)


def validate(state: RecommendState) -> dict[str, Any]:
    """Screen each proposed book against the child + policies; drop the ones that do not fit.

    This is the post-LLM gate: a book the generator proposed only survives if screening keeps it.
    A verdict with no matching proposed book (LLM omission or title drift) defaults to keep, so a
    screening miss never silently drops an otherwise-fit book.
    """
    books = state.get("books") or []
    if not books:
        return {"feedback": ["The generator returned no books."]}

    ctx = cast("dict[str, Any]", state)
    system = prompts.render(
        "recommend.validate",
        child_brief=child_brief(ctx),
        policies_brief=policies_brief(ctx),
        candidates=_render_candidates(books),
    )
    try:
        screening = cast(
            Screening,
            _screen.invoke(
                [*system, *(state.get("messages") or [])],
                config=prompts.config("recommend.validate"),
            ),
        )
        verdicts = screening.verdicts
    except Exception as exc:  # screening unavailable -> keep the proposed list as-is
        logger.warning("recommend.validate failed: %s", type(exc).__name__)
        return {"books": books, "feedback": []}

    verdict_by_title = {v.title.strip().lower(): v for v in verdicts}
    kept: list[dict[str, Any]] = []
    rejected: list[str] = []
    for b in books:
        verdict = verdict_by_title.get((b.get("title") or "").strip().lower())
        if verdict is None or verdict.keep:
            kept.append(b)
        else:
            rejected.append(f"{b.get('title') or '(untitled)'}: {verdict.reason}")

    # Counts only -- reasons can quote profile/policy detail, so they stay out of the log.
    logger.info(
        "recommend.validate: attempt=%d proposed=%d kept=%d rejected=%d",
        state.get("attempts", 0),
        len(books),
        len(kept),
        len(rejected),
    )
    return {"books": kept, "feedback": rejected}


def route_after_validate(state: RecommendState) -> str:
    """Keep the screened list if any book survived or the budget is spent; else regenerate."""
    if state.get("books"):
        return "keep"  # at least one fit book survived screening
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "keep"  # budget spent -- return the result rather than loop forever
    return "retry"


def emit(state: RecommendState) -> dict[str, Any]:
    """Append the surviving booklist to execute's fan-in channel under the "recommend" key."""
    booklist = {"books": state.get("books") or [], "note": state.get("note")}
    return {"results": [{"recommend": booklist}]}


_builder = StateGraph(
    RecommendState, context_schema=AppContext, output_schema=RecommendOutput
)
# The LLM-invoking nodes wrap themselves (like the memory/execute subgraph nodes) so the billing
# ContextVar is live when their token-usage callback fires; emit makes no LLM call.
_builder.add_node("generate", with_turn_context(generate))
_builder.add_node("validate", with_turn_context(validate))
_builder.add_node("emit", emit)

_builder.add_edge(START, "generate")
_builder.add_edge("generate", "validate")
_builder.add_conditional_edges(
    "validate", route_after_validate, {"retry": "generate", "keep": "emit"}
)
_builder.add_edge("emit", END)

recommend_graph = _builder.compile()
