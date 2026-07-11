"""Generic LLM-as-judge: score one capability output against a rubric on 1-5 dimensions.

Shared by every `judge_run.py` node eval (e.g. `capabilities/recommend/judge_run.py`). A node
eval passes the rubric text, the rendered output, a compact context brief, and its dimension
names; this returns one integer per dimension plus a one-sentence justification each, via forced
structured output at temperature 0.

Judge model selection (a best practice, made configurable so it works out of the box):
- Default is `EVAL_JUDGE_MODEL` or the app's own model. This lets evals run with the existing key.
- For a real quality bar you SHOULD point `EVAL_JUDGE_MODEL` at a *stronger, different* model than
  the one that GENERATED the output -- judging with the generator invites self-enhancement bias.

Bias guards baked into the prompt: score each dimension independently (no halo averaging), judge
substance not length (verbosity guard), and follow the rubric's explicit 1/3/5 anchors so scores
are comparable across runs and reviewers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, create_model

# Reuse the app's model id as the default so evals run with the existing key; override with a
# stronger, separate judge model via EVAL_JUDGE_MODEL (see module docstring).
_JUDGE_MODEL_ID = os.getenv("EVAL_JUDGE_MODEL", "claude-sonnet-4-6")
_judge_model = init_chat_model(
    _JUDGE_MODEL_ID, temperature=0, max_retries=3, timeout=60, disable_streaming=True
)


def load_rubric(path: str | Path) -> str:
    """Load a rubric markdown file (e.g. a node's `judge_rubric.md`)."""
    return Path(path).read_text(encoding="utf-8")


def _score_model(dimensions: tuple[str, ...]) -> type[BaseModel]:
    """Build a pydantic model with an int score (1-5) + justification per dimension.

    Constructed dynamically so the judge's structured-output contract always matches exactly the
    dimensions a node declares -- add a dimension to a rubric and the judge is forced to fill it.
    """
    fields: dict[str, Any] = {}
    for dim in dimensions:
        fields[dim] = (
            int,
            Field(ge=1, le=5, description=f"Integer 1-5 for '{dim}' per the rubric."),
        )
        fields[f"{dim}_reason"] = (
            str,
            Field(description=f"One sentence justifying the '{dim}' score."),
        )
    return create_model("JudgeScore", **fields)


def judge(
    rubric: str,
    rendered_output: str,
    context: str,
    dimensions: tuple[str, ...],
) -> dict[str, Any]:
    """Score one output against `rubric` on `dimensions`.

    Returns a flat dict: `{dim: int, f"{dim}_reason": str, ...}`. Callers keep the int scores for
    aggregation (see `metrics.mean_by_dimension` / `pass_rate`) and log the reasons into the
    report for debuggability.
    """
    schema = _score_model(dimensions)
    structured = _judge_model.with_structured_output(schema)
    system = SystemMessage(
        content=(
            "You are a strict, fair evaluator of a children's-book assistant's output. Score ONLY "
            "against the rubric below. Rules: (1) score each dimension INDEPENDENTLY on its own "
            "1-5 scale -- do not let a strong dimension inflate a weak one; (2) anchor each score "
            "to the rubric's explicit 1/3/5 descriptions; (3) judge substance and fit, NOT length "
            "or eloquence; (4) give a one-sentence, concrete justification per dimension.\n\n"
            f"RUBRIC\n{rubric}"
        )
    )
    human = HumanMessage(
        content=(
            f"CONTEXT (the child / family this output was produced for):\n{context}\n\n"
            f"OUTPUT TO SCORE:\n{rendered_output}"
        )
    )
    result = structured.invoke([system, human])
    return result.model_dump()  # type: ignore[union-attr]
