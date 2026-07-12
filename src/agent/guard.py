"""Input safety gate: screen the user's message for prompt-injection / jailbreak attacks.

This is the graph's ENTRY node -- it runs before `load_context`, so a malicious turn never
touches the database or an Anthropic model. It calls Meta's Llama Prompt Guard 2 (86M) via
Groq, a purpose-built classifier that returns the probability that the input is a prompt
attack ("ignore your instructions", "reveal your system prompt", role-play jailbreaks, injected
instructions, ...). Above a threshold we short-circuit to END with a canned refusal.

This is ONE layer of defense in depth, not the authorization boundary: business-semantic
abuse ("bypass the age limit") is not an injection and is caught downstream by the roster /
age / post-LLM gating, per CLAUDE.md. Accordingly, when the check itself cannot run
(missing key, Groq rate-limit/outage, unparsable output) we FAIL OPEN -- allow the turn and
log a warning -- rather than block every request behind a third-party dependency.

PII: only the single latest user message is sent to Groq -- never child/family profile data.
Nothing here logs the message text; we log only the derived score and the block decision.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from langchain.messages import AIMessage, AnyMessage

from .state import FlowState

logger = logging.getLogger(__name__)

# TODO(backend-edge): This node is ONLY the semantic prompt-injection check. The cheap,
# fail-fast STRUCTURAL validation belongs at the backend edge (the trust boundary where
# external input arrives), NOT here -- doing it downstream wastes compute and burns the Groq
# free-tier quota on input that should have been rejected at the door. Still to add there:
#   - length / size cap on the user message (reject oversized input before it reaches us)
#   - encoding / charset validation (reject non-UTF-8 / control-char payloads)
#   - JSON schema validation of the request body (reject malformed / injected structure)
#   - rate limiting (per family/member) and authentication
# Keep this split: structural checks at the edge, this semantic classifier at the agent entry.

# Groq model id for Llama Prompt Guard 2 (86M). Returns choices[0].message.content as a
# string-encoded float in [0, 1]: the probability that the input is a prompt attack.
_MODEL = "meta-llama/llama-prompt-guard-2-86m"

# Canned, non-leaky refusal shown when an input is blocked. Static (no LLM, no PII, no cost)
# and deliberately vague about *why* so it does not coach an attacker.
_REFUSAL = (
    "Sorry, I can't help with that request. I'm here to help you find great books for your "
    "child -- try asking me for a recommendation, or about a specific book."
)


def _threshold() -> float:
    """Attack-probability cutoff (>= is blocked). Override with GUARD_THRESHOLD; default 0.5."""
    try:
        return float(os.getenv("GUARD_THRESHOLD", "0.5"))
    except ValueError:
        logger.warning("Invalid GUARD_THRESHOLD; falling back to 0.5")
        return 0.5


def _enabled() -> bool:
    """Whether screening runs at all. Set GUARD_ENABLED=false to disable (e.g. offline dev)."""
    return os.getenv("GUARD_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }


@lru_cache(maxsize=1)
def _get_client() -> Any | None:
    """Return a cached Groq client, or None when no API key is configured (-> fail open).

    Lazy + cached so importing this module never requires a key (unit tests, offline dev) and
    the client is built at most once per process.
    """
    if not os.getenv("GROQ_API_KEY"):
        logger.warning(
            "GROQ_API_KEY not set; input safety screening is disabled (fail open)."
        )
        return None
    from groq import Groq

    return Groq()


def screen(text: str) -> float | None:
    """Return the attack probability in [0, 1] for `text`, or None if the check could not run.

    None is the fail-open signal (no key, transport/rate-limit error, or an unparsable
    response). Callers treat None as "allow" -- see module docstring. This function never
    raises and never logs `text`.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        completion = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": text}],
        )
        raw = completion.choices[0].message.content
        return float(
            raw
        )  # ValueError if the classifier ever returns a non-numeric body
    except Exception as exc:  # noqa: BLE001 -- any failure is fail-open; log type only, never PII
        logger.warning("Prompt Guard screening failed: %s", type(exc).__name__)
        return None


def _latest_human_text(messages: list[AnyMessage]) -> str | None:
    """Extract the newest human message's text, or None if there is nothing to screen.

    Only string content is screened; multimodal (list) content contributes its text parts.
    """
    for msg in reversed(messages or []):
        if getattr(msg, "type", None) != "human":
            continue
        content = msg.content
        if isinstance(content, str):
            return content or None
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            joined = " ".join(p for p in parts if p).strip()
            return joined or None
        return None
    return None


def guard(state: FlowState) -> dict[str, Any]:
    """Entry node: screen the latest user message; block prompt-injection attempts.

    Always rewrites the `safety` channel so a prior turn's verdict never lingers. On a block,
    appends a canned refusal and the router sends the turn straight to END; otherwise the turn
    proceeds to load_context unchanged.
    """
    text = _latest_human_text(state.get("messages", []))
    if not _enabled() or not text:
        return {"safety": {"blocked": False, "score": None}}

    score = screen(text)
    threshold = _threshold()
    # score is None => check could not run => fail open (allow).
    blocked = score is not None and score >= threshold

    logger.info(
        "guard: blocked=%s score=%s threshold=%.2f",
        blocked,
        f"{score:.4f}" if score is not None else "n/a",
        threshold,
    )

    if blocked:
        return {
            "safety": {"blocked": True, "score": score},
            "messages": [AIMessage(content=_REFUSAL)],
        }
    return {"safety": {"blocked": False, "score": score}}


def route_after_guard(state: FlowState) -> str:
    """Conditional edge: 'blocked' short-circuits to END, 'ok' proceeds to load_context."""
    return "blocked" if (state.get("safety") or {}).get("blocked") else "ok"
