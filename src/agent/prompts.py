"""Versioned prompt registry: the single source of truth for every node/capability prompt.

Prompts live as co-located ``*.prompts.yaml`` files next to the module that uses them (e.g.
``pipeline/respond.prompts.yaml`` beside ``pipeline/respond.py``). Each file declares a
``namespace`` and a set of prompt entries; the entry's dotted id -- ``<namespace>.<key>`` -- is
the stable contract callers reference, independent of where the file physically sits. At import
the loader discovers every such file under the ``agent`` package, compiles each template with
Jinja2, and indexes them by id.

Design (see ROADMAP #1 "Prompt management"):
- **Versioned**: every entry carries an explicit integer ``version``; ``version(id)`` exposes it
  so a turn can record which prompt version produced its output (reproducibility).
- **Templated**: the template body is Jinja2, so conditional wording (a focus-switch note, a
  confirmation outcome, a retry directive) lives in the prompt file, not in Python. The rule of
  thumb: **Python decides what is true (facts/flags), the template decides how to say it**. Data
  rendering that touches DB rows (child/policy briefs) stays in Python and is passed in as a
  variable -- never serialize raw rows into a template.
- **Roles**: an entry is either a ``system`` shorthand (the common case here: one system message,
  with the live conversation appended by the node) or a ``messages`` list of ``{role, template}``
  for multi-role prompts. ``render(id, **vars)`` returns ``list[BaseMessage]`` either way.

No PII concern: this module only reads static template text and renders it with caller-supplied
variables; it logs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined, Template
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import BaseMessage

# autoescape is intentionally OFF: these templates render LLM prompt text, not HTML/XML, so
# HTML-escaping would corrupt the prompt (turn quotes/ampersands into entities). StrictUndefined
# makes a missing variable a hard error at render time rather than a silently blank prompt.
_ENV = Environment(
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    autoescape=False,  # noqa: S701 -- prompt text, not HTML/XML; escaping would corrupt the prompt
)

_ROLE_TO_MESSAGE: dict[str, type[BaseMessage]] = {
    "system": SystemMessage,
    "user": HumanMessage,
    "human": HumanMessage,
    "ai": AIMessage,
    "assistant": AIMessage,
}


@dataclass(frozen=True)
class _Segment:
    """One (role, compiled-template) pair within a prompt."""

    role: str
    template: Template


@dataclass(frozen=True)
class Prompt:
    """A single versioned prompt: an id, a version, and one or more role segments."""

    id: str
    version: int
    segments: tuple[_Segment, ...]

    def render(self, **variables: Any) -> list[BaseMessage]:
        """Render every segment with `variables`, returning the message list to feed the model.

        Trailing newlines are stripped: Jinja `{% %}` block tags leave a structural trailing
        newline that is never meaningful in a prompt, and stripping it keeps templates readable
        (no `{%- -%}` whitespace-control noise) while matching hand-built prompt strings exactly.
        """
        messages: list[BaseMessage] = []
        for seg in self.segments:
            content = seg.template.render(**variables).rstrip("\n")
            messages.append(_ROLE_TO_MESSAGE[seg.role](content=content))
        return messages


_REGISTRY: dict[str, Prompt] | None = None


def _segments_from_spec(pid: str, spec: dict[str, Any]) -> tuple[_Segment, ...]:
    """Build the ordered role segments from one prompt entry (system shorthand or messages list)."""
    if "messages" in spec:
        out: list[_Segment] = []
        for item in spec["messages"]:
            role = str(item["role"]).lower()
            if role not in _ROLE_TO_MESSAGE:
                raise ValueError(f"prompt {pid!r}: unknown role {role!r}")
            out.append(_Segment(role, _ENV.from_string(str(item["template"]))))
        if not out:
            raise ValueError(f"prompt {pid!r}: empty messages list")
        return tuple(out)
    if "system" in spec:
        return (_Segment("system", _ENV.from_string(str(spec["system"]))),)
    raise ValueError(f"prompt {pid!r}: entry must have a 'system' or 'messages' key")


def _load() -> dict[str, Prompt]:
    """Discover every *.prompts.yaml under the agent package and index prompts by id."""
    registry: dict[str, Prompt] = {}
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob("*.prompts.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        namespace = doc.get("namespace")
        if not namespace:
            raise ValueError(f"{path.name}: missing top-level 'namespace'")
        for key, spec in (doc.get("prompts") or {}).items():
            pid = f"{namespace}.{key}"
            if pid in registry:
                raise ValueError(f"duplicate prompt id {pid!r} (in {path.name})")
            if not isinstance(spec.get("version"), int):
                raise ValueError(f"prompt {pid!r}: 'version' must be an int")
            registry[pid] = Prompt(
                id=pid,
                version=spec["version"],
                segments=_segments_from_spec(pid, spec),
            )
    return registry


def _ensure_loaded() -> dict[str, Prompt]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _load()
    return _REGISTRY


def get(prompt_id: str) -> Prompt:
    """Return the Prompt for `prompt_id` (raises KeyError if unknown)."""
    registry = _ensure_loaded()
    try:
        return registry[prompt_id]
    except KeyError:
        raise KeyError(f"unknown prompt id {prompt_id!r}") from None


def render(prompt_id: str, /, **variables: Any) -> list[BaseMessage]:
    """Render `prompt_id` with `variables` into the message list to prepend to a model call."""
    return get(prompt_id).render(**variables)


def version(prompt_id: str) -> int:
    """Return the version of `prompt_id` (for recording which prompt produced a turn)."""
    return get(prompt_id).version
