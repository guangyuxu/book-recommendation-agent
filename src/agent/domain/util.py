"""Small shared helpers for domain tools (set-style list merges)."""

from __future__ import annotations

from collections.abc import Iterable


def merge_unique(existing: Iterable[str] | None, additions: Iterable[str] | None) -> list[str]:
    """Append additions to existing, preserving order and dropping duplicates."""
    out = list(existing or [])
    for item in additions or []:
        if item not in out:
            out.append(item)
    return out


def remove_all(existing: Iterable[str] | None, removals: Iterable[str] | None) -> list[str]:
    """Return existing with every value in removals filtered out."""
    drop = set(removals or [])
    return [item for item in (existing or []) if item not in drop]
