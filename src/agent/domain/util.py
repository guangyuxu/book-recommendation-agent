"""Small shared helpers for domain tools (set-style list merges, date parsing)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date


def parse_iso_date(value: str | None) -> date | None:
    """Parse an ISO 'YYYY-MM-DD' birth date from a tool arg, or None if not given.

    Tool models pass dates as strings; the birth_date columns are DATE. Raise a clear
    error on a malformed value so the failure isn't a cryptic driver-level one later.
    """
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"birth_date must be an ISO date like '2015-04-23', got {value!r}."
        ) from exc


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
