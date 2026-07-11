"""Small shared helpers for domain tools (set-style list merges, date parsing)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date


def parse_iso_date(value: str | None) -> date | None:
    """Parse a birth date from a tool arg, or None if not given.

    Tolerant of partial dates: 'YYYY', 'YYYY-MM', and 'YYYY-MM-DD' are all accepted, with a
    missing month/day defaulting to 01 (a parent may only know the birth year). Tool models
    pass dates as strings; the birth_date columns are DATE. Raise a clear error on a malformed
    value so the failure isn't a cryptic driver-level one later.
    """
    if value is None:
        return None
    parts = str(value).strip().split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 and parts[1] else 1
        day = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        return date(year, month, day)
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"birth_date must be a year or ISO date like '2015' or '2015-04-23', got {value!r}."
        ) from exc


def merge_unique(
    existing: Iterable[str] | None, additions: Iterable[str] | None
) -> list[str]:
    """Append additions to existing, preserving order and dropping duplicates."""
    out = list(existing or [])
    for item in additions or []:
        if item not in out:
            out.append(item)
    return out


def remove_all(
    existing: Iterable[str] | None, removals: Iterable[str] | None
) -> list[str]:
    """Return existing with every value in removals filtered out."""
    drop = set(removals or [])
    return [item for item in (existing or []) if item not in drop]
