"""Dataset loading: read JSONL case files into plain dicts.

JSONL (one JSON object per line) is the house format for eval datasets -- diff-friendly, easy to
append to, and each line stands alone so a malformed row never breaks the rest. Strategies wrap
these dicts in their own pydantic case models (e.g. `s1_classification.schema.S1Case`).
"""

from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a `.jsonl` file into a list of dicts, skipping blank lines.

    Raises a `ValueError` that names the file and line number on a malformed row, so a typo in a
    hand-edited dataset points straight at the offending line instead of a bare JSON error.
    """
    path = Path(path)
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows
