"""Reusable column definitions shared by every model.

Conventions mirrored from the live book_agent schema:
- UUID primary key, generated server-side via gen_random_uuid().
- created_at / updated_at TIMESTAMP, defaulted server-side to now(); updated_at also bumped
  ORM-side via onupdate so writes through the ORM keep it fresh.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Uuid, func, text
from sqlalchemy.orm import Mapped, mapped_column


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime, nullable=False, server_default=func.now())


def _updated_at() -> Mapped[datetime]:
    return mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )
