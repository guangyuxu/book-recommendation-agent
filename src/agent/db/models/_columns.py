"""Reusable column definitions shared by every model.

Conventions mirrored from the live book_agent schema:
- UUID primary key, generated server-side via gen_random_uuid().
- created_at / updated_at TIMESTAMP, defaulted server-side to now(); updated_at also bumped
  ORM-side via onupdate so writes through the ORM keep it fresh.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, Enum, Uuid, func, text
from sqlalchemy.orm import Mapped, mapped_column


class Gender(StrEnum):
    """Allowed gender values (nullable in the DB). StrEnum so the value ('Male'/'Female')
    is what serializes into state/prompts and stores in the column, not the member name."""

    MALE = "Male"
    FEMALE = "Female"


def _gender() -> Mapped[Gender | None]:
    """Nullable gender column, stored as a VARCHAR + CHECK (native_enum=False) so it works on
    both Postgres and the sqlite test DB. values_callable persists the value, not the name."""
    return mapped_column(
        Enum(
            Gender,
            native_enum=False,
            length=16,
            values_callable=lambda enum: [m.value for m in enum],
        )
    )


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
