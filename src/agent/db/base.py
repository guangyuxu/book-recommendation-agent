"""Database infrastructure: engine, session, Base, table creation. Connection string via DATABASE_URL."""

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import parse_qs, unquote, urlsplit

from advanced_alchemy.base import AdvancedDeclarativeBase, CommonTableAttributes
from dotenv import load_dotenv
from sqlalchemy import JSON, Text, create_engine, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()  # so standalone scripts (e.g. table creation) can read .env too

# Cross-dialect JSON: JSONB on Postgres (indexable), plain JSON elsewhere (e.g. sqlite).
JSONType = JSON().with_variant(JSONB(), "postgresql")

# Postgres text[] column type, reused by every model with an array column. Mirrors the
# `text[]` columns in the book_agent schema; on non-Postgres dialects it degrades to JSON.
TextArray = ARRAY(Text).with_variant(JSON(), "sqlite")

BOOK_AGENT_DATABASE_URL = os.getenv("BOOK_AGENT_DATABASE_URL")
if not BOOK_AGENT_DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set; configure it in .env (see the .env example)")

_is_sqlite = BOOK_AGENT_DATABASE_URL.startswith("sqlite")


def _search_path_schema(url: str) -> str | None:
    """Pull the schema out of a libpq `options=-csearch_path=<schema>` query param, if present.

    The URL pins tables to this schema via search_path, but Postgres won't create it for us --
    init_db must create the schema first or CREATE TABLE fails with "no schema has been selected".
    """
    options = parse_qs(urlsplit(url).query).get("options", [None])[0]
    if not options:
        return None
    m = re.search(r"search_path=([^,\s]+)", unquote(options))
    return m.group(1) if m else None


_schema = None if _is_sqlite else _search_path_schema(BOOK_AGENT_DATABASE_URL)
engine = create_engine(
    BOOK_AGENT_DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=not _is_sqlite,  # for Postgres, avoids stale idle connections
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session boundary: commit on success, roll back on error, always close.

    The unit-of-work counterpart to the repositories, whose write methods only flush. Open
    one scope per request/turn, build repositories on the yielded session, and let this commit:

        with session_scope() as s:
            FamilyRepository(session=s).add(Family(family_name="..."))
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class Base(CommonTableAttributes, AdvancedDeclarativeBase):
    """Base class for all ORM models.

    Composed from Advanced Alchemy so models share its `orm_registry`/metadata and gain
    `to_dict()`, while staying compatible with the SQLAlchemySyncRepository. We deliberately
    do NOT inherit a primary-key mixin (e.g. UUIDBase): those pull in Advanced Alchemy's
    `sa_orm_sentinel` column, which our DB-first tables don't have. Instead every model
    declares its own `id`/`created_at`/`updated_at` to mirror the live schema exactly.
    """

    __abstract__ = True


def init_db() -> None:
    """Create tables (dev only). In production prefer Alembic migrations over create_all."""
    from . import models  # noqa: F401  ensure models are registered on Base.metadata

    if _schema:
        # The connection's search_path points at this schema, but it may not exist yet.
        # Create it first (identifier can't be parameterized; _schema comes from our own URL).
        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{_schema}"'))
    Base.metadata.create_all(engine)
