"""Database infrastructure: engine, session, Base, table creation. Connection string via DATABASE_URL."""

import os
import re
from urllib.parse import parse_qs, unquote, urlsplit

from dotenv import load_dotenv
from sqlalchemy import JSON, create_engine, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()  # so standalone scripts (e.g. table creation) can read .env too

# Cross-dialect JSON: JSONB on Postgres (indexable), plain JSON elsewhere (e.g. sqlite).
JSONType = JSON().with_variant(JSONB(), "postgresql")

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


class Base(DeclarativeBase):
    """Base class for all ORM models."""


def init_db() -> None:
    """Create tables (dev only). In production prefer Alembic migrations over create_all."""
    from . import models  # noqa: F401  ensure models are registered on Base.metadata

    if _schema:
        # The connection's search_path points at this schema, but it may not exist yet.
        # Create it first (identifier can't be parameterized; _schema comes from our own URL).
        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{_schema}"'))
    Base.metadata.create_all(engine)
