"""Create DB tables for local dev (idempotent: existing tables are skipped).

Run once before debug_run.py:

    uv run python scripts/create_tables.py

Uses Base.metadata.create_all under the hood -- it only creates MISSING tables
and never drops or alters existing ones. This is a dev convenience only; in
production prefer Alembic migrations (create_all does not evolve changed tables,
and concurrent create_all across replicas can race).
"""

import logging

from dotenv import load_dotenv

load_dotenv()  # pull BOOK_AGENT_DATABASE_URL etc. from .env

from agent.db import init_db  # noqa: E402  (import after load_dotenv so the URL is set)

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_db()
    logger.info("Tables created (existing tables left untouched).")


if __name__ == "__main__":
    main()
