"""One-off migration: add gender/birth_date to family_member and child_profile.

Background: the models grew `gender` (text) and `birth_date` (date) on both
FamilyMember and ChildProfile, and ChildProfile dropped the old stored `birth_year`
and `age` columns (age is now derived in load_context). `create_all` never alters
existing tables, so the live schema was left behind -- SELECTs then referenced
columns that don't exist, surfacing as advanced_alchemy IntegrityError in load_context.

This script is idempotent (IF [NOT] EXISTS everywhere) and best-effort backfills
birth_date from the old data before dropping it:
  - birth_year present  -> birth_date = <birth_year>-01-01
  - else age present    -> birth_date = (current_year - age)-01-01
Jan 1 preserves the displayed age for the remainder of the current year; parents can
re-enter exact dates afterward.

Run once:

    uv run python scripts/migrate_add_gender_birthdate.py
"""

import logging

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text  # noqa: E402

from agent.db.base import engine  # noqa: E402

logger = logging.getLogger(__name__)

DDL = [
    # --- family_member: add new identity columns ---
    "ALTER TABLE family_member ADD COLUMN IF NOT EXISTS gender text",
    "ALTER TABLE family_member ADD COLUMN IF NOT EXISTS birth_date date",
    # --- child_profile: add new columns ---
    "ALTER TABLE child_profile ADD COLUMN IF NOT EXISTS gender text",
    "ALTER TABLE child_profile ADD COLUMN IF NOT EXISTS birth_date date",
    # --- backfill birth_date from legacy birth_year / age (best-effort) ---
    """
    UPDATE child_profile
       SET birth_date = make_date(birth_year, 1, 1)
     WHERE birth_date IS NULL AND birth_year IS NOT NULL
    """,
    """
    UPDATE child_profile
       SET birth_date = make_date(EXTRACT(YEAR FROM now())::int - age, 1, 1)
     WHERE birth_date IS NULL AND age IS NOT NULL
    """,
    # --- drop legacy stored columns now that data is migrated ---
    "ALTER TABLE child_profile DROP COLUMN IF EXISTS age",
    "ALTER TABLE child_profile DROP COLUMN IF EXISTS birth_year",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))
    logger.info(
        "Migration applied: gender/birth_date added; legacy birth_year/age migrated & dropped."
    )


if __name__ == "__main__":
    main()
