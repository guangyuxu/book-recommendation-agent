"""One-off migration: constrain gender to the Gender enum values ('Male' / 'Female').

The models switched gender from free Text to an Enum(Gender, native_enum=False) column,
which on a fresh DB is a VARCHAR + CHECK (gender IN ('Male','Female')). The live columns were
created as plain text by the earlier migration, so this adds the matching CHECK constraint
(NULL still allowed -- gender is optional). Idempotent: drops the constraint first if present.

Run once (after migrate_add_gender_birthdate.py):

    uv run python scripts/migrate_gender_enum.py
"""

import logging

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text  # noqa: E402

from agent.db.base import engine  # noqa: E402

logger = logging.getLogger(__name__)

TABLES = ("family_member", "child_profile")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with engine.begin() as conn:
        for table in TABLES:
            constraint = f"ck_{table}_gender"
            conn.execute(
                text(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}")
            )
            conn.execute(
                text(
                    f"ALTER TABLE {table} ADD CONSTRAINT {constraint} "
                    f"CHECK (gender IN ('Male', 'Female'))"
                )
            )
    logger.info(
        "Migration applied: gender constrained to 'Male'/'Female' (NULL allowed)."
    )


if __name__ == "__main__":
    main()
