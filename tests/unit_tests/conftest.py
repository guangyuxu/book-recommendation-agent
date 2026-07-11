"""Make unit tests hermetic: they run against in-memory sqlite, never a real database.

`agent.db.base` raises at import time when BOOK_AGENT_DATABASE_URL is unset and builds its
engine from it. The unit tests here construct their own sqlite sessions and never use that
module-level engine, but they still import `agent.*`, so collection needs the var to be set.
We default it to in-memory sqlite before any `agent` module is imported. `setdefault` means a
URL already exported in the environment still wins, so nothing changes in CI's integration job
or for anyone deliberately pointing the unit tests at Postgres.
"""

import os

os.environ.setdefault("BOOK_AGENT_DATABASE_URL", "sqlite:///:memory:")
