"""Make unit tests hermetic: no real database and no live accounts service.

`agent.db.base` raises at import time when BOOK_AGENT_DATABASE_URL is unset and builds its
engine from it. The unit tests here construct their own sqlite sessions (for the agent-owned
recommendation/book tables) and never use that module-level engine, but they still import
`agent.*`, so collection needs the var to be set. We default it to in-memory sqlite before any
`agent` module is imported. `setdefault` means a URL already exported in the environment still
wins, so nothing changes in CI's integration job or for anyone deliberately pointing at Postgres.

The family / child / reading / policy tables now live in the accounts service; the agent reaches
them over its internal API (`agent.accounts_client`). Tests never hit a live service: the
`fake_accounts` fixture installs an in-memory stand-in as the process client singleton. We also
default the accounts env vars so `get_client()` never raises during collection.
"""

import os

os.environ.setdefault("BOOK_AGENT_DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("ACCOUNTS_INTERNAL_URL", "http://accounts.test")
os.environ.setdefault("ACCOUNTS_SERVICE_TOKEN", "test-token")

from typing import Any  # noqa: E402
from uuid import uuid4  # noqa: E402

import pytest  # noqa: E402


class FakeAccountsClient:
    """In-memory stand-in for `agent.accounts_client.AccountsClient`.

    Mirrors the internal API's write semantics (upserts replace the fields sent; the domain tools
    are responsible for merging lists before sending) so tests can drive the real domain tools and
    then assert on the resulting state. IDs are generated like the real service would.
    """

    def __init__(self) -> None:
        self.children: dict[str, dict[str, Any]] = {}
        self.reading_profiles: dict[str, dict[str, Any]] = {}
        self.reading_history: dict[str, list[dict[str, Any]]] = {}
        self.members: list[dict[str, Any]] = []
        self.member_profiles: dict[str, dict[str, Any]] = {}
        self.policies: list[dict[str, Any]] = []
        self.calls: list[
            tuple[str, Any]
        ] = []  # (operation, family_id) for scope assertions

    # --- context ---
    def get_context(self, family_id: Any) -> dict[str, Any] | None:
        fid = str(family_id)
        self.calls.append(("get_context", fid))
        children = {
            cid: {**c, "reading_profile": self.reading_profiles.get(cid, {})}
            for cid, c in self.children.items()
            if c.get("family_id") == fid
        }
        members = [
            {**m, "profile": self.member_profiles.get(str(m["id"]), {})}
            for m in self.members
            if m.get("family_id") == fid
        ]
        policies = [p for p in self.policies if p.get("family_id") == fid]
        return {
            "family": {"id": fid},
            "members": members,
            "children": children,
            "policies": policies,
        }

    # --- children ---
    def create_child(self, family_id: Any, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("create_child", str(family_id)))
        cid = str(uuid4())
        child = {"id": cid, "family_id": str(family_id), **body}
        self.children[cid] = child
        self.reading_profiles[cid] = {}
        return child

    def update_child(
        self, family_id: Any, child_id: Any, body: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append(("update_child", str(family_id)))
        child = self.children.setdefault(str(child_id), {"id": str(child_id)})
        child.update(body)
        return child

    def upsert_reading_profile(
        self, family_id: Any, child_id: Any, body: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append(("upsert_reading_profile", str(family_id)))
        prof = self.reading_profiles.setdefault(str(child_id), {})
        prof.update(body)
        return dict(prof)

    # --- reading history ---
    def list_reading_history(
        self, family_id: Any, child_id: Any
    ) -> list[dict[str, Any]]:
        self.calls.append(("list_reading_history", str(family_id)))
        return list(self.reading_history.get(str(child_id), []))

    def create_reading_history(
        self, family_id: Any, child_id: Any, body: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append(("create_reading_history", str(family_id)))
        entry = {"id": str(uuid4()), **body}
        self.reading_history.setdefault(str(child_id), []).append(entry)
        return entry

    def update_reading_history(
        self, family_id: Any, child_id: Any, entry_id: Any, body: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append(("update_reading_history", str(family_id)))
        for entry in self.reading_history.get(str(child_id), []):
            if str(entry["id"]) == str(entry_id):
                entry.update(body)
                return entry
        raise KeyError(entry_id)

    # --- members ---
    def create_member(self, family_id: Any, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("create_member", str(family_id)))
        member = {"id": str(uuid4()), "family_id": str(family_id), **body}
        self.members.append(member)
        return member

    def update_member(
        self, family_id: Any, member_id: Any, body: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append(("update_member", str(family_id)))
        for m in self.members:
            if str(m["id"]) == str(member_id):
                m.update(body)
                return m
        member = {"id": str(member_id), "family_id": str(family_id), **body}
        self.members.append(member)
        return member

    def upsert_member_profile(
        self, family_id: Any, member_id: Any, body: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append(("upsert_member_profile", str(family_id)))
        prof = self.member_profiles.setdefault(str(member_id), {})
        prof.update(body)
        return dict(prof)

    # --- policies ---
    def create_policy(self, family_id: Any, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("create_policy", str(family_id)))
        policy = {
            "id": str(uuid4()),
            "family_id": str(family_id),
            "is_active": True,
            **body,
        }
        self.policies.append(policy)
        return policy

    def update_policy(
        self, family_id: Any, policy_id: Any, body: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append(("update_policy", str(family_id)))
        for p in self.policies:
            if str(p["id"]) == str(policy_id):
                p.update(body)
                return p
        raise KeyError(policy_id)


@pytest.fixture
def fake_accounts() -> Any:
    """Install a FakeAccountsClient as the process accounts-client singleton for the test."""
    import agent.accounts_client as ac

    fake = FakeAccountsClient()
    prior = ac._client
    ac._client = fake
    try:
        yield fake
    finally:
        ac._client = prior
