"""HTTP client for the accounts service's internal (service-to-service) face.

The accounts service is the single writer of the family / member / child / reading-profile /
reading-history / policy tables. The agent no longer touches those tables directly: it reads the
per-turn context bundle and performs every write through `/internal/*`, authenticating with a
service credential (`X-Service-Token`), never a user token. `family_id` is passed as a query
param over the trusted chain (see the accounts internal router).

Config comes from the environment (like `db.base`'s engine):
    ACCOUNTS_INTERNAL_URL    base URL of the accounts internal face, e.g. http://localhost:8001
    ACCOUNTS_SERVICE_TOKEN   shared secret presented as X-Service-Token

Read lazily so importing this module never requires the env to be set (unit tests import the
agent package without a live accounts service); the first real call raises if it is unset.

PII: this client never logs request payloads or response bodies (they carry child/family data).
Only operation names, UUIDs, and HTTP status codes are safe to log.
"""

from __future__ import annotations

import os
from typing import Any
from uuid import UUID

import httpx

_TIMEOUT = httpx.Timeout(10.0)

# JSON-able id: the domain tools hold UUIDs; the graph state carries str. Accept either.
IdLike = UUID | str


class AccountsAPIError(RuntimeError):
    """A non-2xx response from the accounts internal face.

    Carries the HTTP status and the logical operation, never the response body (which may contain
    PII). `str(exc)` is safe to surface to the memory agent and to log.
    """

    def __init__(self, operation: str, status: int) -> None:
        self.operation = operation
        self.status = status
        super().__init__(f"accounts {operation} failed (HTTP {status})")


class AccountsClient:
    """Thin sync wrapper over the accounts `/internal` endpoints used by the agent."""

    def __init__(self, base_url: str, service_token: str) -> None:
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-Service-Token": service_token},
            timeout=_TIMEOUT,
        )

    # --- low-level ---
    def _call(
        self,
        method: str,
        path: str,
        operation: str,
        *,
        family_id: IdLike | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        params = {"family_id": str(family_id)} if family_id is not None else None
        resp = self._http.request(method, path, params=params, json=json)
        if resp.status_code >= 400:
            # Deliberately drop resp.text: it may echo request data / row fields (PII).
            raise AccountsAPIError(operation, resp.status_code)
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    # --- context (read) ---
    def get_context(self, family_id: IdLike) -> dict[str, Any] | None:
        """Return the family's per-turn context bundle, or None if the family is unknown (404)."""
        try:
            return self._call(  # type: ignore[no-any-return]
                "GET", f"/internal/families/{family_id}/context", "get_context"
            )
        except AccountsAPIError as exc:
            if exc.status == 404:
                return None
            raise

    # --- children ---
    def create_child(self, family_id: IdLike, body: dict[str, Any]) -> dict[str, Any]:
        return self._call(  # type: ignore[no-any-return]
            "POST", "/internal/children", "create_child", family_id=family_id, json=body
        )

    def update_child(
        self, family_id: IdLike, child_id: IdLike, body: dict[str, Any]
    ) -> dict[str, Any]:
        return self._call(  # type: ignore[no-any-return]
            "PATCH",
            f"/internal/children/{child_id}",
            "update_child",
            family_id=family_id,
            json=body,
        )

    def upsert_reading_profile(
        self, family_id: IdLike, child_id: IdLike, body: dict[str, Any]
    ) -> dict[str, Any]:
        return self._call(  # type: ignore[no-any-return]
            "PUT",
            f"/internal/children/{child_id}/reading-profile",
            "upsert_reading_profile",
            family_id=family_id,
            json=body,
        )

    # --- reading history ---
    def list_reading_history(
        self, family_id: IdLike, child_id: IdLike
    ) -> list[dict[str, Any]]:
        return self._call(  # type: ignore[no-any-return]
            "GET",
            f"/internal/children/{child_id}/reading-history",
            "list_reading_history",
            family_id=family_id,
        )

    def create_reading_history(
        self, family_id: IdLike, child_id: IdLike, body: dict[str, Any]
    ) -> dict[str, Any]:
        return self._call(  # type: ignore[no-any-return]
            "POST",
            f"/internal/children/{child_id}/reading-history",
            "create_reading_history",
            family_id=family_id,
            json=body,
        )

    def update_reading_history(
        self,
        family_id: IdLike,
        child_id: IdLike,
        entry_id: IdLike,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        return self._call(  # type: ignore[no-any-return]
            "PATCH",
            f"/internal/children/{child_id}/reading-history/{entry_id}",
            "update_reading_history",
            family_id=family_id,
            json=body,
        )

    # --- members ---
    def create_member(self, family_id: IdLike, body: dict[str, Any]) -> dict[str, Any]:
        return self._call(  # type: ignore[no-any-return]
            "POST", "/internal/members", "create_member", family_id=family_id, json=body
        )

    def update_member(
        self, family_id: IdLike, member_id: IdLike, body: dict[str, Any]
    ) -> dict[str, Any]:
        return self._call(  # type: ignore[no-any-return]
            "PATCH",
            f"/internal/members/{member_id}",
            "update_member",
            family_id=family_id,
            json=body,
        )

    def upsert_member_profile(
        self, family_id: IdLike, member_id: IdLike, body: dict[str, Any]
    ) -> dict[str, Any]:
        return self._call(  # type: ignore[no-any-return]
            "PUT",
            f"/internal/members/{member_id}/profile",
            "upsert_member_profile",
            family_id=family_id,
            json=body,
        )

    # --- reading policies ---
    def create_policy(self, family_id: IdLike, body: dict[str, Any]) -> dict[str, Any]:
        return self._call(  # type: ignore[no-any-return]
            "POST",
            "/internal/policies",
            "create_policy",
            family_id=family_id,
            json=body,
        )

    def update_policy(
        self, family_id: IdLike, policy_id: IdLike, body: dict[str, Any]
    ) -> dict[str, Any]:
        return self._call(  # type: ignore[no-any-return]
            "PATCH",
            f"/internal/policies/{policy_id}",
            "update_policy",
            family_id=family_id,
            json=body,
        )


_client: AccountsClient | None = None


def get_client() -> AccountsClient:
    """Return the process-wide accounts client, building it from the environment on first use.

    Raises if the required env vars are unset -- deferred to call time so the agent package stays
    importable without a configured accounts service (e.g. hermetic unit tests inject a fake).
    """
    global _client
    if _client is None:
        base_url = os.getenv("ACCOUNTS_INTERNAL_URL")
        service_token = os.getenv("ACCOUNTS_SERVICE_TOKEN")
        if not base_url or not service_token:
            raise RuntimeError(
                "ACCOUNTS_INTERNAL_URL and ACCOUNTS_SERVICE_TOKEN must be set to reach the "
                "accounts internal API (see .env)."
            )
        _client = AccountsClient(base_url, service_token)
    return _client
