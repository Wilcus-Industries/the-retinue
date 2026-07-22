"""Tests for the authed /api/* surface: bearer auth and POST /api/drain."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from retinue.api import verify_bearer_token
from retinue.app import create_app
from retinue.config import Settings
from retinue.queue import AdhocDrainJob

_TOKEN = "test-api-service-token"


def _make_settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        webhook_secret="test-webhook-secret",
        api_service_token=_TOKEN,
        redis_url="redis://localhost:6379",
        _env_file=None,
    )


@pytest.fixture()
def api_client() -> Iterator[tuple[TestClient, MagicMock]]:
    """Yield the client with ``enqueue_adhoc_drain`` patched and recording."""
    settings = _make_settings()
    enqueue_adhoc = AsyncMock(return_value="jid-adhoc")
    with patch("retinue.api.enqueue_adhoc_drain", enqueue_adhoc):
        app = create_app(settings)
        client = TestClient(app, raise_server_exceptions=True)
        yield client, enqueue_adhoc


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# --- bearer token helper -----------------------------------------------------


def test_verify_bearer_token_accepts_matching_token() -> None:
    """verify_bearer_token accepts a correctly-prefixed, matching token."""
    assert verify_bearer_token(f"Bearer {_TOKEN}", _TOKEN)


def test_verify_bearer_token_rejects_missing_and_wrong() -> None:
    """verify_bearer_token rejects a missing header, wrong token, or wrong scheme."""
    assert not verify_bearer_token(None, _TOKEN)
    assert not verify_bearer_token("Bearer wrong-token", _TOKEN)
    assert not verify_bearer_token(_TOKEN, _TOKEN)  # missing "Bearer " prefix


# --- POST /api/drain ----------------------------------------------------------


def test_drain_with_valid_token_enqueues_adhoc_drain(
    api_client: tuple[TestClient, MagicMock],
) -> None:
    """A valid bearer token enqueues an AdhocDrainJob for the given repo."""
    client, enqueue_adhoc = api_client
    response = client.post(
        "/api/drain",
        json={"repo_full_name": "owner/repo"},
        headers=_auth_headers(_TOKEN),
    )
    assert response.status_code == 202
    enqueue_adhoc.assert_awaited_once()
    assert enqueue_adhoc.call_args[0][1] == AdhocDrainJob(repo_full_name="owner/repo")


def test_drain_with_missing_token_returns_401_and_no_enqueue(
    api_client: tuple[TestClient, MagicMock],
) -> None:
    """A missing bearer token returns 401 and enqueues nothing."""
    client, enqueue_adhoc = api_client
    response = client.post("/api/drain", json={"repo_full_name": "owner/repo"})
    assert response.status_code == 401
    enqueue_adhoc.assert_not_called()


def test_drain_with_wrong_token_returns_401_and_no_enqueue(
    api_client: tuple[TestClient, MagicMock],
) -> None:
    """A wrong bearer token returns 401 and enqueues nothing."""
    client, enqueue_adhoc = api_client
    response = client.post(
        "/api/drain",
        json={"repo_full_name": "owner/repo"},
        headers=_auth_headers("not-the-token"),
    )
    assert response.status_code == 401
    enqueue_adhoc.assert_not_called()


def test_webhook_route_unaffected_by_api_auth(
    api_client: tuple[TestClient, MagicMock],
) -> None:
    """The webhook route is mounted independently and ignores the API bearer token.

    A request to /webhook with no Authorization header and a bad HMAC signature still
    gets the webhook's own 401 (not the API auth's), proving the two auth schemes don't
    interfere with each other.
    """
    client, _enqueue_adhoc = api_client
    response = client.post(
        "/webhook",
        content=b'{"action": "opened"}',
        headers={
            "X-GitHub-Event": "issues",
            "X-Hub-Signature-256": "sha256=bad",
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 401
    assert response.text == "Invalid webhook signature"
