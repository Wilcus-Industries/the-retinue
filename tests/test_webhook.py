"""Tests for the webhook endpoint: signature validation, issues filtering, enqueue."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from retinue.app import create_app
from retinue.config import Settings
from retinue.queue import PrdJob
from retinue.webhook import compute_signature, verify_signature

_SECRET = "test-webhook-secret"


def _make_settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        webhook_secret=_SECRET,
        redis_url="redis://localhost:6379",
        _env_file=None,
    )


def _sign(payload: bytes, secret: str) -> str:
    sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


def _issues_payload(
    action: str = "opened", issue_number: int = 1
) -> dict:  # type: ignore[type-arg]
    return {
        "action": action,
        "issue": {"number": issue_number},
        "repository": {"full_name": "owner/repo"},
    }


@pytest.fixture()
def app_client() -> Iterator[tuple[TestClient, MagicMock]]:
    """Yield (TestClient, mock_enqueue) with the patch active for the whole test."""
    settings = _make_settings()
    mock_enqueue = AsyncMock(return_value="jid-test")
    with patch("retinue.webhook.enqueue_prd", mock_enqueue):
        app = create_app(settings)
        client = TestClient(app, raise_server_exceptions=True)
        yield client, mock_enqueue


# --- signature helpers ------------------------------------------------------


def test_compute_signature_matches_github_format() -> None:
    """compute_signature returns the ``sha256=<hex>`` header GitHub sends."""
    payload = b'{"hello": "world"}'
    assert compute_signature(payload, _SECRET) == _sign(payload, _SECRET)


def test_verify_round_trips_with_compute_signature() -> None:
    """verify_signature accepts a header produced by compute_signature."""
    payload = b"some-body-bytes"
    assert verify_signature(payload, _SECRET, compute_signature(payload, _SECRET))


def test_verify_rejects_missing_and_bad() -> None:
    """verify_signature returns False for a missing or mismatched header."""
    payload = b"body"
    assert not verify_signature(payload, _SECRET, None)
    assert not verify_signature(payload, _SECRET, "sha256=deadbeef")


# --- endpoint behaviour -----------------------------------------------------


def test_valid_issues_webhook_returns_202_and_enqueues_one(
    app_client: tuple[TestClient, MagicMock],
) -> None:
    """A validly signed issues webhook returns 202 and enqueues exactly one job."""
    client, mock_enqueue = app_client
    payload = json.dumps(_issues_payload(action="opened", issue_number=5)).encode()
    headers = {
        "X-GitHub-Event": "issues",
        "X-Hub-Signature-256": _sign(payload, _SECRET),
        "Content-Type": "application/json",
    }
    response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code == 202
    mock_enqueue.assert_awaited_once()
    enqueued_job = mock_enqueue.call_args[0][1]
    assert enqueued_job == PrdJob(
        repo_full_name="owner/repo", issue_number=5, action="opened"
    )


def test_invalid_signature_returns_401_and_enqueues_nothing(
    app_client: tuple[TestClient, MagicMock],
) -> None:
    """An invalid signature returns 401 and no job is enqueued."""
    client, mock_enqueue = app_client
    payload = json.dumps(_issues_payload()).encode()
    headers = {
        "X-GitHub-Event": "issues",
        "X-Hub-Signature-256": "sha256=bad",
        "Content-Type": "application/json",
    }
    response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code == 401
    mock_enqueue.assert_not_called()


def test_missing_signature_returns_401_and_enqueues_nothing(
    app_client: tuple[TestClient, MagicMock],
) -> None:
    """A missing signature header returns 401 and no job is enqueued."""
    client, mock_enqueue = app_client
    payload = json.dumps(_issues_payload()).encode()
    headers = {
        "X-GitHub-Event": "issues",
        "Content-Type": "application/json",
    }
    response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code == 401
    mock_enqueue.assert_not_called()


def test_non_issues_event_ignored(
    app_client: tuple[TestClient, MagicMock],
) -> None:
    """A validly signed non-issues event returns 204 without enqueuing."""
    client, mock_enqueue = app_client
    payload = json.dumps({"action": "opened"}).encode()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": _sign(payload, _SECRET),
        "Content-Type": "application/json",
    }
    response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code == 204
    mock_enqueue.assert_not_called()


def test_enqueue_failure_returns_5xx(
    app_client: tuple[TestClient, MagicMock],
) -> None:
    """If enqueue raises, the handler returns 5xx (not 202) so GitHub redelivers."""
    settings = _make_settings()
    failing_enqueue = AsyncMock(side_effect=RuntimeError("redis down"))
    payload = json.dumps(_issues_payload()).encode()
    headers = {
        "X-GitHub-Event": "issues",
        "X-Hub-Signature-256": _sign(payload, _SECRET),
        "Content-Type": "application/json",
    }
    with patch("retinue.webhook.enqueue_prd", failing_enqueue):
        app = create_app(settings)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code >= 500
    failing_enqueue.assert_called_once()


# --- lifespan / pool wiring -------------------------------------------------


def test_lifespan_creates_and_closes_pool() -> None:
    """The lifespan creates an Arq pool on startup and closes it on shutdown."""
    settings = _make_settings()
    mock_pool = AsyncMock()
    mock_pool.close = AsyncMock()

    with (
        patch("arq.create_pool", return_value=mock_pool) as mock_create,
        patch("retinue.webhook.enqueue_prd", AsyncMock(return_value="jid")),
    ):
        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=True):
            mock_create.assert_called_once()
        mock_pool.close.assert_called_once()


def test_webhook_reads_pool_from_app_state() -> None:
    """The webhook handler uses the pool the lifespan placed on app.state."""
    settings = _make_settings()
    mock_pool = AsyncMock()
    mock_pool.close = AsyncMock()
    captured_pools: list[object] = []

    async def fake_enqueue(pool: object, job: PrdJob) -> str:
        captured_pools.append(pool)
        return "jid"

    payload = json.dumps(_issues_payload()).encode()
    headers = {
        "X-GitHub-Event": "issues",
        "X-Hub-Signature-256": _sign(payload, _SECRET),
        "Content-Type": "application/json",
    }
    with (
        patch("arq.create_pool", return_value=mock_pool),
        patch("retinue.webhook.enqueue_prd", side_effect=fake_enqueue),
    ):
        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=True) as client:
            response = client.post("/webhook", content=payload, headers=headers)

    assert response.status_code == 202
    assert captured_pools == [mock_pool]
