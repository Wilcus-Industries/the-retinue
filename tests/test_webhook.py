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
from retinue.queue import MergedPrJob, PrdJob, ReviewJob
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
    action: str = "opened",
    issue_number: int = 1,
    *,
    labels: list[str] | None = None,
) -> dict:  # type: ignore[type-arg]
    """A signed ``issues`` payload; defaults to a ``prd``-labeled issue.

    ``labels`` defaults to ``["prd"]`` so the common path through the gate is the
    PRD-correct one; pass ``[]`` for an unlabeled issue.
    """
    label_names = ["prd"] if labels is None else labels
    return {
        "action": action,
        "issue": {
            "number": issue_number,
            "labels": [{"name": name} for name in label_names],
        },
        "repository": {"full_name": "owner/repo"},
    }


def _pull_request_payload(
    action: str = "closed", *, merged: bool = True, number: int = 42
) -> dict:  # type: ignore[type-arg]
    return {
        "action": action,
        "pull_request": {"number": number, "merged": merged},
        "repository": {"full_name": "owner/repo"},
    }


def _review_payload(
    *, number: int = 42, state: str = "changes_requested", body: str = "blocking: x"
) -> dict:  # type: ignore[type-arg]
    return {
        "action": "submitted",
        "review": {"state": state, "body": body},
        "pull_request": {"number": number},
        "repository": {"full_name": "owner/repo"},
    }


def _post(client: TestClient, event: str, payload: dict):  # type: ignore[type-arg, no-untyped-def]
    body = json.dumps(payload).encode()
    return client.post(
        "/webhook",
        content=body,
        headers={
            "X-GitHub-Event": event,
            "X-Hub-Signature-256": _sign(body, _SECRET),
            "Content-Type": "application/json",
        },
    )


@pytest.fixture()
def app_client() -> Iterator[tuple[TestClient, MagicMock]]:
    """Yield (TestClient, mock_enqueue) with the patch active for the whole test."""
    settings = _make_settings()
    mock_enqueue = AsyncMock(return_value="jid-test")
    with patch("retinue.webhook.enqueue_prd", mock_enqueue):
        app = create_app(settings)
        client = TestClient(app, raise_server_exceptions=True)
        yield client, mock_enqueue


@pytest.fixture()
def dispatch_client() -> Iterator[tuple[TestClient, MagicMock, MagicMock, MagicMock]]:
    """Yield the client with all three enqueue seams patched and recording."""
    settings = _make_settings()
    enqueue_prd = AsyncMock(return_value="jid-prd")
    enqueue_review = AsyncMock(return_value="jid-review")
    enqueue_merged = AsyncMock(return_value="jid-merge")
    with (
        patch("retinue.webhook.enqueue_prd", enqueue_prd),
        patch("retinue.webhook.enqueue_review", enqueue_review),
        patch("retinue.webhook.enqueue_merged_pr", enqueue_merged),
    ):
        app = create_app(settings)
        client = TestClient(app, raise_server_exceptions=True)
        yield client, enqueue_prd, enqueue_review, enqueue_merged


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


def test_prd_labeled_issue_returns_202_and_enqueues_one(
    app_client: tuple[TestClient, MagicMock],
) -> None:
    """A ``prd``-labeled, relevant-action issue returns 202 and enqueues one job."""
    client, mock_enqueue = app_client
    payload = json.dumps(
        _issues_payload(action="opened", issue_number=5, labels=["prd"])
    ).encode()
    response = _post(client, "issues", json.loads(payload.decode()))
    assert response.status_code == 202
    mock_enqueue.assert_awaited_once()
    enqueued_job = mock_enqueue.call_args[0][1]
    assert enqueued_job == PrdJob(
        repo_full_name="owner/repo", issue_number=5, action="opened"
    )


@pytest.mark.parametrize("action", ["opened", "reopened", "edited", "labeled"])
def test_prd_labeled_relevant_actions_enqueue(
    app_client: tuple[TestClient, MagicMock], action: str
) -> None:
    """Each relevant action on a ``prd``-labeled issue enqueues exactly one job."""
    client, mock_enqueue = app_client
    response = _post(client, "issues", _issues_payload(action=action, labels=["prd"]))
    assert response.status_code == 202
    mock_enqueue.assert_awaited_once()
    assert mock_enqueue.call_args[0][1].action == action


def test_unlabeled_issue_acks_204_and_enqueues_nothing(
    app_client: tuple[TestClient, MagicMock],
) -> None:
    """An issue without the ``prd`` label is acked 204 and enqueues nothing."""
    client, mock_enqueue = app_client
    response = _post(client, "issues", _issues_payload(action="opened", labels=[]))
    assert response.status_code == 204
    mock_enqueue.assert_not_called()


def test_non_prd_label_acks_204_and_enqueues_nothing(
    app_client: tuple[TestClient, MagicMock],
) -> None:
    """An issue labeled with something other than ``prd`` enqueues nothing."""
    client, mock_enqueue = app_client
    response = _post(
        client, "issues", _issues_payload(action="opened", labels=["bug", "backlog"])
    )
    assert response.status_code == 204
    mock_enqueue.assert_not_called()


@pytest.mark.parametrize("action", ["closed", "assigned", "deleted", "unlabeled"])
def test_prd_labeled_irrelevant_action_acks_204(
    app_client: tuple[TestClient, MagicMock], action: str
) -> None:
    """A ``prd``-labeled issue on a non-relevant action is acked 204, nothing enqueued."""
    client, mock_enqueue = app_client
    response = _post(client, "issues", _issues_payload(action=action, labels=["prd"]))
    assert response.status_code == 204
    mock_enqueue.assert_not_called()


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


# --- pull_request / pull_request_review dispatch ----------------------------


def test_merged_pull_request_enqueues_reap(
    dispatch_client: tuple[TestClient, MagicMock, MagicMock, MagicMock],
) -> None:
    """A closed+merged pull_request returns 202 and enqueues exactly one reap job."""
    client, enqueue_prd, enqueue_review, enqueue_merged = dispatch_client
    response = _post(client, "pull_request", _pull_request_payload(number=42))
    assert response.status_code == 202
    enqueue_merged.assert_awaited_once()
    assert enqueue_merged.call_args[0][1] == MergedPrJob(
        repo_full_name="owner/repo", pr_number=42
    )
    enqueue_prd.assert_not_called()
    enqueue_review.assert_not_called()


def test_closed_unmerged_pull_request_is_ignored(
    dispatch_client: tuple[TestClient, MagicMock, MagicMock, MagicMock],
) -> None:
    """A closed-but-not-merged pull_request is acked 204 and reaps nothing."""
    client, _prd, _review, enqueue_merged = dispatch_client
    response = _post(
        client, "pull_request", _pull_request_payload(action="closed", merged=False)
    )
    assert response.status_code == 204
    enqueue_merged.assert_not_called()


def test_opened_pull_request_is_ignored(
    dispatch_client: tuple[TestClient, MagicMock, MagicMock, MagicMock],
) -> None:
    """A non-close pull_request action (opened) is acked 204 and reaps nothing."""
    client, _prd, _review, enqueue_merged = dispatch_client
    response = _post(
        client, "pull_request", _pull_request_payload(action="opened", merged=False)
    )
    assert response.status_code == 204
    enqueue_merged.assert_not_called()


def test_pull_request_review_enqueues_loopback(
    dispatch_client: tuple[TestClient, MagicMock, MagicMock, MagicMock],
) -> None:
    """A pull_request_review returns 202 and enqueues the loopback with its state/body."""
    client, enqueue_prd, enqueue_review, enqueue_merged = dispatch_client
    response = _post(
        client,
        "pull_request_review",
        _review_payload(number=42, state="changes_requested", body="blocking nit"),
    )
    assert response.status_code == 202
    enqueue_review.assert_awaited_once()
    assert enqueue_review.call_args[0][1] == ReviewJob(
        repo_full_name="owner/repo",
        pr_number=42,
        review_state="changes_requested",
        review_body="blocking nit",
    )
    enqueue_prd.assert_not_called()
    enqueue_merged.assert_not_called()


def test_unsigned_pull_request_returns_401(
    dispatch_client: tuple[TestClient, MagicMock, MagicMock, MagicMock],
) -> None:
    """The HMAC contract is identical for the new events: a bad signature is 401."""
    client, _prd, _review, enqueue_merged = dispatch_client
    body = json.dumps(_pull_request_payload()).encode()
    response = client.post(
        "/webhook",
        content=body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": "sha256=bad",
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 401
    enqueue_merged.assert_not_called()


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
