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
from retinue.queue import AdhocDrainJob, MergedPrJob
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
    """An ``issues`` payload; defaults to a ``ready-for-agent``-labeled issue.

    ``labels`` defaults to ``["ready-for-agent"]`` so the common path through the gate
    is the trigger one; pass ``[]`` for an unlabeled issue.
    """
    label_names = ["ready-for-agent"] if labels is None else labels
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
def dispatch_client() -> Iterator[tuple[TestClient, MagicMock, MagicMock]]:
    """Yield the client with both enqueue seams patched and recording.

    Returns ``(client, enqueue_adhoc, enqueue_merged)``.
    """
    settings = _make_settings()
    enqueue_adhoc = AsyncMock(return_value="jid-adhoc")
    enqueue_merged = AsyncMock(return_value="jid-merge")
    with (
        patch("retinue.webhook.enqueue_adhoc_drain", enqueue_adhoc),
        patch("retinue.webhook.enqueue_merged_pr", enqueue_merged),
    ):
        app = create_app(settings)
        client = TestClient(app, raise_server_exceptions=True)
        yield client, enqueue_adhoc, enqueue_merged


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


# --- ad-hoc drain dispatch (ready-for-agent issues) -------------------------


@pytest.mark.parametrize("action", ["opened", "reopened", "edited", "labeled"])
def test_ready_for_agent_issue_enqueues_one_adhoc_drain(
    dispatch_client: tuple[TestClient, MagicMock, MagicMock], action: str
) -> None:
    """Each relevant action on a ``ready-for-agent`` issue enqueues one ad-hoc drain."""
    client, enqueue_adhoc, enqueue_merged = dispatch_client
    response = _post(
        client, "issues", _issues_payload(action=action, labels=["ready-for-agent"])
    )
    assert response.status_code == 202
    enqueue_adhoc.assert_awaited_once()
    assert enqueue_adhoc.call_args[0][1] == AdhocDrainJob(repo_full_name="owner/repo")
    enqueue_merged.assert_not_called()


@pytest.mark.parametrize(
    "labels", [[], ["bug", "backlog"], ["custom-trigger"]], ids=["none", "other", "custom"]
)
def test_relevant_action_kicks_a_drain_regardless_of_label(
    dispatch_client: tuple[TestClient, MagicMock, MagicMock], labels: list[str]
) -> None:
    """A relevant issue action kicks a drain no matter what labels the issue carries.

    The kick is only a per-repo "drain this repo" signal; the drain itself re-lists and
    re-filters the repo's ready issues by its own configured ``trigger_label``. Gating the
    kick on the hardcoded ``ready-for-agent`` label would starve a BYOK repo that
    configures a custom trigger label (or hasn't been labeled yet) of any webhook-driven
    drain.
    """
    client, enqueue_adhoc, enqueue_merged = dispatch_client
    response = _post(client, "issues", _issues_payload(action="opened", labels=labels))
    assert response.status_code == 202
    enqueue_adhoc.assert_awaited_once()
    assert enqueue_adhoc.call_args[0][1] == AdhocDrainJob(repo_full_name="owner/repo")
    enqueue_merged.assert_not_called()


@pytest.mark.parametrize("action", ["closed", "assigned", "deleted", "unlabeled"])
def test_ready_for_agent_irrelevant_action_acks_204(
    dispatch_client: tuple[TestClient, MagicMock, MagicMock], action: str
) -> None:
    """A ``ready-for-agent`` issue on a non-relevant action is acked 204, nothing enqueued."""
    client, enqueue_adhoc, _merged = dispatch_client
    response = _post(
        client, "issues", _issues_payload(action=action, labels=["ready-for-agent"])
    )
    assert response.status_code == 204
    enqueue_adhoc.assert_not_called()


# --- pull_request dispatch --------------------------------------------------


def test_merged_pull_request_enqueues_reap(
    dispatch_client: tuple[TestClient, MagicMock, MagicMock],
) -> None:
    """A closed+merged pull_request returns 202 and enqueues exactly one reap job."""
    client, enqueue_adhoc, enqueue_merged = dispatch_client
    response = _post(client, "pull_request", _pull_request_payload(number=42))
    assert response.status_code == 202
    enqueue_merged.assert_awaited_once()
    assert enqueue_merged.call_args[0][1] == MergedPrJob(
        repo_full_name="owner/repo", pr_number=42
    )
    enqueue_adhoc.assert_not_called()


def test_closed_unmerged_pull_request_is_ignored(
    dispatch_client: tuple[TestClient, MagicMock, MagicMock],
) -> None:
    """A closed-but-not-merged pull_request is acked 204 and reaps nothing."""
    client, _adhoc, enqueue_merged = dispatch_client
    response = _post(
        client, "pull_request", _pull_request_payload(action="closed", merged=False)
    )
    assert response.status_code == 204
    enqueue_merged.assert_not_called()


def test_opened_pull_request_is_ignored(
    dispatch_client: tuple[TestClient, MagicMock, MagicMock],
) -> None:
    """A non-close pull_request action (opened) is acked 204 and reaps nothing."""
    client, _adhoc, enqueue_merged = dispatch_client
    response = _post(
        client, "pull_request", _pull_request_payload(action="opened", merged=False)
    )
    assert response.status_code == 204
    enqueue_merged.assert_not_called()


# --- signature / other-event behaviour --------------------------------------


def test_invalid_signature_returns_401_and_enqueues_nothing(
    dispatch_client: tuple[TestClient, MagicMock, MagicMock],
) -> None:
    """An invalid signature returns 401 and no job is enqueued."""
    client, enqueue_adhoc, enqueue_merged = dispatch_client
    payload = json.dumps(_issues_payload()).encode()
    headers = {
        "X-GitHub-Event": "issues",
        "X-Hub-Signature-256": "sha256=bad",
        "Content-Type": "application/json",
    }
    response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code == 401
    enqueue_adhoc.assert_not_called()
    enqueue_merged.assert_not_called()


def test_missing_signature_returns_401_and_enqueues_nothing(
    dispatch_client: tuple[TestClient, MagicMock, MagicMock],
) -> None:
    """A missing signature header returns 401 and no job is enqueued."""
    client, enqueue_adhoc, enqueue_merged = dispatch_client
    payload = json.dumps(_issues_payload()).encode()
    headers = {
        "X-GitHub-Event": "issues",
        "Content-Type": "application/json",
    }
    response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code == 401
    enqueue_adhoc.assert_not_called()
    enqueue_merged.assert_not_called()


def test_non_issue_non_pr_event_ignored(
    dispatch_client: tuple[TestClient, MagicMock, MagicMock],
) -> None:
    """A validly signed event that is neither issues nor pull_request returns 204."""
    client, enqueue_adhoc, enqueue_merged = dispatch_client
    payload = json.dumps({"action": "submitted"}).encode()
    headers = {
        "X-GitHub-Event": "pull_request_review",
        "X-Hub-Signature-256": _sign(payload, _SECRET),
        "Content-Type": "application/json",
    }
    response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code == 204
    enqueue_adhoc.assert_not_called()
    enqueue_merged.assert_not_called()


def test_enqueue_failure_returns_5xx() -> None:
    """If enqueue raises, the handler returns 5xx (not 202) so GitHub redelivers."""
    settings = _make_settings()
    failing_enqueue = AsyncMock(side_effect=RuntimeError("redis down"))
    payload = json.dumps(_issues_payload()).encode()
    headers = {
        "X-GitHub-Event": "issues",
        "X-Hub-Signature-256": _sign(payload, _SECRET),
        "Content-Type": "application/json",
    }
    with patch("retinue.webhook.enqueue_adhoc_drain", failing_enqueue):
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
        patch("retinue.webhook.enqueue_adhoc_drain", AsyncMock(return_value="jid")),
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

    async def fake_enqueue(pool: object, job: AdhocDrainJob) -> str:
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
        patch("retinue.webhook.enqueue_adhoc_drain", side_effect=fake_enqueue),
    ):
        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=True) as client:
            response = client.post("/webhook", content=payload, headers=headers)

    assert response.status_code == 202
    assert captured_pools == [mock_pool]
