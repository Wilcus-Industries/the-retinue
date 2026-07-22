"""Tests for the authed API surface: bearer verification and POST /api/drain."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from retinue.api import verify_bearer_token
from retinue.app import create_app
from retinue.config import Settings
from retinue.queue import RUN_ADHOC_DRAIN_TASK

_TOKEN = "tok-secret"


def _make_settings(*, api_service_token: str = _TOKEN) -> Settings:
    return Settings(  # type: ignore[call-arg]
        webhook_secret="x",
        api_service_token=api_service_token,
        _env_file=None,
    )


class _FakePool:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, dict[str, object]]] = []

    async def delete(self, key: str) -> None:
        pass

    async def enqueue_job(self, task: str, **kwargs: object) -> SimpleNamespace:
        self.enqueued.append((task, kwargs))
        return SimpleNamespace(job_id="jid-adhoc")


# ---------------------------------------------------------------------------
# verify_bearer_token unit tests
# ---------------------------------------------------------------------------


def test_verify_bearer_token_matches() -> None:
    assert verify_bearer_token("secret", "secret") is True


def test_verify_bearer_token_mismatch() -> None:
    assert verify_bearer_token("wrong", "secret") is False


def test_verify_bearer_token_empty_configured() -> None:
    assert verify_bearer_token("anything", "") is False


def test_verify_bearer_token_none_presented() -> None:
    assert verify_bearer_token(None, "secret") is False


# ---------------------------------------------------------------------------
# POST /api/drain integration tests (TestClient over create_app)
# ---------------------------------------------------------------------------


def _client_with_pool(settings: Settings) -> tuple[TestClient, _FakePool]:
    app = create_app(settings)
    pool = _FakePool()
    app.state.arq_pool = pool
    return TestClient(app, raise_server_exceptions=True), pool


def test_valid_bearer_enqueues_drain() -> None:
    client, pool = _client_with_pool(_make_settings())
    resp = client.post(
        "/api/drain",
        json={"repo_full_name": "owner/repo"},
        headers={"Authorization": f"Bearer {_TOKEN}"},
    )
    assert resp.status_code == 202
    assert len(pool.enqueued) == 1
    task, kwargs = pool.enqueued[0]
    assert task == RUN_ADHOC_DRAIN_TASK
    assert kwargs["repo_full_name"] == "owner/repo"
    assert kwargs["_job_id"] == "adhoc-drain:owner/repo"


def test_missing_token_returns_401_no_enqueue() -> None:
    client, pool = _client_with_pool(_make_settings())
    resp = client.post("/api/drain", json={"repo_full_name": "owner/repo"})
    assert resp.status_code == 401
    assert pool.enqueued == []


def test_wrong_token_returns_401_no_enqueue() -> None:
    client, pool = _client_with_pool(_make_settings())
    resp = client.post(
        "/api/drain",
        json={"repo_full_name": "owner/repo"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401
    assert pool.enqueued == []


def test_malformed_scheme_returns_401() -> None:
    client, pool = _client_with_pool(_make_settings())
    resp = client.post(
        "/api/drain",
        json={"repo_full_name": "owner/repo"},
        headers={"Authorization": _TOKEN},
    )
    assert resp.status_code == 401
    assert pool.enqueued == []


def test_fail_closed_when_token_empty() -> None:
    client, pool = _client_with_pool(_make_settings(api_service_token=""))
    resp = client.post(
        "/api/drain",
        json={"repo_full_name": "owner/repo"},
        headers={"Authorization": "Bearer anything"},
    )
    assert resp.status_code == 401
    assert pool.enqueued == []
