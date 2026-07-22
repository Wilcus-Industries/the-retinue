"""Tests for the authed /api/* surface: bearer auth and POST /api/drain.

``POST /api/drain`` is exercised against a real ``ArqRedis`` pool backed by an
in-process ``fakeredis`` server (the same pattern as ``tests/test_roundtrip.py`` and
``tests/test_adhoc_e2e.py``) rather than a mocked ``enqueue_adhoc_drain`` — the point is
to prove the real ``request.app.state.arq_pool`` wiring and the per-repo dedup
``_job_id`` land correctly, which a mocked enqueue call would not catch.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

import fakeredis
import pytest
import pytest_asyncio
from arq import ArqRedis
from arq.jobs import Job
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from retinue.api import verify_bearer_token
from retinue.app import create_app
from retinue.config import Settings
from retinue.queue import RUN_ADHOC_DRAIN_TASK
from retinue.run_ledger import RunLedgerStore, RunState, run_ledger_store_path

_TOKEN = "test-api-service-token"


def _make_settings(tmp_path: Path | None = None) -> Settings:
    # A tmp_path pins the dedupe db (and so the run-ledger's state dir) into the test's
    # own dir, isolating each test's ledger file; the default keeps existing callers valid.
    dedupe_db_path = (
        str(tmp_path / "retinue-dedupe.sqlite3")
        if tmp_path is not None
        else "retinue-dedupe.sqlite3"
    )
    return Settings(  # type: ignore[call-arg]
        webhook_secret="test-webhook-secret",
        api_service_token=_TOKEN,
        redis_url="redis://localhost:6379",
        dedupe_db_path=dedupe_db_path,
        _env_file=None,
    )


@pytest_asyncio.fixture()
async def arq_pool() -> AsyncIterator[ArqRedis]:
    """An ArqRedis backed by an isolated in-process fakeredis server (the real spine)."""
    server = fakeredis.FakeServer()
    fake = fakeredis.FakeAsyncRedis(server=server)
    pool = ArqRedis(pool_or_conn=fake.connection_pool)
    try:
        yield pool
    finally:
        # Idempotent: a test may have already closed the shared pool.
        with contextlib.suppress(Exception):
            await pool.aclose()


@pytest_asyncio.fixture()
async def api_app(arq_pool: ArqRedis) -> FastAPI:
    """The app with its real ``arq_pool`` wired to the fake Redis (no mocked enqueue)."""
    app = create_app(_make_settings())
    app.state.arq_pool = arq_pool
    return app


@pytest_asyncio.fixture()
async def api_client(api_app: FastAPI) -> AsyncIterator[AsyncClient]:
    """An httpx client driving the real ASGI app over ``ASGITransport``."""
    transport = ASGITransport(app=api_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


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


@pytest.mark.asyncio
async def test_drain_with_valid_token_enqueues_adhoc_drain(
    api_client: AsyncClient, arq_pool: ArqRedis
) -> None:
    """A valid bearer token lands a real AdhocDrainJob on the arq pool.

    Queries the fake Redis back by the returned job id to prove the real
    ``enqueue_adhoc_drain`` -> ``ArqRedis`` wiring ran (task name, repo kwarg, and the
    per-repo dedup ``_job_id``), not just that a mock was called.
    """
    response = await api_client.post(
        "/api/drain",
        json={"repo_full_name": "owner/repo"},
        headers=_auth_headers(_TOKEN),
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert job_id == "adhoc-drain:owner/repo"

    info = await Job(job_id, arq_pool).info()
    assert info is not None
    assert info.function == RUN_ADHOC_DRAIN_TASK
    assert info.kwargs == {"repo_full_name": "owner/repo"}


@pytest.mark.asyncio
async def test_drain_with_missing_token_returns_401_and_no_enqueue(
    api_client: AsyncClient, arq_pool: ArqRedis
) -> None:
    """A missing bearer token returns 401 and enqueues nothing."""
    response = await api_client.post(
        "/api/drain", json={"repo_full_name": "owner/repo"}
    )
    assert response.status_code == 401
    assert await Job("adhoc-drain:owner/repo", arq_pool).info() is None


@pytest.mark.asyncio
async def test_drain_with_wrong_token_returns_401_and_no_enqueue(
    api_client: AsyncClient, arq_pool: ArqRedis
) -> None:
    """A wrong bearer token returns 401 and enqueues nothing."""
    response = await api_client.post(
        "/api/drain",
        json={"repo_full_name": "owner/repo"},
        headers=_auth_headers("not-the-token"),
    )
    assert response.status_code == 401
    assert await Job("adhoc-drain:owner/repo", arq_pool).info() is None


@pytest.mark.asyncio
async def test_webhook_route_unaffected_by_api_auth(api_client: AsyncClient) -> None:
    """The webhook route is mounted independently and ignores the API bearer token.

    A request to /webhook with no Authorization header and a bad HMAC signature still
    gets the webhook's own 401 (not the API auth's), proving the two auth schemes don't
    interfere with each other.
    """
    response = await api_client.post(
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


# --- GET /api/runs ------------------------------------------------------------


@pytest.mark.asyncio
async def test_runs_returns_the_recorded_rows(tmp_path: Path) -> None:
    """Authed ``GET /api/runs`` returns the run-ledger rows recorded on the shared file.

    Builds its own app (no Redis needed for ``/runs``) and seeds the same temp-file ledger
    ``create_app`` reads, proving the reader picks up the worker-written rows.
    """
    settings = _make_settings(tmp_path)
    store = RunLedgerStore(run_ledger_store_path(settings))
    await store.record(repo_full_name="owner/repo", issue=7, state=RunState.BUILDING)
    await store.record(repo_full_name="owner/repo", issue=8, state=RunState.QUEUED)

    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/runs", headers=_auth_headers(_TOKEN))

    assert resp.status_code == 200
    by_issue = {row["issue"]: row for row in resp.json()}
    assert by_issue[7]["state"] == "building"
    assert by_issue[8]["state"] == "queued"
    assert by_issue[7]["repo"] == "owner/repo"
    assert by_issue[8]["url"] is None


@pytest.mark.asyncio
async def test_runs_requires_a_valid_bearer_token(tmp_path: Path) -> None:
    """``GET /api/runs`` 401s with no header or a wrong token (router-level auth)."""
    app = create_app(_make_settings(tmp_path))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        no_header = await client.get("/api/runs")
        wrong = await client.get("/api/runs", headers=_auth_headers("wrong-token"))

    assert no_header.status_code == 401
    assert wrong.status_code == 401
