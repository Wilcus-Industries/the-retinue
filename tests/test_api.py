"""Tests for the authed /api/* surface: bearer auth, POST /api/drain, GET /api/budget.

``POST /api/drain`` is exercised against a real ``ArqRedis`` pool backed by an
in-process ``fakeredis`` server (the same pattern as ``tests/test_roundtrip.py`` and
``tests/test_adhoc_e2e.py``) rather than a mocked ``enqueue_adhoc_drain`` — the point is
to prove the real ``request.app.state.arq_pool`` wiring and the per-repo dedup
``_job_id`` land correctly, which a mocked enqueue call would not catch.

``GET /api/budget`` (issue #90) is exercised against a real temp-file
:class:`retinue.budget.BudgetLedger` — entries are seeded through a ledger bound to the
same ``budget_db_path`` before the request, so the assertion proves the API process
actually reads the shared ledger rather than a mock of it.
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
from retinue.budget import AuthMode, BudgetLedger, SystemClock
from retinue.config import Settings
from retinue.queue import RUN_ADHOC_DRAIN_TASK
from retinue.run_ledger import RunLedgerStore, RunState, run_ledger_store_path

_TOKEN = "test-api-service-token"


def _make_settings(
    tmp_path: Path | None = None, *, budget_db_path: str | None = None
) -> Settings:
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
        budget_db_path=budget_db_path or "retinue-budget.sqlite3",
        weekly_budget=1000.0,
        budget_daily_cap_fraction=0.12,
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


@pytest.fixture()
def budget_db_path(tmp_path: Path) -> Path:
    """The temp-file path the app's budget ledger and a seeding ledger both bind to."""
    return tmp_path / "budget.sqlite3"


@pytest_asyncio.fixture()
async def api_app(arq_pool: ArqRedis, budget_db_path: Path) -> AsyncIterator[FastAPI]:
    """The app with its real ``arq_pool`` wired to the fake Redis (no mocked enqueue).

    ``ASGITransport`` never runs the app's lifespan, so the ``budget_ledger`` connection
    a ``GET /api/budget`` request opens is never closed by the lifespan ``finally``. The
    teardown here closes ``app.state.budget_ledger`` so that aiosqlite connection does not
    leak past the test (the source of the suite's "Event loop is closed" ResourceWarning).
    """
    app = create_app(_make_settings(budget_db_path=str(budget_db_path)))
    app.state.arq_pool = arq_pool
    try:
        yield app
    finally:
        await app.state.budget_ledger.close()


@pytest_asyncio.fixture()
async def api_client(api_app: FastAPI) -> AsyncIterator[AsyncClient]:
    """An httpx client driving the real ASGI app over ``ASGITransport``."""
    transport = ASGITransport(app=api_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_create_app_exposes_budget_ledger_for_teardown(budget_db_path: Path) -> None:
    """create_app publishes its BudgetLedger on app.state so callers can close it.

    ASGITransport-driven tests never run the lifespan, so without a reachable handle the
    ledger connection opened by a /api/budget request would leak. Exposing it on
    app.state.budget_ledger is what lets the api_app fixture close it in teardown.
    """
    app = create_app(_make_settings(budget_db_path=str(budget_db_path)))
    assert isinstance(app.state.budget_ledger, BudgetLedger)


# --- bearer token helper -----------------------------------------------------


def test_verify_bearer_token_accepts_matching_token() -> None:
    """verify_bearer_token accepts a correctly-prefixed, matching token."""
    assert verify_bearer_token(f"Bearer {_TOKEN}", _TOKEN)


def test_verify_bearer_token_rejects_missing_and_wrong() -> None:
    """verify_bearer_token rejects a missing header, wrong token, or wrong scheme."""
    assert not verify_bearer_token(None, _TOKEN)
    assert not verify_bearer_token("Bearer wrong-token", _TOKEN)
    assert not verify_bearer_token(_TOKEN, _TOKEN)  # missing "Bearer " prefix


def test_verify_bearer_token_rejects_non_ascii_token_without_raising() -> None:
    """A non-ASCII bearer token is rejected, not a TypeError.

    ``hmac.compare_digest`` raises ``TypeError`` when handed a ``str`` containing
    non-ASCII characters; an attacker sending ``Authorization: Bearer café`` must get
    a clean False (→ 401) rather than an unhandled exception (→ 500).
    """
    assert not verify_bearer_token("Bearer café-ÿ", _TOKEN)
    # And when the *expected* token is non-ASCII, a matching presentation still works.
    assert verify_bearer_token("Bearer café-ÿ", "café-ÿ")


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


# --- GET /api/escalations ------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalations_returns_only_escalated_rows_with_issue_urls(
    tmp_path: Path,
) -> None:
    """``GET /api/escalations`` returns only ``escalated`` rows, each with an issue URL.

    Seeds the same shared ledger file with a mix of states (queued, building, pr_opened,
    escalated) — the endpoint must return the escalated row alone, not the others.
    """
    settings = _make_settings(tmp_path)
    store = RunLedgerStore(run_ledger_store_path(settings))
    await store.record(repo_full_name="owner/repo", issue=7, state=RunState.QUEUED)
    await store.record(repo_full_name="owner/repo", issue=8, state=RunState.BUILDING)
    await store.record(
        repo_full_name="owner/repo",
        issue=9,
        state=RunState.PR_OPENED,
        url="https://github.com/owner/repo/pull/1",
    )
    await store.record(
        repo_full_name="owner/repo",
        issue=31,
        state=RunState.ESCALATED,
        url="https://github.com/owner/repo/issues/31",
    )

    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/escalations", headers=_auth_headers(_TOKEN))

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["issue"] == 31
    assert body[0]["repo"] == "owner/repo"
    assert body[0]["url"] == "https://github.com/owner/repo/issues/31"


@pytest.mark.asyncio
async def test_escalations_requires_a_valid_bearer_token(tmp_path: Path) -> None:
    """``GET /api/escalations`` 401s with no header or a wrong token (router-level auth)."""
    app = create_app(_make_settings(tmp_path))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        no_header = await client.get("/api/escalations")
        wrong = await client.get(
            "/api/escalations", headers=_auth_headers("wrong-token")
        )

    assert no_header.status_code == 401
    assert wrong.status_code == 401


# --- GET /api/budget ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_returns_trailing_spend_and_cap_from_the_shared_ledger(
    api_client: AsyncClient, budget_db_path: Path
) -> None:
    """A valid bearer token reads the trailing-24h spend and cap off the real ledger.

    Entries are seeded through a *separate* :class:`BudgetLedger` bound to the same
    ``budget_db_path`` the app's own ledger reads — proving the API process reads the
    worker's on-disk ledger, not an in-memory mock of it.
    """
    seeding_ledger = BudgetLedger(
        budget_db_path,
        clock=SystemClock(),
        auth_mode=AuthMode.API_KEY,
        weekly_budget=1000.0,
        daily_cap_fraction=0.12,
    )
    try:
        await seeding_ledger.record_spend(amount=5.0)
        await seeding_ledger.record_spend(amount=3.0)
        expected_trailing = await seeding_ledger.trailing_24h_spend()
        expected_cap = seeding_ledger.cap()
    finally:
        await seeding_ledger.close()

    response = await api_client.get("/api/budget", headers=_auth_headers(_TOKEN))

    assert response.status_code == 200
    body = response.json()
    assert body["trailing_24h_spend"] == pytest.approx(expected_trailing)
    assert body["trailing_24h_spend"] == pytest.approx(8.0)
    assert body["cap"] == pytest.approx(expected_cap)
    assert body["cap"] == pytest.approx(120.0)  # 1000.0 * 0.12


@pytest.mark.asyncio
async def test_budget_with_missing_token_returns_401(api_client: AsyncClient) -> None:
    """A missing bearer token returns 401 and no budget data leaks."""
    response = await api_client.get("/api/budget")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_budget_with_wrong_token_returns_401(api_client: AsyncClient) -> None:
    """A wrong bearer token returns 401 and no budget data leaks."""
    response = await api_client.get(
        "/api/budget", headers=_auth_headers("not-the-token")
    )
    assert response.status_code == 401
