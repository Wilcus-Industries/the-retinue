"""FastAPI application factory.

Wires config and the webhook router into a single ASGI app. The factory pattern
lets tests inject a custom Settings instance without touching environment
variables.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import arq
from fastapi import FastAPI

from retinue.api import make_api_router
from retinue.budget import AuthMode, BudgetLedger, SystemClock
from retinue.config import Settings
from retinue.run_ledger import RunLedgerStore, run_ledger_store_path
from retinue.webhook import make_webhook_router

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construct and return the Retinue FastAPI application.

    A lifespan hook creates an Arq Redis pool from ``settings.redis_url`` on
    startup and closes it on shutdown, storing it in ``app.state.arq_pool``.
    Tests may set ``app.state.arq_pool`` directly after calling this function to
    inject a mock pool without triggering a real Redis connection.

    Args:
        settings: Optional Settings override (defaults to reading from env/.env).

    Returns:
        A configured FastAPI instance.
    """
    if settings is None:
        settings = Settings()  # type: ignore[call-arg]

    # Capture settings in the lifespan closure so the Redis URL is available.
    _settings = settings

    # The API process's own read-side handle onto the shared budget SQLite ledger
    # (GET /api/budget, issue #90). It only *queries* the ledger — never records a charge —
    # but the connection itself is read-write (BudgetLedger opens WAL and ensures the schema
    # on first use), so it is NOT a read-only handle and would fail on a :ro mount (see #88).
    # Bound to the same ``budget_db_path`` the worker
    # writes, but a distinct connection: this process never touches the worker's
    # in-memory BudgetGovernor. weekly_budget/daily_cap_fraction come from this
    # process's own Settings (the same env), so cap() is computable with no worker
    # dependency. Construction is synchronous (the aiosqlite connection opens lazily
    # on first use), so it's available even when the app's lifespan never runs (e.g.
    # tests driving the app directly via ASGITransport).
    budget_ledger = BudgetLedger(
        _settings.budget_db_path,
        clock=SystemClock(),
        auth_mode=AuthMode.from_config(_settings.auth_mode),
        weekly_budget=_settings.weekly_budget,
        daily_cap_fraction=_settings.budget_daily_cap_fraction,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        pool = await arq.create_pool(
            arq.connections.RedisSettings.from_dsn(_settings.redis_url)
        )
        app.state.arq_pool = pool
        logger.info("Arq Redis pool created (%s)", _settings.redis_url)
        try:
            yield
        finally:
            # Nest so the budget ledger is closed unconditionally: a raise from
            # pool.close() must not skip the ledger close (and leak its connection).
            try:
                await pool.close()
            finally:
                await budget_ledger.close()
            logger.info("Arq Redis pool closed")

    app = FastAPI(title="The Retinue", version="0.1.0", lifespan=lifespan)
    # Default to None so the attribute always exists even if the lifespan hasn't
    # run (e.g. in tests that patch enqueue_prd and never need the real pool).
    app.state.arq_pool = None

    webhook_router = make_webhook_router(webhook_secret=settings.webhook_secret)
    app.include_router(webhook_router)

    # Reader side of the cross-process run-ledger (the worker is the writer). Construction
    # is pure — no I/O until a request hits /runs — so this is safe at app-build time.
    run_ledger = RunLedgerStore(run_ledger_store_path(settings))
    api_router = make_api_router(
        api_service_token=settings.api_service_token,
        run_ledger=run_ledger,
        budget_ledger=budget_ledger,
    )
    app.include_router(api_router)

    return app
