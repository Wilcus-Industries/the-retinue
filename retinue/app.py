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

from retinue.config import Settings
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
            await pool.close()
            logger.info("Arq Redis pool closed")

    app = FastAPI(title="The Retinue", version="0.1.0", lifespan=lifespan)
    # Default to None so the attribute always exists even if the lifespan hasn't
    # run (e.g. in tests that patch enqueue_prd and never need the real pool).
    app.state.arq_pool = None

    webhook_router = make_webhook_router(webhook_secret=settings.webhook_secret)
    app.include_router(webhook_router)

    return app
