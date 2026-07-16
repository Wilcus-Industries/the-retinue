"""Tests for the GET /health liveness endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from retinue.app import create_app
from retinue.config import Settings


def _make_settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        webhook_secret="test-secret",
        redis_url="redis://localhost:6379",
        _env_file=None,
    )


def test_health_returns_200_with_ok() -> None:
    app = create_app(_make_settings())
    client = TestClient(app, raise_server_exceptions=True)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
