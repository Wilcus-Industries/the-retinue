"""Tests for Settings: env/.env loading of the webhook secret and Redis URL."""

from __future__ import annotations

from pathlib import Path

import pytest

from retinue.config import Settings


def test_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings reads the webhook secret and Redis URL from the environment."""
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    monkeypatch.setenv("REDIS_URL", "redis://example:6380/1")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.webhook_secret == "s3cret"
    assert settings.redis_url == "redis://example:6380/1"


def test_redis_url_defaults_to_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    """When REDIS_URL is unset, the Redis URL defaults to localhost:6379."""
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    monkeypatch.delenv("REDIS_URL", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.redis_url == "redis://localhost:6379"


def test_settings_loads_from_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Settings reads values from a .env file when env vars are absent."""
    monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("WEBHOOK_SECRET=from-dotenv\nREDIS_URL=redis://dotenv:6379\n")
    settings = Settings(_env_file=str(env_file))  # type: ignore[call-arg]
    assert settings.webhook_secret == "from-dotenv"
    assert settings.redis_url == "redis://dotenv:6379"


def test_missing_webhook_secret_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing webhook secret is a configuration error."""
    monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
    with pytest.raises(ValueError):
        Settings(_env_file=None)  # type: ignore[call-arg]
