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


def test_budget_settings_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The service-level budget settings carry sensible defaults (api_key, 12% cap)."""
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.auth_mode == "api_key"
    assert settings.weekly_budget == 0.0
    assert settings.budget_db_path == "retinue-budget.sqlite3"
    assert settings.budget_daily_cap_fraction == 0.12


def test_budget_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The budget settings load from the environment (subscription/token mode)."""
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    monkeypatch.setenv("AUTH_MODE", "subscription")
    monkeypatch.setenv("WEEKLY_BUDGET", "1000000")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.auth_mode == "subscription"
    assert settings.weekly_budget == 1_000_000.0


def test_adapter_settings_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The new adapter wiring fields default to empty (opt-in, never required)."""
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.github_app_id == ""
    assert settings.github_app_private_key_path == ""
    assert settings.anthropic_api_key == ""
    assert settings.anthropic_oauth_token == ""
    assert settings.ntfy_topic == ""
    assert settings.pushover_token == ""
    assert settings.pushover_user == ""


def test_heimdall_bot_login_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """The heimdall bot login defaults to ``heimdall[bot]`` (the inbound review filter)."""
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.heimdall_bot_login == "heimdall[bot]"


def test_heimdall_bot_login_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The heimdall bot login is overridable from the environment."""
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    monkeypatch.setenv("HEIMDALL_BOT_LOGIN", "watcher[bot]")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.heimdall_bot_login == "watcher[bot]"


def test_adapter_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """GitHub App, Anthropic, and push channel settings load from the environment."""
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    monkeypatch.setenv("GITHUB_APP_ID", "123456")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", "/secrets/app.pem")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-xxx")
    monkeypatch.setenv("NTFY_TOPIC", "retinue-alerts")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.github_app_id == "123456"
    assert settings.github_app_private_key_path == "/secrets/app.pem"
    assert settings.anthropic_api_key == "sk-ant-api-xxx"
    assert settings.ntfy_topic == "retinue-alerts"


def test_oauth_token_reads_claude_code_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The subscription OAuth token reads the conventional CLAUDE_CODE_OAUTH_TOKEN."""
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-yyy")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.anthropic_oauth_token == "sk-ant-oat-yyy"


def test_anthropic_credential_follows_auth_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """``anthropic_credential`` resolves the key/token the active auth mode uses."""
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-xxx")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-yyy")

    monkeypatch.setenv("AUTH_MODE", "api_key")
    assert Settings(_env_file=None).anthropic_credential == "sk-ant-api-xxx"  # type: ignore[call-arg]

    monkeypatch.setenv("AUTH_MODE", "subscription")
    assert Settings(_env_file=None).anthropic_credential == "sk-ant-oat-yyy"  # type: ignore[call-arg]
