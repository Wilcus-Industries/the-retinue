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


@pytest.mark.parametrize("value", ["-1", "-0.5"])
def test_negative_weekly_budget_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A negative weekly budget is a config error; only 0.0 (disabled) or >0 are valid."""
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    monkeypatch.setenv("WEEKLY_BUDGET", value)
    with pytest.raises(ValueError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_zero_weekly_budget_is_the_disabled_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WEEKLY_BUDGET=0 loads cleanly: it is the metering-disabled sentinel, not an error."""
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    monkeypatch.setenv("WEEKLY_BUDGET", "0")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.weekly_budget == 0.0


def test_job_timeout_default_exceeds_arq_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real claude build outlasts arq's 300s default, so the job timeout must be larger.

    A drain kicked at the default 300s is cancelled mid-implement (the build needs many
    minutes), so the worker-global ``job_timeout`` is driven from this setting with a
    default well above 300.
    """
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.job_timeout_seconds > 300
    assert settings.job_timeout_seconds == 1800


def test_job_timeout_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The job timeout loads from the environment for longer or shorter builds."""
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "3600")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.job_timeout_seconds == 3600


def test_implement_max_turns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The implementer's agent-loop cap carries its default (mirrors the orchestrator)."""
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.implement_max_turns == 80


@pytest.mark.parametrize("value", ["0", "-1"])
def test_non_positive_job_timeout_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A zero or negative job timeout is a config error, not a silent load."""
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError):
        Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize("value", ["0", "-1"])
def test_non_positive_implement_max_turns_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A zero or negative implement cap is a config error, not a silent load."""
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    monkeypatch.setenv("IMPLEMENT_MAX_TURNS", value)
    with pytest.raises(ValueError):
        Settings(_env_file=None)  # type: ignore[call-arg]


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
