"""Application configuration loaded from environment variables and a .env file."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Retinue runtime configuration.

    All values can be set via environment variables (upper-cased field names) or a
    ``.env`` file. The webhook secret has no default and must be supplied; secrets
    should live in ``.env`` or be injected by the deployment environment.
    """

    webhook_secret: str = Field(..., description="GitHub webhook HMAC secret")
    redis_url: str = Field(
        default="redis://localhost:6379", description="Redis connection URL"
    )
    dedupe_db_path: str = Field(
        default="retinue-dedupe.sqlite3",
        description="Path to the SQLite file backing PRD-event deduplication",
    )

    # Budget governor (issue #14). The budget is service-level — shared across the
    # orchestrator and cron lanes — so it lives here, not in the per-repo config.
    # ``auth_mode`` selects the metering unit: an API key meters dollars against the
    # weekly-$ budget; subscription OAuth meters tokens against the weekly-token budget.
    auth_mode: str = Field(
        default="api_key",
        description="Spend-metering auth mode: 'api_key' (dollars) or "
        "'subscription' (tokens)",
    )
    weekly_budget: float = Field(
        default=0.0,
        description="Service-level weekly budget — dollars in api_key mode, tokens in "
        "subscription mode. The 24h cap is a fraction of this.",
    )
    budget_db_path: str = Field(
        default="retinue-budget.sqlite3",
        description="Path to the SQLite file backing the rolling-24h spend ledger",
    )
    budget_daily_cap_fraction: float = Field(
        default=0.12,
        description="Fraction of the weekly budget spendable per rolling-24h window",
    )

    # extra="ignore": the .env may be shared with deployment tooling that carries
    # keys which are not Settings fields, so unknown keys are ignored rather than
    # rejected (pydantic-settings defaults to extra="forbid").
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
