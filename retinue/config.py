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

    # GitHub App identity (issue #16). The worker authenticates as the App installation
    # to mint clone/issue tokens; the private key is read from a file path (never inlined
    # into env) so the PEM stays a mounted secret rather than a leaked literal.
    github_app_id: str = Field(
        default="", description="GitHub App numeric id (the JWT 'iss' claim)"
    )
    github_app_private_key_path: str = Field(
        default="",
        description="Path to the GitHub App RSA private key (PEM) used to sign app JWTs",
    )

    # Anthropic auth. BOTH credentials are carried so a deployment can run either metering
    # mode without re-keying: ``api_key`` mode reads ``anthropic_api_key`` (dollars),
    # ``subscription`` mode reads ``anthropic_oauth_token`` (tokens, an ``sk-ant-oat...``
    # OAuth token on Authorization: Bearer). ``anthropic_credential`` resolves the one the
    # active ``auth_mode`` uses.
    anthropic_api_key: str = Field(
        default="", description="Anthropic API key, used in api_key auth mode"
    )
    anthropic_oauth_token: str = Field(
        default="",
        description="Claude subscription OAuth token (sk-ant-oat...), used in "
        "subscription auth mode",
        alias="claude_code_oauth_token",
    )

    # Push channel (issue #16): exactly one of ntfy or Pushover backs the notify push
    # sink. ntfy needs a topic (+ optional token for a protected topic); Pushover needs
    # both an app token and a user/group key.
    ntfy_topic: str = Field(
        default="", description="ntfy topic the push sink publishes to (ntfy backend)"
    )
    ntfy_token: str = Field(
        default="", description="Optional ntfy access token for a protected topic"
    )
    pushover_token: str = Field(
        default="", description="Pushover application API token (Pushover backend)"
    )
    pushover_user: str = Field(
        default="", description="Pushover user/group key (Pushover backend)"
    )

    @property
    def anthropic_credential(self) -> str:
        """The Anthropic credential the active ``auth_mode`` authenticates with.

        ``subscription`` mode rides the OAuth token (``Authorization: Bearer`` + the
        OAuth beta header); ``api_key`` mode rides the raw API key (``x-api-key``).
        """
        if self.auth_mode == "subscription":
            return self.anthropic_oauth_token
        return self.anthropic_api_key

    # extra="ignore": the .env may be shared with deployment tooling that carries
    # keys which are not Settings fields, so unknown keys are ignored rather than
    # rejected (pydantic-settings defaults to extra="forbid").
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        # The OAuth token field carries an alias (CLAUDE_CODE_OAUTH_TOKEN) so the
        # conventional env var name resolves it; allow population by the field name too
        # so ANTHROPIC_OAUTH_TOKEN and direct construction both work.
        "populate_by_name": True,
    }
