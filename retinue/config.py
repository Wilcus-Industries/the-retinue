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

    # extra="ignore": the .env may be shared with deployment tooling that carries
    # keys which are not Settings fields, so unknown keys are ignored rather than
    # rejected (pydantic-settings defaults to extra="forbid").
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
