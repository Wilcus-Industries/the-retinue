"""Per-repo opt-in config: the ``.github/retinue.yml`` schema and its loader.

Presence of ``.github/retinue.yml`` in a repo is the opt-in signal: a repo with
no file is skipped upstream, and a repo whose file is malformed is skipped here.
:func:`load_repo_config` never raises on bad input — it returns ``None`` and logs,
so a single broken config cannot crash the worker.

The schema mirrors the :class:`retinue.config.Settings` pydantic style but models a
YAML document rather than environment variables. Validation is strict (unknown keys
are rejected) so a typo'd field is surfaced as a skip, not silently dropped.
"""

from __future__ import annotations

import logging

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)

# A cron cadence is the classic five whitespace-separated fields
# (minute hour day-of-month month day-of-week).
_CRON_FIELD_COUNT = 5


class SecretsConfig(BaseModel):
    """Secrets and secret references declared by a repo's config.

    The YAML block carries inline ``NAME: value`` pairs alongside a reserved
    ``refs`` list, e.g.::

        secrets:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          refs:
            - vault://team/retinue/github-token

    Inline pairs are gathered into :attr:`values`; ``refs`` is pulled out into its
    own list. Real secrets are expected to be ``${{ secrets.NAME }}`` placeholders
    rather than literals; the loader carries them, it does not resolve them.

    Attributes:
        values: Inline secret name -> value/placeholder mappings.
        refs: External secret references (e.g. ``vault://...``) resolved downstream.
    """

    model_config = ConfigDict(extra="forbid")

    values: dict[str, str] = Field(default_factory=dict)
    refs: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _gather_inline_values(cls, data: object) -> object:
        """Fold inline ``NAME: value`` pairs into ``values``, keeping ``refs`` apart.

        Lets the YAML use the flat, ergonomic shape (secret names as siblings of
        ``refs``) while the model keeps inline values and refs cleanly separated.
        """
        if not isinstance(data, dict):
            return data
        refs = data.get("refs", [])
        values = {
            key: value for key, value in data.items() if key not in ("refs", "values")
        }
        # An explicit ``values`` mapping, if given, is merged on top of inline pairs.
        values.update(data.get("values", {}) or {})
        return {"values": values, "refs": refs}


class RepoConfig(BaseModel):
    """Validated contents of a repo's ``.github/retinue.yml``.

    Attributes:
        staging_branch: Branch the retinue integrates work onto (default ``staging``).
        retry_cap: Max retries per unit of work before giving up (default ``3``).
        max_parallel: Optional cap on concurrent work; unset means no explicit cap.
        cron: Optional five-field cron cadence for scheduled runs.
        models: Role -> model-id overrides (e.g. ``{"planner": "claude-opus-4"}``).
        secrets: Secrets and secret-references block.
    """

    model_config = ConfigDict(extra="forbid")

    staging_branch: str = "staging"
    retry_cap: int = Field(default=3, ge=0)
    max_parallel: int | None = Field(default=None, gt=0)
    cron: str | None = None
    models: dict[str, str] = Field(default_factory=dict)
    secrets: SecretsConfig = Field(default_factory=SecretsConfig)

    @field_validator("cron")
    @classmethod
    def _validate_cron(cls, value: str | None) -> str | None:
        """Reject a cron string that is not the standard five whitespace fields."""
        if value is not None and len(value.split()) != _CRON_FIELD_COUNT:
            raise ValueError(
                f"cron must have {_CRON_FIELD_COUNT} fields, got {len(value.split())}"
            )
        return value


def load_repo_config(text: str) -> RepoConfig | None:
    """Parse and validate a repo's ``retinue.yml`` text into a :class:`RepoConfig`.

    The presence-as-opt-in and never-crash contract live here: any failure to parse
    or validate is logged and turned into ``None`` so the caller treats the repo as
    a skip rather than propagating an exception into the worker.

    Args:
        text: Raw contents of the repo's ``.github/retinue.yml``.

    Returns:
        A validated :class:`RepoConfig`, or ``None`` when the document is malformed,
        is not a mapping, or violates the schema.
    """
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        logger.warning("Skipping repo: malformed retinue.yml (YAML error): %s", exc)
        return None

    # An empty file parses to None; treat it as an all-defaults config.
    if document is None:
        document = {}
    if not isinstance(document, dict):
        logger.warning(
            "Skipping repo: retinue.yml top level is %s, expected a mapping",
            type(document).__name__,
        )
        return None

    try:
        return RepoConfig.model_validate(document)
    except ValidationError as exc:
        logger.warning("Skipping repo: invalid retinue.yml schema: %s", exc)
        return None
