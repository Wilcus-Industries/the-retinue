"""Per-repo opt-in config: the ``.github/retinue.yml`` schema and its loader.

Presence of ``.github/retinue.yml`` in a repo is the opt-in signal: a repo with
no file is skipped upstream, and a repo whose file is malformed is skipped here.
:func:`load_repo_config` never raises on bad input — it returns ``None`` and logs,
so a single broken config cannot crash the worker.

The schema mirrors the :class:`retinue.config.Settings` pydantic style but models a
YAML document rather than environment variables. Validation is strict (unknown keys
are rejected) so a typo'd field is surfaced as a skip, not silently dropped.

It also carries the optional ``routing:`` block — a repo's per-issue model/effort
routing table (see :class:`RoutingConfig`); absence means routing is off.
"""

from __future__ import annotations

import logging
import re

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from retinue.roles import EFFORT_HIGH, EFFORT_MAX, EFFORT_XHIGH, Role

logger = logging.getLogger(__name__)

# A cron cadence is the classic five whitespace-separated fields
# (minute hour day-of-month month day-of-week).
_CRON_FIELD_COUNT = 5

# Reasoning-effort tiers a routing-table entry's ``effort:`` may name. Reuses the
# three tiers the role registry actually assigns (:data:`retinue.roles.EFFORT_HIGH`
# et al.) plus ``low``/``medium``, the two lighter tiers a repo's routing table may
# pick for cheap work — the PRD's full five-tier set.
_VALID_EFFORT_TIERS = frozenset({"low", "medium", EFFORT_HIGH, EFFORT_XHIGH, EFFORT_MAX})

# Role keys a routing level's ``roles:`` map may use — the current agent-role
# registry, so a routing table can only target roles that exist.
_VALID_ROLE_NAMES = frozenset(role.value for role in Role)

# A level name must be a lowercase, hyphen-separated slug — the same shape as this
# repo's existing kebab-case labels (e.g. ``ready-for-agent``), since it becomes half
# of the ``level:<name>`` GitHub label the classifier will apply (PRD #58).
_LEVEL_SLUG_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


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


class ModelEffort(BaseModel):
    """A model id with an optional reasoning-effort override.

    Shared shape for a routing table's ``classifier:`` override and each entry in a
    level's ``roles:`` map — both are ``{model, effort?}``.

    Attributes:
        model: The model id to use.
        effort: Optional reasoning-effort tier, one of ``low`` / ``medium`` / ``high``
            / ``xhigh`` / ``max``. Unset means the role's own registry tier applies.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    effort: str | None = None

    @field_validator("effort")
    @classmethod
    def _validate_effort(cls, value: str | None) -> str | None:
        """Reject an effort tier outside the known five-tier set."""
        if value is not None and value not in _VALID_EFFORT_TIERS:
            raise ValueError(
                f"effort must be one of {sorted(_VALID_EFFORT_TIERS)}, got {value!r}"
            )
        return value


class RoutingLevel(BaseModel):
    """One named level of a repo's routing table.

    Attributes:
        description: Prose describing what kind of work belongs to this level; the
            classifier's prompt is built from every level's description.
        roles: Role name -> model/effort override for this level. Partial: a role not
            named here falls back to the role-registry default at resolution time.
    """

    model_config = ConfigDict(extra="forbid")

    description: str
    roles: dict[str, ModelEffort] = Field(default_factory=dict)

    @field_validator("roles")
    @classmethod
    def _validate_role_keys(
        cls, value: dict[str, ModelEffort]
    ) -> dict[str, ModelEffort]:
        """Reject a role key that isn't in the role registry."""
        unknown = sorted(set(value) - _VALID_ROLE_NAMES)
        if unknown:
            raise ValueError(
                f"unknown role key(s) {unknown}, expected one of "
                f"{sorted(_VALID_ROLE_NAMES)}"
            )
        return value


class RoutingConfig(BaseModel):
    """The ``routing:`` block: a repo's per-issue model/effort routing table.

    Absence of a ``routing:`` block on :class:`RepoConfig` means routing is off and
    every role resolves the plain registry default; presence means every level is
    validated up front so a broken table is caught at config-load time, not mid-run.

    Attributes:
        default: The fallback level name, used on classification failure and as the
            role map for roles that run wider than one issue (slicer, resolver,
            staging-PR reviewer). Must name a key in :attr:`levels`.
        classifier: Optional model/effort override for the classifier role itself.
            Unset means the classifier's own registry default applies.
        levels: Level name -> :class:`RoutingLevel`, at least one entry.
    """

    model_config = ConfigDict(extra="forbid")

    default: str
    classifier: ModelEffort | None = None
    levels: dict[str, RoutingLevel]

    @field_validator("levels")
    @classmethod
    def _validate_levels(
        cls, value: dict[str, RoutingLevel]
    ) -> dict[str, RoutingLevel]:
        """Reject an empty table and any level name that isn't a lowercase slug."""
        if not value:
            raise ValueError("routing.levels must name at least one level")
        bad_names = sorted(
            name for name in value if not _LEVEL_SLUG_RE.fullmatch(name)
        )
        if bad_names:
            raise ValueError(
                f"level name(s) {bad_names} must be lowercase label-safe slugs "
                "(letters, digits, hyphens)"
            )
        return value

    @model_validator(mode="after")
    def _validate_default_exists(self) -> RoutingConfig:
        """Reject a ``default:`` that doesn't name one of the declared levels."""
        if self.default not in self.levels:
            raise ValueError(
                f"routing.default {self.default!r} must name an existing level"
            )
        return self


class RepoConfig(BaseModel):
    """Validated contents of a repo's ``.github/retinue.yml``.

    Attributes:
        staging_branch: Branch the retinue integrates work onto (default ``staging``).
        retry_cap: Max retries per unit of work before giving up (default ``3``).
        max_parallel: Optional cap on concurrent work; unset means no explicit cap.
        cron: Optional five-field cron cadence for scheduled runs.
        models: Role -> model-id overrides, keyed by the :class:`retinue.roles.Role`
            value (``slicer`` / ``implementer`` / ``resolver`` / ``reviewer``), e.g.
            ``{"implementer": "claude-opus-4-8"}``. Applied over the role registry's
            default model by :func:`retinue.roles.resolve_model`.
        secrets: Secrets and secret-references block.
        routing: Optional per-issue model/effort routing table (the ``routing:``
            block); absent means routing is off and every role resolves the plain
            registry default.
    """

    model_config = ConfigDict(extra="forbid")

    staging_branch: str = "staging"
    retry_cap: int = Field(default=3, ge=0)
    max_parallel: int | None = Field(default=None, gt=0)
    cron: str | None = None
    models: dict[str, str] = Field(default_factory=dict)
    secrets: SecretsConfig = Field(default_factory=SecretsConfig)
    routing: RoutingConfig | None = None

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
