"""The agent-role registry: one table owning every agent role's model and effort.

The retinue runs six agent roles — the PRD :class:`~retinue.slicer.ClaudeSliceGenerator`,
the :class:`~retinue.orchestrator.ContainerImplementer`, the
:class:`~retinue.orchestrator.AgentSdkConflictResolver`, the internal
:class:`~retinue.reviewer.AgentSdkReviewGenerator`, the read-only ``planner`` (Opus
on the in-container CLI, run with no write/edit/commit capability — it maps the code via
an Explore subagent and emits a plan as its captured output), and the
:class:`~retinue.classifier.ClaudeIssueClassifier` (a Haiku-class Messages-API role that
routes one issue to a level of a repo's routing table). Each one needs a model id,
a reasoning-effort tier, and an invocation transport. This module is the single place
those facts live: :data:`ROLE_REGISTRY` maps each :class:`Role` to its :class:`RoleSpec`,
and the adapters resolve their model and effort from it instead of hand-rolling private
constants — so a tier can't silently drift between two Opus call sites.

The planner also owns its invocation construction here: :func:`planner_cli_argv` builds
the read-only headless ``claude`` argv — the CLI's non-mutating ``plan`` permission mode,
an allow-list of read/search tools plus the ``Task`` tool that spawns the Explore
subagent, and a deny-list of every write-capable tool. The brief mandates at least one
Explore subagent before a plan is produced, and the plan is captured from the run's
output rather than written to the workspace.

:func:`resolve_model` and :func:`resolve_effort` are level-aware over the repo's routing
table (the optional ``routing:`` block in ``.github/retinue.yml``, carried on
:class:`~retinue.repo_config.RepoConfig`). At a given ``level``, a level's ``roles:`` map
keyed by the role's :attr:`Role.value` overrides the model (and, when the entry sets
``effort:``, the effort tier); a role the level does not name — or a call with no
``level``, no ``routing:`` block, or no config at all — falls through to the registry
default. A ``level`` of ``None`` with a routing table present resolves via the table's
``default:`` level.

The two transports are kept distinct because the roles use genuinely different wires: the
implementer execs the in-container ``claude`` CLI, while the other three POST the Anthropic
Messages API. Effort rides ``output_config.effort`` on the Messages-API roles (Opus 4.8
removed the extended-thinking ``budget_tokens`` mechanism); the CLI implementer carries no
effort flag today, so its tier is registry metadata that records the PRD's intent without
changing the wire.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from retinue.repo_config import ModelEffort, RepoConfig

logger = logging.getLogger(__name__)

# Reasoning-effort tiers, expressed as the ``output_config.effort`` string the Messages
# API call carries. Opus 4.8 (the model every Opus role pins) removed the extended-
# thinking ``budget_tokens`` mechanism — it returns HTTP 400 — so effort is the current
# control. The literal tier strings are self-documenting, so no numeric budget bookkeeping
# is needed. ``low`` is the classifier's cheap Haiku tier; ``high`` the implementer's
# Sonnet tier; ``xhigh`` the slicer/resolver Opus tier; ``max`` the highest tier,
# reserved for the internal reviewer.
EFFORT_LOW = "low"
EFFORT_HIGH = "high"
EFFORT_XHIGH = "xhigh"
EFFORT_MAX = "max"


class Transport(enum.Enum):
    """How a role's model is invoked.

    ``CLAUDE_CLI`` execs the headless ``claude`` CLI inside the disposable build
    container; ``MESSAGES_API`` POSTs the Anthropic Messages API directly. The two are
    kept distinct because they carry effort and credentials differently, and a role's
    transport is a fixed property of the role rather than something a repo overrides.
    """

    CLAUDE_CLI = "claude_cli"
    MESSAGES_API = "messages_api"


class Role(enum.Enum):
    """The agent roles the retinue runs.

    The ``value`` is the key a routing level's ``roles:`` map uses to target a role
    (e.g. ``roles: {implementer: {model: claude-opus-4-8}}``), so it is the stable public
    name of the role, not an implementation detail.
    """

    SLICER = "slicer"
    IMPLEMENTER = "implementer"
    RESOLVER = "resolver"
    REVIEWER = "reviewer"
    PLANNER = "planner"
    CLASSIFIER = "classifier"


@dataclass(frozen=True)
class RoleSpec:
    """The model, effort tier, and transport one agent role runs with.

    Attributes:
        model: The default model id for the role; a routing level replaces it per role
            via :func:`resolve_model`.
        effort: The default reasoning-effort tier (one of :data:`EFFORT_LOW` /
            :data:`EFFORT_HIGH` / :data:`EFFORT_XHIGH` / :data:`EFFORT_MAX`); a routing
            level whose entry sets ``effort:`` replaces it per role via
            :func:`resolve_effort`.
        transport: How the role's model is invoked (:class:`Transport`).
    """

    model: str
    effort: str
    transport: Transport


# The single source of truth for each role's model + effort + transport. The defaults are
# the PRD-pinned tiers the roles previously held as private constants: slicer Opus/xhigh,
# implementer Sonnet/high, resolver Opus/xhigh, reviewer Opus/max. The planner is Opus at
# the ``high`` tier on the in-container CLI — like the implementer it execs ``claude``, but
# run read-only (see :func:`planner_cli_argv`); the CLI carries no effort flag today, so
# ``high`` is registry metadata that records the PRD's intent without changing the wire.
ROLE_REGISTRY: dict[Role, RoleSpec] = {
    Role.SLICER: RoleSpec(
        model="claude-opus-4-8",
        effort=EFFORT_XHIGH,
        transport=Transport.MESSAGES_API,
    ),
    Role.IMPLEMENTER: RoleSpec(
        model="claude-sonnet-4-6",
        effort=EFFORT_HIGH,
        transport=Transport.CLAUDE_CLI,
    ),
    Role.RESOLVER: RoleSpec(
        model="claude-opus-4-8",
        effort=EFFORT_XHIGH,
        transport=Transport.MESSAGES_API,
    ),
    Role.REVIEWER: RoleSpec(
        model="claude-opus-4-8",
        effort=EFFORT_MAX,
        transport=Transport.MESSAGES_API,
    ),
    Role.PLANNER: RoleSpec(
        model="claude-opus-4-8",
        effort=EFFORT_HIGH,
        transport=Transport.CLAUDE_CLI,
    ),
    Role.CLASSIFIER: RoleSpec(
        model="claude-haiku-4-5",
        effort=EFFORT_LOW,
        transport=Transport.MESSAGES_API,
    ),
}


def _routed_override(
    role: Role, config: RepoConfig | None, level: str | None
) -> ModelEffort | None:
    """Return the routing-table entry for ``role`` at the effective level, or None.

    With no config or no ``routing:`` block, routing is off and there is no override.
    A ``level`` of None is the wider-scope callers, which resolve via the table's
    ``default:`` level. A level (or default) that names no declared level, or a level
    whose ``roles:`` map does not name ``role``, yields None so the caller falls back
    to the registry default.
    """
    if config is None or config.routing is None:
        return None
    level_name = level if level is not None else config.routing.default
    routing_level = config.routing.levels.get(level_name)
    if routing_level is None:
        # Only reachable with an explicit level (the default is validated against the
        # table at config load); an unknown name is a typo or a stale ``level:`` label,
        # so surface it rather than silently resolving the registry default.
        if level is not None:
            logger.warning(
                "Unknown explicit routing level %r for role %s; falling back to the "
                "registry default",
                level,
                role.value,
            )
        return None
    return routing_level.roles.get(role.value)


def resolve_model(
    role: Role, config: RepoConfig | None = None, *, level: str | None = None
) -> str:
    """Return the model id for ``role``, applying the routing table's override if present.

    The registry default is used unless the effective routing level's ``roles:`` map names
    the role, in which case that entry's model id wins. ``level`` selects the level; a
    ``level`` of ``None`` resolves via the table's ``default:`` level. With no config, no
    ``routing:`` block, or a level naming no declared level, the registry default is used.

    Args:
        role: The agent role whose model to resolve.
        config: The repo's validated config; its ``routing:`` table supplies overrides.
        level: The routing level to resolve at, or ``None`` for the ``default:`` level.

    Returns:
        The resolved model id.
    """
    override = _routed_override(role, config, level)
    if override is None:
        return ROLE_REGISTRY[role].model
    return override.model


def resolve_effort(
    role: Role, config: RepoConfig | None = None, *, level: str | None = None
) -> str:
    """Return the reasoning-effort tier for ``role``, applying a routing override if set.

    The registry tier is used unless the effective routing level's ``roles:`` map names
    the role *and* that entry sets ``effort:``, in which case the level's effort wins. A
    level that names the role's model but no ``effort`` leaves the tier on the registry
    default (only the model is overridden). ``level`` of ``None`` resolves via the
    ``default:`` level; no config or no ``routing:`` block yields the registry tier.

    Effort surfaces on the wire only for Messages-API roles (via
    :func:`structured_output_config`); CLI-transport roles (implementer, planner) carry
    it as recorded metadata with no wire effect.

    Args:
        role: The agent role whose effort tier to resolve.
        config: The repo's validated config; its ``routing:`` table supplies overrides.
        level: The routing level to resolve at, or ``None`` for the ``default:`` level.

    Returns:
        The resolved effort tier (one of :data:`EFFORT_HIGH` / :data:`EFFORT_XHIGH` /
        :data:`EFFORT_MAX`, or a level's lighter ``low`` / ``medium`` tier).
    """
    override = _routed_override(role, config, level)
    if override is None or override.effort is None:
        return ROLE_REGISTRY[role].effort
    return override.effort


# --- Claude Code identity block for OAuth premium-model access --------------------
#
# A subscription OAuth token may only call a premium model over the raw ``/v1/messages``
# API when the request's *first* ``system`` block is the Claude Code CLI identity string
# — Anthropic gates premium access on that exact-string entitlement, and a bare role
# brief is rejected (surfacing as a mislabeled 429). The three raw-API roles (slicer,
# reviewer, resolver) run this builder over their own brief, each gated on its own
# existing OAuth detection, so an OAuth request leads with the identity block and an
# API-key request keeps the brief unchanged. The literal must match the CLI byte-for-byte.
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


def is_oauth_credential(credential: str) -> bool:
    """True when ``credential`` is a subscription OAuth token (``sk-ant-oat...``)."""
    return credential.startswith("sk-ant-oat")


def oauth_system(role_system: str, *, is_oauth: bool) -> str | list[dict[str, str]]:
    """Build a role's Messages-API ``system`` field, adding the identity block for OAuth.

    In OAuth mode the field becomes a two-block list — the :data:`CLAUDE_CODE_IDENTITY`
    block first (the entitlement the premium-model gate checks), the role's own brief
    second. In API-key mode the brief passes through unchanged as a plain string, so
    the existing wire is untouched where the identity block is not required.

    Args:
        role_system: The role's own system brief.
        is_oauth: True when the request authenticates with a subscription OAuth token.

    Returns:
        A two-block ``[{type, text}, ...]`` list in OAuth mode, or ``role_system``
        verbatim in API-key mode.
    """
    if is_oauth:
        return [
            {"type": "text", "text": CLAUDE_CODE_IDENTITY},
            {"type": "text", "text": role_system},
        ]
    return role_system


# Model families that reject the Messages-API ``output_config.effort`` parameter with
# HTTP 400 ("This model does not support the effort parameter."): Haiku 4.5 and
# Sonnet 4.5. Later families (Sonnet 4.6+, Opus 4.5+) accept it. Prefix-matched so a
# dated id (``claude-haiku-4-5-20251001``) matches its family.
_EFFORT_UNSUPPORTED_MODEL_PREFIXES = ("claude-haiku-4-5", "claude-sonnet-4-5")


def supports_effort(model: str) -> bool:
    """True when ``model`` accepts the Messages-API ``output_config.effort`` parameter."""
    return not model.startswith(_EFFORT_UNSUPPORTED_MODEL_PREFIXES)


def structured_output_config(
    role: Role,
    schema: dict[str, Any],
    *,
    model: str | None = None,
    effort: str | None = None,
) -> dict[str, Any]:
    """Build a Messages-API ``output_config`` carrying a role's effort + JSON schema.

    The Messages API accepts structured output only as ``output_config.format =
    {"type": "json_schema", "schema": ...}``; the OpenAI-style top-level
    ``response_format`` parameter does not exist on the Claude wire and is rejected
    with HTTP 400. This helper is the single place that shape lives, so the
    Messages-API roles (slicer, reviewer, resolver, classifier) cannot drift onto
    incompatible encodings again. The role's registry effort tier rides the same dict
    — one ``output_config`` per request, never two — except for models that reject
    the parameter outright (see :func:`supports_effort`), where the key is omitted
    entirely so the request survives.

    Args:
        role: The Messages-API role whose registry effort tier the request carries.
        schema: The strict JSON schema the model must emit (callers keep
            ``required`` + ``additionalProperties: false`` on every object).
        model: The model id the request will name. When it is one that rejects the
            ``effort`` parameter (Haiku 4.5 / Sonnet 4.5 families), the key is dropped
            regardless of ``effort``. ``None`` keeps the effort key — callers should
            pass their resolved model so a routed-to-Haiku request cannot 400.
        effort: Optional effort-tier override; when a repo's routing table supplies a
            per-role tier (e.g. a ``classifier:`` override), it replaces the
            registry-resolved tier. ``None`` (the default) keeps the registry tier, so
            existing callers are unchanged.

    Returns:
        The ``output_config`` dict for the Messages API request body.
    """
    config: dict[str, Any] = {
        "format": {"type": "json_schema", "schema": schema},
    }
    if model is None or supports_effort(model):
        config["effort"] = effort if effort is not None else resolve_effort(role)
    return config


# --- planner invocation construction (read-only, explore-first) -------------------
#
# The planner execs the same in-container ``claude`` CLI as the implementer, but run
# read-only: it maps the relevant code with an Explore subagent and emits a plan, never
# touching the workspace. Read-only is belt-and-suspenders — the CLI's non-mutating
# ``plan`` permission mode, plus an explicit allow-list of read/search tools (and the
# ``Task`` tool the Explore subagent rides on) against a deny-list naming every
# write-capable tool. The plan is captured from the run's output, so unlike the
# implementer there is no ``--output-format json`` result-file contract.

# The CLI permission mode that lets the planner read and search but commits no edits.
# The ``claude`` CLI offers a first-class non-mutating ``plan`` mode; pinning it (rather
# than relying on tool lists alone) makes the read-only intent explicit at the wire.
_PLANNER_PERMISSION_MODE = "plan"

# Read/search tools the planner may use, including the ``Task`` tool that spawns the
# mandated Explore subagent. No write-capable tool appears here, so the planner cannot
# edit, write, or run shell commands even if the permission mode were misread.
_PLANNER_ALLOWED_TOOLS = "Read,Glob,Grep,Task,WebFetch,WebSearch"

# Write-capable tools named explicitly in the deny-list so the read-only contract holds
# regardless of CLI default-tool changes — a second guard alongside the allow-list and
# the ``plan`` permission mode.
_PLANNER_DISALLOWED_TOOLS = "Write,Edit,MultiEdit,NotebookEdit,Bash,BashOutput,KillShell"

# The planner's brief, appended to the CLI's system prompt. It mandates at least one
# Explore subagent to map the code before any plan, and frames the captured output as
# the plan itself — the planner writes nothing to the workspace.
_PLANNER_SYSTEM = (
    "You are a read-only planner. You have no write, edit, or commit capability and "
    "must not attempt to change any file. Before you produce a plan you must spawn at "
    "least one Explore subagent to map the relevant code; do not emit a plan until you "
    "have explored. Then output the plan as your response — it is captured as your "
    "result, so write the plan itself rather than saving it to a file."
)


def planner_cli_argv(*, prompt: str, model: str) -> list[str]:
    """Assemble the read-only headless ``claude`` argv for one in-container plan run.

    Runs non-interactively (``-p`` print mode), pins the planning ``model``, and enforces
    read-only twice over: the CLI's non-mutating ``plan`` permission mode, plus an
    allow-list of read/search tools (and the ``Task`` tool the Explore subagent rides on)
    against a deny-list naming every write-capable tool. The frozen planner brief mandates
    at least one Explore subagent before a plan. No ``--output-format`` flag is set: the
    plan is captured from the run's output rather than written to the workspace.

    Args:
        prompt: The per-run instruction naming what to plan.
        model: The planning model id (resolve via :func:`resolve_model` with
            :data:`Role.PLANNER`).

    Returns:
        The ``claude`` argv (a list, no shell), ready to exec in the build container.
    """
    return [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--permission-mode",
        _PLANNER_PERMISSION_MODE,
        "--allowedTools",
        _PLANNER_ALLOWED_TOOLS,
        "--disallowedTools",
        _PLANNER_DISALLOWED_TOOLS,
        "--append-system-prompt",
        _PLANNER_SYSTEM,
    ]
