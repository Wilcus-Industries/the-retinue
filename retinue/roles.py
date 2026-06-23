"""The agent-role registry: one table owning every agent role's model and effort.

The retinue runs four agent roles — the PRD :class:`~retinue.slicer.ClaudeSliceGenerator`,
the :class:`~retinue.orchestrator.ContainerImplementer`, the
:class:`~retinue.orchestrator.AgentSdkConflictResolver`, and the internal
:class:`~retinue.reviewer.AgentSdkReviewGenerator`. Each one needs a model id, a
reasoning-effort tier, and an invocation transport. This module is the single place
those facts live: :data:`ROLE_REGISTRY` maps each :class:`Role` to its :class:`RoleSpec`,
and the four adapters resolve their model and effort from it instead of hand-rolling
private constants — so a tier can't silently drift between two Opus call sites.

:func:`resolve_model` applies a repo's ``models`` override (the optional ``role ->
model-id`` block in ``.github/retinue.yml``, carried on :class:`~retinue.repo_config.RepoConfig`)
on top of the registry default, keyed by the role's :attr:`Role.value`. Effort is the
registry's alone — :func:`resolve_effort` never reads the override, so a repo can swap a
role's model without disturbing the rigor tier the PRD pinned.

The two transports are kept distinct because the roles use genuinely different wires: the
implementer execs the in-container ``claude`` CLI, while the other three POST the Anthropic
Messages API. Effort rides ``output_config.effort`` on the Messages-API roles (Opus 4.8
removed the extended-thinking ``budget_tokens`` mechanism); the CLI implementer carries no
effort flag today, so its tier is registry metadata that records the PRD's intent without
changing the wire.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from retinue.repo_config import RepoConfig

# Reasoning-effort tiers, expressed as the ``output_config.effort`` string the Messages
# API call carries. Opus 4.8 (the model every Opus role pins) removed the extended-
# thinking ``budget_tokens`` mechanism — it returns HTTP 400 — so effort is the current
# control. The literal tier strings are self-documenting, so no numeric budget bookkeeping
# is needed. ``high`` is the implementer's Sonnet tier; ``xhigh`` the slicer/resolver
# Opus tier; ``max`` the highest tier, reserved for the internal reviewer.
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
    """The four agent roles the retinue runs.

    The ``value`` is the key a repo's ``models`` override block uses to target a role
    (e.g. ``models: {implementer: claude-opus-4-8}``), so it is the stable public name
    of the role, not an implementation detail.
    """

    SLICER = "slicer"
    IMPLEMENTER = "implementer"
    RESOLVER = "resolver"
    REVIEWER = "reviewer"


@dataclass(frozen=True)
class RoleSpec:
    """The model, effort tier, and transport one agent role runs with.

    Attributes:
        model: The default model id for the role; a repo's ``models`` override replaces
            it per :func:`resolve_model`.
        effort: The reasoning-effort tier (one of :data:`EFFORT_HIGH` / :data:`EFFORT_XHIGH`
            / :data:`EFFORT_MAX`); registry-owned, never overridden.
        transport: How the role's model is invoked (:class:`Transport`).
    """

    model: str
    effort: str
    transport: Transport


# The single source of truth for each role's model + effort + transport. The defaults are
# the PRD-pinned tiers the four roles previously held as private constants: slicer
# Opus/xhigh, implementer Sonnet/high, resolver Opus/xhigh, reviewer Opus/max.
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
}


def resolve_model(role: Role, config: RepoConfig | None = None) -> str:
    """Return the model id for ``role``, applying a repo's ``models`` override if present.

    The registry default is used unless ``config.models`` carries an entry keyed by the
    role's :attr:`Role.value`, in which case that model id wins. ``config`` is optional so
    a call site with no repo config (or a fake) still resolves the default.

    Args:
        role: The agent role whose model to resolve.
        config: The repo's validated config; its ``models`` block overrides the default.

    Returns:
        The resolved model id.
    """
    default = ROLE_REGISTRY[role].model
    if config is None:
        return default
    return config.models.get(role.value, default)


def resolve_effort(role: Role, config: RepoConfig | None = None) -> str:
    """Return the reasoning-effort tier for ``role`` from the registry.

    Effort is registry-owned: a repo's ``models`` block overrides only the model, never
    the rigor tier the PRD pinned, so ``config`` is accepted for call-site symmetry with
    :func:`resolve_model` but does not change the result.

    Args:
        role: The agent role whose effort tier to resolve.
        config: Accepted for symmetry; the effort tier ignores it.

    Returns:
        The role's effort tier (one of :data:`EFFORT_HIGH` / :data:`EFFORT_XHIGH` /
        :data:`EFFORT_MAX`).
    """
    return ROLE_REGISTRY[role].effort
