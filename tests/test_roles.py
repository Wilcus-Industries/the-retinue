"""Tests for the agent-role registry.

The registry owns the four agent roles (slicer, implementer, conflict resolver,
internal reviewer) and resolves each one's model id and reasoning-effort tier from a
single table. A ``repo_config.models`` entry overrides a role's model; the effort tier
is the registry's alone. No network, Agent SDK, or gh is touched — this is a pure
lookup over the table.
"""

from __future__ import annotations

import pytest

from retinue.repo_config import RepoConfig
from retinue.roles import (
    EFFORT_HIGH,
    EFFORT_MAX,
    EFFORT_XHIGH,
    ROLE_REGISTRY,
    Role,
    Transport,
    resolve_effort,
    resolve_model,
)


def test_every_role_has_a_registry_entry() -> None:
    """The registry covers exactly the four agent roles, no more, no fewer."""
    assert set(ROLE_REGISTRY) == set(Role)


@pytest.mark.parametrize(
    ("role", "model", "effort", "transport"),
    [
        (Role.SLICER, "claude-opus-4-8", EFFORT_XHIGH, Transport.MESSAGES_API),
        (Role.IMPLEMENTER, "claude-sonnet-4-6", EFFORT_HIGH, Transport.CLAUDE_CLI),
        (Role.RESOLVER, "claude-opus-4-8", EFFORT_XHIGH, Transport.MESSAGES_API),
        (Role.REVIEWER, "claude-opus-4-8", EFFORT_MAX, Transport.MESSAGES_API),
    ],
)
def test_default_tiers_match_the_prd(
    role: Role, model: str, effort: str, transport: Transport
) -> None:
    """Each role's default model, effort tier, and transport are the PRD-pinned values.

    These are the defaults the four existing roles already used as hand-rolled
    constants; the registry must preserve them so consolidation drifts no behavior.
    """
    spec = ROLE_REGISTRY[role]
    assert spec.model == model
    assert spec.effort == effort
    assert spec.transport is transport


def test_resolve_model_returns_the_registry_default_without_override() -> None:
    """With no ``models`` override, the resolved model is the registry default."""
    config = RepoConfig()
    for role in Role:
        assert resolve_model(role, config) == ROLE_REGISTRY[role].model


def test_resolve_model_applies_a_repo_config_override() -> None:
    """A ``repo_config.models`` entry keyed by the role overrides that role's model."""
    config = RepoConfig(models={Role.IMPLEMENTER.value: "claude-opus-4-8"})
    assert resolve_model(Role.IMPLEMENTER, config) == "claude-opus-4-8"
    # An unrelated role is untouched by another role's override.
    assert resolve_model(Role.SLICER, config) == ROLE_REGISTRY[Role.SLICER].model


def test_resolve_model_ignores_an_unrelated_override_key() -> None:
    """A ``models`` key that names no known role leaves every role on its default."""
    config = RepoConfig(models={"planner": "claude-opus-4"})
    for role in Role:
        assert resolve_model(role, config) == ROLE_REGISTRY[role].model


def test_resolve_effort_is_registry_only_and_ignores_overrides() -> None:
    """Effort is the registry's alone — ``models`` overrides the model, never the tier."""
    config = RepoConfig(models={Role.REVIEWER.value: "claude-opus-4-8"})
    assert resolve_effort(Role.REVIEWER) == EFFORT_MAX
    assert resolve_effort(Role.REVIEWER, config) == EFFORT_MAX
