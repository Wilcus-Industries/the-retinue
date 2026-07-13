"""Tests for the agent-role registry.

The registry owns the five agent roles (slicer, implementer, conflict resolver,
internal reviewer, and read-only planner) and resolves each one's model id and
reasoning-effort tier from a single table. A ``repo_config.models`` entry overrides a
role's model; the effort tier is the registry's alone. The planner additionally owns a
read-only CLI invocation builder, asserted here for its no-write/explore-first contract.
No network, Agent SDK, or gh is touched — this is a pure lookup over the table plus
argv string assembly.
"""

from __future__ import annotations

import pytest

from retinue.repo_config import RepoConfig
from retinue.roles import (
    CLAUDE_CODE_IDENTITY,
    EFFORT_HIGH,
    EFFORT_MAX,
    EFFORT_XHIGH,
    ROLE_REGISTRY,
    Role,
    Transport,
    oauth_system,
    planner_cli_argv,
    resolve_effort,
    resolve_model,
    structured_output_config,
)


def test_every_role_has_a_registry_entry() -> None:
    """The registry covers exactly the agent roles, no more, no fewer."""
    assert set(ROLE_REGISTRY) == set(Role)


@pytest.mark.parametrize(
    ("role", "model", "effort", "transport"),
    [
        (Role.SLICER, "claude-opus-4-8", EFFORT_XHIGH, Transport.MESSAGES_API),
        (Role.IMPLEMENTER, "claude-sonnet-4-6", EFFORT_HIGH, Transport.CLAUDE_CLI),
        (Role.RESOLVER, "claude-opus-4-8", EFFORT_XHIGH, Transport.MESSAGES_API),
        (Role.REVIEWER, "claude-opus-4-8", EFFORT_MAX, Transport.MESSAGES_API),
        (Role.PLANNER, "claude-opus-4-8", EFFORT_HIGH, Transport.CLAUDE_CLI),
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
    config = RepoConfig(models={"nonexistent-role": "claude-opus-4"})
    for role in Role:
        assert resolve_model(role, config) == ROLE_REGISTRY[role].model


def test_resolve_effort_is_registry_only_and_ignores_overrides() -> None:
    """Effort is the registry's alone — ``models`` overrides the model, never the tier."""
    config = RepoConfig(models={Role.REVIEWER.value: "claude-opus-4-8"})
    assert resolve_effort(Role.REVIEWER) == EFFORT_MAX
    assert resolve_effort(Role.REVIEWER, config) == EFFORT_MAX


def test_planner_role_resolves_opus_high_on_the_cli_transport() -> None:
    """The planner is Opus at the ``high`` tier, invoked on the in-container CLI."""
    spec = ROLE_REGISTRY[Role.PLANNER]
    assert resolve_model(Role.PLANNER) == "claude-opus-4-8"
    assert resolve_effort(Role.PLANNER) == EFFORT_HIGH
    assert spec.transport is Transport.CLAUDE_CLI


def test_planner_argv_runs_headless_with_the_pinned_model() -> None:
    """The argv runs the headless CLI in print mode and pins the planning model."""
    argv = planner_cli_argv(prompt="plan issue #5", model="m")

    assert argv[0] == "claude"
    assert argv[1:3] == ["-p", "plan issue #5"]
    assert "--model" in argv and "m" in argv


def test_planner_argv_runs_read_only_with_no_write_capability() -> None:
    """The planner is granted read/search tools only — no write/edit/commit ever.

    Read-only is enforced two ways that must agree: the permission mode is the CLI's
    non-mutating ``plan`` mode, and no write-capable tool (Write/Edit/Bash) appears in
    the allow-list while each is named in the deny-list.
    """
    argv = planner_cli_argv(prompt="p", model="m")

    assert "--permission-mode" in argv
    mode = argv[argv.index("--permission-mode") + 1]
    assert mode == "plan"

    allowed = argv[argv.index("--allowedTools") + 1]
    disallowed = argv[argv.index("--disallowedTools") + 1]
    for write_tool in ("Write", "Edit", "Bash"):
        assert write_tool not in allowed
        assert write_tool in disallowed


def test_planner_argv_permits_the_explore_subagent() -> None:
    """The Task tool that spawns the Explore subagent is in the allow-list."""
    argv = planner_cli_argv(prompt="p", model="m")

    allowed = argv[argv.index("--allowedTools") + 1]
    assert "Task" in allowed
    for read_tool in ("Read", "Glob", "Grep"):
        assert read_tool in allowed


def test_planner_instruction_requires_an_explore_subagent_before_a_plan() -> None:
    """The appended brief mandates ≥1 Explore subagent before any plan is emitted."""
    argv = planner_cli_argv(prompt="p", model="m")
    system = argv[argv.index("--append-system-prompt") + 1]

    assert "Explore" in system
    lowered = system.lower()
    assert "at least one" in lowered
    assert "before" in lowered and "plan" in lowered


def test_planner_argv_writes_nothing_to_the_workspace() -> None:
    """The plan is the captured stdout — the argv asks for no file output of its own.

    The planner has no ``--output-format json`` result-file contract and grants no
    write tool, so nothing is written to the workspace; the orchestrator captures the
    plan from the run's output.
    """
    argv = planner_cli_argv(prompt="p", model="m")

    assert "--output-format" not in argv
    disallowed = argv[argv.index("--disallowedTools") + 1]
    assert "Write" in disallowed


def test_oauth_system_prepends_identity_block_in_oauth_mode() -> None:
    """OAuth mode turns the system field into a two-block list, identity block first.

    A subscription OAuth token may only reach a premium model over the raw Messages API
    when the request's leading ``system`` block is the Claude Code identity string; the
    role's own brief follows it as the second block.
    """
    assert oauth_system("role prompt", is_oauth=True) == [
        {"type": "text", "text": CLAUDE_CODE_IDENTITY},
        {"type": "text", "text": "role prompt"},
    ]


def test_oauth_system_passes_plain_string_through_in_api_key_mode() -> None:
    """api_key mode leaves the role prompt as the unchanged plain string (never a list)."""
    result = oauth_system("role prompt", is_oauth=False)
    assert result == "role prompt"
    assert not isinstance(result, list)


def test_structured_output_config_carries_effort_and_json_schema_format() -> None:
    """The shared helper emits the canonical ``output_config`` wire shape.

    The Messages API accepts structured output only as
    ``output_config.format = {type: json_schema, schema: ...}`` — the OpenAI-style
    top-level ``response_format`` is not a Claude API parameter and 400s. The helper
    is the single place that shape lives, with the role's registry effort riding the
    same dict so a role sends exactly one output_config.
    """
    schema = {"type": "object", "required": [], "additionalProperties": False}

    config = structured_output_config(Role.REVIEWER, schema)

    assert config == {
        "effort": EFFORT_MAX,
        "format": {"type": "json_schema", "schema": schema},
    }


@pytest.mark.parametrize(
    ("role", "effort"),
    [
        (Role.SLICER, EFFORT_XHIGH),
        (Role.RESOLVER, EFFORT_XHIGH),
        (Role.REVIEWER, EFFORT_MAX),
    ],
)
def test_structured_output_config_resolves_effort_per_role(
    role: Role, effort: str
) -> None:
    """Each Messages-API role's effort tier comes from the registry, not the caller."""
    assert structured_output_config(role, {"type": "object"})["effort"] == effort


def test_claude_code_identity_is_the_exact_cli_string() -> None:
    """The identity block is the exact Claude Code CLI system string.

    The entitlement match is exact-string, so this pins the literal the CLI sends —
    any drift would silently lose premium-model access for the OAuth roles.
    """
    assert (
        CLAUDE_CODE_IDENTITY
        == "You are Claude Code, Anthropic's official CLI for Claude."
    )
