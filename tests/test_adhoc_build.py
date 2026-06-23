"""Tests for the ad-hoc build primitive (issue #29).

The flow is plan -> materialize -> implement -> done-check -> push, all in one
disposable container. Every collaborator is faked: a fake planner records the issue it
was asked to plan and returns a canned plan string, a fake implementer records the issue
it was asked to build, the done-check runs against the faked container reused from the
done-check tests, and the container records every command so the order (clone -> branch
off staging -> plan -> materialize-plan-file -> implement -> done-check -> push-on-green)
is assertable. No Agent SDK, no Docker, no gh, no network.

A green done-check pushes ``issue-<N>`` (cut off ``config.staging_branch``); a red
done-check pushes nothing.
"""

from __future__ import annotations

import base64

import pytest

from retinue.adhoc_build import (
    PLAN_FILE,
    AdhocBuildResult,
    AdhocIssue,
    ContainerPlanner,
    PlanError,
    Planner,
    _materialize_plan_command,
    _plan_prompt,
    _slice_for_issue,
    build_adhoc_issue,
)
from retinue.container import Container, RunResult
from retinue.done_check import DoneCheckReport
from retinue.orchestrator import Implementer, Slice, _implement_prompt
from retinue.repo_config import RepoConfig
from retinue.roles import Role, planner_cli_argv, resolve_model
from tests.test_done_check import (
    CLAUDE_MD,
    FakeAuth,
    FakeRuntime,
    _resolver,
    _sink,
)

PLAN_TEXT = "1. write a failing test\n2. make it pass"


class FakePlanner:
    """Records the issue it was asked to plan; returns a canned plan, marks the container."""

    def __init__(self, plan: str = PLAN_TEXT) -> None:
        self.planned: list[AdhocIssue] = []
        self._plan = plan

    async def plan(self, issue: AdhocIssue, *, container: Container) -> str:
        self.planned.append(issue)
        await container.run_command(["plan", issue.branch])
        return self._plan

    def auth_env(self) -> dict[str, str]:
        return {}


class FakeImplementer:
    """Records the slice it was asked to build; marks the container with an implement event."""

    def __init__(self) -> None:
        self.built: list[Slice] = []
        self.plan_paths: list[str | None] = []

    async def implement(
        self, slice_: Slice, *, container: Container, plan_path: str | None = None
    ) -> None:
        self.built.append(slice_)
        self.plan_paths.append(plan_path)
        await container.run_command(["implement", slice_.branch])

    def auth_env(self) -> dict[str, str]:
        return {}


def _issue(issue_number: int = 29) -> AdhocIssue:
    return AdhocIssue(repo_full_name="owner/repo", issue_number=issue_number)


async def _build(
    *,
    runtime: FakeRuntime,
    planner: Planner | None = None,
    implementer: Implementer | None = None,
    config: RepoConfig | None = None,
    captured: list[DoneCheckReport] | None = None,
    issue: AdhocIssue | None = None,
) -> AdhocBuildResult:
    return await build_adhoc_issue(
        issue or _issue(),
        config or RepoConfig(),
        CLAUDE_MD,
        planner=planner or FakePlanner(),
        implementer=implementer or FakeImplementer(),
        auth=FakeAuth(),
        runtime=runtime,
        resolve_secret=_resolver({}),
        report=_sink(captured if captured is not None else []),
    )


# --- branch naming ---------------------------------------------------------------


def test_adhoc_issue_branch_name() -> None:
    """The branch an ad-hoc issue is built on is ``issue-<N>``."""
    assert _issue(29).branch == "issue-29"
    assert _issue(7).branch == "issue-7"


# --- order: plan then implement, both in one container ----------------------------


@pytest.mark.asyncio
async def test_runs_planner_then_implementer_in_one_container() -> None:
    """The primitive plans, then implements, both inside a single started container."""
    planner = FakePlanner()
    implementer = FakeImplementer()
    runtime = FakeRuntime()

    await _build(runtime=runtime, planner=planner, implementer=implementer)

    # Exactly one container was started for the whole build.
    assert runtime.log.count("start:" + runtime.log[0].split(":", 1)[1]) >= 1
    assert sum(1 for event in runtime.log if event.startswith("start:")) == 1
    # The planner ran before the implementer.
    plan_idx = runtime.log.index("run:plan issue-29")
    impl_idx = runtime.log.index("run:implement issue-29")
    assert plan_idx < impl_idx
    # Each was asked about exactly this issue.
    assert planner.planned == [_issue()]
    assert implementer.built == [Slice("owner/repo", 29, 29)]


# --- the plan is materialized into the file the implementer reads -----------------


@pytest.mark.asyncio
async def test_plan_is_materialized_into_the_plan_file_before_implementing() -> None:
    """The captured plan is written to PLAN_FILE in-container before the implementer runs."""
    planner = FakePlanner(plan=PLAN_TEXT)
    runtime = FakeRuntime()

    await _build(runtime=runtime, planner=planner)

    expected_blob = base64.b64encode(PLAN_TEXT.encode()).decode()
    materialize = "run:" + " ".join(_materialize_plan_command(PLAN_TEXT))
    assert materialize in runtime.log
    # The blob carries the exact plan bytes.
    assert expected_blob in materialize
    # Materialization happens after the plan is captured and before the implementer runs.
    mat_idx = runtime.log.index(materialize)
    plan_idx = runtime.log.index("run:plan issue-29")
    impl_idx = runtime.log.index("run:implement issue-29")
    assert plan_idx < mat_idx < impl_idx


def test_plan_prompt_names_the_plan_file_and_issue() -> None:
    """The planner prompt points at the issue and the materialized plan file."""
    prompt = _plan_prompt(_issue(29))
    assert "#29" in prompt
    assert PLAN_FILE in prompt


# --- the implementer is pointed at the plan file it must consume ------------------


@pytest.mark.asyncio
async def test_implementer_is_pointed_at_the_plan_file() -> None:
    """The ad-hoc lane tells the implementer to read PLAN_FILE, not just write it.

    Closes #29 AC2: consumption, not mere materialization — the implementer the ad-hoc
    lane runs receives ``PLAN_FILE`` as its plan path, so it is instructed to read the
    plan the planner wrote before building.
    """
    implementer = FakeImplementer()
    runtime = FakeRuntime()

    await _build(runtime=runtime, implementer=implementer)

    assert implementer.plan_paths == [PLAN_FILE]


def test_implement_prompt_with_plan_path_instructs_reading_the_plan() -> None:
    """A plan_path makes the implement prompt point the subagent at PLAN_FILE first."""
    prompt = _implement_prompt(_slice_for_issue(_issue(29)), plan_path=PLAN_FILE)

    assert PLAN_FILE in prompt
    assert "read" in prompt.lower()


# --- branch cut off staging -------------------------------------------------------


@pytest.mark.asyncio
async def test_issue_branch_is_cut_off_the_staging_branch() -> None:
    """The implementer's ``issue-<N>`` branch is created off ``config.staging_branch``."""
    runtime = FakeRuntime()
    config = RepoConfig(staging_branch="trunk")

    await _build(runtime=runtime, config=config)

    # The branch is checked out off origin/<staging_branch>.
    assert "run:git checkout -B issue-29 origin/trunk" in runtime.log
    assert "run:git fetch origin trunk" in runtime.log


# --- green pushes, red pushes nothing ---------------------------------------------


@pytest.mark.asyncio
async def test_green_done_check_pushes_the_issue_branch() -> None:
    """A green done-check pushes ``issue-<N>`` to origin and reports passed."""
    runtime = FakeRuntime()
    captured: list[DoneCheckReport] = []

    result = await _build(runtime=runtime, captured=captured)

    assert result.passed is True
    assert "run:git push origin issue-29" in runtime.log
    assert [r.passed for r in captured] == [True]


@pytest.mark.asyncio
async def test_red_done_check_pushes_nothing() -> None:
    """A red done-check pushes nothing and reports failed."""
    runtime = FakeRuntime(results={"uv": RunResult(exit_code=1, stderr="boom")})
    captured: list[DoneCheckReport] = []

    result = await _build(runtime=runtime, captured=captured)

    assert result.passed is False
    assert not any(event.startswith("run:git push") for event in runtime.log)
    assert [r.passed for r in captured] == [False]


# --- the container is always destroyed --------------------------------------------


@pytest.mark.asyncio
async def test_container_is_destroyed_on_every_path() -> None:
    """The disposable container is torn down even when the done-check is red."""
    runtime = FakeRuntime(results={"uv": RunResult(exit_code=1, stderr="boom")})

    await _build(runtime=runtime)

    assert runtime.container is not None
    assert runtime.container.destroyed is True


@pytest.mark.asyncio
async def test_planner_and_implementer_credentials_are_injected_at_start() -> None:
    """Both the planner's and the implementer's auth env ride the container at start."""

    class CredPlanner(FakePlanner):
        def auth_env(self) -> dict[str, str]:
            return {"PLANNER_TOKEN": "p"}

    class CredImplementer(FakeImplementer):
        def auth_env(self) -> dict[str, str]:
            return {"IMPLEMENTER_TOKEN": "i"}

    runtime = FakeRuntime()
    await _build(runtime=runtime, planner=CredPlanner(), implementer=CredImplementer())

    assert runtime.started_env is not None
    assert runtime.started_env["PLANNER_TOKEN"] == "p"
    assert runtime.started_env["IMPLEMENTER_TOKEN"] == "i"


# --- ContainerPlanner: the real planner adapter -----------------------------------


class ScriptedContainer:
    """A container that returns a scripted RunResult for the planner exec."""

    def __init__(self, result: RunResult) -> None:
        self._result = result
        self.commands: list[list[str]] = []

    async def run_command(self, command: list[str]) -> RunResult:
        self.commands.append(command)
        return self._result

    async def destroy(self) -> None:  # pragma: no cover - unused in these tests
        pass


@pytest.mark.asyncio
async def test_container_planner_captures_the_plan_from_stdout() -> None:
    """The real planner execs the read-only argv and returns the run's captured output."""
    container = ScriptedContainer(RunResult(exit_code=0, stdout="the plan"))
    planner = ContainerPlanner(credential="cred", auth_mode="api_key", model="m")

    plan = await planner.plan(_issue(29), container=container)

    assert plan == "the plan"
    assert container.commands == [
        planner_cli_argv(prompt=_plan_prompt(_issue(29)), model="m")
    ]


@pytest.mark.asyncio
async def test_container_planner_raises_on_a_nonzero_exit() -> None:
    """A planner exec that exits non-zero raises rather than returning a half plan."""
    container = ScriptedContainer(RunResult(exit_code=2, stderr="kaboom"))
    planner = ContainerPlanner(credential="cred")

    with pytest.raises(PlanError):
        await planner.plan(_issue(29), container=container)


def test_container_planner_defaults_to_the_planner_role_model() -> None:
    """The planner's model defaults to the planner role registry entry."""
    planner = ContainerPlanner(credential="cred")
    assert planner.model == resolve_model(Role.PLANNER)


def test_container_planner_routes_subscription_credential() -> None:
    """A subscription auth mode threads the credential as the OAuth env var."""
    planner = ContainerPlanner(credential="tok", auth_mode="subscription")
    assert planner.auth_env() == {"CLAUDE_CODE_OAUTH_TOKEN": "tok"}
