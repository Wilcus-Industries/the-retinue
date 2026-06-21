"""Tests for the disposable-container done-check orchestration (issue #4).

The orchestration is auth -> clone -> inject -> run -> report -> teardown. Every
collaborator is faked: a fake :class:`InstallationAuth` mints a canned token, a fake
:class:`ContainerRuntime` records the command order and whether the container was
destroyed, a dict-backed secret resolver stands in for the secret store, and a list
sink captures what was reported. No Docker, no network.
"""

from __future__ import annotations

import pytest

from retinue.container import Container, RunResult
from retinue.done_check import (
    DEFAULT_IMAGE,
    DoneCheckError,
    DoneCheckReport,
    MissingSecretError,
    ReportSink,
    SecretResolver,
    parse_done_check,
    run_done_check,
)
from retinue.github_app import InstallationToken
from retinue.repo_config import RepoConfig, SecretsConfig

CLAUDE_MD = """# CLAUDE.md

## Definition of done

```
uv run pytest
uv run ruff check .
```
"""


class FakeAuth:
    """Mints a canned installation token and records that auth was called."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def installation_token(self, repo_full_name: str) -> InstallationToken:
        self.calls.append(repo_full_name)
        return InstallationToken(
            token="ghs_faketoken",
            clone_url=f"https://x-access-token:ghs_faketoken@github.com/{repo_full_name}.git",
        )


class FakeContainer:
    """In-memory container that scripts per-command results and records teardown.

    ``results`` maps the first argv token (e.g. "git", "uv") to the
    :class:`RunResult` to return; an unscripted command returns success. ``log``
    appends each event so a test can assert clone-before-run and that destroy ran.
    """

    def __init__(self, log: list[str], results: dict[str, RunResult]) -> None:
        self._log = log
        self._results = results
        self.destroyed = False

    async def run_command(self, command: list[str]) -> RunResult:
        self._log.append("run:" + " ".join(command))
        return self._results.get(command[0], RunResult(exit_code=0))

    async def destroy(self) -> None:
        self.destroyed = True
        self._log.append("destroy")


class FakeRuntime:
    """Spawns one :class:`FakeContainer`, recording the start event and injected env."""

    def __init__(self, results: dict[str, RunResult] | None = None) -> None:
        self.log: list[str] = []
        self.started_env: dict[str, str] | None = None
        self.container: FakeContainer | None = None
        self._results = results or {}

    async def start(self, *, image: str, env: dict[str, str]) -> Container:
        self.log.append(f"start:{image}")
        self.started_env = env
        self.container = FakeContainer(self.log, self._results)
        return self.container


def _resolver(known: dict[str, str]) -> SecretResolver:
    async def resolve(name: str) -> str | None:
        return known.get(name)

    return resolve


def _sink(captured: list[DoneCheckReport]) -> ReportSink:
    async def report(result: DoneCheckReport) -> None:
        captured.append(result)

    return report


def _config(secrets: SecretsConfig | None = None) -> RepoConfig:
    return RepoConfig(secrets=secrets or SecretsConfig())


# --- parse_done_check ------------------------------------------------------------


def test_parse_done_check_reads_commands_under_heading() -> None:
    """The done-check is the fenced block under 'Definition of done', one cmd/line."""
    commands = parse_done_check(CLAUDE_MD)
    assert commands == [["uv", "run", "pytest"], ["uv", "run", "ruff", "check", "."]]


def test_parse_done_check_raises_without_block() -> None:
    """A CLAUDE.md with no done-check block is an error, not a silent empty list."""
    with pytest.raises(DoneCheckError):
        parse_done_check("# CLAUDE.md\n\nNo commands here.\n")


# --- happy path: ordering + report + teardown ------------------------------------


@pytest.mark.asyncio
async def test_orchestration_runs_in_order_reports_and_tears_down() -> None:
    """Auth -> clone -> run -> report -> teardown happens in order on the happy path."""
    auth = FakeAuth()
    runtime = FakeRuntime()
    captured: list[DoneCheckReport] = []

    result = await run_done_check(
        "owner/repo",
        _config(),
        CLAUDE_MD,
        auth=auth,
        runtime=runtime,
        resolve_secret=_resolver({}),
        report=_sink(captured),
    )

    assert auth.calls == ["owner/repo"]
    # Start (auth done) -> clone -> the two done-check commands -> destroy, in order.
    assert runtime.log == [
        f"start:{DEFAULT_IMAGE}",
        "run:git clone https://x-access-token:ghs_faketoken@github.com/owner/repo.git .",
        "run:uv run pytest",
        "run:uv run ruff check .",
        "destroy",
    ]
    assert runtime.container is not None and runtime.container.destroyed
    assert captured == [result]
    assert result.passed and not result.escalated


@pytest.mark.asyncio
async def test_clone_runs_before_done_check_commands() -> None:
    """The clone must precede every done-check command in the recorded order."""
    runtime = FakeRuntime()
    await run_done_check(
        "owner/repo",
        _config(),
        CLAUDE_MD,
        auth=FakeAuth(),
        runtime=runtime,
        resolve_secret=_resolver({}),
        report=_sink([]),
    )
    clone_index = next(i for i, e in enumerate(runtime.log) if "git clone" in e)
    pytest_index = next(i for i, e in enumerate(runtime.log) if "uv run pytest" in e)
    assert clone_index < pytest_index


# --- secret injection ------------------------------------------------------------


@pytest.mark.asyncio
async def test_secrets_are_injected_into_container_env() -> None:
    """Resolved inline secrets and refs land in the container's env at start."""
    runtime = FakeRuntime()
    config = _config(
        SecretsConfig(
            values={"OPENAI_API_KEY": "${{ secrets.OPENAI_API_KEY }}"},
            refs=["vault://team/token"],
        )
    )
    resolver = _resolver(
        {"OPENAI_API_KEY": "sk-real", "vault://team/token": "vault-secret"}
    )

    await run_done_check(
        "owner/repo",
        config,
        CLAUDE_MD,
        auth=FakeAuth(),
        runtime=runtime,
        resolve_secret=resolver,
        report=_sink([]),
    )

    assert runtime.started_env == {
        "OPENAI_API_KEY": "sk-real",
        "vault://team/token": "vault-secret",
    }


@pytest.mark.asyncio
async def test_missing_secret_escalates_and_never_starts_container() -> None:
    """A missing required secret escalates before any container is started."""
    runtime = FakeRuntime()
    captured: list[DoneCheckReport] = []
    config = _config(SecretsConfig(values={"OPENAI_API_KEY": "${{ secrets.X }}"}))

    with pytest.raises(MissingSecretError):
        await run_done_check(
            "owner/repo",
            config,
            CLAUDE_MD,
            auth=FakeAuth(),
            runtime=runtime,
            resolve_secret=_resolver({}),  # cannot resolve OPENAI_API_KEY
            report=_sink(captured),
        )

    # No container was started, so nothing ran and nothing needs teardown.
    assert runtime.log == []
    # The escalation is observable on the sink.
    assert len(captured) == 1
    assert captured[0].escalated and not captured[0].passed
    assert "OPENAI_API_KEY" in captured[0].detail


# --- teardown is guaranteed on failure paths -------------------------------------


@pytest.mark.asyncio
async def test_container_torn_down_when_done_check_fails() -> None:
    """A failing done-check still reports (passed=False) and tears the container down."""
    runtime = FakeRuntime(results={"uv": RunResult(exit_code=1, stderr="boom")})
    captured: list[DoneCheckReport] = []

    result = await run_done_check(
        "owner/repo",
        _config(),
        CLAUDE_MD,
        auth=FakeAuth(),
        runtime=runtime,
        resolve_secret=_resolver({}),
        report=_sink(captured),
    )

    assert not result.passed
    assert runtime.container is not None and runtime.container.destroyed
    assert captured == [result]
    assert "boom" in result.detail


@pytest.mark.asyncio
async def test_container_torn_down_when_clone_raises() -> None:
    """A clone failure raises DoneCheckError but teardown still runs (finally)."""
    runtime = FakeRuntime(results={"git": RunResult(exit_code=128, stderr="no auth")})

    with pytest.raises(DoneCheckError):
        await run_done_check(
            "owner/repo",
            _config(),
            CLAUDE_MD,
            auth=FakeAuth(),
            runtime=runtime,
            resolve_secret=_resolver({}),
            report=_sink([]),
        )

    assert runtime.container is not None and runtime.container.destroyed


@pytest.mark.asyncio
async def test_container_torn_down_when_report_raises() -> None:
    """If the report sink raises, the container is still destroyed by the finally."""
    runtime = FakeRuntime()

    async def exploding_report(result: DoneCheckReport) -> None:
        raise RuntimeError("sink down")

    with pytest.raises(RuntimeError, match="sink down"):
        await run_done_check(
            "owner/repo",
            _config(),
            CLAUDE_MD,
            auth=FakeAuth(),
            runtime=runtime,
            resolve_secret=_resolver({}),
            report=exploding_report,
        )

    assert runtime.container is not None and runtime.container.destroyed
