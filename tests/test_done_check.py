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
    EnvSecretResolver,
    GhCliError,
    GhReportSink,
    MissingSecretError,
    ReportSink,
    SecretResolver,
    parse_done_check,
    parse_secret_ref,
    render_report_argv,
    render_report_body,
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


# --- real resolver: parse_secret_ref (pure) --------------------------------------


def test_parse_secret_ref_unwraps_placeholder() -> None:
    """A ``${{ secrets.NAME }}`` placeholder normalises to the bare key NAME."""
    assert parse_secret_ref("${{ secrets.OPENAI_API_KEY }}") == "OPENAI_API_KEY"


def test_parse_secret_ref_unwraps_env_scheme() -> None:
    """An ``env://NAME`` reference normalises to the bare key NAME."""
    assert parse_secret_ref("env://GITHUB_TOKEN") == "GITHUB_TOKEN"


def test_parse_secret_ref_passes_through_other_refs_verbatim() -> None:
    """A bare name or an unknown scheme is used as the key verbatim (after strip)."""
    assert parse_secret_ref("  vault://team/token  ") == "vault://team/token"
    assert parse_secret_ref("GITHUB_TOKEN") == "GITHUB_TOKEN"


# --- real resolver: EnvSecretResolver lookup -------------------------------------


@pytest.mark.asyncio
async def test_env_resolver_reads_placeholder_from_environment() -> None:
    """A placeholder resolves against the injected environment, not os.environ."""
    resolver = EnvSecretResolver(environ={"OPENAI_API_KEY": "sk-real"})
    assert await resolver("${{ secrets.OPENAI_API_KEY }}") == "sk-real"


@pytest.mark.asyncio
async def test_env_resolver_config_layer_wins_over_environment() -> None:
    """Deployment config takes precedence over the same key in the environment."""
    resolver = EnvSecretResolver(
        config={"TOKEN": "from-config"}, environ={"TOKEN": "from-env"}
    )
    assert await resolver("env://TOKEN") == "from-config"


@pytest.mark.asyncio
async def test_env_resolver_returns_none_on_miss_without_logging_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unresolved reference returns None and logs the name only, never a value."""
    resolver = EnvSecretResolver(config={"OTHER": "super-secret"}, environ={})
    with caplog.at_level("WARNING"):
        assert await resolver("${{ secrets.MISSING }}") is None
    assert "MISSING" in caplog.text
    assert "super-secret" not in caplog.text


@pytest.mark.asyncio
async def test_env_resolver_satisfies_secretresolver_protocol_in_orchestration() -> None:
    """The real resolver drops into run_done_check wherever the fake one did."""
    runtime = FakeRuntime()
    config = _config(SecretsConfig(values={"API_KEY": "${{ secrets.API_KEY }}"}))
    resolve_secret: SecretResolver = EnvSecretResolver(environ={"API_KEY": "sk-real"})

    await run_done_check(
        "owner/repo",
        config,
        CLAUDE_MD,
        auth=FakeAuth(),
        runtime=runtime,
        resolve_secret=resolve_secret,
        report=_sink([]),
    )

    assert runtime.started_env == {"API_KEY": "sk-real"}


# --- real report sink: GhReportSink (pure parts + fake runner) --------------------


class FakeGhRunner:
    """Records the argv + env it was called with; returns canned stdout, no process."""

    def __init__(self, *, fail_with: GhCliError | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self._fail_with = fail_with

    async def __call__(self, argv: object, env: object) -> bytes:
        assert isinstance(argv, (list, tuple))
        assert isinstance(env, dict)
        self.calls.append(([str(a) for a in argv], dict(env)))
        if self._fail_with is not None:
            raise self._fail_with
        return b"https://github.com/owner/repo/issues/1#issuecomment-1\n"


def _report(*, passed: bool = True, escalated: bool = False) -> DoneCheckReport:
    return DoneCheckReport(
        repo_full_name="owner/repo",
        passed=passed,
        escalated=escalated,
        detail="all done-check commands passed",
    )


def test_render_report_body_headers_each_outcome() -> None:
    """Passed, failed, and escalated each get a distinct scannable header + the detail."""
    assert render_report_body(_report(passed=True)).startswith("## Done-check passed")
    assert render_report_body(_report(passed=False)).startswith("## Done-check failed")
    escalated = render_report_body(_report(passed=False, escalated=True))
    assert escalated.startswith("## Done-check escalated")
    # The report detail always rides along in the body.
    assert "all done-check commands passed" in render_report_body(_report())


def test_render_report_argv_targets_repo_and_passes_body_via_flag() -> None:
    """The argv is a no-shell ``gh issue comment`` with the body behind ``--body``."""
    argv = render_report_argv(_report())
    assert argv[:5] == ["gh", "issue", "comment", "--repo", "owner/repo"]
    assert argv[5] == "--body"
    assert argv[6] == render_report_body(_report())


@pytest.mark.asyncio
async def test_gh_report_sink_assembles_command_and_injects_token_as_gh_token() -> None:
    """The sink posts the rendered comment and carries the token in GH_TOKEN, off argv."""
    runner = FakeGhRunner()
    sink = GhReportSink(token="ghs_secret", runner=runner)

    await sink(_report(passed=False))

    assert len(runner.calls) == 1
    argv, env = runner.calls[0]
    assert argv == render_report_argv(_report(passed=False))
    assert env == {"GH_TOKEN": "ghs_secret"}
    # The token rides in the env only — never on the command line.
    assert "ghs_secret" not in argv


@pytest.mark.asyncio
async def test_gh_report_sink_without_token_uses_ambient_auth() -> None:
    """With no token the child env carries no GH_TOKEN, falling back to ambient auth."""
    runner = FakeGhRunner()
    await GhReportSink(runner=runner)(_report())
    _, env = runner.calls[0]
    assert env == {}


@pytest.mark.asyncio
async def test_gh_report_sink_propagates_gh_failure() -> None:
    """A non-zero ``gh`` exit surfaces as GhCliError rather than being swallowed."""
    failure = GhCliError(["gh"], returncode=1, stderr="not found")
    sink = GhReportSink(token="t", runner=FakeGhRunner(fail_with=failure))
    with pytest.raises(GhCliError):
        await sink(_report())


@pytest.mark.asyncio
async def test_gh_report_sink_satisfies_reportsink_protocol_in_orchestration() -> None:
    """The real sink drops into run_done_check wherever the fake one did."""
    runner = FakeGhRunner()
    report: ReportSink = GhReportSink(token="t", runner=runner)

    result = await run_done_check(
        "owner/repo",
        _config(),
        CLAUDE_MD,
        auth=FakeAuth(),
        runtime=FakeRuntime(),
        resolve_secret=_resolver({}),
        report=report,
    )

    assert result.passed
    argv, _ = runner.calls[0]
    assert argv == render_report_argv(result)
