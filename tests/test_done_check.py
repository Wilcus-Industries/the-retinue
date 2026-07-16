"""Tests for the done-check building blocks (issue #4, post B-full).

The orchestrator now owns the per-slice container lifecycle (see
``tests/test_orchestrator.py``); this module covers the reusable, container-agnostic
pieces that lifecycle drives: parsing the gate from ``CLAUDE.md``, resolving its secrets
(escalating a missing one before any container starts), running the commands in a
container the caller owns, and the gh report sink. Every collaborator is faked — no
Docker, no network.

The in-memory ``FakeAuth`` / ``FakeContainer`` / ``FakeRuntime`` fakes and the
``_resolver`` / ``_sink`` / ``CLAUDE_MD`` helpers are reused by the orchestrator and
reviewer tests, so they live in ``tests/fakes.py``.
"""

from __future__ import annotations

import pytest

from retinue.container import RunResult
from retinue.done_check import (
    DoneCheckError,
    DoneCheckReport,
    EnvSecretResolver,
    GhReportSink,
    MissingSecretError,
    SecretResolver,
    parse_done_check,
    parse_secret_ref,
    render_report_argv,
    render_report_body,
    resolve_secrets_or_escalate,
    run_done_check_commands,
)
from retinue.gh import GhCliError
from retinue.repo_config import RepoConfig, SecretsConfig
from tests.fakes import CLAUDE_MD, FakeContainer, _resolver, _sink


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


# --- run_done_check_commands -----------------------------------------------------


@pytest.mark.asyncio
async def test_run_done_check_commands_passes_when_all_green() -> None:
    """Every command exiting 0 yields ``(True, ...)`` and runs each in order."""
    log: list[str] = []
    container = FakeContainer(log, {})

    passed, detail = await run_done_check_commands(
        container, [["uv", "run", "pytest"], ["uv", "run", "ruff", "check", "."]]
    )

    assert passed is True
    assert "passed" in detail
    assert log == ["run:uv run pytest", "run:uv run ruff check ."]


@pytest.mark.asyncio
async def test_run_done_check_commands_stops_at_first_failure() -> None:
    """A failing command yields ``(False, detail)`` and the later commands never run."""
    log: list[str] = []
    container = FakeContainer(log, {"uv": RunResult(exit_code=1, stderr="boom")})

    passed, detail = await run_done_check_commands(
        container, [["uv", "run", "pytest"], ["echo", "later"]]
    )

    assert passed is False
    assert "boom" in detail
    # The second command never ran (stopped at the first failure).
    assert log == ["run:uv run pytest"]


@pytest.mark.asyncio
async def test_failure_detail_includes_stdout_failure_summary() -> None:
    """The failure detail must carry stdout, where pytest writes which test failed.

    pytest/ruff/mypy print findings to stdout; uv prints setup/download progress to
    stderr. A detail built from stderr alone shows only setup noise (the real dogfood
    bug), so the actual failing-test line on stdout has to ride along.
    """
    log: list[str] = []
    container = FakeContainer(
        log,
        {
            "uv": RunResult(
                exit_code=1,
                stdout="FAILED tests/test_widget.py::test_frobnicate - assert 1 == 2",
                stderr="Downloading mypy (14.4MiB)\nInstalled 62 packages",
            )
        },
    )

    passed, detail = await run_done_check_commands(container, [["uv", "run", "pytest"]])

    assert passed is False
    assert "FAILED tests/test_widget.py::test_frobnicate" in detail


@pytest.mark.asyncio
async def test_done_check_blanks_anthropic_credential_for_each_command() -> None:
    """Each done-check command runs with the Anthropic auth credential unset.

    The build container carries the implementer's CLAUDE_CODE_OAUTH_TOKEN /
    ANTHROPIC_API_KEY so claude can authenticate; pytest must NOT inherit it (the
    dogfood leak: ``test_adapter_settings_default`` read the live token from env and
    printed it into the report). Blanking it per-command is the source fix.
    """
    log: list[str] = []
    container = FakeContainer(log, {})

    await run_done_check_commands(container, [["uv", "run", "pytest"]])

    env = container.command_env["uv"]
    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == ""


@pytest.mark.asyncio
async def test_done_check_redacts_injected_secret_values_from_detail() -> None:
    """An injected secret echoed by a failing command is scrubbed from the detail.

    Shape-based redaction only catches known credential shapes; a repo-declared secret
    or the webhook secret has no shape. The retinue knows the exact values it injected,
    so it redacts those literally — the report never even holds the value.
    """
    log: list[str] = []
    secret = "wh00k-Sup3r-S3cret-value-no-shape"
    container = FakeContainer(
        log,
        {"uv": RunResult(exit_code=1, stdout=f"assert '{secret}' == ''")},
    )

    passed, detail = await run_done_check_commands(
        container, [["uv", "run", "pytest"]], secret_values=[secret]
    )

    assert passed is False
    assert secret not in detail
    assert "[REDACTED]" in detail


@pytest.mark.asyncio
async def test_failure_detail_keeps_tail_not_head() -> None:
    """When output is long, the detail keeps the tail — pytest's summary prints last."""
    log: list[str] = []
    head = "\n".join(f"setup line {i}" for i in range(500))
    container = FakeContainer(
        log,
        {"uv": RunResult(exit_code=1, stdout=f"{head}\n=== 1 failed, 622 passed ===")},
    )

    passed, detail = await run_done_check_commands(container, [["uv", "run", "pytest"]])

    assert passed is False
    assert "=== 1 failed, 622 passed ===" in detail
    assert "setup line 0" not in detail


# --- resolve_secrets_or_escalate -------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_secrets_returns_env_when_all_resolved() -> None:
    """Resolved inline secrets and refs come back as the env to inject; nothing reported."""
    captured: list[DoneCheckReport] = []
    config = _config(
        SecretsConfig(
            values={"OPENAI_API_KEY": "${{ secrets.OPENAI_API_KEY }}"},
            refs=["vault://team/token"],
        )
    )
    resolver = _resolver(
        {"OPENAI_API_KEY": "sk-real", "vault://team/token": "vault-secret"}
    )

    env = await resolve_secrets_or_escalate(
        "owner/repo", 47, config, resolver, _sink(captured)
    )

    assert env == {"OPENAI_API_KEY": "sk-real", "vault://team/token": "vault-secret"}
    assert captured == []  # nothing to escalate


@pytest.mark.asyncio
async def test_resolve_secrets_escalates_and_raises_on_missing() -> None:
    """A missing required secret posts an escalation report and re-raises."""
    captured: list[DoneCheckReport] = []
    config = _config(SecretsConfig(values={"OPENAI_API_KEY": "${{ secrets.X }}"}))

    with pytest.raises(MissingSecretError):
        await resolve_secrets_or_escalate(
            "owner/repo", 47, config, _resolver({}), _sink(captured)
        )

    assert len(captured) == 1
    assert captured[0].escalated and not captured[0].passed
    assert captured[0].issue_number == 47  # the escalation comment targets the issue
    assert "OPENAI_API_KEY" in captured[0].detail


@pytest.mark.asyncio
async def test_real_env_resolver_drops_into_resolve_secrets() -> None:
    """The real EnvSecretResolver satisfies the resolver seam in the resolve step."""
    config = _config(SecretsConfig(values={"API_KEY": "${{ secrets.API_KEY }}"}))
    resolve_secret: SecretResolver = EnvSecretResolver(environ={"API_KEY": "sk-real"})

    env = await resolve_secrets_or_escalate(
        "owner/repo", 47, config, resolve_secret, _sink([])
    )

    assert env == {"API_KEY": "sk-real"}


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
        issue_number=47,
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


def test_render_report_body_redacts_secret_shaped_strings() -> None:
    """The posted body scrubs credential-shaped strings — defense in depth.

    Even with the env-strip in place, a test could echo some other secret; the report
    is the last gate before a value reaches GitHub, so it redacts Anthropic keys,
    GitHub tokens, and PEM private-key blocks rather than posting them.
    """
    detail = (
        "auth failed with sk-ant-oat01-K_0RRfidvjYPNy2KfcFJ-DUUBjQAA\n"
        "clone used ghp_abcdEFGH1234567890abcdEFGH1234567890\n"
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAK\n-----END RSA PRIVATE KEY-----"
    )
    report = DoneCheckReport(
        repo_full_name="owner/repo",
        issue_number=47,
        passed=False,
        escalated=False,
        detail=detail,
    )

    body = render_report_body(report)

    assert "sk-ant-oat01-K_0RRfidvjYPNy2KfcFJ-DUUBjQAA" not in body
    assert "ghp_abcdEFGH1234567890abcdEFGH1234567890" not in body
    assert "MIIEowIBAAK" not in body
    assert "[REDACTED]" in body
    # The non-secret context around the secrets survives so the report stays useful.
    assert "auth failed" in body


def test_render_report_argv_targets_repo_and_passes_body_via_flag() -> None:
    """The argv is a no-shell ``gh issue comment <number>`` with the body behind ``--body``.

    ``gh issue comment`` requires the issue number as a positional (it errors
    "accepts 1 arg(s), received 0" without it), so the report's ``issue_number`` must
    ride in the argv ahead of the flags.
    """
    argv = render_report_argv(_report())
    assert argv[:6] == ["gh", "issue", "comment", "47", "--repo", "owner/repo"]
    assert argv[6] == "--body"
    assert argv[7] == render_report_body(_report())


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
