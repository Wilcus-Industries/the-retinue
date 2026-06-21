"""Run a repo's done-check in a fresh disposable container, then report and tear down.

This is the heart of issue #4. For an accepted PRD the worker:

1. **auth** — mints a GitHub App installation token (:mod:`retinue.github_app`),
2. **clone** — clones the repo into a fresh container over that token,
3. **inject** — places the config's secrets into the container env (a missing required
   secret escalates *before* the doomed check runs),
4. **run** — runs the done-check command read from the repo's ``CLAUDE.md``,
5. **report** — posts the outcome to an observable sink (commit status / issue comment),
6. **teardown** — destroys the container.

Every collaborator — auth, the container runtime, the secret resolver, and the report
sink — is injected, so the whole orchestration is exercised in tests with no Docker and
no network. Teardown is guaranteed by ``try/finally``: the container is destroyed on
every path, including when clone, run, or report raises.
"""

from __future__ import annotations

import logging
import re
import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from retinue.container import Container, ContainerRuntime, RunResult
from retinue.github_app import InstallationAuth
from retinue.repo_config import RepoConfig

logger = logging.getLogger(__name__)

# Default image for the disposable container; the real value is deployment config.
DEFAULT_IMAGE = "ghcr.io/the-retinue/done-check-runner:latest"

# A fenced code block in CLAUDE.md, e.g. ```\nuv run pytest\n```. The done-check is the
# first such block under a "Definition of done" heading; see _extract_done_check.
_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_DONE_HEADING = re.compile(r"^#{1,6}\s+definition of done\b", re.IGNORECASE | re.MULTILINE)

# An async callable that resolves a secret name to its concrete value, or None when the
# secret cannot be resolved. The ``${{ secrets.NAME }}`` placeholders in a repo config
# are resolved here (e.g. from the deployment's secret store); the resolver is injected
# so tests supply a dict-backed fake.
SecretResolver = Callable[[str], Awaitable[str | None]]

# An async sink that records an observable result (a commit status or issue comment).
# Injected so tests assert what was reported without a real gh call.
ReportSink = Callable[["DoneCheckReport"], Awaitable[None]]


class MissingSecretError(Exception):
    """A required secret could not be resolved, so the done-check must not run.

    Raised during the inject step *before* a container does any work, so a doomed
    check escalates rather than running without its secrets. Carries the offending
    secret name for the report.
    """

    def __init__(self, secret_name: str) -> None:
        super().__init__(f"required secret could not be resolved: {secret_name}")
        self.secret_name = secret_name


class DoneCheckError(Exception):
    """The done-check could not be determined or run (distinct from it failing).

    Used when the repo's ``CLAUDE.md`` carries no recognisable done-check command;
    a check that runs and *fails* is a normal result, not this error.
    """


@dataclass(frozen=True)
class DoneCheckReport:
    """The observable outcome of a done-check run, posted to the report sink.

    Attributes:
        repo_full_name: The repo the check ran against, e.g. "owner/repo".
        passed: True only when every done-check command exited 0.
        escalated: True when a required secret was missing (the check never ran).
        detail: Human-readable summary for the commit status / issue comment.
    """

    repo_full_name: str
    passed: bool
    escalated: bool
    detail: str


def parse_done_check(claude_md: str) -> list[list[str]]:
    """Extract the done-check commands from a repo's ``CLAUDE.md``.

    The done-check is the first fenced code block under a "Definition of done"
    heading; each non-blank line is one command, split with shell quoting rules.

    Args:
        claude_md: The raw contents of the repo's ``CLAUDE.md``.

    Returns:
        A list of commands, each an argv list (e.g. ``[["uv", "run", "pytest"]]``).

    Raises:
        DoneCheckError: When no done-check block is found.
    """
    heading = _DONE_HEADING.search(claude_md)
    search_from = heading.end() if heading else 0
    block = _FENCE.search(claude_md, search_from)
    if block is None:
        raise DoneCheckError("no done-check code block found in CLAUDE.md")

    commands = [shlex.split(line) for line in block.group(1).splitlines() if line.strip()]
    if not commands:
        raise DoneCheckError("done-check code block in CLAUDE.md is empty")
    return commands


async def _resolve_secrets(
    config: RepoConfig, resolve_secret: SecretResolver
) -> dict[str, str]:
    """Resolve a config's declared secrets into concrete env values.

    Inline ``${{ secrets.NAME }}`` placeholders and external ``refs`` are both
    resolved through ``resolve_secret``. A name that cannot be resolved is a hard
    error so the check never starts without it.

    Raises:
        MissingSecretError: When any declared secret resolves to None.
    """
    env: dict[str, str] = {}
    for name in config.secrets.values:
        value = await resolve_secret(name)
        if value is None:
            raise MissingSecretError(name)
        env[name] = value
    for ref in config.secrets.refs:
        value = await resolve_secret(ref)
        if value is None:
            raise MissingSecretError(ref)
        env[ref] = value
    return env


async def _clone_repo(container: Container, clone_url: str) -> None:
    """Clone the repo into the container over the installation token URL."""
    result = await container.run_command(["git", "clone", clone_url, "."])
    if not result.ok:
        raise DoneCheckError(f"clone failed (exit {result.exit_code}): {result.stderr}")


async def _run_done_check(
    container: Container, commands: list[list[str]]
) -> tuple[bool, str]:
    """Run each done-check command in order; stop at the first failure.

    Returns:
        ``(passed, detail)`` — ``passed`` is True only if every command exited 0.
    """
    for command in commands:
        result: RunResult = await container.run_command(command)
        if not result.ok:
            joined = " ".join(command)
            return False, f"`{joined}` failed (exit {result.exit_code})\n{result.stderr}"
    return True, "all done-check commands passed"


async def run_done_check(
    repo_full_name: str,
    config: RepoConfig,
    claude_md: str,
    *,
    auth: InstallationAuth,
    runtime: ContainerRuntime,
    resolve_secret: SecretResolver,
    report: ReportSink,
    image: str = DEFAULT_IMAGE,
) -> DoneCheckReport:
    """Run a repo's done-check in a disposable container and report the outcome.

    Orchestrates auth -> clone -> inject -> run -> report -> teardown. Secrets are
    resolved and a missing required secret escalates *before* any container starts,
    so a doomed check never runs. The container is started only once secrets are in
    hand, and is destroyed in a ``finally`` so it is never leaked on any path.

    Args:
        repo_full_name: The repo to check, e.g. "owner/repo".
        config: The accepted repo config (its ``secrets`` block is injected).
        claude_md: The repo's ``CLAUDE.md`` text, carrying the done-check command.
        auth: Mints the installation token used to clone (the auth seam).
        runtime: Spawns the disposable container (the Docker seam).
        resolve_secret: Resolves declared secret names/refs to values.
        report: Sink the outcome is posted to (commit status / issue comment).
        image: Container image to run the check in.

    Returns:
        The :class:`DoneCheckReport` that was posted to ``report``.

    Raises:
        MissingSecretError: A required secret was missing; an escalation report is
            posted before the error propagates, and no container is started.
        DoneCheckError: The done-check could not be parsed, the clone failed, or a
            command could not be run.
    """
    commands = parse_done_check(claude_md)

    # Inject step first: resolving secrets up front means a missing required secret
    # escalates before a container ever spins up — no doomed check, nothing to tear
    # down. The escalation is itself observable on the report sink.
    try:
        env = await _resolve_secrets(config, resolve_secret)
    except MissingSecretError as exc:
        escalation = DoneCheckReport(
            repo_full_name=repo_full_name,
            passed=False,
            escalated=True,
            detail=f"escalated: required secret {exc.secret_name!r} is missing",
        )
        await report(escalation)
        logger.warning("Escalating %s: missing secret %s", repo_full_name, exc.secret_name)
        raise

    token = await auth.installation_token(repo_full_name)
    container = await runtime.start(image=image, env=env)
    try:
        await _clone_repo(container, token.clone_url)
        passed, detail = await _run_done_check(container, commands)
        report_result = DoneCheckReport(
            repo_full_name=repo_full_name,
            passed=passed,
            escalated=False,
            detail=detail,
        )
        await report(report_result)
        logger.info(
            "Done-check for %s: %s", repo_full_name, "passed" if passed else "failed"
        )
        return report_result
    finally:
        # Guaranteed teardown: the disposable container is destroyed on every path,
        # including when clone, run, or report raises.
        await container.destroy()
