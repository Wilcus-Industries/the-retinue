"""Run a repo's done-check in a fresh disposable container, then report and tear down.

This is the heart of issue #4. For an accepted PRD the worker:

1. **auth** — mints a GitHub App installation token (:mod:`retinue.github_app`),
2. **clone** — clones the repo into a fresh container over that token,
3. **inject** — places the config's secrets into the container env (a missing required
   secret escalates *before* the doomed check runs),
4. **run** — runs the done-check command read from the repo's ``CLAUDE.md``,
5. **report** — posts the outcome to an observable sink (an issue comment via ``gh``),
6. **teardown** — destroys the container.

Every collaborator — auth, the container runtime, the secret resolver, and the report
sink — is injected, so the whole orchestration is exercised in tests with no Docker and
no network. Teardown is guaranteed by ``try/finally``: the container is destroyed on
every path, including when clone, run, or report raises.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
from collections.abc import Awaitable, Callable, Mapping, Sequence
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


# A reference to a secret as it appears in a repo config: either an inline
# ``${{ secrets.NAME }}`` placeholder (the ergonomic shape repos write) or an external
# ``scheme://path`` ref. ``parse_secret_ref`` normalises both into the lookup key the
# resolver reads from its store; anything else is taken as a literal env-var name.
_PLACEHOLDER = re.compile(r"^\$\{\{\s*secrets\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}$")
_ENV_REF = re.compile(r"^env://(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")


def parse_secret_ref(ref: str) -> str:
    """Normalise a config secret reference into the store lookup key.

    Three shapes are understood, all of which name a key in the resolver's backing
    store (env + injected config); resolution itself is a pure dict lookup:

    * ``${{ secrets.NAME }}`` — the inline placeholder repos write; the key is ``NAME``.
    * ``env://NAME`` — an explicit environment reference; the key is ``NAME``.
    * anything else (e.g. a bare ``NAME`` or a ``vault://...`` ref) — used verbatim
      as the key, so the store can carry pre-provisioned external secrets under that
      exact name.

    This function is pure and value-free: it parses a *reference*, never a secret
    value, so it is safe to exercise directly in tests.

    Args:
        ref: The reference as written in the repo config.

    Returns:
        The lookup key to read from the resolver's backing store.
    """
    placeholder = _PLACEHOLDER.match(ref.strip())
    if placeholder is not None:
        return placeholder.group(1)
    env_ref = _ENV_REF.match(ref.strip())
    if env_ref is not None:
        return env_ref.group("name")
    return ref.strip()


class EnvSecretResolver:
    """Production :data:`SecretResolver`: resolve named secrets from env + config.

    The backing store is the process environment overlaid with an optional static
    ``config`` mapping (deployment-provided secrets that are not env vars). A config
    entry wins over the same key in the environment so a deployment can pin a value
    explicitly. ``external_dep none``: no network, no vault client — every resolution
    is a local lookup, which is exactly what makes the orchestration exercisable.

    Values are never logged: a miss logs only the *reference* (a name, not a secret),
    and a hit logs nothing about the value. The instance is callable so it drops in
    wherever a :data:`SecretResolver` is expected.
    """

    def __init__(
        self,
        *,
        config: Mapping[str, str] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """Build a resolver over a static config layered on the environment.

        Args:
            config: Deployment-provided ``key -> value`` secrets that take precedence
                over the environment. Defaults to empty.
            environ: The environment mapping to read; defaults to ``os.environ`` so
                tests can inject a fixed mapping instead of mutating real env.
        """
        self._config = dict(config or {})
        self._environ = environ if environ is not None else os.environ

    async def __call__(self, ref: str) -> str | None:
        """Resolve ``ref`` to its concrete value, or ``None`` when unknown.

        The reference is normalised by :func:`parse_secret_ref`, then looked up in the
        config layer first and the environment second. A miss is logged by *name*
        only — never a value — and returns ``None`` so the caller escalates.
        """
        key = parse_secret_ref(ref)
        value = self._config.get(key)
        if value is None:
            value = self._environ.get(key)
        if value is None:
            logger.warning("Secret reference %r could not be resolved", ref)
            return None
        return value


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


# An async runner for a ``gh`` argv: returns the command's stdout bytes, raising on a
# non-zero exit. Injected so the command-assembly, auth env, and body rendering of
# :class:`GhReportSink` are exercised without spawning a real process, mirroring the
# injected-seam style of :class:`retinue.cron.GhCli`. The default
# (:func:`_run_gh_subprocess`) shells out to ``gh``.
GhRunner = Callable[[Sequence[str], Mapping[str, str]], Awaitable[bytes]]


class GhCliError(RuntimeError):
    """A ``gh`` invocation behind :class:`GhReportSink` failed (non-zero exit).

    Carries the argv and the captured stderr so a failed report post is debuggable
    rather than a bare ``CalledProcessError``.
    """

    def __init__(self, argv: Sequence[str], *, returncode: int, stderr: str) -> None:
        self.argv = list(argv)
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"gh exited {returncode} for {' '.join(argv)}: {stderr.strip()}")


# Status markers for the rendered report body. A passed check, a failed check, and an
# escalation (missing secret, the check never ran) each get a distinct, scannable header.
_STATUS_PASSED = "Done-check passed"
_STATUS_FAILED = "Done-check failed"
_STATUS_ESCALATED = "Done-check escalated"


def render_report_body(report: DoneCheckReport) -> str:
    """Render a :class:`DoneCheckReport` into the Markdown body posted to GitHub.

    The header names the outcome (passed / failed / escalated) so it is scannable in
    the GitHub UI, and the report's ``detail`` carries the per-command summary. Pure
    and value-free, so it is exercised directly in tests without a real ``gh`` call.

    Args:
        report: The outcome to render.

    Returns:
        The Markdown comment body for the report.
    """
    if report.escalated:
        status = _STATUS_ESCALATED
    elif report.passed:
        status = _STATUS_PASSED
    else:
        status = _STATUS_FAILED
    return f"## {status}\n\n{report.detail}"


def render_report_argv(report: DoneCheckReport) -> list[str]:
    """Assemble the ``gh issue comment`` argv that posts ``report`` to its repo.

    Posts to the repo's tracking issue numbered by the report; the body comes from
    :func:`render_report_body` and is passed via ``--body`` so it is never interpolated
    into a shell. Pure, so command assembly is unit-testable without a real ``gh``.
    """
    return [
        "gh",
        "issue",
        "comment",
        "--repo",
        report.repo_full_name,
        "--body",
        render_report_body(report),
    ]


async def _run_gh_subprocess(argv: Sequence[str], env: Mapping[str, str]) -> bytes:
    """Spawn ``gh`` with ``env`` layered over the ambient env; return stdout bytes.

    The default :data:`GhRunner`. Uses :func:`asyncio.create_subprocess_exec` (no shell,
    so the repo name and body are never interpolated into a command line) and raises
    :class:`GhCliError` on a non-zero exit so a failed report post fails loudly.
    """
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **env},
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise GhCliError(
            argv,
            returncode=process.returncode or -1,
            stderr=stderr.decode(errors="replace"),
        )
    return stdout


class GhReportSink:
    """The production :data:`ReportSink`: posts a done-check outcome via the ``gh`` CLI.

    Runs ``gh issue comment --repo <repo> --body <rendered>`` so the outcome of a
    done-check (passed, failed, or escalated) lands as an observable comment on the
    repo's tracking issue. Authenticates by injecting the GitHub token into the child
    env as ``GH_TOKEN`` (the variable the ``gh`` CLI reads), so no token ever lands on
    the command line.

    The subprocess spawn is the one impure edge, factored behind the injected ``runner``
    so command assembly, the auth env, and body rendering are unit-testable without a
    real ``gh``, Docker, or network — mirroring :class:`retinue.cron.GhCli`. The instance
    is callable, so it drops into :func:`run_done_check` wherever the fake sink did.

    Args:
        token: The GitHub token ``gh`` authenticates with, placed in the child env as
            ``GH_TOKEN``. ``None`` runs ``gh`` with the ambient auth (a logged-in CLI).
        runner: The injected argv runner; defaults to the real subprocess spawn.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        runner: GhRunner | None = None,
    ) -> None:
        self._token = token
        self._runner = runner or _run_gh_subprocess

    async def __call__(self, report: DoneCheckReport) -> None:
        """Post ``report`` to its repo's tracking issue as a ``gh`` comment.

        Assembles the ``gh issue comment`` argv and runs it through the injected runner
        with the auth env. The stdout (the created comment's URL) is discarded.

        Raises:
            GhCliError: ``gh`` exited non-zero (propagated from the runner).
        """
        argv = render_report_argv(report)
        await self._runner(argv, self._auth_env())
        logger.info(
            "Posted done-check report for %s (passed=%s, escalated=%s)",
            report.repo_full_name,
            report.passed,
            report.escalated,
        )

    def _auth_env(self) -> Mapping[str, str]:
        """The child-process env carrying the token as ``GH_TOKEN`` (empty when none).

        The token goes in the env, never on the argv, so it never lands in a process
        listing or a log of the command.
        """
        return {"GH_TOKEN": self._token} if self._token else {}


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
