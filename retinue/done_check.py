"""Done-check building blocks: parse the gate, resolve its secrets, run it, report it.

Issue #4 originally owned the whole disposable-container orchestration here. Since the
B-full refactor the shared container-build lifecycle owns the per-issue flow (clone → branch
→ implement → done-check → push → destroy, see
:func:`retinue.container_build.build_issue_in_container`), so this module is now the set of
reusable, container-agnostic pieces that lifecycle drives:

1. **parse** — :func:`parse_done_check` reads the done-check command from ``CLAUDE.md``,
2. **inject** — :func:`resolve_secrets_or_escalate` resolves the config's declared secrets
   (a missing required secret escalates on the report sink *before* any container starts),
3. **run** — :func:`run_done_check_commands` runs the commands inside a container the
   caller owns and already cloned the slice's changes into, yielding ``(passed, detail)``,
4. **report** — :func:`render_report_body` / :class:`GhReportSink` post the outcome to an
   observable sink (an issue comment via ``gh``).

Every collaborator — the secret resolver, the container, and the report sink — is
injected, so each piece is exercised in tests with no Docker and no network.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from retinue.container import Container, RunResult
from retinue.gh import GhBytesRunner, auth_env, run_gh_subprocess
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
        issue_number: The issue the outcome comment is posted on (``gh issue comment``
            requires it as a positional).
        passed: True only when every done-check command exited 0.
        escalated: True when a required secret was missing (the check never ran).
        detail: Human-readable summary for the commit status / issue comment.
    """

    repo_full_name: str
    issue_number: int
    passed: bool
    escalated: bool
    detail: str


# Status markers for the rendered report body. A passed check, a failed check, and an
# escalation (missing secret, the check never ran) each get a distinct, scannable header.
_STATUS_PASSED = "Done-check passed"
_STATUS_FAILED = "Done-check failed"
_STATUS_ESCALATED = "Done-check escalated"

# Credential shapes scrubbed from a report body before it is posted to GitHub. The report
# echoes a failed command's output, which can carry a secret a test printed (the dogfood
# leak: pytest read the live Anthropic token from env and the failure detail posted it).
# This is the last gate before a value reaches GitHub, so known credential shapes — and any
# PEM private-key block — are replaced wholesale.
_REDACTION = "[REDACTED]"
_SECRET_PATTERNS = (
    # PEM private-key blocks (DOTALL: spans the base64 body across newlines). Matched
    # first so the whole block collapses to one marker rather than per-line noise.
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
    re.compile(r"sk-ant-[A-Za-z0-9._-]+"),  # Anthropic api / oat tokens
    re.compile(r"github_pat_[A-Za-z0-9_]+"),  # GitHub fine-grained PAT
    re.compile(r"gh[pousr]_[A-Za-z0-9]+"),  # GitHub classic / app / refresh tokens
)


def _redact_secrets(text: str) -> str:
    """Replace credential-shaped substrings in ``text`` with a redaction marker."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_REDACTION, text)
    return text


# Below this length a "secret" value is treated as too short to redact safely: it would
# match common substrings in unrelated output (e.g. a 3-char value nuking real text).
# Real credentials are far longer, so the floor loses nothing.
_MIN_REDACT_LEN = 4


def _redact_values(text: str, secret_values: Iterable[str]) -> str:
    """Replace exact occurrences of each injected secret value with the marker.

    Shape-based redaction only catches known credential shapes; value redaction catches
    *any* secret the retinue injected (a repo-declared secret, the webhook secret) by its
    literal value, regardless of shape. Longest-first so a value that contains another is
    redacted whole rather than leaving a fragment.
    """
    for value in sorted(set(secret_values), key=len, reverse=True):
        if len(value) >= _MIN_REDACT_LEN:
            text = text.replace(value, _REDACTION)
    return text


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
    return f"## {status}\n\n{_redact_secrets(report.detail)}"


def render_report_argv(report: DoneCheckReport) -> list[str]:
    """Assemble the ``gh issue comment`` argv that posts ``report`` to its repo.

    Posts to the issue numbered by the report; the issue number is the required
    positional ``gh issue comment`` takes (it errors "accepts 1 arg(s), received 0"
    without it), and the body comes from :func:`render_report_body`, passed via
    ``--body`` so it is never interpolated into a shell. Pure, so command assembly is
    unit-testable without a real ``gh``.
    """
    return [
        "gh",
        "issue",
        "comment",
        str(report.issue_number),
        "--repo",
        report.repo_full_name,
        "--body",
        render_report_body(report),
    ]


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
    is callable, so it drops into the orchestrator's build flow wherever the fake did.

    Args:
        token: The GitHub token ``gh`` authenticates with, placed in the child env as
            ``GH_TOKEN``. ``None`` runs ``gh`` with the ambient auth (a logged-in CLI).
        runner: The injected argv runner; defaults to the real subprocess spawn.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        runner: GhBytesRunner | None = None,
    ) -> None:
        self._token = token
        self._runner = runner or run_gh_subprocess

    async def __call__(self, report: DoneCheckReport) -> None:
        """Post ``report`` to its repo's tracking issue as a ``gh`` comment.

        Assembles the ``gh issue comment`` argv and runs it through the injected runner
        with the auth env. The stdout (the created comment's URL) is discarded.

        Raises:
            GhCliError: ``gh`` exited non-zero (propagated from the runner).
        """
        argv = render_report_argv(report)
        await self._runner(argv, auth_env(self._token))
        logger.info(
            "Posted done-check report for %s (passed=%s, escalated=%s)",
            report.repo_full_name,
            report.passed,
            report.escalated,
        )


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


# The implementer's Anthropic credential rides the build container's env so claude can
# authenticate; the done-check commands (pytest/ruff/mypy) never need it. Blanking these
# names per command keeps the live token out of the checked repo's process env — the
# dogfood leak was a test reading CLAUDE_CODE_OAUTH_TOKEN from env and echoing it.
_AUTH_ENV_VARS = ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")


async def run_done_check_commands(
    container: Container,
    commands: list[list[str]],
    *,
    secret_values: Iterable[str] = (),
) -> tuple[bool, str]:
    """Run each done-check command in order in ``container``; stop at the first failure.

    The container is owned by the caller (the orchestrator's per-slice build container,
    which has already cloned the repo and let the implementer commit the slice), so the
    commands run over the *real* changes rather than a pristine clone. Each command runs
    with the Anthropic auth credential blanked in its environment (it is the implementer's,
    not the check's), so a test can never read it back out of the env.

    Args:
        container: The build container the commands exec in.
        commands: The done-check commands, run in order until one fails.
        secret_values: The exact secret values injected into the container (repo secrets,
            the auth credential). A failing command's output is scrubbed of these before
            it is returned, so the detail — and the report built from it — never carries a
            secret, whatever shape it has.

    Returns:
        ``(passed, detail)`` — ``passed`` is True only if every command exited 0.
    """
    scrub_env = {name: "" for name in _AUTH_ENV_VARS}
    for command in commands:
        result: RunResult = await container.run_command(command, env=scrub_env)
        if not result.ok:
            detail = _format_failure_detail(command, result)
            return False, _redact_values(detail, secret_values)
    return True, "all done-check commands passed"


# The failure detail is capped to a readable tail: GitHub comments allow ~65k chars, but
# the diagnostic value is in the last lines, and a wall of setup output buries it.
_DETAIL_MAX_LINES = 80
_DETAIL_MAX_CHARS = 8000


def _format_failure_detail(command: Sequence[str], result: RunResult) -> str:
    """Render a failed command's output into the report detail.

    pytest/ruff/mypy write their findings to *stdout*; setup tooling (uv) writes
    download/venv progress to *stderr*. Ordering stderr-then-stdout and keeping the
    *tail* puts the actual failure summary — printed last, on stdout — at the end of
    the detail instead of burying it under setup noise (the dogfood bug: a failure
    comment that showed only ``Installed 62 packages``).
    """
    joined = " ".join(command)
    output = "\n".join(
        part for part in (result.stderr.strip(), result.stdout.strip()) if part
    )
    tail = _tail(output, max_lines=_DETAIL_MAX_LINES, max_chars=_DETAIL_MAX_CHARS)
    if not tail:
        return f"`{joined}` failed (exit {result.exit_code})"
    return f"`{joined}` failed (exit {result.exit_code})\n```\n{tail}\n```"


def _tail(text: str, *, max_lines: int, max_chars: int) -> str:
    """Keep the last ``max_lines`` lines of ``text``, then clamp to ``max_chars``."""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    clipped = "\n".join(lines)
    if len(clipped) > max_chars:
        clipped = clipped[-max_chars:]
    return clipped


async def resolve_secrets_or_escalate(
    repo_full_name: str,
    issue_number: int,
    config: RepoConfig,
    resolve_secret: SecretResolver,
    report: ReportSink,
) -> dict[str, str]:
    """Resolve the config's declared secrets; on a miss, escalate then re-raise.

    Resolving secrets up front lets the caller inject them into the container env at
    ``start`` *and* lets a missing required secret escalate before any container spins
    up — no doomed check, nothing to tear down. The escalation is observable on the
    report sink; the :class:`MissingSecretError` still propagates so the caller skips
    the build.

    Args:
        repo_full_name: The repo whose secrets are being resolved (for the escalation).
        issue_number: The issue the escalation comment is posted on.
        config: The accepted repo config (its ``secrets`` block is resolved).
        resolve_secret: Resolves declared secret names/refs to values.
        report: Sink the escalation is posted to on a miss.

    Returns:
        The resolved ``{env_name: value}`` map to inject into the container env.

    Raises:
        MissingSecretError: A required secret was missing; an escalation report is
            posted before the error propagates.
    """
    try:
        return await _resolve_secrets(config, resolve_secret)
    except MissingSecretError as exc:
        await report(
            DoneCheckReport(
                repo_full_name=repo_full_name,
                issue_number=issue_number,
                passed=False,
                escalated=True,
                detail=f"escalated: required secret {exc.secret_name!r} is missing",
            )
        )
        logger.warning("Escalating %s: missing secret %s", repo_full_name, exc.secret_name)
        raise
