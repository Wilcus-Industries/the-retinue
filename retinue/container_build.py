"""The per-issue container-build lifecycle the scheduler build lane runs.

The build lane runs one issue's whole build inside **one disposable container** that is
destroyed on every path: parse the done-check, resolve the config's secrets, start the
container (secrets + git committer identity + the in-container agents' credentials all in
its env), clone the repo and check out a fresh ``issue-<N>`` branch off the target base,
exec the implementer, guard against a hollow implement (zero commits landed), run the
repo's done-check over the real changes, push the branch only on green, and post the
outcome to the report sink. :func:`build_issue_in_container` owns that lifecycle; the
caller — the **ad-hoc lane** (:func:`retinue.adhoc_build.build_adhoc_issue`), which
branches off the repo's target branch — owns only what genuinely differs, passed as hooks:
a read-only planner before the implement (``pre_implement``), the materialized plan the
implementer reads (``plan_path``), and the in-session review gate after a green check
(``on_green``).

The building blocks the lifecycle drives — the :class:`Slice` unit, the
:class:`Implementer` seam, the git command builders, and the in-container git helpers —
live here too, so the lane draws on one public set instead of reaching into private
internals. Every side-effecting collaborator is injected, so the whole flow is exercised
in tests with no Agent SDK, no Docker, no gh, and no network.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from retinue.classifier import ClassifyInput
from retinue.container import Container, ContainerRuntime
from retinue.done_check import (
    DoneCheckReport,
    ReportSink,
    SecretResolver,
    parse_done_check,
    resolve_secrets_or_escalate,
    run_done_check_commands,
)
from retinue.github_app import InstallationAuth
from retinue.repo_config import RepoConfig
from retinue.roles import Role, resolve_model
from retinue.vocab import issue_branch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Slice:
    """One ready slice: a single issue an implementer builds on its own branch.

    Attributes:
        repo_full_name: The target repo, e.g. "owner/repo".
        issue_number: The slice's GitHub issue number; the implementer commits to
            the ``issue-<N>`` branch derived from it.
        prd_number: The parent PRD number; the integration branch is
            ``retinue/prd-<prd_number>``. The ad-hoc lane, which has no parent PRD,
            sets it to the issue number itself.
    """

    repo_full_name: str
    issue_number: int
    prd_number: int

    @property
    def branch(self) -> str:
        """The branch the implementer commits the slice to: ``issue-<N>``."""
        return issue_branch(self.issue_number)


class Implementer(Protocol):
    """Spawns one implementer subagent that builds a slice. The Agent SDK seam.

    A production implementation execs the headless ``claude`` CLI *inside the disposable
    build container* the lifecycle passes in; the subagent implements TDD-first and
    commits to the slice's ``issue-<N>`` branch already checked out there. Tests inject a
    fake that records the request (and may mark the container log) without any real spawn.
    The contract is the commit on ``slice.branch``; the lifecycle does not read a return
    value, it gates on the hollow-implement guard and the done-check that follow.
    """

    async def implement(
        self, slice_: Slice, *, container: Container, plan_path: str | None = None
    ) -> None:
        """Build ``slice_`` in ``container``, committing to its ``issue-<N>`` branch.

        ``plan_path`` is the in-container path of a materialized implementation plan the
        subagent must read before building. A caller that materializes no plan passes
        nothing (``None``), leaving the prompt unchanged; the ad-hoc lane passes its
        ``PLAN_FILE`` so the subagent is pointed at the plan the planner wrote.
        """
        ...

    def auth_env(self) -> dict[str, str]:
        """The env the agent authenticates with, merged into the container at start.

        Returned by the implementer (which owns the Anthropic credential) so the
        lifecycle can inject it into the build container's environment at ``start``
        without knowing how the credential is routed. A fake that needs no credential
        returns an empty mapping.
        """
        ...


class ImplementError(RuntimeError):
    """The container-exec implementer run ended in an error rather than a clean build.

    Distinct from a *clean-but-insufficient* build, which the lifecycle catches via the
    done-check that follows: this is the ``claude`` CLI exec itself failing (a non-zero
    exit code, or a json result flagged ``is_error``), or a hollow implement that landed
    zero commits (:func:`ensure_commits_landed`), so the build surfaces the failure
    rather than proceeding over a half-built or untouched tree.
    """


class GitOpsError(RuntimeError):
    """A ``git`` command failed with a hard error.

    A hard error (unknown ref, not a repository, checkout failure) means the branch could
    not be advanced at all, so it propagates and fails the build rather than being swallowed.
    """


# Identity used for the git commits the retinue records. Builds are non-interactive, so a
# committer identity must be configured or ``git commit`` refuses to run.
GIT_AUTHOR_NAME = "the-retinue"
GIT_AUTHOR_EMAIL = "retinue@users.noreply.github.com"

# The committer identity injected into the build container's env so the *agent's* own
# ``git commit`` (and the push) run non-interactively. The container env is fixed at
# ``start``, so the identity must ride it there rather than per-command ``-c`` flags the
# agent would not use; git reads these four vars without any repo config.
GIT_COMMITTER_ENV = {
    "GIT_AUTHOR_NAME": GIT_AUTHOR_NAME,
    "GIT_AUTHOR_EMAIL": GIT_AUTHOR_EMAIL,
    "GIT_COMMITTER_NAME": GIT_AUTHOR_NAME,
    "GIT_COMMITTER_EMAIL": GIT_AUTHOR_EMAIL,
}


def clone_command(clone_url: str) -> list[str]:
    """Argv that clones the repo (over the installation-token URL) into the workspace."""
    return ["git", "clone", clone_url, "."]


def push_branch_command(branch: str) -> list[str]:
    """Argv that pushes ``branch`` to ``origin`` (authenticated by the cloned remote URL)."""
    return ["git", "push", "origin", branch]


def create_branch_commands(branch: str, base: str) -> list[list[str]]:
    """Argv list that creates ``branch`` off ``base`` and checks it out.

    ``base`` is referenced via ``origin/<base>`` so the branch is rooted on the freshly
    cloned remote tip rather than whatever happens to be checked out, then a local
    ``branch`` is created at that point and made current.
    """
    return [
        ["git", "fetch", "origin", base],
        ["git", "checkout", "-B", branch, f"origin/{base}"],
    ]


def implement_env(credential: str, auth_mode: str) -> dict[str, str]:
    """Build the env the ``claude`` CLI authenticates with, routing the credential by mode.

    ``api_key`` mode threads the credential as ``ANTHROPIC_API_KEY``; ``subscription`` mode
    threads it as ``CLAUDE_CODE_OAUTH_TOKEN`` (the Claude subscription OAuth env var the
    headless CLI reads). Only the credential env var is set here — the lifecycle merges
    it into the build container's environment at ``start``.
    """
    if auth_mode == "subscription":
        return {"CLAUDE_CODE_OAUTH_TOKEN": credential}
    return {"ANTHROPIC_API_KEY": credential}


def write_file_command(path: str, content: str) -> list[str]:
    """Argv that writes ``content`` to ``path`` inside the container, byte-exact.

    ``run_command`` execs the argv directly (no shell, no stdin), so arbitrary file
    bodies can't be passed as a here-doc or piped. The content is base64-encoded and
    decoded in-container via positional args (``$1``/``$2``) — never interpolated into
    the command string — so conflict markers, quotes, and newlines survive untouched and
    nothing in the file body is interpreted as shell syntax.
    """
    blob = base64.b64encode(content.encode()).decode()
    script = 'printf %s "$1" | base64 -d > "$2"'
    return ["sh", "-c", script, "sh", blob, path]


async def clone_and_branch(
    container: Container, clone_url: str, *, branch: str, base: str
) -> None:
    """Clone the repo into ``container`` and check out a fresh ``branch`` off ``base``."""
    clone = await container.run_command(clone_command(clone_url))
    if not clone.ok:
        raise GitOpsError(f"clone failed (exit {clone.exit_code}): {clone.stderr}")
    for command in create_branch_commands(branch, base):
        result = await container.run_command(command)
        if not result.ok:
            raise GitOpsError(
                f"failed to create slice branch {branch} off {base} "
                f"(exit {result.exit_code}): {result.stderr}"
            )


async def push_branch(container: Container, branch: str) -> None:
    """Push ``branch`` to ``origin`` from inside ``container``; raise on failure."""
    result = await container.run_command(push_branch_command(branch))
    if not result.ok:
        raise GitOpsError(
            f"failed to push {branch} (exit {result.exit_code}): {result.stderr}"
        )


async def ensure_commits_landed(
    container: Container, *, branch: str, base: str
) -> None:
    """Raise :class:`ImplementError` when the implement run committed nothing.

    A hollow implement — the agent no-ops and exits 0 — leaves the tree at
    ``origin/<base>``; the done-check then passes vacuously over the untouched tree and
    an empty branch merges. Counting the commits since ``origin/<base>`` right after the
    implement catches that before the done-check runs. A probe that itself fails (bad
    exit, empty stdout) also raises: an unreadable count must not pass as "commits
    exist".
    """
    result = await container.run_command(
        ["git", "rev-list", "--count", f"origin/{base}..HEAD"]
    )
    count = result.stdout.strip()
    if not result.ok or count in ("", "0"):
        raise ImplementError(
            f"implementer for {branch} landed no commits "
            f"(rev-list exit {result.exit_code}, count {count!r})"
        )


async def build_issue_in_container(
    slice_: Slice,
    config: RepoConfig,
    claude_md: str,
    *,
    base: str,
    implementer: Implementer,
    auth: InstallationAuth,
    runtime: ContainerRuntime,
    resolve_secret: SecretResolver,
    report: ReportSink,
    image: str,
    lane_label: str = "Slice",
    extra_auth_envs: Sequence[Mapping[str, str]] = (),
    pre_implement: Callable[[Container], Awaitable[None]] | None = None,
    plan_path: str | None = None,
    on_green: Callable[[Container], Awaitable[None]] | None = None,
) -> bool:
    """Run one issue's full build in a single disposable container; return green/red.

    Owns the whole per-issue lifecycle, destroying the container on every path:

    1. parse the done-check and resolve the config's secrets (a missing one escalates on
       the report sink and propagates *before* any container starts),
    2. start the container with the secrets, the git committer identity, and the
       in-container agents' credentials (``extra_auth_envs`` plus the implementer's) all
       in its env (the env is fixed at ``start``),
    3. clone the repo and check out a fresh ``issue-<N>`` branch off ``base`` — the PRD
       lane's integration branch, or the ad-hoc lane's staging branch,
    4. run the lane's ``pre_implement`` hook when given (the ad-hoc planner + plan
       materialization),
    5. exec the implementer (``claude``) inside the container to build and commit the
       issue, pointed at ``plan_path`` when the lane materialized a plan,
    6. guard against a hollow implement: zero commits since ``origin/<base>`` raises
       :class:`ImplementError` before a vacuous done-check can pass,
    7. run the done-check over the real changes and post the outcome,
    8. push ``issue-<N>`` to ``origin`` only when the done-check is green (a red build
       pushes nothing),
    9. on green only, run the lane's ``on_green`` hook when given (the ad-hoc advisory
       review).

    Args:
        slice_: The issue to build (repo, issue number, branch derivation).
        config: The accepted repo config; its ``secrets`` are injected into the container.
        claude_md: The repo's ``CLAUDE.md`` text, carrying the done-check command.
        base: The branch ``issue-<N>`` is cut off (and the hollow-implement probe's base).
        implementer: Execs the implementer subagent in the container (the Agent SDK seam).
        auth: Mints the installation token used to clone (the auth seam).
        runtime: Spawns the disposable build container (the Docker seam).
        resolve_secret: Resolves the config's declared secret names/refs to values.
        report: Sink the done-check outcome is posted to (commit status / comment).
        image: Container image the build runs in.
        lane_label: The lane's name for the done-check log line ("Slice" / "Ad-hoc issue").
        extra_auth_envs: Credential envs of additional in-container agents (the ad-hoc
            planner), merged into the container env before the implementer's own.
        pre_implement: Hook run after clone+branch and before the implement.
        plan_path: In-container path of a materialized plan, threaded to the implementer.
        on_green: Hook run after a green done-check was pushed and reported.

    Returns:
        True only when the done-check passed (and the branch was pushed); False on red.
    """
    commands = parse_done_check(claude_md)
    env = await resolve_secrets_or_escalate(
        slice_.repo_full_name, slice_.issue_number, config, resolve_secret, report
    )
    auth_envs = [*extra_auth_envs, implementer.auth_env()]
    start_env = {**env, **GIT_COMMITTER_ENV}
    for auth_env in auth_envs:
        start_env.update(auth_env)
    # The exact secret values injected into the container, scrubbed from a failing
    # done-check's report (repo-declared secrets plus the agents' credentials).
    secret_values = [*env.values(), *(v for a in auth_envs for v in a.values())]
    token = await auth.installation_token(slice_.repo_full_name)
    container = await runtime.start(image=image, env=start_env)
    try:
        await clone_and_branch(
            container, token.clone_url, branch=slice_.branch, base=base
        )
        if pre_implement is not None:
            await pre_implement(container)
        if plan_path is None:
            # The bare call shape (no plan_path) is kept so an injected implementer that
            # predates the plan_path parameter still satisfies the seam.
            await implementer.implement(slice_, container=container)
        else:
            await implementer.implement(
                slice_, container=container, plan_path=plan_path
            )
        await ensure_commits_landed(container, branch=slice_.branch, base=base)
        passed, detail = await run_done_check_commands(
            container, commands, secret_values=secret_values
        )
        if passed:
            await push_branch(container, slice_.branch)
        await report(
            DoneCheckReport(
                repo_full_name=slice_.repo_full_name,
                issue_number=slice_.issue_number,
                passed=passed,
                escalated=False,
                detail=detail,
            )
        )
        logger.info(
            "%s %s done-check %s",
            lane_label,
            slice_.branch,
            "passed" if passed else "failed",
        )
        if passed and on_green is not None:
            await on_green(container)
        return passed
    finally:
        # Guaranteed teardown: the disposable container is destroyed on every path,
        # including when clone, a hook, implement, the done-check, the push, or report
        # raises.
        await container.destroy()


# --- real container-exec implementer (production adapter behind the Implementer seam) ---
#
# The production :class:`Implementer` execs the headless ``claude`` CLI *inside the
# disposable build container* the lifecycle owns — the "shell out to claude" discipline the
# PRD cites. The repo is already cloned and the ``issue-<N>`` branch checked out in that
# container, so the agent edits files and commits over the real tree the done-check then
# runs against. Confining the autonomous AI step to a throwaway container keeps it off the
# worker host and its mounted ``docker.sock``. The one side effect is the exec, taken behind
# the injected :class:`~retinue.container.Container`, so the flow is exercisable without a
# live model, the CLI, Docker, gh, or network. The bug-prone pure parts — prompt assembly,
# argv construction, the api_key-vs-subscription env auth routing, and reading the CLI result
# — are factored into the free functions below so they are unit-tested in isolation.
#
# Auth mirrors :class:`retinue.config.Settings`: ``auth_mode="api_key"`` threads the
# credential to the CLI as ``ANTHROPIC_API_KEY``; ``auth_mode="subscription"`` threads it as
# ``CLAUDE_CODE_OAUTH_TOKEN`` (the subscription OAuth env var). The credential rides the
# container env (fixed at ``start``), so the implementer exposes it via :meth:`auth_env` for
# the lifecycle to merge in, rather than passing it per-exec. The contract is the commit on
# ``slice.branch``; the lifecycle gates on the done-check that follows, so a run the CLI
# finishes "successfully" but that fails to satisfy the repo is still caught downstream.
# A non-zero CLI exit (or a json result flagged ``is_error``) raises :class:`ImplementError`.
#
# The implementing model and effort tier come from the :data:`~retinue.roles.Role.IMPLEMENTER`
# registry entry (Sonnet 4.6 at the ``high`` tier by default), resolved at construction time
# so a repo's routing level can swap the model at the wiring site. The in-container
# ``claude`` CLI carries no effort flag today, so the ``high`` tier is registry metadata that
# records the PRD's intent without changing the wire.

# The implementer's brief, appended to the CLI's system prompt. Frozen (no per-slice
# interpolation) so the prefix is stable; the slice specifics ride in the per-slice prompt.
_IMPLEMENT_SYSTEM = (
    "You are an autonomous implementer. Build the requested GitHub issue inside the "
    "repository you are running in. Default to test-driven development: when the change "
    "has testable behavior, write or update a failing test first, then write code until "
    "it passes. A documentation- or config-only change has nothing to test — make the "
    "change directly rather than inventing a test for it. Either way, ensure the repo's "
    "own checks pass before you commit. Make the smallest change that satisfies the "
    "issue; do not refactor unrelated code. When the work is complete and committed to "
    "the issue's branch, stop."
)

# Hard cap on the implementer's agent loop. Without it the headless ``claude`` run is
# bounded only by the arq job_timeout, which cancels the *whole* job (container and all)
# mid-implement — so a thrashing run (e.g. a doc task re-running the full check suite each
# turn) is killed before the done-check ever runs. The cap makes the agent stop and lets
# the done-check report on whatever was committed. Tunable via ``implement_max_turns``.
_DEFAULT_IMPLEMENT_MAX_TURNS = 80


# Fetch one issue's facts (title/body/labels) — the seam the implementer bakes the issue
# content into its prompt through. Defined here (with the implementer that carries it) so
# :class:`ContainerImplementer` can carry it without an import cycle.
IssueFactsSource = Callable[[str, int], Awaitable[ClassifyInput]]


def _implement_prompt(
    slice_: Slice, *, plan_path: str | None = None, facts: ClassifyInput | None = None
) -> str:
    """Assemble the per-slice prompt: which issue to build, on which branch.

    Names the target repo, the issue number to implement, and the ``issue-<N>`` branch the
    work must be committed to, so the spawned subagent commits where the lifecycle expects
    to find it. When ``plan_path`` is given (the ad-hoc lane), the prompt leads with an
    instruction to read that materialized plan first; with no ``plan_path`` the prompt is
    the bare form.

    When ``facts`` is given, the issue's title and body are appended as the authoritative
    spec. This is load-bearing: the build container has no ``gh`` and no GitHub token in
    its env (the installation token only rides the clone URL), so the agent cannot read
    the issue itself — without the baked content it no-ops and the slice builds hollow.
    """
    plan_preamble = (
        f"Read the implementation plan at '{plan_path}' first, then implement it. "
        if plan_path is not None
        else ""
    )
    facts_section = (
        ""
        if facts is None
        else (
            "\n\nThe issue's title and body follow; they are the authoritative "
            "specification for this change. This container cannot reach GitHub, so "
            "work from them rather than trying to fetch the issue.\n\n"
            f"Issue title: {facts.title}\n\n"
            f"Issue body:\n{facts.body}"
        )
    )
    return (
        f"{plan_preamble}"
        f"Implement issue #{slice_.issue_number} of {slice_.repo_full_name}. "
        f"Commit your work to the '{slice_.branch}' branch (already checked out). "
        "Implement it test-driven when the change has testable behavior; a documentation- "
        "or config-only change needs no test. Ensure the repo's checks pass before "
        f"committing.{facts_section}"
    )


def _claude_argv(*, prompt: str, model: str, max_turns: int) -> list[str]:
    """Assemble the headless ``claude`` CLI argv for one in-container implement run.

    Runs non-interactively (``-p`` print mode), pins the implementing ``model``, and runs
    with ``--permission-mode bypassPermissions``: ``acceptEdits`` only auto-accepts *file
    edits*, leaving every Bash call — ``git commit``, the repo's checks — blocked pending
    an approval a headless run can never give, so the agent edits its whole run and exits
    0 with zero commits (a hollow-implement cause, seen live). The container is
    disposable and isolated, so bypassing permissions is the intended trade. Caps the
    agent loop at ``max_turns`` so a runaway/thrashing run stops instead of being killed
    mid-implement by the arq job_timeout, appends the frozen implementer brief to the
    system prompt, and emits a machine-readable json result so the exit can be
    cross-checked. The CLI runs in the container's working dir (the cloned repo), so no
    cwd flag is needed.
    """
    return [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--permission-mode",
        "bypassPermissions",
        "--max-turns",
        str(max_turns),
        "--append-system-prompt",
        _IMPLEMENT_SYSTEM,
        "--output-format",
        "json",
    ]


def _claude_result_is_error(stdout: str) -> bool:
    """Whether the CLI's ``--output-format json`` result flags the run as errored.

    The headless CLI emits a json object carrying an ``is_error`` boolean. A non-json or
    empty stdout is not treated as an error here — the exit code is the primary signal —
    so this only catches a run that exited 0 yet reported an internal error. An
    unparseable or empty stdout is unexpected given ``--output-format json`` was
    requested, so it is logged as a warning (the exit code still decides) rather than
    silently passing.
    """
    try:
        result = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        logger.warning(
            "Implementer CLI stdout was not parseable JSON despite --output-format "
            "json (exit code stays authoritative): %r",
            stdout,
        )
        return False
    return bool(isinstance(result, dict) and result.get("is_error"))


# The result text can carry the agent's whole closing message; the log keeps enough to
# diagnose a wrong-but-clean run without flooding the line.
_RESULT_SNIPPET_CHARS = 500


def _claude_result_summary(stdout: str) -> str:
    """A log-ready summary of the CLI's json result: turn count + result snippet.

    Returns an empty string when the stdout is not the expected json object, so the
    completion log line degrades gracefully rather than raising over telemetry.
    """
    try:
        result = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(result, dict):
        return ""
    turns = result.get("num_turns")
    text = str(result.get("result", ""))[:_RESULT_SNIPPET_CHARS]
    return f" ({turns} turns): {text}"


@dataclass(frozen=True)
class ContainerImplementer:
    """Real :class:`Implementer`: build a slice by exec-ing ``claude`` in the build container.

    Satisfies the implementer protocol ``implement(slice_, *, container) -> None`` so it
    drops in where the fake implementer sits in tests and at the wiring site. It execs the
    headless ``claude`` CLI inside the already-cloned, branch-checked-out container, hands it
    the per-slice prompt, and lets it implement TDD-first and commit to ``slice.branch``. The
    lifecycle gates on the done-check that follows, so the contract here is only that the
    exec ran and committed; a non-zero exit (or an ``is_error`` json result) raises
    :class:`ImplementError`.

    Attributes:
        credential: The Anthropic credential (API key or subscription OAuth token).
        auth_mode: ``"api_key"`` (credential rides ``ANTHROPIC_API_KEY``) or
            ``"subscription"`` (credential rides ``CLAUDE_CODE_OAUTH_TOKEN``).
        model: The implementing model id; defaults to the
            :data:`~retinue.roles.Role.IMPLEMENTER` registry entry (Sonnet 4.6), which a
            repo's routing level can replace at the wiring site.
        max_turns: Hard cap on the agent loop, threaded to ``claude --max-turns`` so a
            runaway implement stops itself rather than being killed (with its done-check)
            by the arq job_timeout. The wiring site passes ``settings.implement_max_turns``.
        issue_facts: Fetches the issue's title/body on the worker (which has ``gh`` and
            the installation token) so they are baked into the prompt — the container
            cannot reach GitHub itself. ``None`` keeps the bare prompt.
    """

    credential: str
    auth_mode: str = "api_key"
    model: str = field(default_factory=lambda: resolve_model(Role.IMPLEMENTER))
    max_turns: int = _DEFAULT_IMPLEMENT_MAX_TURNS
    issue_facts: IssueFactsSource | None = None

    async def implement(
        self, slice_: Slice, *, container: Container, plan_path: str | None = None
    ) -> None:
        """Exec ``claude`` in ``container`` to build ``slice_``; raise on an errored run.

        ``plan_path``, when given, names a materialized plan the per-slice prompt instructs
        the subagent to read before building (the ad-hoc lane); the bare call passes nothing.
        """
        facts: ClassifyInput | None = None
        if self.issue_facts is not None:
            facts = await self.issue_facts(
                slice_.repo_full_name, slice_.issue_number
            )
        prompt = _implement_prompt(slice_, plan_path=plan_path, facts=facts)
        argv = _claude_argv(prompt=prompt, model=self.model, max_turns=self.max_turns)
        # The runner container execs as root, and the CLI refuses bypassPermissions
        # under root unless IS_SANDBOX=1 marks the env as a disposable sandbox —
        # which this container is.
        result = await container.run_command(argv, env={"IS_SANDBOX": "1"})
        if not result.ok:
            raise ImplementError(
                f"implementer for {slice_.branch} exited {result.exit_code}: "
                f"{result.stderr}"
            )
        if _claude_result_is_error(result.stdout):
            raise ImplementError(
                f"implementer for {slice_.branch} reported an error: {result.stdout}"
            )
        # The CLI's stdout is consumed here and the container is destroyed after the
        # build, so this line is the only forensic trace of what the agent reported —
        # without it a clean-but-wrong run (e.g. "I could not commit") is invisible.
        logger.info(
            "Implementer for %s completed in-container%s",
            slice_.branch,
            _claude_result_summary(result.stdout),
        )

    def auth_env(self) -> dict[str, str]:
        """The credential env the lifecycle merges into the build container at start."""
        return implement_env(self.credential, self.auth_mode)
