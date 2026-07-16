"""The per-issue container-build lifecycle shared by the PRD and ad-hoc lanes.

Both build lanes run one issue's whole build inside **one disposable container** that is
destroyed on every path: parse the done-check, resolve the config's secrets, start the
container (secrets + git committer identity + the in-container agents' credentials all in
its env), clone the repo and check out a fresh ``issue-<N>`` branch off the lane's base,
exec the implementer, guard against a hollow implement (zero commits landed), run the
repo's done-check over the real changes, push the branch only on green, and post the
outcome to the report sink. :func:`build_issue_in_container` owns that lifecycle; the
lanes own only what genuinely differs, passed as hooks:

- the **PRD lane** (:func:`retinue.orchestrator.build_slice` / ``build_prd``) branches off
  the integration branch ``retinue/prd-<n>`` and passes no hooks,
- the **ad-hoc lane** (:func:`retinue.adhoc_build.build_adhoc_issue`) branches off the
  repo's staging branch, runs a read-only planner before the implement
  (``pre_implement``), points the implementer at the materialized plan (``plan_path``),
  and runs an advisory review after a green check (``on_green``).

The building blocks the lifecycle drives — the :class:`Slice` unit, the
:class:`Implementer` seam, the git command builders, and the in-container git helpers —
live here too, so both lanes (and the orchestrator's merge seam) share one public set
instead of reaching into each other's privates. Every side-effecting collaborator is
injected, so the whole flow is exercised in tests with no Agent SDK, no Docker, no gh,
and no network.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

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
        subagent must read before building. The PRD lane passes nothing (``None``), so its
        prompt is unchanged; the ad-hoc lane passes its ``PLAN_FILE`` so the subagent is
        pointed at the plan the planner wrote.
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
    """A ``git`` command failed for a reason other than a recoverable merge conflict.

    Distinct from :class:`retinue.orchestrator.MergeConflict`: a conflict is handed to
    the resolver, but a hard error (unknown ref, not a repository, checkout failure)
    means the branch could not be advanced at all, so it propagates rather than
    masquerading as a conflict the resolver could fix.
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
            # The bare call shape is kept for the PRD lane so an injected implementer
            # that predates the plan_path parameter still satisfies the seam.
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
