"""Ad-hoc build primitive: plan -> materialize -> implement -> commit, gated on done-check.

The ad-hoc lane (:mod:`retinue.lane`) routes a standalone ``ready-for-agent`` issue — one
with no ``Part of #<prd>`` link — here. Unlike a PRD slice there is no integration branch
and no merge: the issue is built directly on an ``issue-<N>`` branch cut off the repo's
``config.staging_branch``, and a green build pushes that branch for a human to open a PR
from. The whole build runs in **one disposable container** that is destroyed on every path
(:func:`build_adhoc_issue`):

1. **clone + branch** — the container clones the repo over the installation token and
   checks out a fresh ``issue-<N>`` branch off ``config.staging_branch``,
2. **plan** — the read-only planner (the :data:`~retinue.roles.Role.PLANNER` registry
   entry, Opus on the in-container CLI) maps the code with an Explore subagent and emits a
   plan, captured from its output (it writes nothing to the workspace),
3. **materialize** — the captured plan is written byte-exact into :data:`PLAN_FILE`, the
   one file the implementer reads, so the plan crosses from the read-only planner to the
   write-capable implementer through the workspace rather than a second model call,
4. **implement** — the same implementer the PRD lane uses (Sonnet/high on the in-container
   CLI) implements TDD-first and commits to the ``issue-<N>`` branch,
5. **done-check** — the repo's done-check runs in the *same* container over the real
   changes, and the outcome is posted to the report sink,
6. **push** — only on a green done-check is ``issue-<N>`` pushed to origin; a red check
   pushes nothing.

Every side-effecting collaborator — the planner spawn, the implementer spawn, the
container runtime, the auth, the secret resolver, and the report sink — is injected, so
the whole flow is exercised in tests with no Agent SDK, no Docker, no gh, and no network.
The container/git/done-check/credential mechanics are reused wholesale from
:mod:`retinue.orchestrator` and :mod:`retinue.done_check`; this module only adds the
planner seam, the plan materialization, and the no-merge plan->execute ordering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from retinue.container import Container, ContainerRuntime
from retinue.done_check import (
    DEFAULT_IMAGE,
    DoneCheckReport,
    ReportSink,
    SecretResolver,
    parse_done_check,
    resolve_secrets_or_escalate,
    run_done_check_commands,
)
from retinue.github_app import InstallationAuth
from retinue.orchestrator import (
    _GIT_COMMITTER_ENV,
    Implementer,
    Slice,
    _clone_and_branch,
    _implement_env,
    _push_branch,
    _write_file_command,
)
from retinue.repo_config import RepoConfig
from retinue.roles import Role, planner_cli_argv, resolve_model

logger = logging.getLogger(__name__)

# The single file the captured plan is materialized into and the implementer reads. It
# lives under a dot-dir so it is unobtrusive in the worktree; the path is the contract
# between the read-only planner and the write-capable implementer, named in the planner's
# prompt so the plan it emits is framed as "what will land in this file".
PLAN_FILE = ".retinue/plan.md"


@dataclass(frozen=True)
class AdhocIssue:
    """One standalone issue the ad-hoc lane builds directly off staging.

    Attributes:
        repo_full_name: The target repo, e.g. "owner/repo".
        issue_number: The issue's GitHub number; the build commits to the derived
            ``issue-<N>`` branch.
    """

    repo_full_name: str
    issue_number: int

    @property
    def branch(self) -> str:
        """The branch the build commits to: ``issue-<N>``."""
        return f"issue-{self.issue_number}"


class Planner(Protocol):
    """Spawns one read-only planner that produces a plan for an issue. The planner seam.

    A production implementation execs the read-only headless ``claude`` CLI *inside the
    disposable build container* the orchestration passes in (the same container the
    implementer later builds in), maps the code with an Explore subagent, and returns the
    plan captured from the run's output. Tests inject a fake that records the request and
    returns a canned plan without any real spawn. The plan text is the contract — the
    orchestration materializes it into :data:`PLAN_FILE` for the implementer to read.
    """

    async def plan(self, issue: AdhocIssue, *, container: Container) -> str:
        """Plan ``issue`` in ``container`` (read-only) and return the captured plan."""
        ...

    def auth_env(self) -> dict[str, str]:
        """The env the planner authenticates with, merged into the container at start."""
        ...


def _plan_prompt(issue: AdhocIssue) -> str:
    """Assemble the planner's per-issue prompt: which issue to plan, where the plan lands.

    Names the issue to plan and tells the planner its emitted plan will be materialized
    into :data:`PLAN_FILE` for the implementer to read, so the planner frames its output
    as the implementation plan that file will carry rather than incidental prose.
    """
    return (
        f"Produce an implementation plan for issue #{issue.issue_number} of "
        f"{issue.repo_full_name}. Your plan will be saved to '{PLAN_FILE}' and read by "
        "the implementer, so write the plan itself as your response."
    )


def _materialize_plan_command(plan: str) -> list[str]:
    """Argv that writes the captured ``plan`` into :data:`PLAN_FILE`, byte-exact.

    Reuses the orchestrator's base64 in-container file writer so the plan's markdown —
    backticks, quotes, newlines — survives untouched and nothing in it is interpreted as
    shell syntax. The parent dot-dir is created first so the write into it can't fail.
    """
    return _write_file_command(PLAN_FILE, plan)


_ENSURE_PLAN_DIR_COMMAND = ["mkdir", "-p", ".retinue"]


class PlanError(RuntimeError):
    """The read-only planner run ended in an error rather than producing a plan.

    Raised when the in-container ``claude`` plan exec exits non-zero, so the build surfaces
    the failure rather than materializing an empty or half plan and implementing against it.
    """


@dataclass(frozen=True)
class ContainerPlanner:
    """Real :class:`Planner`: produce a plan by exec-ing the read-only ``claude`` CLI.

    Satisfies the planner protocol ``plan(issue, *, container) -> str`` so it drops in
    where the fake planner sits in tests and at the wiring site. It execs the read-only
    headless ``claude`` argv (:func:`retinue.roles.planner_cli_argv`) inside the already
    cloned, branch-checked-out container and returns the plan captured from the run's
    stdout. A non-zero exit raises :class:`PlanError`. The plan is captured from output,
    so unlike the implementer there is no ``--output-format json`` result contract.

    Attributes:
        credential: The Anthropic credential (API key or subscription OAuth token).
        auth_mode: ``"api_key"`` (credential rides ``ANTHROPIC_API_KEY``) or
            ``"subscription"`` (credential rides ``CLAUDE_CODE_OAUTH_TOKEN``).
        model: The planning model id; defaults to the
            :data:`~retinue.roles.Role.PLANNER` registry entry (Opus 4.8), which a repo's
            ``models`` override can replace at the wiring site.
    """

    credential: str
    auth_mode: str = "api_key"
    model: str = field(default_factory=lambda: resolve_model(Role.PLANNER))

    async def plan(self, issue: AdhocIssue, *, container: Container) -> str:
        """Exec the read-only ``claude`` planner in ``container``; return its plan."""
        argv = planner_cli_argv(prompt=_plan_prompt(issue), model=self.model)
        result = await container.run_command(argv)
        if not result.ok:
            raise PlanError(
                f"planner for {issue.branch} exited {result.exit_code}: {result.stderr}"
            )
        logger.info("Planner for %s produced a plan in-container", issue.branch)
        return result.stdout

    def auth_env(self) -> dict[str, str]:
        """The credential env the orchestration merges into the build container at start."""
        return _implement_env(self.credential, self.auth_mode)


@dataclass(frozen=True)
class AdhocBuildResult:
    """Result of building one ad-hoc issue.

    Attributes:
        branch: The ``issue-<N>`` branch the build targeted (pushed only when ``passed``).
        passed: True when the done-check was green (and the branch was pushed); False on
            a red check, where nothing was pushed.
    """

    branch: str
    passed: bool


async def build_adhoc_issue(
    issue: AdhocIssue,
    config: RepoConfig,
    claude_md: str,
    *,
    planner: Planner,
    implementer: Implementer,
    auth: InstallationAuth,
    runtime: ContainerRuntime,
    resolve_secret: SecretResolver,
    report: ReportSink,
    image: str = DEFAULT_IMAGE,
) -> AdhocBuildResult:
    """Build one ad-hoc issue in one container: plan -> materialize -> implement -> push.

    Runs the whole build in a single disposable container, destroyed on every path:

    1. parse the done-check and resolve the config's secrets (a missing one escalates on
       the report sink and propagates *before* any container starts),
    2. start the container with the secrets, the git committer identity, and *both* the
       planner's and the implementer's credential env (the env is fixed at ``start``),
    3. clone the repo and check out a fresh ``issue-<N>`` branch off ``config.staging_branch``,
    4. run the read-only planner to produce a plan, captured from its output,
    5. materialize the plan into :data:`PLAN_FILE` for the implementer to read,
    6. exec the implementer to build and commit the issue on ``issue-<N>``,
    7. run the done-check over the real changes and post the outcome,
    8. push ``issue-<N>`` to origin only when the done-check is green (a red build pushes
       nothing).

    Args:
        issue: The ad-hoc issue to build (repo, issue number).
        config: The accepted repo config; its ``staging_branch`` is the issue-branch base
            and its ``secrets`` are injected into the container.
        claude_md: The repo's ``CLAUDE.md`` text, carrying the done-check command.
        planner: Execs the read-only planner in the container (the planner seam).
        implementer: Execs the implementer subagent in the container (the Agent SDK seam).
        auth: Mints the installation token used to clone (the auth seam).
        runtime: Spawns the disposable build container (the Docker seam).
        resolve_secret: Resolves the config's declared secret names/refs to values.
        report: Sink the done-check outcome is posted to (commit status / comment).
        image: Container image the build runs in.

    Returns:
        An :class:`AdhocBuildResult`: ``passed=True`` when the green branch was pushed,
        ``passed=False`` when a red done-check pushed nothing.

    Raises:
        Propagates whatever the build container raises (e.g. a missing secret, a clone
        failure, or a :class:`PlanError` from the planner exec).
    """
    commands = parse_done_check(claude_md)
    env = await resolve_secrets_or_escalate(
        issue.repo_full_name, config, resolve_secret, report
    )
    start_env = {
        **env,
        **_GIT_COMMITTER_ENV,
        **planner.auth_env(),
        **implementer.auth_env(),
    }
    token = await auth.installation_token(issue.repo_full_name)
    container = await runtime.start(image=image, env=start_env)
    try:
        await _clone_and_branch(
            container,
            token.clone_url,
            branch=issue.branch,
            base=config.staging_branch,
        )
        plan = await planner.plan(issue, container=container)
        await _materialize_plan(container, plan)
        await implementer.implement(_slice_for_issue(issue), container=container)
        passed, detail = await run_done_check_commands(container, commands)
        if passed:
            await _push_branch(container, issue.branch)
        await report(
            DoneCheckReport(
                repo_full_name=issue.repo_full_name,
                passed=passed,
                escalated=False,
                detail=detail,
            )
        )
        logger.info(
            "Ad-hoc issue %s done-check %s",
            issue.branch,
            "passed" if passed else "failed",
        )
        return AdhocBuildResult(branch=issue.branch, passed=passed)
    finally:
        # Guaranteed teardown: the disposable container is destroyed on every path,
        # including when clone, plan, implement, the done-check, or push raises.
        await container.destroy()


async def _materialize_plan(container: Container, plan: str) -> None:
    """Write the captured plan into :data:`PLAN_FILE` inside ``container``, byte-exact."""
    await container.run_command(_ENSURE_PLAN_DIR_COMMAND)
    await container.run_command(_materialize_plan_command(plan))


def _slice_for_issue(issue: AdhocIssue) -> Slice:
    """Adapt an :class:`AdhocIssue` to the :class:`~retinue.orchestrator.Slice` seam.

    The implementer seam is shared with the PRD lane, whose contract is a ``Slice``. An
    ad-hoc issue has no parent PRD, so it stands on its own integration target: the
    per-issue PRD number is the issue number itself (the same convention the cron lane uses
    for a standalone backlog nit), which only feeds the ``issue-<N>`` branch the implementer
    already commits to here — the ad-hoc lane never merges onto that target.
    """
    return Slice(
        repo_full_name=issue.repo_full_name,
        issue_number=issue.issue_number,
        prd_number=issue.issue_number,
    )
