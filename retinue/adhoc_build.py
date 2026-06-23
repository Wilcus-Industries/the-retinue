"""Ad-hoc build primitive: plan -> materialize -> implement -> review, gated on done-check.

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
   CLI) is pointed at :data:`PLAN_FILE` via its ``plan_path`` and told to read the plan
   before building, then implements TDD-first and commits to the ``issue-<N>`` branch,
5. **done-check** — the repo's done-check runs in the *same* container over the real
   changes, and the outcome is posted to the report sink,
6. **push** — only on a green done-check is ``issue-<N>`` pushed to origin; a red check
   pushes nothing,
7. **review** — after a green push, the internal reviewer (the
   :data:`~retinue.roles.Role.REVIEWER` Opus role) reviews the ``issue-<N>`` diff and
   files each finding as a flat ``review-fix`` + ``ready-for-agent`` follow-up issue that
   loops back as ordinary ad-hoc work. The review is **advisory**: it never blocks the
   build or the push, and any error it raises is swallowed. The review-fix chain is bounded
   by the per-unit retry cap (the same persisted counter triage uses), so a review-fix
   issue cannot spawn review fixes without limit.

Every side-effecting collaborator — the planner spawn, the implementer spawn, the
reviewer, the container runtime, the auth, the secret resolver, and the report sink — is
injected, so the whole flow is exercised in tests with no Agent SDK, no Docker, no gh, and
no network. The container/git/done-check/credential mechanics are reused wholesale from
:mod:`retinue.orchestrator` and :mod:`retinue.done_check`, and the review filing reuses the
internal reviewer's :class:`~retinue.reviewer.ReviewGenerator` seam, ``review-fix`` label,
and gh issue creator; this module only adds the planner seam, the plan materialization,
threading :data:`PLAN_FILE` into the implementer as its ``plan_path``, the no-merge
plan->execute ordering, and the advisory, retry-cap-bounded third review pass.
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
from retinue.impl_retry import ImplRetryStore, impl_retry_key
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
from retinue.reviewer import (
    READY_LABEL,
    REVIEW_FIX_LABEL,
    IssueCreator,
    IssueDraft,
    ReviewGenerator,
    ReviewInput,
)
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


class AdhocReviewer(Protocol):
    """Reviews a freshly-built ``issue-<N>`` diff and files review-fix follow-ups.

    The third pass of the ad-hoc build. After a green build pushes ``issue-<N>``, the
    orchestration hands the issue and the build container here; a production
    implementation diffs the issue branch over the staging base, runs the internal
    reviewer over that diff, and files each finding as a flat ``review-fix`` +
    ``ready-for-agent`` follow-up issue (no ``Part of #`` footer — ad-hoc work has no
    parent PRD — and no Blocked-by wiring; the fix loops back as ordinary ad-hoc work).
    The review is advisory: :func:`build_adhoc_issue` swallows whatever ``review`` raises
    so a reviewer error never undoes the green build or its push. Tests inject a fake that
    records the request without the Agent SDK, gh, or network.
    """

    async def review(self, issue: AdhocIssue, *, container: Container) -> None:
        """Review ``issue``'s diff in ``container`` and file review-fix follow-ups."""
        ...

    def auth_env(self) -> dict[str, str]:
        """The credential env merged into the build container at start, for symmetry."""
        ...


def _issue_diff_command(branch: str, base: str) -> list[str]:
    """Argv for the diff a freshly-built ``branch`` contributed over the staging ``base``.

    Uses the three-dot form (``base...branch``) so the diff is the issue branch's own
    contribution since it was cut off the staging base — the work the build produced —
    rather than also folding in whatever else advanced staging in parallel. The branch is
    the *local* ``issue-<N>`` ref the build just committed to (no ``origin/`` prefix): the
    review runs in the same container that built it, so the local tip is the surface to
    review.
    """
    return ["git", "diff", f"{base}...{branch}"]


@dataclass(frozen=True)
class ContainerAdhocReviewer:
    """Real :class:`AdhocReviewer`: review the issue diff and file flat review-fix issues.

    Satisfies the reviewer protocol ``review(issue, *, container) -> None`` so it drops in
    where the fake sits in tests and at the wiring site. The flow reuses the internal
    reviewer wholesale:

    1. diff the ``issue-<N>`` branch over ``config.staging_branch`` inside the build
       container (the same container that built it, so the local tip is reviewed),
    2. run the injected :class:`~retinue.reviewer.ReviewGenerator` (the headless Agent-SDK
       reviewer, Opus/max) over that diff, and
    3. file each finding via the injected :class:`~retinue.slicer.IssueCreator` (gh) as a
       flat ``review-fix`` + ``ready-for-agent`` issue — no ``Part of #`` footer and no
       Blocked-by wiring, since ad-hoc work has no parent PRD; the fix loops back as
       ordinary ad-hoc work.

    The review-fix chain is bounded by the **per-unit retry cap**: the reviewer reuses the
    persisted :class:`~retinue.impl_retry.ImplRetryStore` keyed by the issue's
    :func:`~retinue.impl_retry.impl_retry_key`. A unit that has already spent its
    ``config.retry_cap`` review budget files nothing more, so a review-fix issue cannot
    spawn review fixes without limit. A review that files at least one fix records one
    attempt against the cap; a clean review consumes no budget.

    Attributes:
        repo_full_name: The target repo the review-fix issues are filed against.
        config: The accepted repo config; ``staging_branch`` is the diff base and
            ``retry_cap`` bounds the review-fix chain.
        generate: The headless Agent-SDK reviewer (the :class:`ReviewGenerator` seam).
        create_issue: The gh issue creator filing each flat review-fix issue (slicer's
            seam, reused from the internal reviewer).
        retry_store: The persisted per-unit attempt counter bounding the chain.
        credential: The reviewing model's credential; for seam symmetry with the planner
            and implementer (the generator holds its own credential over HTTP).
        auth_mode: ``"api_key"`` or ``"subscription"``; routes :meth:`auth_env`.
    """

    repo_full_name: str
    config: RepoConfig
    generate: ReviewGenerator
    create_issue: IssueCreator
    retry_store: ImplRetryStore
    credential: str = ""
    auth_mode: str = "api_key"

    async def review(self, issue: AdhocIssue, *, container: Container) -> None:
        """Review ``issue``'s diff and file each finding as a flat review-fix issue."""
        key = impl_retry_key(_slice_for_issue(issue))
        spent = await self.retry_store.count(key)
        if spent >= self.config.retry_cap:
            # The unit has spent its whole review budget; do not file more fixes — this
            # bounds the review-fix chain so it cannot loop without limit.
            logger.info(
                "Skipping ad-hoc review of %s: retry budget spent (%d/%d)",
                issue.branch,
                spent,
                self.config.retry_cap,
            )
            return
        diff = await self._issue_diff(issue, container)
        plan = await self.generate(
            ReviewInput(
                repo_full_name=self.repo_full_name,
                prd_number=issue.issue_number,
                merged_issues=[],
                diff=diff,
            )
        )
        if not plan.findings:
            logger.info("Ad-hoc review of %s found nothing to fix", issue.branch)
            return
        for finding in plan.findings:
            await self.create_issue(
                IssueDraft(
                    title=finding.title,
                    body=finding.body.rstrip(),
                    labels=[READY_LABEL, REVIEW_FIX_LABEL],
                )
            )
        # One review pass that filed fixes spends one unit of the retry budget, so a
        # review-fix issue that loops back gets at most ``retry_cap`` review passes.
        await self.retry_store.record_attempt(key)
        logger.info(
            "Ad-hoc review of %s filed %d review-fix follow-up(s)",
            issue.branch,
            len(plan.findings),
        )

    async def _issue_diff(self, issue: AdhocIssue, container: Container) -> str:
        """Return the issue branch's contribution over the staging base, from the build."""
        result = await container.run_command(
            _issue_diff_command(issue.branch, self.config.staging_branch)
        )
        return result.stdout

    def auth_env(self) -> dict[str, str]:
        """The credential env the orchestration merges into the build container at start."""
        if not self.credential:
            return {}
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
    reviewer: AdhocReviewer | None = None,
    image: str = DEFAULT_IMAGE,
) -> AdhocBuildResult:
    """Build one ad-hoc issue in one container: plan -> implement -> push -> review.

    Runs the whole build in a single disposable container, destroyed on every path:

    1. parse the done-check and resolve the config's secrets (a missing one escalates on
       the report sink and propagates *before* any container starts),
    2. start the container with the secrets, the git committer identity, and the planner's,
       the implementer's, and the reviewer's credential env (the env is fixed at ``start``),
    3. clone the repo and check out a fresh ``issue-<N>`` branch off ``config.staging_branch``,
    4. run the read-only planner to produce a plan, captured from its output,
    5. materialize the plan into :data:`PLAN_FILE` for the implementer to read,
    6. exec the implementer — pointed at :data:`PLAN_FILE` via ``plan_path`` so it reads
       the plan first — to build and commit the issue on ``issue-<N>``,
    7. run the done-check over the real changes and post the outcome,
    8. push ``issue-<N>`` to origin only when the done-check is green (a red build pushes
       nothing),
    9. on a green build only, run the injected ``reviewer`` (when present) over the
       ``issue-<N>`` diff to file review-fix follow-ups. The review is **advisory**: any
       error it raises is logged and swallowed so it never undoes the green build or its
       push, and a red build is never reviewed (there is no built work to review).

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
        reviewer: The internal reviewer run after a green build (the reviewer seam);
            absent means no third review pass (and no review-fix follow-ups).
        image: Container image the build runs in.

    Returns:
        An :class:`AdhocBuildResult`: ``passed=True`` when the green branch was pushed,
        ``passed=False`` when a red done-check pushed nothing.

    Raises:
        Propagates whatever the build container raises (e.g. a missing secret, a clone
        failure, or a :class:`PlanError` from the planner exec). The advisory review pass
        is the one step that never propagates: its errors are swallowed.
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
        **(reviewer.auth_env() if reviewer is not None else {}),
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
        await implementer.implement(
            _slice_for_issue(issue), container=container, plan_path=PLAN_FILE
        )
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
        if passed and reviewer is not None:
            await _review_advisory(reviewer, issue, container)
        return AdhocBuildResult(branch=issue.branch, passed=passed)
    finally:
        # Guaranteed teardown: the disposable container is destroyed on every path,
        # including when clone, plan, implement, the done-check, or push raises.
        await container.destroy()


async def _review_advisory(
    reviewer: AdhocReviewer, issue: AdhocIssue, container: Container
) -> None:
    """Run the third review pass, swallowing any error so it never blocks the build.

    The review is advisory — it files review-fix follow-ups but must not undo the green
    build or its push — so a reviewer failure is logged and dropped rather than propagated
    out of :func:`build_adhoc_issue`. The container is still torn down by the build's
    ``finally`` regardless.
    """
    try:
        await reviewer.review(issue, container=container)
    except Exception:
        # Review is advisory: never let it block the build/PR path. KeyboardInterrupt/
        # SystemExit are not caught — only Exception — so process control still propagates.
        logger.warning(
            "Ad-hoc review of %s failed; continuing (review is advisory)",
            issue.branch,
            exc_info=True,
        )


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
