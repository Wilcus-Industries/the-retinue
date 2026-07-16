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
   before building, then implements TDD-first and commits to the ``issue-<N>`` branch; an
   implement run that lands **zero commits** on the branch fails the build (the shared
   hollow-implement guard) instead of pushing an empty branch,
5. **done-check** — the repo's done-check runs in the *same* container over the real
   changes, and the outcome is posted to the report sink,
6. **push** — only on a green done-check is ``issue-<N>`` pushed to origin; a red check
   pushes nothing,
7. **review** — after a green push, the internal reviewer (the
   :data:`~retinue.roles.Role.REVIEWER` Opus role) reviews the ``issue-<N>`` diff and
   files each finding as a flat ``review-fix`` + ``ready-for-agent`` follow-up issue that
   loops back as ordinary ad-hoc work. The review is **advisory**: it never blocks the
   build or the push, and any error it raises is swallowed. The review-fix **chain** is
   bounded by a lineage marker, not by the issue number: each filed review-fix issue
   carries a ``Chain-depth: <n>`` line in its body (:func:`render_chain_depth`), and a
   build whose issue is already at depth ``config.retry_cap`` files no further fixes. So
   the chain ``#29 -> #501 -> #503 -> ...`` terminates after ``retry_cap`` hops even though
   each hop is a fresh GitHub issue with its own number. This bound is self-carrying — it
   lives in the issue body, touches no shared store, and so never collides with triage's
   build-retry budget (:func:`~retinue.triage.triage_implementer`). It is only *live*,
   though, if the lane reads the marker back when it rebuilds a fetched issue: the ad-hoc
   drain (#32) must construct each issue with :meth:`AdhocIssue.from_fetched_issue`, the
   one constructor that parses ``Chain-depth:`` out of the fetched body into
   :attr:`AdhocIssue.chain_depth`. Building ``AdhocIssue`` by hand instead defaults every
   hop to depth 0 and makes the bound inert.

Every side-effecting collaborator — the planner spawn, the implementer spawn, the
reviewer, the container runtime, the auth, the secret resolver, and the report sink — is
injected, so the whole flow is exercised in tests with no Agent SDK, no Docker, no gh, and
no network. The whole container lifecycle — start, clone+branch, implement, the
hollow-implement guard, done-check, push-on-green, report, teardown — is the shared
:func:`retinue.container_build.build_issue_in_container` (the same lifecycle the PRD lane
runs), and the review filing reuses the internal reviewer's
:class:`~retinue.reviewer.ReviewGenerator` seam, ``review-fix`` label, and gh issue
creator; this module only adds the planner seam, the plan materialization, threading
:data:`PLAN_FILE` into the implementer as its ``plan_path``, the no-merge plan->execute
ordering, and the advisory, chain-depth-bounded third review pass.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from retinue.container import Container, ContainerRuntime
from retinue.container_build import (
    GitOpsError,
    Implementer,
    Slice,
    build_issue_in_container,
    implement_env,
    write_file_command,
)
from retinue.done_check import (
    DEFAULT_IMAGE,
    ReportSink,
    SecretResolver,
)
from retinue.github_app import InstallationAuth
from retinue.repo_config import RepoConfig
from retinue.reviewer import (
    REVIEW_FIX_LABEL,
    IssueCreator,
    IssueDraft,
    ReviewGenerator,
    ReviewInput,
)
from retinue.roles import Role, planner_cli_argv, resolve_model
from retinue.vocab import READY_LABEL, issue_branch

logger = logging.getLogger(__name__)

# The single file the captured plan is materialized into and the implementer reads. It
# lives under a dot-dir so it is unobtrusive in the worktree; the path is the contract
# between the read-only planner and the write-capable implementer, named in the planner's
# prompt so the plan it emits is framed as "what will land in this file".
PLAN_FILE = ".retinue/plan.md"

# The lineage marker stamped into a filed review-fix issue's body and read back when that
# issue loops in as ad-hoc work. It carries the chain's *depth* — how many review-fix hops
# separate this issue from the chain origin — so each hop inherits one decreasing budget
# and the chain terminates at ``config.retry_cap`` regardless of the fresh GitHub issue
# number each hop is filed under. Rendered like the slicer's ``Part of #<prd>`` footer and
# parsed back with :data:`_CHAIN_DEPTH_RE`. An issue with no marker is depth 0 (a chain
# origin: a hand-filed nit or the first ad-hoc build).
_CHAIN_DEPTH_PREFIX = "Chain-depth:"
_CHAIN_DEPTH_RE = re.compile(rf"^{re.escape(_CHAIN_DEPTH_PREFIX)}\s*(\d+)\s*$", re.MULTILINE)


def render_chain_depth(depth: int) -> str:
    """Render the ``Chain-depth: <depth>`` lineage marker for a filed review-fix body."""
    return f"{_CHAIN_DEPTH_PREFIX} {depth}"


def parse_chain_depth(body: str) -> int:
    """Read the chain depth from an issue ``body``; ``0`` when no marker is present.

    An issue carrying no :data:`_CHAIN_DEPTH_PREFIX` marker is a chain origin (a
    hand-filed nit or the first ad-hoc build), so it starts the chain at depth 0.
    """
    match = _CHAIN_DEPTH_RE.search(body)
    return int(match.group(1)) if match else 0


@dataclass(frozen=True)
class AdhocIssue:
    """One standalone issue the ad-hoc lane builds directly off staging.

    Attributes:
        repo_full_name: The target repo, e.g. "owner/repo".
        issue_number: The issue's GitHub number; the build commits to the derived
            ``issue-<N>`` branch.
        chain_depth: How many review-fix hops separate this issue from its chain origin.
            A hand-filed or first-built issue is depth 0; a review-fix this build files
            is stamped (and loops back as) the next depth. The review pass uses it — not
            the per-issue GitHub number — to bound the ``#29 -> #501 -> #503 -> ...``
            chain, since each hop is filed under a fresh number with no shared state.
    """

    repo_full_name: str
    issue_number: int
    chain_depth: int = 0

    @classmethod
    def from_fetched_issue(
        cls, repo_full_name: str, issue_number: int, body: str
    ) -> AdhocIssue:
        """Build the issue from a fetched GitHub issue, reading its lineage depth.

        The canonical constructor the ad-hoc drain (:mod:`retinue.lane`'s ranked drain,
        #32) must call instead of ``AdhocIssue(repo_full_name=..., issue_number=...)``.
        It parses the ``Chain-depth: <n>`` marker the prior review-fix hop stamped into
        ``body`` (:func:`parse_chain_depth`) and threads it into :attr:`chain_depth`, so
        the #39 review-fix chain bound — which lives entirely in the issue body — is read
        back and stops being inert. Building the issue by hand instead would default every
        hop to depth 0, leaving the ``#29 -> #501 -> #503 -> ...`` chain unbounded.

        Args:
            repo_full_name: The target repo, e.g. ``"owner/repo"``.
            issue_number: The fetched issue's GitHub number.
            body: The fetched issue body; its ``Chain-depth:`` marker carries the depth
                (a marker-less body is a chain origin, depth 0).
        """
        return cls(
            repo_full_name=repo_full_name,
            issue_number=issue_number,
            chain_depth=parse_chain_depth(body),
        )

    @property
    def branch(self) -> str:
        """The branch the build commits to: ``issue-<N>``."""
        return issue_branch(self.issue_number)


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
    return write_file_command(PLAN_FILE, plan)


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
            routing level can replace at the wiring site.
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
        return implement_env(self.credential, self.auth_mode)


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
        """The reviewer's credential env.

        Retained for wiring symmetry with the planner and implementer seams, but the ad-hoc
        build no longer injects it into the container — the reviewer's model call goes out
        over HTTP with the generator's own credential, so an in-container credential is unused.
        """
        ...


def _issue_diff_command(branch: str, base: str) -> list[str]:
    """Argv for the diff a freshly-built ``branch`` contributed over the staging ``base``.

    Uses the three-dot form (``origin/<base>...branch``) so the diff is the issue branch's
    own contribution since it was cut off the staging base — the work the build produced —
    rather than also folding in whatever else advanced staging in parallel. Both sides must
    resolve against the refs the build container actually has:
    :func:`retinue.container_build.clone_and_branch` runs ``git fetch origin <base>`` then
    ``git checkout -B issue-<N> origin/<base>``, so it creates the remote-tracking
    ``origin/<base>`` ref and the local ``issue-<N>`` branch but **no** bare local
    ``<base>`` (that exists only when ``staging_branch`` happens to be the clone's default
    HEAD). The base is therefore ``origin/<base>`` — mirroring the orchestrator's
    :func:`~retinue.orchestrator._branch_diff_command`, whose base side is
    also a resolvable ref — while the branch stays the *local* ``issue-<N>`` ref the build
    just committed to (no ``origin/`` prefix): the review runs in the same container that
    built it, so the local tip is the surface to review.
    """
    return ["git", "diff", f"origin/{base}...{branch}"]


@dataclass(frozen=True)
class ContainerAdhocReviewer:
    """Real :class:`AdhocReviewer`: review the issue diff and file flat review-fix issues.

    Satisfies the reviewer protocol ``review(issue, *, container) -> None`` so it drops in
    where the fake sits in tests and at the wiring site. The flow reuses the internal
    reviewer wholesale:

    1. diff the local ``issue-<N>`` branch over the remote-tracking
       ``origin/<config.staging_branch>`` ref inside the build container (the same
       container that built it, so the local tip is reviewed against a ref it actually
       has); a failed diff raises rather than yielding an empty review surface,
    2. run the injected :class:`~retinue.reviewer.ReviewGenerator` (the headless Agent-SDK
       reviewer, Opus/max) over that diff, and
    3. file each finding via the injected :class:`~retinue.slicer.IssueCreator` (gh) as a
       flat ``review-fix`` + ``ready-for-agent`` issue — no ``Part of #`` footer and no
       Blocked-by wiring, since ad-hoc work has no parent PRD; the fix loops back as
       ordinary ad-hoc work.

    The review-fix **chain** is bounded by the issue's :attr:`~AdhocIssue.chain_depth`, a
    ``Chain-depth: <n>`` lineage marker each filed fix carries in its body
    (:func:`render_chain_depth`) and the next ad-hoc build reads back
    (:func:`parse_chain_depth`). A build whose issue is already at depth
    ``config.retry_cap`` files no further fixes, and every fix this build files is stamped
    one depth deeper — so the chain ``#29 -> #501 -> #503 -> ...`` terminates after
    ``retry_cap`` hops even though each hop is a fresh GitHub issue number. The bound is
    carried in the issue body, not in a shared counter, so it never raids — and is never
    raided by — triage's build-retry budget
    (:func:`~retinue.triage.triage_implementer`). A clean review (no findings) does not
    extend the chain.

    Attributes:
        repo_full_name: The target repo the review-fix issues are filed against.
        config: The accepted repo config; ``staging_branch`` is the diff base and
            ``retry_cap`` bounds the review-fix chain depth.
        generate: The headless Agent-SDK reviewer (the :class:`ReviewGenerator` seam).
        create_issue: The gh issue creator filing each flat review-fix issue (slicer's
            seam, reused from the internal reviewer).
        credential: The reviewing model's credential; retained for wiring symmetry with the
            planner and implementer, but NOT injected into the build container — the reviewer
            runs over HTTP with the generator's own credential.
        auth_mode: ``"api_key"`` or ``"subscription"``; routes :meth:`auth_env`.
    """

    repo_full_name: str
    config: RepoConfig
    generate: ReviewGenerator
    create_issue: IssueCreator
    credential: str = ""
    auth_mode: str = "api_key"

    async def review(self, issue: AdhocIssue, *, container: Container) -> None:
        """Review ``issue``'s diff and file each finding as a flat review-fix issue."""
        if issue.chain_depth >= self.config.retry_cap:
            # The chain has reached its bound; do not file more fixes. Keyed on the
            # issue's own lineage depth, not the GitHub number, so a freshly-numbered
            # review-fix issue still inherits the chain's spent budget and the loop
            # ``#29 -> #501 -> #503 -> ...`` terminates.
            logger.info(
                "Skipping ad-hoc review of %s: chain depth reached (%d/%d)",
                issue.branch,
                issue.chain_depth,
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
        next_depth = issue.chain_depth + 1
        for finding in plan.findings:
            await self.create_issue(
                IssueDraft(
                    title=finding.title,
                    body=f"{finding.body.rstrip()}\n\n{render_chain_depth(next_depth)}",
                    labels=[READY_LABEL, REVIEW_FIX_LABEL],
                )
            )
        logger.info(
            "Ad-hoc review of %s filed %d review-fix follow-up(s) at chain depth %d",
            issue.branch,
            len(plan.findings),
            next_depth,
        )

    async def _issue_diff(self, issue: AdhocIssue, container: Container) -> str:
        """Return the issue branch's contribution over the staging base, from the build.

        A non-zero diff exit (e.g. an unresolvable base ref) raises rather than returning
        the empty stdout: a failed diff must not be silently treated as an empty review
        surface — that would leave the very branch the reviewer is meant to review
        unreviewed with no error surfaced. The advisory wrapper
        (:func:`_review_advisory`) still swallows the raised error, but logs it.
        """
        command = _issue_diff_command(issue.branch, self.config.staging_branch)
        result = await container.run_command(command)
        if not result.ok:
            raise GitOpsError(
                f"ad-hoc review diff for {issue.branch} exited "
                f"{result.exit_code}: {result.stderr}"
            )
        return result.stdout

    def auth_env(self) -> dict[str, str]:
        """The reviewer's credential env (empty when no credential is set).

        Retained for wiring symmetry with the planner and implementer seams; the ad-hoc
        build no longer injects it in-container, since the reviewer's model call goes out
        over HTTP with the generator's own credential.
        """
        if not self.credential:
            return {}
        return implement_env(self.credential, self.auth_mode)


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

    Runs the shared per-issue lifecycle
    (:func:`retinue.container_build.build_issue_in_container`) in a single disposable
    container, destroyed on every path — parse the done-check, resolve the secrets, start
    the container, clone and cut ``issue-<N>`` off ``config.staging_branch``, implement,
    guard against a hollow implement (zero commits fails the build), done-check, push on
    green, report — with the ad-hoc lane's hooks threaded in:

    1. **pre-implement**: run the read-only planner to produce a plan, captured from its
       output, and materialize it into :data:`PLAN_FILE` for the implementer to read,
    2. **plan_path**: the implementer is pointed at :data:`PLAN_FILE` so it reads the plan
       first,
    3. **credentials**: the planner's and the implementer's credential env ride the
       container at ``start`` — the two roles that exec in-container; the reviewer runs
       over HTTP, so its credential is not injected,
    4. **on green**: run the injected ``reviewer`` (when present) over the ``issue-<N>``
       diff to file review-fix follow-ups. The review is **advisory**: any error it raises
       is logged and swallowed so it never undoes the green build or its push, and a red
       build is never reviewed (there is no built work to review).

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
        failure, a :class:`PlanError` from the planner exec, or an
        :class:`~retinue.container_build.ImplementError` from a hollow implement). The
        advisory review pass is the one step that never propagates: its errors are
        swallowed.
    """

    async def plan_and_materialize(container: Container) -> None:
        plan = await planner.plan(issue, container=container)
        await _materialize_plan(container, plan)

    on_green: Callable[[Container], Awaitable[None]] | None = None
    if reviewer is not None:
        present_reviewer = reviewer

        async def review_on_green(container: Container) -> None:
            await _review_advisory(present_reviewer, issue, container)

        on_green = review_on_green

    passed = await build_issue_in_container(
        _slice_for_issue(issue),
        config,
        claude_md,
        base=config.staging_branch,
        implementer=implementer,
        auth=auth,
        runtime=runtime,
        resolve_secret=resolve_secret,
        report=report,
        image=image,
        lane_label="Ad-hoc issue",
        extra_auth_envs=[planner.auth_env()],
        pre_implement=plan_and_materialize,
        plan_path=PLAN_FILE,
        on_green=on_green,
    )
    return AdhocBuildResult(branch=issue.branch, passed=passed)


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
