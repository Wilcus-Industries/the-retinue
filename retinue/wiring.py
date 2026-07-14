"""Production wiring for the orchestrator build + cron lanes (the budget/triage glue).

Two bindings live here, each taking the implementer-spawn seam as their one injected
dependency (the Agent-SDK subagent is owned by a separate layer) and wiring every other
collaborator to its real adapter:

* :func:`bind_build_prd` — the orchestrator build lane. It gates the run on the shared
  :class:`retinue.budget.BudgetGovernor` (deferring a run whose estimate would start it
  over the rolling-24h cap), then runs :func:`retinue.orchestrator.build_prd` with the
  implementer wrapped in :func:`retinue.triage.triage_implementer` so a hard failure or
  mis-scope is reasoned about against the persisted :class:`retinue.impl_retry.ImplRetryStore`
  cap rather than blindly retried.
* :func:`bind_cron_tick` — the cron backlog lane. It binds :func:`retinue.cron.run_cron_tick`
  to its real collaborators (the gh backlog query, the shared governor, the clock, the
  single-run lock, and the downstream build) so a scheduled tick drains one backlog issue.

The budget *meter* (mid-run pause/resume) is exposed through the governor passed in, which
the cron lane and the build lane share — one service-level governor across both.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from retinue.adhoc_drain import AdhocBuild, AdhocGh, AdhocPrOpen, run_adhoc_drain
from retinue.budget import BudgetGovernor, Clock
from retinue.container import Container, ContainerRuntime
from retinue.cron import CronBuild, CronGh, CronTickResult, run_cron_tick
from retinue.done_check import ReportSink, SecretResolver
from retinue.github_app import InstallationAuth
from retinue.impl_retry import ImplRetryStore
from retinue.notify import Notifier
from retinue.orchestrator import (
    GitOps,
    Implementer,
    PrdBuildResult,
    PrdSlice,
    RoundReviewer,
    Slice,
    build_prd,
    integration_branch,
)
from retinue.repo_config import RepoConfig
from retinue.reviewer import (
    BlockedByEditor,
    ReviewGenerator,
    ReviewInput,
    review_round,
)
from retinue.slicer import IssueCreator
from retinue.triage import TriageImplementer, triage_implementer

logger = logging.getLogger(__name__)


class RoundDiffSource(Protocol):
    """Produces a merged round's diff for the internal reviewer. The diff seam.

    A production implementation runs ``git diff`` in the merge container that advanced
    the integration branch (so the reviewer reads the round's merged work from the same
    clone); tests inject a fake returning a canned diff. ``merged_branches`` are the
    round's merged ``issue-<N>`` branches; ``base`` is the integration branch.
    """

    async def round_diff(self, *, merged_branches: list[str], base: str) -> str:
        """Return the merged diff of ``merged_branches`` over the integration ``base``."""
        ...


@dataclass(frozen=True)
class BoundBuildResult:
    """Outcome of a budget-gated orchestrator build.

    Attributes:
        deferred: True when the budget gate held the run back (over the 24h cap); the
            build never ran and ``prd_build`` is ``None``.
        defer_until: When the window frees on a deferred run; ``None`` otherwise.
        prd_build: The orchestrator's :class:`PrdBuildResult` when the run executed.
    """

    deferred: bool
    defer_until: datetime | None = None
    prd_build: PrdBuildResult | None = None


def bind_build_prd(
    *,
    implementer: Implementer | TriageImplementer,
    governor: BudgetGovernor,
    notifier: Notifier,
    create_issue: IssueCreator,
    retry_store_path: Path,
    estimated_amount: float,
    git: GitOps,
    auth: InstallationAuth,
    runtime: ContainerRuntime,
    resolve_secret: SecretResolver,
    report: ReportSink,
    review_reviewer: ReviewerFactory | None = None,
) -> Callable[..., Awaitable[BoundBuildResult]]:
    """Bind the budget-gated, triage-wrapped orchestrator build.

    Returns an async ``(repo_full_name, prd_number, slices, config, claude_md) ->
    BoundBuildResult``. It first asks the shared ``governor`` to gate the run by its
    ``estimated_amount``; a deferred gate returns immediately without building (and
    charges nothing), while an admitted run's estimate is recorded on the shared
    rolling-24h ledger at the gate. Otherwise it runs
    :func:`retinue.orchestrator.build_prd`, wrapping each implementer attempt in
    :func:`retinue.triage.triage_implementer` so a failure/mis-scope is reasoned about
    against the persisted retry cap (retry / reslice / escalate) instead of a blind loop.

    Args:
        implementer: The implementer-spawn seam (Agent SDK), triaged on failure.
        governor: The shared service-level budget governor.
        notifier: The escalation fan-out used by triage's escalate path.
        create_issue: The gh issue creator used by triage's reslice path.
        retry_store_path: SQLite file backing the persisted per-slice retry counter.
        estimated_amount: The run's estimated charge, gated against — and recorded on —
            the rolling-24h ledger when the run is admitted.
        git: Integration-branch git operations (the merge seam).
        auth: Mints the installation token used to clone.
        runtime: Spawns the disposable done-check container.
        resolve_secret: Resolves the config's declared secret names/refs.
        report: Sink the done-check outcome is posted to.
        review_reviewer: A factory ``(repo_full_name, prd_number) -> RoundReviewer`` built
            per build (the reviewer is per-PRD), run after each round's merge; absent means
            no per-round review (and no review-fix follow-up slices).

    Returns:
        An async build callable returning a :class:`BoundBuildResult`.
    """
    retry_store = ImplRetryStore(retry_store_path)

    async def run(
        *,
        repo_full_name: str,
        prd_number: int,
        slices: list[PrdSlice],
        config: RepoConfig,
        claude_md: str,
    ) -> BoundBuildResult:
        gate = await governor.gate(estimated_amount=estimated_amount)
        if gate.deferred:
            logger.info(
                "Budget gate deferred PRD #%d (%s) until %s",
                prd_number,
                repo_full_name,
                gate.defer_until,
            )
            return BoundBuildResult(deferred=True, defer_until=gate.defer_until)

        triaged = _TriagingImplementer(
            implementer=implementer,
            config=config,
            notifier=notifier,
            create_issue=create_issue,
            retry_store=retry_store,
        )
        reviewer = (
            review_reviewer(repo_full_name, prd_number)
            if review_reviewer is not None
            else None
        )
        prd_build = await build_prd(
            slices,
            config,
            claude_md,
            implementer=triaged,
            git=git,
            auth=auth,
            runtime=runtime,
            resolve_secret=resolve_secret,
            report=report,
            lock=nullcontext(),
            review_round=reviewer,
        )
        return BoundBuildResult(deferred=False, prd_build=prd_build)

    return run


# A factory the build lane calls per build to construct the per-PRD internal reviewer.
# The reviewer is per-PRD (its ``Part of #<prd>`` footer and diff base depend on the
# specific PRD), but ``bind_build_prd`` is bound per repo, so the reviewer is built lazily
# at run time from the build's ``(repo_full_name, prd_number)``.
ReviewerFactory = Callable[[str, int], RoundReviewer]


@dataclass
class _BoundRoundReviewer:
    """Production :class:`RoundReviewer`: review a merged round, enqueue review-fix slices.

    After a round merges, :func:`retinue.orchestrator.build_prd` hands the round's merged
    issue numbers here. This adapter produces the round's merged diff (the injected
    :class:`RoundDiffSource`, the merge container's ``git diff``), drives
    :func:`retinue.reviewer.review_round` over the real Agent-SDK reviewer + gh issue
    creator + Blocked-by editor to file ``review-fix`` follow-ups (and wire them into the
    flagged dependents' ``## Blocked by`` on GitHub), then returns one independently-ready
    :class:`PrdSlice` per filed issue so it builds in a *subsequent* round of the same run.

    Attributes:
        diff_source: Produces the round's merged diff from the merge container.
        generate: The headless Agent-SDK reviewer (the :class:`ReviewGenerator` seam).
        create_issue: The gh issue creator filing each review-fix issue (slicer's seam).
        edit_blocked_by: The gh issue-body editor wiring the fix into dependents.
        repo_full_name: The target repo the review-fix issues are filed against.
        prd_number: The parent PRD the review-fix issues link back to (``Part of #``).
    """

    diff_source: RoundDiffSource
    generate: ReviewGenerator
    create_issue: IssueCreator
    edit_blocked_by: BlockedByEditor
    repo_full_name: str
    prd_number: int

    async def review(self, *, merged_issues: list[int]) -> list[PrdSlice]:
        """Review ``merged_issues``' merged diff; return review-fix slices to enqueue.

        The round diff is taken over the PRD's integration branch — each merged
        ``issue-<N>`` was rooted off that branch's tip, so the three-dot diff there is the
        round's contribution.
        """
        diff = await self.diff_source.round_diff(
            merged_branches=[f"issue-{n}" for n in merged_issues],
            base=integration_branch(self.prd_number),
        )
        result = await review_round(
            ReviewInput(
                repo_full_name=self.repo_full_name,
                prd_number=self.prd_number,
                merged_issues=list(merged_issues),
                diff=diff,
            ),
            generate=self.generate,
            create_issue=self.create_issue,
            edit_blocked_by=self.edit_blocked_by,
        )
        return [
            PrdSlice(
                repo_full_name=self.repo_full_name,
                issue_number=number,
                prd_number=self.prd_number,
            )
            for number in result.filed_issues
        ]


def bind_round_reviewer(
    *,
    diff_source: RoundDiffSource,
    generate: ReviewGenerator,
    create_issue: IssueCreator,
    edit_blocked_by: BlockedByEditor,
    repo_full_name: str,
    prd_number: int,
) -> RoundReviewer:
    """Bind the internal reviewer run after each round's merge in the live build.

    Wires the real per-round reviewer for one repo/PRD: the merge container's diff source,
    the Agent-SDK reviewer, and the gh issue creator + Blocked-by editor. The returned
    :class:`RoundReviewer` is passed to :func:`bind_build_prd` so ``build_prd`` reviews
    every merged round and the review-fix follow-ups it files build in a later round.

    Args:
        diff_source: Produces the round's merged diff (the merge container's ``git diff``).
        generate: The headless Agent-SDK reviewer (the :class:`ReviewGenerator` seam).
        create_issue: The gh issue creator filing each review-fix issue (slicer's seam).
        edit_blocked_by: The gh issue-body editor wiring the fix into dependents.
        repo_full_name: The target repo the review-fix issues are filed against.
        prd_number: The parent PRD the review-fix issues link back to; the round diff is
            taken over its integration branch.

    Returns:
        A :class:`RoundReviewer` the build lane runs after each round's merge.
    """
    return _BoundRoundReviewer(
        diff_source=diff_source,
        generate=generate,
        create_issue=create_issue,
        edit_blocked_by=edit_blocked_by,
        repo_full_name=repo_full_name,
        prd_number=prd_number,
    )


@dataclass
class _TriagingImplementer:
    """An :class:`Implementer` that routes each attempt through triage reasoning.

    Satisfies the orchestrator's ``implement(slice_, *, container, plan_path) -> None``
    contract by delegating to :func:`retinue.triage.triage_implementer`, which drives the
    real implementer in the build ``container`` and, on a failure or returned notes, decides
    retry / reslice / escalate against the persisted cap. The orchestrator gates on the
    done-check that follows, so a triaged-and-built slice proceeds normally and an
    escalated one leaves no commit to push or merge. ``auth_env`` is forwarded to the
    wrapped implementer so the orchestrator can inject the agent's credential at start.

    This is the PRD lane's implementer, where there is no materialized plan file, so
    ``plan_path`` is accepted for protocol conformance and ignored — the ad-hoc lane's
    plan threading is a :mod:`retinue.adhoc_build` concern, not a triaged-build one.
    """

    implementer: Implementer | TriageImplementer
    config: RepoConfig
    notifier: Notifier
    create_issue: IssueCreator
    retry_store: ImplRetryStore

    async def implement(
        self, slice_: Slice, *, container: Container, plan_path: str | None = None
    ) -> None:
        await triage_implementer(
            slice_,
            self.config,
            implementer=self.implementer,
            notifier=self.notifier,
            create_issue=self.create_issue,
            retry_store=self.retry_store,
            container=container,
        )

    def auth_env(self) -> dict[str, str]:
        return self.implementer.auth_env()


def bind_cron_tick(
    *,
    gh: CronGh,
    governor: BudgetGovernor,
    clock: Clock,
    build: CronBuild,
    lock: AbstractAsyncContextManager[object],
    quota_every: int = 5,
) -> Callable[..., Awaitable[CronTickResult]]:
    """Bind the cron backlog-drain tick to its real collaborators.

    Returns an async ``(repo_full_name, tick_number, estimated_amount) -> CronTickResult``
    that drives :func:`retinue.cron.run_cron_tick`: gate on the shared ``governor``, pick
    the next backlog issue by weighted score (or the quota floor on every Nth tick), and
    run its downstream ``build``. The governor is the *same* service-level governor the
    orchestrator build lane shares, so both lanes meter against one rolling-24h window.

    Args:
        gh: The backlog gh seam (lists ``backlog`` issues with labels + ages).
        governor: The shared service-level budget governor.
        clock: The injected time source for age-weighting.
        build: The downstream build chain run for the picked issue.
        lock: The single-run lock (raises when a tick is already in flight).
        quota_every: Take the oldest low-priority issue on every Nth tick.

    Returns:
        An async cron-tick callable returning a :class:`retinue.cron.CronTickResult`.
    """

    async def tick(
        *, repo_full_name: str, tick_number: int, estimated_amount: float
    ) -> CronTickResult:
        return await run_cron_tick(
            repo_full_name=repo_full_name,
            gh=gh,
            governor=governor,
            clock=clock,
            build=build,
            tick_number=tick_number,
            estimated_amount=estimated_amount,
            lock=lock,
            quota_every=quota_every,
        )

    return tick


def bind_adhoc_drain(
    *,
    gh: AdhocGh,
    build: AdhocBuild,
    open_pr: AdhocPrOpen,
    governor: BudgetGovernor,
    estimated_amount: float,
    lock: AbstractAsyncContextManager[object],
) -> Callable[..., Awaitable[None]]:
    """Bind the ad-hoc drain to its real collaborators behind one ``(repo, config)`` callable.

    Returns an async ``(*, repo_full_name, config) -> None`` — the
    :data:`retinue.heartbeat.HeartbeatDrain` shape — that drives
    :func:`retinue.adhoc_drain.run_adhoc_drain` with the gh seam, the per-issue build, the
    shared service-level governor, and the single-run lock already bound. The two callers of
    the bound drain fire the *same* drain: the webhook's low-latency ad-hoc kick
    (:func:`retinue.worker.run_adhoc_drain_job`) and the heartbeat's safety-net sweep
    (issue #43 wires the heartbeat's ``drain`` to this same seam), so a kicked drain and a
    swept drain are identical work under one single-run lock.

    The governor is the *same* service-level governor the orchestrator and cron lanes share,
    so every ad-hoc build meters against the one rolling-24h window. ``prd_in_flight`` is left
    at the drain's default (False): the kick path has no live PRD-build signal to thread, so
    the drain ranks and builds every ad-hoc issue rather than deferring to a PRD it cannot
    observe — PRD-first preemption stays a heartbeat-side refinement.

    Args:
        gh: The ad-hoc gh seam (lists ``ready-for-agent`` issues; answers the flight-state
            classification query).
        build: The downstream ad-hoc build+PR primitive run per buildable issue.
        open_pr: The PR-open-only recovery run per stranded issue (a green branch with no
            PR), opening its PR without a rebuild.
        governor: The shared service-level budget governor each build meters through.
        estimated_amount: The per-build charge metered against the shared cap.
        lock: The single-run lock; a second concurrent drain for the repo raises
            :class:`retinue.adhoc_drain.AdhocDrainBusyError`.

    Returns:
        An async drain callable taking ``repo_full_name`` and the repo's ``config``.
    """

    async def drain(*, repo_full_name: str, config: RepoConfig) -> None:
        await run_adhoc_drain(
            repo_full_name=repo_full_name,
            gh=gh,
            build=build,
            open_pr=open_pr,
            config=config,
            governor=governor,
            estimated_amount=estimated_amount,
            lock=lock,
        )

    return drain
