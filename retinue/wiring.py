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

from retinue.budget import BudgetGovernor, Clock
from retinue.container import ContainerRuntime
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
    Slice,
    build_prd,
)
from retinue.repo_config import RepoConfig
from retinue.slicer import IssueCreator
from retinue.triage import TriageImplementer, triage_implementer

logger = logging.getLogger(__name__)


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
) -> Callable[..., Awaitable[BoundBuildResult]]:
    """Bind the budget-gated, triage-wrapped orchestrator build.

    Returns an async ``(repo_full_name, prd_number, slices, config, claude_md) ->
    BoundBuildResult``. It first asks the shared ``governor`` to gate the run by its
    ``estimated_amount``; a deferred gate returns immediately without building. Otherwise
    it runs :func:`retinue.orchestrator.build_prd`, wrapping each implementer attempt in
    :func:`retinue.triage.triage_implementer` so a failure/mis-scope is reasoned about
    against the persisted retry cap (retry / reslice / escalate) instead of a blind loop.

    Args:
        implementer: The implementer-spawn seam (Agent SDK), triaged on failure.
        governor: The shared service-level budget governor.
        notifier: The escalation fan-out used by triage's escalate path.
        create_issue: The gh issue creator used by triage's reslice path.
        retry_store_path: SQLite file backing the persisted per-slice retry counter.
        estimated_amount: The run's estimated charge, gated against the rolling-24h cap.
        git: Integration-branch git operations (the merge seam).
        auth: Mints the installation token used to clone.
        runtime: Spawns the disposable done-check container.
        resolve_secret: Resolves the config's declared secret names/refs.
        report: Sink the done-check outcome is posted to.

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
        )
        return BoundBuildResult(deferred=False, prd_build=prd_build)

    return run


@dataclass
class _TriagingImplementer:
    """An :class:`Implementer` that routes each attempt through triage reasoning.

    Satisfies the orchestrator's ``implement(slice_) -> None`` contract by delegating to
    :func:`retinue.triage.triage_implementer`, which drives the real implementer and, on a
    failure or returned notes, decides retry / reslice / escalate against the persisted
    cap. The orchestrator gates on the done-check that follows, so a triaged-and-built
    slice proceeds normally and an escalated one leaves no commit to merge.
    """

    implementer: Implementer | TriageImplementer
    config: RepoConfig
    notifier: Notifier
    create_issue: IssueCreator
    retry_store: ImplRetryStore

    async def implement(self, slice_: Slice) -> None:
        await triage_implementer(
            slice_,
            self.config,
            implementer=self.implementer,
            notifier=self.notifier,
            create_issue=self.create_issue,
            retry_store=self.retry_store,
        )


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
