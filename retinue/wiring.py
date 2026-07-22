"""Production wiring for the retinue's lanes (the scheduler/cron glue).

One binding per lane lives here — the composition root between the worker's startup and
the lanes' pure drivers:

* :func:`bind_adhoc_drain` — the unified scheduler lane. It binds
  :func:`retinue.adhoc_drain.run_adhoc_drain` to the per-repo gh seams (the trigger-label
  issue listing and the readiness gh seam), resolves ``target_branch`` (``None`` -> the
  repo's default branch), builds the real build+PR primitive, and runs the drain behind one
  ``(*, repo_full_name, config)`` callable the webhook kick and the heartbeat sweep both
  fire.
* :func:`bind_cron_tick` — the backlog cron lane. It binds
  :func:`retinue.cron.run_cron_tick` to its real collaborators (the gh backlog query, the
  shared governor, the clock, the single-run lock, and the downstream build) so a scheduled
  tick lists and locks one backlog issue.

The budget gate is enforced through the governor passed in — the one service-level
governor the worker's startup constructs, shared across every lane.
"""

from __future__ import annotations

import logging
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

from retinue.adhoc_drain import AdhocDrainLock, run_adhoc_drain
from retinue.budget import (
    ADHOC_DRAIN_ESTIMATED_AMOUNT,
    CRON_PROMOTION_ESTIMATED_AMOUNT,
    BudgetGovernor,
    SystemClock,
)
from retinue.cron import CronLock, CronTickResult, run_cron_tick
from retinue.github_app import InstallationAuth
from retinue.heartbeat import HeartbeatCronTick, HeartbeatDrain
from retinue.reconcile import GhRunner, ReconcileGhRunner
from retinue.repo_config import RepoConfig
from retinue.run_ledger import RunLedgerStore

if TYPE_CHECKING:
    from retinue.config import Settings
    from retinue.pipeline import ClaudeMdFetcher, PipelineFactory

logger = logging.getLogger(__name__)


def _default_branch_argv(repo_full_name: str) -> list[str]:
    """Assemble the ``gh repo view`` argv reading a repo's default branch name."""
    return [
        "repo",
        "view",
        repo_full_name,
        "--json",
        "defaultBranchRef",
        "--jq",
        ".defaultBranchRef.name",
    ]


async def _resolve_target_branch(
    config: RepoConfig, repo_full_name: str, runner: GhRunner
) -> RepoConfig:
    """Resolve a ``None`` ``target_branch`` to the repo's default branch, re-stamping config.

    A repo config that leaves ``target_branch`` unset (``None``) wants its own default
    branch as the build base and PR target; that concrete name is a ``gh`` lookup made once
    at the wiring boundary and stamped back onto the config so build-time code can call
    :meth:`~retinue.repo_config.RepoConfig.require_target_branch` without cutting
    ``issue-<N>`` off ``origin/None``. An already-set ``target_branch`` is left untouched.
    """
    if config.target_branch is not None:
        return config
    stdout = await runner(_default_branch_argv(repo_full_name))
    branch = stdout.strip()
    if not branch:
        raise ValueError(
            f"could not resolve the default branch for {repo_full_name}"
        )
    return config.model_copy(update={"target_branch": branch})


def bind_cron_tick(
    settings: Settings,
    auth: InstallationAuth,
    *,
    governor: BudgetGovernor,
    fetch_claude_md: ClaudeMdFetcher,
    quota_every: int = 5,
) -> HeartbeatCronTick:
    """Bind the heartbeat's backlog cron tick to ``run_cron_tick`` over the real adapters.

    Returns an async ``(*, repo_full_name, tick_number) -> CronTickResult`` — the
    :data:`retinue.heartbeat.HeartbeatCronTick` shape — that drives one repo's backlog
    lane through :func:`retinue.cron.run_cron_tick`: gate on the shared ``governor``, pick
    the next backlog issue by weighted score (or the quota floor on every Nth tick), and
    run its downstream build. The tick's own work is label surgery with no model spend, so
    it gates at :data:`retinue.budget.CRON_PROMOTION_ESTIMATED_AMOUNT` (zero) — the
    governor is the *same* service-level governor the ad-hoc lane meters for the real
    build, so the budget is one rolling-24h window charged exactly once per promoted
    issue; a per-repo single-run lock registry lets two repos tick concurrently while a
    repo's own ticks serialize.

    The backlog cron lane's job is a label-surgery *trickle* promotion: the picked backlog
    nit is promoted into the scheduler queue by swapping ``backlog`` for the repo's
    ``config.trigger_label`` in one ``gh issue edit`` (:class:`retinue.cron.GhCliBacklogPromoter`),
    so the real build stays with the unified scheduler and the promotion leaves a GitHub
    audit trail. The repo's config rides each tick so the promotion applies that repo's own
    trigger label.

    Args:
        settings: The runtime settings carrying the Anthropic config.
        auth: The GitHub App installation auth used to mint per-repo tokens.
        governor: The shared service-level budget governor.
        fetch_claude_md: Reads each repo's ``CLAUDE.md`` (kept for parity with the drain
            bind; the trickle promotion does not build, so it is unused here).
        quota_every: Take the oldest low-priority issue on every Nth tick.

    Returns:
        The bound cron tick — an async
        ``(*, repo_full_name, tick_number, config) -> CronTickResult``.
    """
    # Deferred: retinue.pipeline imports this module, so a module-level import of the cron
    # gh seams would risk a cycle; resolving them at bind time also keeps them
    # monkeypatchable module attributes (tests patch retinue.cron.GhCli before startup).
    from retinue.cron import GhCli, GhCliBacklogPromoter

    locks: dict[str, AbstractAsyncContextManager[object]] = {}

    async def cron_tick(
        *, repo_full_name: str, tick_number: int, config: RepoConfig
    ) -> CronTickResult:
        token = (await auth.installation_token(repo_full_name)).token
        gh = GhCli(token=token)
        promoter = GhCliBacklogPromoter(
            trigger_label=config.trigger_label, token=token
        )
        return await run_cron_tick(
            repo_full_name=repo_full_name,
            gh=gh,
            governor=governor,
            clock=SystemClock(),
            build=promoter.promote,
            tick_number=tick_number,
            estimated_amount=CRON_PROMOTION_ESTIMATED_AMOUNT,
            lock=locks.setdefault(repo_full_name, CronLock()),
            quota_every=quota_every,
        )

    return cron_tick


def bind_adhoc_drain(
    settings: Settings,
    auth: InstallationAuth,
    *,
    governor: BudgetGovernor,
    run_ledger: RunLedgerStore,
    pipeline_factory: PipelineFactory,
    fetch_claude_md: ClaudeMdFetcher,
) -> HeartbeatDrain:
    """Bind the production scheduler drain to a ``(*, repo_full_name, config)`` callable.

    Returns an async ``(*, repo_full_name, config) -> None`` — the
    :data:`retinue.heartbeat.HeartbeatDrain` shape — that drives
    :func:`retinue.adhoc_drain.run_adhoc_drain`. The two callers fire the *same* drain: the
    webhook's low-latency kick (:func:`retinue.worker.run_adhoc_drain_job`) and the
    heartbeat's safety-net sweep, so a kicked drain and a swept drain are identical work
    under one single-run lock. The per-repo lock registry lets two repos drain concurrently
    while a repo's own kicked and swept drains serialize.

    Each call mints a per-repo installation token, then constructs the per-repo gh seams
    (:class:`retinue.adhoc_drain.GhCli` for the trigger-labeled issue listing and
    :class:`retinue.readiness.GhCli` for the readiness truth over the same token), resolves
    ``target_branch`` (``None`` -> the repo's default branch) and re-stamps the config,
    reuses the worker's ``pipeline_factory`` to build the repo's pipeline (its
    ``process_adhoc_pr`` opens the PR), and binds the real build+PR primitive
    (:func:`retinue.pipeline.bind_adhoc_build`) and the PR-open-only stranded-branch
    recovery (:func:`retinue.pipeline.bind_adhoc_pr_open`).

    The governor is the *same* service-level governor the cron lane shares, so every build
    meters against the one rolling-24h window.

    Args:
        settings: The runtime settings carrying the Anthropic config.
        auth: The GitHub App installation auth used to mint per-repo tokens.
        governor: The shared service-level budget governor each build meters through.
        run_ledger: The injected run-ledger store the drain records coarse run-state into
            for the API to read (built once at worker startup, like ``governor``).
        pipeline_factory: The worker's pipeline factory, reused so the ad-hoc PR step rides
            the same per-repo pipeline.
        fetch_claude_md: Reads each repo's ``CLAUDE.md`` (the done-check command source).

    Returns:
        The bound drain — an async ``(*, repo_full_name, config) -> None``.
    """
    # Deferred: retinue.pipeline imports this module, so a module-level import of its
    # ad-hoc binders / gh seams would be a cycle. Resolving at bind time also keeps the gh
    # seams monkeypatchable module attributes (tests patch retinue.adhoc_drain.GhCli /
    # retinue.readiness.GhCli / retinue.pipeline.bind_adhoc_build before startup).
    from retinue.adhoc_drain import GhCli as AdhocGhCli
    from retinue.pipeline import bind_adhoc_build, bind_adhoc_pr_open
    from retinue.readiness import GhCli as ReadinessGhCli

    locks: dict[str, AdhocDrainLock] = {}

    async def drain(*, repo_full_name: str, config: RepoConfig) -> None:
        token = (await auth.installation_token(repo_full_name)).token
        config = await _resolve_target_branch(
            config, repo_full_name, ReconcileGhRunner(token)
        )
        gh = AdhocGhCli(token=token)
        readiness_gh = ReadinessGhCli(token=token)
        pipeline = await pipeline_factory(repo_full_name, config)
        build = bind_adhoc_build(
            settings,
            auth,
            pipeline=pipeline,
            repo_full_name=repo_full_name,
            token=token,
            config=config,
            claude_md=await fetch_claude_md(repo_full_name),
        )
        await run_adhoc_drain(
            repo_full_name=repo_full_name,
            gh=gh,
            readiness_gh=readiness_gh,
            build=build,
            open_pr=bind_adhoc_pr_open(pipeline),
            config=config,
            governor=governor,
            ledger=run_ledger,
            estimated_amount=ADHOC_DRAIN_ESTIMATED_AMOUNT,
            lock=locks.setdefault(repo_full_name, AdhocDrainLock()),
        )

    return drain
