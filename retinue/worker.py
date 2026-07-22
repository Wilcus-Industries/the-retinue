"""Arq worker: the reap + scheduler-drain tasks and the heartbeat, plus WorkerSettings.

The worker dequeues two webhook-driven jobs — a merge reap (``reap_pr_job``) and a
low-latency scheduler-drain kick (``run_adhoc_drain_job``) — and runs a worker-global
heartbeat that sweeps every opted-in repo's drain and backlog cron tick as a safety net.

Launch the worker with:
    arq retinue.worker.WorkerSettings
or via the ``retinue-worker`` console script.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from arq import cron
from arq.connections import RedisSettings
from arq.worker import func as arq_func
from arq.worker import run_worker

import retinue.github_app as github_app
from retinue.adhoc_drain import AdhocDrainBusyError
from retinue.budget import AuthMode, BudgetGovernor, BudgetLedger, SystemClock
from retinue.config import Settings
from retinue.github_app import (
    InstallationAuth,
    InstalledRepos,
    github_claude_md_fetcher,
    github_config_fetcher,
)
from retinue.handoff import MergedPullRequest
from retinue.heartbeat import (
    DueRepo,
    HeartbeatDrain,
    RepoEnumerator,
    heartbeat_tick,
)
from retinue.pipeline import PipelineFactory, build_pipeline_factory
from retinue.repo_config import RepoConfig, load_repo_config
from retinue.run_ledger import RunLedgerStore, run_ledger_store_path
from retinue.wiring import bind_adhoc_drain, bind_cron_tick

logger = logging.getLogger(__name__)

# Async callable that returns a repo's raw ``.github/retinue.yml`` text, or None when the
# repo has no such file. The GitHub fetch is injected at this seam so the opt-in resolution
# is testable without network; :func:`retinue.github_app.github_config_fetcher` builds the
# real one.
ConfigFetcher = Callable[[str], Awaitable[str | None]]

# The worker-global heartbeat fires every Nth minute (the global tick). This is the coarse
# arq-level cadence; ``repo_config.cron`` is the finer per-repo "is this repo due?" filter
# under it (:func:`retinue.heartbeat.cron_due`). A 15-minute tick keeps the safety-net sweep
# responsive (a missed-webhook issue is caught within the quarter-hour) without busy-looping.
HEARTBEAT_MINUTES = frozenset(range(0, 60, 15))


async def reap_pr_job(
    ctx: dict[str, Any],
    *,
    repo_full_name: str,
    pr_number: int,
) -> None:
    """Arq task: reap a human-merged PR (close the issue(s), then reap the parent).

    Resolves the PR's issue mapping from the run-state store recorded when the build opened
    the PR, then drives the pipeline reap. With no pipeline wired the event is logged and
    dropped.

    Args:
        ctx: Arq worker context carrying ``fetch_config`` and ``pipeline_factory``.
        repo_full_name: e.g. "owner/repo".
        pr_number: The merged PR number.
    """
    config = await _config_for(ctx, repo_full_name)
    if config is None:
        return
    pipeline_factory: PipelineFactory | None = ctx.get("pipeline_factory")
    if pipeline_factory is None:
        logger.info(
            "No pipeline wired; dropping merge of %s PR #%d", repo_full_name, pr_number
        )
        return

    pipeline = await pipeline_factory(repo_full_name, config)
    mapping = await pipeline.round_for_pr(
        repo_full_name=repo_full_name, pr_number=pr_number
    )
    if mapping is None:
        logger.info("Merge of unknown PR %s #%d; skipping reap", repo_full_name, pr_number)
        return
    prd_number, slice_numbers = mapping
    merged = MergedPullRequest(
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        prd_number=prd_number,
        slice_issues=slice_numbers,
    )
    result = await pipeline.reap_pr(merged)
    logger.info(
        "Reap for %s PR #%d: %s", repo_full_name, pr_number, result.outcome.value
    )


# The kick task and the heartbeat sweep read the bound drain from ``ctx`` under the *same*
# :data:`retinue.heartbeat.HeartbeatDrain` shape — an async ``(*, repo_full_name, config) ->
# None`` produced by :func:`retinue.wiring.bind_adhoc_drain` — so a webhook kick and a swept
# sweep fire the identical bound drain; a bare worker with none wired no-ops the kick.
async def run_adhoc_drain_job(
    ctx: dict[str, Any],
    *,
    repo_full_name: str,
) -> None:
    """Arq task: run one scheduler drain for a repo, kicked by the webhook.

    The webhook enqueues this low-latency kick when a trigger-labeled issue event arrives,
    so the repo's scheduler backlog drains without waiting for the cron heartbeat (a flurry
    of events for one repo coalesces to a single drain via the queue's per-repo job id). The
    bound drain is read from ``ctx`` — the *same* drain the heartbeat fires — so a kick and a
    sweep are identical work under one single-run lock.

    Two skips keep one kick from crashing the worker: with no drain wired (a bare worker) the
    kick logs and returns, and a repo no longer opted in is a skip rather than a drain of a
    de-opted repo.

    Args:
        ctx: Arq worker context; may carry ``adhoc_drain`` (the bound drain) and
            ``fetch_config`` (the opt-in config seam).
        repo_full_name: e.g. "owner/repo".
    """
    drain: HeartbeatDrain | None = ctx.get("adhoc_drain")
    if drain is None:
        logger.info("No ad-hoc drain wired; dropping kick for %s", repo_full_name)
        return
    config = await _config_for(ctx, repo_full_name)
    if config is None:
        return
    try:
        await drain(repo_full_name=repo_full_name, config=config)
    except AdhocDrainBusyError:
        # A drain for this repo is already in flight (the kick collided with the heartbeat
        # sweep or another kick under the single-run lock). The in-flight drain covers this
        # repo's ready work, so the redundant kick is an expected skip — swallow it rather
        # than fail the arq task and trigger a retry.
        logger.info(
            "Ad-hoc drain already in flight for %s; skipping the redundant kick",
            repo_full_name,
        )


async def _config_for(ctx: dict[str, Any], repo_full_name: str) -> RepoConfig | None:
    """Fetch and parse a repo's opt-in config.

    A repo with no config or a malformed one is treated as not opted in (``None``), so a
    merge/kick event for a de-opted repo is a skip rather than a crash.
    """
    fetch_config: ConfigFetcher | None = ctx.get("fetch_config")
    if fetch_config is None:
        return None
    raw = await fetch_config(repo_full_name)
    if raw is None:
        return None
    return load_repo_config(raw)


def _configure_logging() -> None:
    """Send INFO-level logs to stdout so the worker's progress is observable.

    ``run_worker`` does not configure logging (only arq's CLI does), so without this the
    root logger sits at WARNING with no handler and every ``logger.info`` line is dropped.
    Install a single stdout handler at INFO; ``force=True`` rebinds any pre-existing root
    config to the current stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


async def on_startup(ctx: dict[str, Any]) -> None:
    """Populate the worker context with the live pipeline, drain, and heartbeat wiring.

    When GitHub App auth resolves (:func:`_load_github_client`), it installs the config
    fetcher, the real ``pipeline_factory`` over the production adapters, the bound scheduler
    drain the webhook kick fires, and the worker-global heartbeat's four collaborators
    (:func:`retinue.heartbeat.heartbeat_tick` reads them from ``ctx``): the real wall-clock,
    the installed-and-opted-in repo enumerator, the *same* bound drain the kick fires, and a
    bound backlog cron tick. With no auth wired, the fetcher defaults to not-opted-in and
    neither the pipeline nor the heartbeat is installed (so the registered cron tick safely
    no-ops).
    """
    global settings
    if settings is None:
        settings = _load_settings()
    wiring = _load_github_client()
    if wiring is None:
        # No GitHub App auth wired yet: fall back to the safe not-opted-in default so
        # nothing is processed without an explicit config, rather than crashing the worker.
        ctx["fetch_config"] = _no_config_fetcher
        return
    auth, client = wiring
    fetch_claude_md = github_claude_md_fetcher(auth, client)
    # The ONE service-level budget governor: the pipeline factory's build gate, the
    # scheduler drain, and the cron tick all meter this same instance, so every lane charges
    # one shared rolling-24h ledger over ``settings.budget_db_path``.
    governor = BudgetGovernor(
        BudgetLedger(
            settings.budget_db_path,
            clock=SystemClock(),
            auth_mode=AuthMode.from_config(settings.auth_mode),
            weekly_budget=settings.weekly_budget,
            daily_cap_fraction=settings.budget_daily_cap_fraction,
        )
    )
    ctx["governor"] = governor
    # The cross-process run-ledger the drain records into and the API reads back; built here
    # (writer side) from the same shared state dir the web reader resolves.
    run_ledger = RunLedgerStore(run_ledger_store_path(settings))
    pipeline_factory = build_pipeline_factory(
        settings, auth, governor=governor, fetch_claude_md=fetch_claude_md
    )
    ctx["github_client"] = client
    ctx["fetch_config"] = github_config_fetcher(auth, client)
    ctx["pipeline_factory"] = pipeline_factory
    # The webhook's low-latency kick (run_adhoc_drain_job) reads ``adhoc_drain`` from ctx;
    # bind it here so a kick on a deployed worker actually drains. The heartbeat's
    # safety-net sweep fires this *same* bound drain (below).
    adhoc_drain = bind_adhoc_drain(
        settings,
        auth,
        governor=governor,
        run_ledger=run_ledger,
        pipeline_factory=pipeline_factory,
        fetch_claude_md=fetch_claude_md,
    )
    ctx["adhoc_drain"] = adhoc_drain
    # The worker-global heartbeat (heartbeat_tick) reads its four collaborators from ctx.
    ctx["heartbeat_clock"] = SystemClock()
    ctx["heartbeat_enumerate_repos"] = _bind_heartbeat_enumerate(
        auth, fetch_config=ctx["fetch_config"]
    )
    ctx["heartbeat_drain"] = adhoc_drain
    ctx["heartbeat_cron_tick"] = bind_cron_tick(
        settings,
        auth,
        governor=governor,
        fetch_claude_md=fetch_claude_md,
    )


def _bind_heartbeat_enumerate(
    auth: InstallationAuth, *, fetch_config: ConfigFetcher
) -> RepoEnumerator:
    """Bind the heartbeat's opted-in repo enumerator over the GitHub-App installed set.

    Returns an async ``() -> list[DueRepo]`` that lists the App's installed repositories,
    fetches each repo's ``.github/retinue.yml`` through the *same* opt-in ``fetch_config``
    seam, and yields a :class:`~retinue.heartbeat.DueRepo` for each repo with an accepted
    :class:`~retinue.repo_config.RepoConfig`. A repo not opted in or with a malformed config
    is dropped, so the sweep only touches opted-in repos.

    The enumeration seam is the production :class:`retinue.github_app.GitHubInstallationAuth`,
    which also satisfies :class:`~retinue.github_app.InstalledRepos`; a bare auth without it
    yields an empty sweep rather than crashing the tick.
    """

    async def enumerate_repos() -> list[DueRepo]:
        if not isinstance(auth, InstalledRepos):
            return []
        due: list[DueRepo] = []
        for repo_full_name in await auth.installed_repositories():
            raw = await fetch_config(repo_full_name)
            if raw is None:
                continue
            config = load_repo_config(raw)
            if config is None:
                continue
            due.append(DueRepo(repo_full_name=repo_full_name, config=config))
        return due

    return enumerate_repos


async def on_shutdown(ctx: dict[str, Any]) -> None:
    """Close what :func:`on_startup` opened: the GitHub HTTP client."""
    client: httpx.AsyncClient | None = ctx.get("github_client")
    if client is not None:
        await client.aclose()


async def _no_config_fetcher(repo_full_name: str) -> str | None:
    """Fallback fetcher used when no GitHub App auth is wired: treat repo as opted out."""
    logger.debug("No GitHub auth wired; treating %s as not opted in", repo_full_name)
    return None


def _load_github_client() -> tuple[InstallationAuth, httpx.AsyncClient] | None:
    """Construct the production GitHub installation auth and HTTP client, if available.

    Returns ``None`` in two cases, both of which make the worker fall back to the safe
    not-opted-in default rather than crashing: when no concrete ``build_installation_auth``
    is wired at all, and — the fresh-deploy case — when the builder exists but raises
    :class:`~retinue.github_app.InstallationAuthError` because the GitHub App is not yet
    configured. The builder is invoked lazily, per startup, so registering the task (e.g. in
    tests) needs no GitHub App credentials and opens no network client at import time.
    """
    builder = getattr(github_app, "build_installation_auth", None)
    if builder is None:
        return None
    try:
        auth = builder()
    except github_app.InstallationAuthError:
        return None
    return auth, httpx.AsyncClient(timeout=30.0)


# Module-level settings, instantiated lazily in main() so importing this module does not
# require the env vars to be present (e.g. when registering the task in tests).
settings: Any = None


def _load_settings() -> Any:
    return Settings()  # type: ignore[call-arg]


def main() -> None:
    """Console-script entrypoint: start the Arq worker with WorkerSettings.

    Resolves ``WorkerSettings.redis_settings`` from the configured ``REDIS_URL`` here, at
    process start: arq reads that class attribute when it constructs the Worker, before
    ``on_startup`` runs, so the override must be applied now.
    """
    _configure_logging()

    global settings
    if settings is None:
        settings = _load_settings()
    WorkerSettings.redis_settings = RedisSettings.from_dsn(settings.redis_url)
    # arq reads job_timeout off the class before on_startup; its 300s default cancels a
    # real build mid-implement, so override it here at process start.
    WorkerSettings.job_timeout = settings.job_timeout_seconds

    run_worker(WorkerSettings)  # type: ignore[arg-type]


class WorkerSettings:
    """Arq WorkerSettings: registers the reap + drain tasks and the heartbeat.

    Launch the worker process with:
        arq retinue.worker.WorkerSettings
    """

    # The re-kicked drain registers with ``keep_result=0``: arq's enqueue dedups on the
    # completed job's lingering *result* key too (default 1h), so keeping results would
    # silently swallow a post-drain webhook kick.
    functions = [
        reap_pr_job,
        arq_func(run_adhoc_drain_job, keep_result=0),
    ]
    # The worker-global cron heartbeat: fires every Nth minute as the safety-net sweep for
    # issues labeled while the webhook was missed (firing the scheduler drain for each due
    # repo) and drives the backlog cron lane. Its collaborators are read from ``ctx`` in
    # :func:`heartbeat_tick`; a bare worker with none wired ticks harmlessly.
    cron_jobs = [cron(heartbeat_tick, minute=set(HEARTBEAT_MINUTES))]
    on_startup = on_startup
    on_shutdown = on_shutdown
    # Overridden from JOB_TIMEOUT_SECONDS in main() at process start; the default here must
    # already outlast a real build, since arq reads it before on_startup and its own 300s
    # default would cancel the drain mid-implement.
    job_timeout: int = 1800
    # Overridden from the configured REDIS_URL in main() at process start.
    redis_settings: RedisSettings = RedisSettings()
