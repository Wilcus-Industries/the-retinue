"""Arq worker: the process_prd task and WorkerSettings.

The walking-skeleton worker dequeues a PRD job and logs the repository, issue
number, and action. Later issues replace the body of :func:`process_prd` with the
real PRD pipeline; the queue contract (task name and kwargs) stays the same.

Launch the worker with:
    arq retinue.worker.WorkerSettings
or via the ``retinue-worker`` console script.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import enum
import logging
import re
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from arq import cron
from arq.connections import RedisSettings
from arq.worker import Retry
from arq.worker import func as arq_func

from retinue.budget import SystemClock
from retinue.cron import CronTickResult
from retinue.dedupe import PrdDedupeStore, prd_dedupe_key
from retinue.github_app import InstallationAuth
from retinue.handoff import MergedPullRequest
from retinue.heartbeat import (
    DueRepo,
    HeartbeatCronTick,
    HeartbeatDrain,
    RepoEnumerator,
    heartbeat_tick,
)
from retinue.loopback import (
    HeimdallFinding,
    HeimdallReview,
    ReviewState,
    Severity,
    round_key,
)
from retinue.pipeline import (
    ClaudeMdFetcher,
    Pipeline,
    build_pipeline_factory,
    run_state_store_path,
)
from retinue.queue import RESUME_ROUND_TASK, RESUME_ROUNDS_TASK, PrdJob
from retinue.reconcile import RunStateStore
from retinue.repo_config import RepoConfig, load_repo_config

logger = logging.getLogger(__name__)

# Async callable that returns a repo's raw ``.github/retinue.yml`` text, or None
# when the repo has no such file. The GitHub fetch is injected at this seam so the
# gate is testable without network; :func:`github_config_fetcher` builds the real one.
ConfigFetcher = Callable[[str], Awaitable[str | None]]

# Builds a :class:`~retinue.pipeline.Pipeline` for an accepted repo. Async so the
# production factory can mint a per-repo installation token before constructing the gh
# adapters. Injected onto the Arq context by :func:`on_startup` so the worker tasks stay
# testable with a fake factory; production binds it to a factory over the real adapters.
PipelineFactory = Callable[[str, RepoConfig], Awaitable[Pipeline]]

# Async callable returning a PRD/issue body text given (repo, issue number). The gh
# issue read the slicer needs, injected so the worker is testable without a live gh.
IssueBodyFetcher = Callable[[str, int], Awaitable[str]]

# The worker-global heartbeat fires every Nth minute (the global tick). This is the coarse
# arq-level cadence; ``repo_config.cron`` is the finer per-repo "is this repo due?" filter
# under it (:func:`retinue.heartbeat.cron_due`). A 15-minute tick keeps the safety-net sweep
# responsive (a missed-webhook issue is caught within the quarter-hour) without busy-looping.
HEARTBEAT_MINUTES = frozenset(range(0, 60, 15))

# Path of the opt-in config file inside each repo, fetched over the contents API.
RETINUE_CONFIG_PATH = ".github/retinue.yml"
# Path of the repo's CLAUDE.md, fetched over the contents API to source the done-check
# command the build gates on (a missing file reads as empty text).
CLAUDE_MD_PATH = "CLAUDE.md"
GITHUB_API_BASE_URL = "https://api.github.com"


class GateOutcome(enum.Enum):
    """Why the opt-in gate accepted or skipped a PRD."""

    ACCEPTED = "accepted"
    NOT_OPTED_IN = "not_opted_in"
    MALFORMED_CONFIG = "malformed_config"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class GateResult:
    """Result of gating one PRD.

    Attributes:
        outcome: Why the PRD was accepted or skipped.
        config: The parsed repo config when ``outcome`` is ``ACCEPTED``, else None.
    """

    outcome: GateOutcome
    config: RepoConfig | None = None


async def gate_prd(
    job: PrdJob,
    *,
    fetch_config: ConfigFetcher,
    dedupe: PrdDedupeStore,
) -> GateResult:
    """Decide whether a dequeued PRD should be processed, and parse its config.

    Three gates, in order: opt-in (a ``.github/retinue.yml`` exists), validity (it
    parses against the schema), and novelty (not already deduped). A repo with no
    file or a malformed file is an observable skip — logged, never raised — so one
    bad repo cannot crash the worker. The dedupe slot is claimed only for an
    accepted PRD, so a malformed config does not block a later fixed redelivery.

    Args:
        job: The dequeued PRD job.
        fetch_config: Async callable returning the repo's ``retinue.yml`` text, or
            None when the repo has no config file.
        dedupe: The SQLite-backed dedupe store.

    Returns:
        A :class:`GateResult`; ``config`` is set only when ``outcome`` is ACCEPTED.
    """
    ref = f"{job.repo_full_name}#{job.issue_number}"

    raw = await fetch_config(job.repo_full_name)
    if raw is None:
        logger.info("Skipping %s: repo not opted in (no .github/retinue.yml)", ref)
        return GateResult(GateOutcome.NOT_OPTED_IN)

    config = load_repo_config(raw)
    if config is None:
        logger.warning("Skipping %s: malformed .github/retinue.yml", ref)
        return GateResult(GateOutcome.MALFORMED_CONFIG)

    if not await dedupe.claim(prd_dedupe_key(job)):
        logger.info("Skipping %s: duplicate PRD already processed", ref)
        return GateResult(GateOutcome.DUPLICATE)

    return GateResult(GateOutcome.ACCEPTED, config=config)


async def process_prd(
    ctx: dict[str, Any],
    *,
    repo_full_name: str,
    issue_number: int,
    action: str,
) -> None:
    """Arq task: gate a dequeued PRD on opt-in + validity + novelty, then process.

    Loads the repo's ``.github/retinue.yml`` (presence = opt-in), validates it, and
    deduplicates the PRD before doing real work. A repo with no config, a malformed
    config, or an already-processed PRD is an observable skip. The config-fetcher
    and dedupe store are read from ``ctx`` (populated by ``on_startup``); when absent
    (e.g. the bare round-trip test) the task falls back to logging only.

    Args:
        ctx: Arq worker context; may carry ``fetch_config`` and ``dedupe``.
        repo_full_name: e.g. "owner/repo".
        issue_number: The GitHub issue number.
        action: The webhook action that triggered the job.
    """
    job = PrdJob(
        repo_full_name=repo_full_name, issue_number=issue_number, action=action
    )

    fetch_config: ConfigFetcher | None = ctx.get("fetch_config")
    dedupe: PrdDedupeStore | None = ctx.get("dedupe")
    if fetch_config is None or dedupe is None:
        # No gate wired in (bare worker / round-trip skeleton): log and return.
        logger.info(
            "Processing PRD for %s#%d action=%s",
            repo_full_name,
            issue_number,
            action,
        )
        return

    result = await gate_prd(job, fetch_config=fetch_config, dedupe=dedupe)
    if result.outcome is not GateOutcome.ACCEPTED or result.config is None:
        return

    logger.info(
        "Processing PRD for %s#%d action=%s (staging_branch=%s)",
        repo_full_name,
        issue_number,
        action,
        result.config.staging_branch,
    )

    pipeline_factory: PipelineFactory | None = ctx.get("pipeline_factory")
    if pipeline_factory is None:
        # No real pipeline wired (bare worker / round-trip skeleton): the gate ran and
        # accepted, but there is nothing downstream to drive. Stop after the accept log.
        return

    pipeline = await pipeline_factory(repo_full_name, result.config)
    try:
        prd_result = await pipeline.process_prd_job(
            repo_full_name=repo_full_name,
            prd_number=issue_number,
            prd_body=await _load_prd_body(ctx, repo_full_name, issue_number),
        )
    except Exception:
        # The dedupe claim landed at the gate, before any durable run state. A crash
        # before the round persisted its slices would otherwise burn the PRD forever
        # (every redelivery reads as a duplicate), so release the claim; once slices
        # are durable the resume sweep owns the round and the claim must stand.
        if not await pipeline.has_round(
            repo_full_name=repo_full_name, prd_number=issue_number
        ):
            await dedupe.release(prd_dedupe_key(job))
            logger.warning(
                "PRD %s#%d crashed before any durable slice; dedupe claim released",
                repo_full_name,
                issue_number,
            )
        raise
    logger.info(
        "Pipeline for %s#%d: sliced=%s pr_opened=%s deferred=%s",
        repo_full_name,
        issue_number,
        prd_result.sliced,
        prd_result.pr_opened,
        prd_result.deferred,
    )
    if prd_result.deferred:
        await _enqueue_resume_round(
            ctx,
            repo_full_name=repo_full_name,
            prd_number=issue_number,
            defer_until=prd_result.defer_until,
        )


async def process_review_job(
    ctx: dict[str, Any],
    *,
    repo_full_name: str,
    pr_number: int,
    review_state: str,
    review_body: str = "",
) -> None:
    """Arq task: drive the heimdall loopback for one ``pull_request_review`` event.

    Parses the review into a :class:`~retinue.loopback.HeimdallReview` and hands it to
    the wired pipeline (rebuild / converge / escalate). The repo config is fetched the
    same way the PRD gate fetches it; a repo no longer opted in is a skip. With no
    pipeline wired (the bare worker) the event is logged and dropped.

    Args:
        ctx: Arq worker context carrying ``fetch_config`` and ``pipeline_factory``.
        repo_full_name: e.g. "owner/repo".
        pr_number: The reviewed PR number.
        review_state: The gh review state.
        review_body: The review body heimdall findings are parsed from.
    """
    config = await _config_for(ctx, repo_full_name)
    if config is None:
        return
    pipeline_factory: PipelineFactory | None = ctx.get("pipeline_factory")
    if pipeline_factory is None:
        logger.info("No pipeline wired; dropping review for %s PR #%d", repo_full_name, pr_number)
        return

    pipeline = await pipeline_factory(repo_full_name, config)
    mapping = await pipeline.round_for_pr(
        repo_full_name=repo_full_name, pr_number=pr_number
    )
    if mapping is None:
        # A review on a PR the retinue never opened (not in run-state): not ours to act on.
        logger.info("Review for unknown PR %s #%d; skipping", repo_full_name, pr_number)
        return
    prd_number, _slice_numbers = mapping

    review = parse_heimdall_review(
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        prd_number=prd_number,
        review_state=review_state,
        review_body=review_body,
    )
    # GitHub redelivers review webhooks; serializing per PR keeps the loopback's
    # read-count-then-record-round window atomic so two deliveries cannot both see a
    # stale round count and double-consume the rebuild budget.
    locks: dict[str, asyncio.Lock] = ctx.setdefault("review_locks", {})
    lock = locks.setdefault(round_key(repo_full_name, pr_number), asyncio.Lock())
    async with lock:
        result = await pipeline.process_review(review)
    logger.info(
        "Loopback for %s PR #%d: %s", repo_full_name, pr_number, result.outcome.value
    )


async def reap_pr_job(
    ctx: dict[str, Any],
    *,
    repo_full_name: str,
    pr_number: int,
) -> None:
    """Arq task: reap a human-merged PR (close slice issues, then reap the PRD).

    Resolves the PR's PRD and slice issues from the run-state store recorded when the
    PRD round opened the PR, then drives the pipeline reap. With no pipeline wired the
    event is logged and dropped.

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
        logger.info("No pipeline wired; dropping merge of %s PR #%d", repo_full_name, pr_number)
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
    """Arq task: run one ad-hoc drain for a repo, kicked by the webhook (``RUN_ADHOC_DRAIN_TASK``).

    The webhook enqueues this low-latency kick when a ``ready-for-agent`` (non-``prd``) issue
    event arrives, so the repo's ad-hoc backlog drains without waiting for the cron heartbeat
    (a flurry of events for one repo coalesces to a single drain via the queue's per-repo job
    id). The bound drain is read from ``ctx`` (populated by :func:`on_startup` under live
    GitHub-App auth) — the *same* drain the heartbeat fires (issue #43) — so a kick and a
    sweep are identical work under one single-run lock.

    Two skips keep one kick from crashing the worker, mirroring the other tasks' discipline:
    with no drain wired (a bare worker / round-trip skeleton) the kick logs and returns, and a
    repo no longer opted in (no fetchable config) is a skip rather than a drain of a de-opted
    repo.

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
    await drain(repo_full_name=repo_full_name, config=config)


# A deferral with no reported window (the ledger couldn't say when it frees) retries on
# this fallback cadence; a reported window schedules the resume for exactly then.
_DEFERRED_RESUME_FALLBACK_SECONDS = 3600
# A still-deferred resume retries via arq's Retry; this caps the retries (a day of
# hourly fallbacks) so a wedged budget can't respin the job forever — the round's
# run-state row survives for the next restart's sweep either way.
_RESUME_ROUND_MAX_TRIES = 24


def _resume_round_job_id(repo_full_name: str, prd_number: int) -> str:
    """The per-round Arq job id coalescing concurrent deferred-resume enqueues."""
    return f"resume-round:{repo_full_name}#{prd_number}"


async def _enqueue_resume_round(
    ctx: dict[str, Any],
    *,
    repo_full_name: str,
    prd_number: int,
    defer_until: datetime | None,
) -> None:
    """Enqueue the deferred round's resume for when the budget window frees.

    Deliberately NOT a re-enqueue of ``PROCESS_PRD_TASK`` — that would re-slice the PRD
    into duplicate issues. ``resume_round_job`` reconciles against GitHub truth and
    re-drives only the unbuilt phase. The per-round ``_job_id`` coalesces a burst of
    deferrals into one scheduled resume (the task registers with ``keep_result=0`` so a
    finished run's result key never swallows the next enqueue).
    """
    redis = ctx.get("redis")
    if redis is None:
        logger.warning(
            "No redis on the worker context; deferred PRD %s#%d was not re-enqueued",
            repo_full_name,
            prd_number,
        )
        return
    schedule: dict[str, Any] = (
        {"_defer_until": defer_until}
        if defer_until is not None
        else {"_defer_by": _DEFERRED_RESUME_FALLBACK_SECONDS}
    )
    await redis.enqueue_job(
        RESUME_ROUND_TASK,
        repo_full_name=repo_full_name,
        prd_number=prd_number,
        _job_id=_resume_round_job_id(repo_full_name, prd_number),
        **schedule,
    )


async def resume_round_job(
    ctx: dict[str, Any], *, repo_full_name: str, prd_number: int
) -> None:
    """Arq task: re-drive one budget-deferred PRD round once its window frees.

    Enqueued by :func:`process_prd` / :func:`resume_rounds_job` when the budget gate
    deferred a round's build. Re-gates through :meth:`retinue.pipeline.Pipeline.resume_round`
    (reconcile against GitHub truth, then rebuild only the unfinished slices); a resume
    that is *still* deferred raises :class:`arq.worker.Retry` so arq re-schedules this
    same job — re-enqueueing under the running job's own ``_job_id`` would be silently
    dropped (the live job key still exists). The usual skips apply: a de-opted repo or a
    bare worker logs and returns.

    Args:
        ctx: Arq worker context carrying ``fetch_config`` and ``pipeline_factory``.
        repo_full_name: e.g. "owner/repo".
        prd_number: The deferred PRD's tracking issue number.
    """
    config = await _config_for(ctx, repo_full_name)
    if config is None:
        logger.info(
            "Skipping deferred resume of %s PRD #%d: repo no longer opted in",
            repo_full_name,
            prd_number,
        )
        return
    pipeline_factory: PipelineFactory | None = ctx.get("pipeline_factory")
    if pipeline_factory is None:
        logger.info(
            "No pipeline wired; dropping deferred resume of %s PRD #%d",
            repo_full_name,
            prd_number,
        )
        return

    pipeline = await pipeline_factory(repo_full_name, config)
    outcome = await pipeline.resume_round(
        repo_full_name=repo_full_name, prd_number=prd_number
    )
    if outcome.deferred:
        defer = _retry_defer_seconds(outcome.defer_until)
        logger.info(
            "Resume of %s PRD #%d still budget-deferred; retrying in %.0fs",
            repo_full_name,
            prd_number,
            defer,
        )
        raise Retry(defer=defer)
    logger.info(
        "Deferred resume of %s PRD #%d completed at %s",
        repo_full_name,
        prd_number,
        outcome.reconcile.phase.value,
    )


def _retry_defer_seconds(defer_until: datetime | None) -> float:
    """Seconds until the budget window frees, floored at a minute; 1h with no window."""
    if defer_until is None:
        return float(_DEFERRED_RESUME_FALLBACK_SECONDS)
    return max(60.0, (defer_until - datetime.now(UTC)).total_seconds())


async def resume_rounds_job(ctx: dict[str, Any]) -> None:
    """Arq task: the crash-resume startup sweep over every persisted PRD round.

    Enqueued by :func:`on_startup` (``RESUME_ROUNDS_TASK``) so a restart resumes the
    rounds a crash left in flight without blocking worker boot. Each round from the
    run-state store is resumed through its repo's pipeline
    (:meth:`retinue.pipeline.Pipeline.resume_round` — reconcile against GitHub truth,
    then re-drive the phase). The sweep is crash-safe per round: one round's failure is
    logged and skipped so the rest still resume, and its row survives for a later sweep.
    A round of a repo no longer opted in is skipped the same way. With no run-state or
    pipeline wired (a bare worker) the sweep logs and returns.

    Args:
        ctx: Arq worker context; may carry ``run_state``, ``fetch_config``, and
            ``pipeline_factory``.
    """
    run_state: RunStateStore | None = ctx.get("run_state")
    pipeline_factory: PipelineFactory | None = ctx.get("pipeline_factory")
    if run_state is None or pipeline_factory is None:
        logger.info("No run-state/pipeline wired; skipping the resume sweep")
        return

    rounds = await run_state.all_rounds()
    logger.info("Resume sweep: %d persisted round(s) to reconcile", len(rounds))
    for round_ in rounds:
        config = await _config_for(ctx, round_.repo_full_name)
        if config is None:
            logger.info(
                "Skipping resume of %s PRD #%d: repo no longer opted in",
                round_.repo_full_name,
                round_.prd_number,
            )
            continue
        try:
            pipeline = await pipeline_factory(round_.repo_full_name, config)
            outcome = await pipeline.resume_round(
                repo_full_name=round_.repo_full_name, prd_number=round_.prd_number
            )
        except Exception:
            # Crash-safe per round: the row survives for a later sweep to retry.
            logger.exception(
                "Resume of %s PRD #%d failed; continuing the sweep",
                round_.repo_full_name,
                round_.prd_number,
            )
            continue
        if outcome.deferred:
            # A budget-deferred resume would otherwise sit until the *next* restart
            # happens to sweep it; schedule its own resume for when the window frees.
            await _enqueue_resume_round(
                ctx,
                repo_full_name=round_.repo_full_name,
                prd_number=round_.prd_number,
                defer_until=outcome.defer_until,
            )
        logger.info(
            "Resumed %s PRD #%d at %s%s",
            round_.repo_full_name,
            round_.prd_number,
            outcome.reconcile.phase.value,
            " (budget-deferred)" if outcome.deferred else "",
        )


async def _config_for(ctx: dict[str, Any], repo_full_name: str) -> RepoConfig | None:
    """Fetch and parse a repo's opt-in config the same way the PRD gate does.

    A repo with no config or a malformed one is treated as not opted in (``None``), so a
    review/merge event for a de-opted repo is a skip rather than a crash.
    """
    fetch_config: ConfigFetcher | None = ctx.get("fetch_config")
    if fetch_config is None:
        return None
    raw = await fetch_config(repo_full_name)
    if raw is None:
        return None
    return load_repo_config(raw)


async def _load_prd_body(
    ctx: dict[str, Any], repo_full_name: str, issue_number: int
) -> str:
    """Read the PRD issue body the slicer slices, via the injected fetcher.

    The body fetch is an injected ``fetch_prd_body`` seam (the gh issue read), so the
    worker stays testable without a live gh; the bare worker has no fetcher and yields
    an empty body, which the slicer escalates as too thin.
    """
    fetch_body: IssueBodyFetcher | None = ctx.get("fetch_prd_body")
    if fetch_body is None:
        return ""
    return await fetch_body(repo_full_name, issue_number)


# gh review states map onto the loopback's three-valued review state. ``dismissed`` and
# any unrecognised state are read as a plain comment (no verdict), so an odd state never
# blocks or converges on its own.
_REVIEW_STATE_MAP = {
    "approved": ReviewState.APPROVED,
    "changes_requested": ReviewState.REQUEST_CHANGES,
    "request_changes": ReviewState.REQUEST_CHANGES,
    "commented": ReviewState.COMMENTED,
}

# Heimdall never submits an APPROVED review: its clean pass is a COMMENTED review whose
# body is "Heimdall review: no concerns found across any lens." — this marker (matched
# case-insensitively) is what flags that COMMENT as a verdict so the loopback converges
# on it. Heimdall's verdict-less "review failed" COMMENT note carries neither this
# marker nor finding lines, so it stays a no-verdict.
_CLEAN_PASS_MARKER = "no concerns found"


def parse_heimdall_review(
    *,
    repo_full_name: str,
    pr_number: int,
    prd_number: int,
    review_state: str,
    review_body: str,
) -> HeimdallReview:
    """Parse a ``pull_request_review`` into a :class:`~retinue.loopback.HeimdallReview`.

    Maps the gh review ``state`` onto the loopback's three-valued state and reads
    heimdall's findings out of the review body — one finding per
    ``<severity>: <summary>`` line (severity is one of low/medium/high/critical,
    case-insensitive); a line without that shape is ignored as prose. A body carrying
    heimdall's clean-pass marker (:data:`_CLEAN_PASS_MARKER`) sets ``clean_pass`` so
    the findings-free clean COMMENT still reads as a verdict. The integration branch
    is derived from the PRD number (``retinue/prd-<n>``). Pure and value-free, so it
    is unit-tested without a live gh.

    Args:
        repo_full_name: e.g. "owner/repo".
        pr_number: The reviewed PR number.
        prd_number: The parent PRD (resolved from run-state); also the issue an
            escalation comments/labels and the integration branch's number.
        review_state: The gh review state string.
        review_body: The review body to parse findings from.

    Returns:
        The parsed :class:`HeimdallReview`.
    """
    from retinue.orchestrator import integration_branch

    state = _REVIEW_STATE_MAP.get(review_state.lower(), ReviewState.COMMENTED)
    return HeimdallReview(
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        prd_number=prd_number,
        prd_issue_number=prd_number,
        integration_branch=integration_branch(prd_number),
        state=state,
        findings=_parse_findings(review_body),
        clean_pass=_CLEAN_PASS_MARKER in review_body.lower(),
    )


# A finding line is ``<severity>: <summary>``, tolerating common markdown dressing —
# a leading bullet (``-``/``*``/``+``) or ordinal (``1.``/``2)``), and ``**bold**``
# around the severity — so a prose-formatted heimdall review still parses instead of
# silently reading as zero findings.
_FINDING_LINE = re.compile(
    r"^\s*(?:[-*+]|\d+[.)])?\s*\*{0,2}(low|medium|high|critical)\*{0,2}\s*:\s*(.+)$",
    re.IGNORECASE,
)


def _parse_findings(review_body: str) -> list[HeimdallFinding]:
    """Parse ``<severity>: <summary>`` lines from a review body into findings."""
    findings: list[HeimdallFinding] = []
    for line in review_body.splitlines():
        match = _FINDING_LINE.match(line)
        if match is None:
            continue
        severity = Severity[match.group(1).upper()]
        findings.append(
            HeimdallFinding(summary=match.group(2).strip(), severity=severity)
        )
    return findings


def _configure_logging() -> None:
    """Send INFO-level logs to stdout so the worker's progress is observable.

    ``run_worker`` does not configure logging (only arq's CLI does), so without
    this the root logger sits at WARNING with no handler and every ``logger.info``
    line is dropped. Install a single stdout handler at INFO; ``force=True``
    rebinds any pre-existing root config to the current stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


def _repo_contents_url(repo_full_name: str, path: str) -> str:
    """Build the GitHub contents-API URL for a file at ``path`` in a repo."""
    return f"{GITHUB_API_BASE_URL}/repos/{repo_full_name}/contents/{path}"


def _contents_url(repo_full_name: str) -> str:
    """Build the GitHub contents-API URL for a repo's ``.github/retinue.yml``."""
    return _repo_contents_url(repo_full_name, RETINUE_CONFIG_PATH)


def _auth_headers(token: str) -> dict[str, str]:
    """Build the request headers authorising a GitHub contents-API read.

    Uses the documented ``Bearer`` scheme and pins the v3 contents media type and API
    version so the response shape (a base64 ``content`` field) is stable.
    """
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _decode_contents_payload(payload: dict[str, Any]) -> str:
    """Decode a GitHub contents-API payload into the raw file text.

    The contents API returns the file body base64-encoded in ``content`` (with
    embedded newlines) under ``encoding: base64``. Decode it to UTF-8 text — the same
    raw YAML the fake fetcher hands :func:`gate_prd` for parsing.

    Raises:
        ValueError: when the payload is not a base64-encoded file (unexpected shape,
            e.g. a directory listing or an unknown encoding).
    """
    encoding = payload.get("encoding")
    if encoding != "base64" or not isinstance(payload.get("content"), str):
        raise ValueError(f"unexpected contents payload encoding: {encoding!r}")
    try:
        return base64.b64decode(payload["content"]).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"undecodable contents payload: {exc}") from exc


def github_config_fetcher(
    auth: InstallationAuth, client: httpx.AsyncClient
) -> ConfigFetcher:
    """Build the production config fetcher backed by the GitHub contents API.

    The returned async callable mints an installation token for the repo, reads
    ``.github/retinue.yml`` over the contents API, and returns its raw YAML text — the
    exact shape :func:`gate_prd` expects. A 404 (no such file) maps to ``None`` so the
    gate reads the repo as not opted in, matching the injected fake. Any other HTTP
    error is raised: a transient failure must retry the job, not be silently mistaken
    for an opted-out repo.

    Args:
        auth: Mints an installation token scoped to the target repo.
        client: A shared httpx client used for the contents read.

    Returns:
        A :class:`ConfigFetcher` returning the raw config text, or ``None`` on 404.
    """

    async def fetch(repo_full_name: str) -> str | None:
        installation = await auth.installation_token(repo_full_name)
        response = await client.get(
            _contents_url(repo_full_name),
            headers=_auth_headers(installation.token),
        )
        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        response.raise_for_status()
        return _decode_contents_payload(response.json())

    return fetch


def _issue_url(repo_full_name: str, issue_number: int) -> str:
    """Build the GitHub issues-API URL for one issue."""
    return f"{GITHUB_API_BASE_URL}/repos/{repo_full_name}/issues/{issue_number}"


def _github_issue_body_fetcher(
    auth: InstallationAuth, client: httpx.AsyncClient
) -> IssueBodyFetcher:
    """Build the production issue-body fetcher backed by the GitHub issues API.

    Returns an async ``(repo, issue) -> body`` that mints an installation token and reads
    the issue's ``body`` so the slicer slices the real PRD text. A missing body (``null``)
    reads as empty string — the slicer escalates an empty PRD as too thin. Any HTTP error
    is raised so the job retries rather than slicing a phantom body.

    Args:
        auth: Mints an installation token scoped to the target repo.
        client: A shared httpx client used for the issue read.

    Returns:
        An :data:`IssueBodyFetcher` returning the issue body text.
    """

    async def fetch(repo_full_name: str, issue_number: int) -> str:
        installation = await auth.installation_token(repo_full_name)
        response = await client.get(
            _issue_url(repo_full_name, issue_number),
            headers=_auth_headers(installation.token),
        )
        response.raise_for_status()
        return str(response.json().get("body") or "")

    return fetch


def _github_claude_md_fetcher(
    auth: InstallationAuth, client: httpx.AsyncClient
) -> ClaudeMdFetcher:
    """Build the production ``CLAUDE.md`` fetcher backed by the GitHub contents API.

    Returns an async ``(repo) -> claude_md`` that mints an installation token and reads
    the target repo's root ``CLAUDE.md`` so the build's done-check command is parsed from
    the *real* repo text (not an empty default). A repo with no ``CLAUDE.md`` (404) reads
    as empty text — the pipeline then finds no parseable done-check gate and escalates
    (:meth:`retinue.pipeline.Pipeline._has_done_check_gate`) rather than running a phantom
    gate or crash-looping the build. Any other HTTP error is raised so the job retries
    rather than building against a degraded, empty done-check spec.

    Args:
        auth: Mints an installation token scoped to the target repo.
        client: A shared httpx client used for the contents read.

    Returns:
        A :data:`~retinue.pipeline.ClaudeMdFetcher` returning the ``CLAUDE.md`` text.
    """

    async def fetch(repo_full_name: str) -> str:
        installation = await auth.installation_token(repo_full_name)
        response = await client.get(
            _repo_contents_url(repo_full_name, CLAUDE_MD_PATH),
            headers=_auth_headers(installation.token),
        )
        if response.status_code == httpx.codes.NOT_FOUND:
            return ""
        response.raise_for_status()
        return _decode_contents_payload(response.json())

    return fetch


async def on_startup(ctx: dict[str, Any]) -> None:
    """Populate the worker context with the gate's collaborators and the live pipeline.

    Installs the config fetcher and the SQLite-backed dedupe store onto ``ctx`` so
    :func:`process_prd` can gate each dequeued PRD on opt-in, validity, and novelty. When
    GitHub App auth resolves (:func:`_load_github_client`), it also installs the real
    PRD-body fetcher and a ``pipeline_factory`` over the production adapters — the factory
    sources each repo's ``CLAUDE.md`` (the done-check command) and binds the live build
    lane — so an accepted PRD drives the real slice -> build -> staging-PR pipeline. The
    same auth branch binds the webhook's ad-hoc drain *and* the worker-global heartbeat's
    four collaborators (:func:`retinue.heartbeat.heartbeat_tick` reads them from ``ctx``):
    the real wall-clock, the installed-and-opted-in repo enumerator, the *same* bound
    ad-hoc drain the kick fires, and a bound :func:`retinue.cron.run_cron_tick`. The same
    branch installs the run-state store and enqueues the crash-resume sweep
    (:func:`resume_rounds_job`) so rounds a crash left in flight are re-driven. With no
    auth wired, the fetcher defaults to not-opted-in and neither the pipeline nor the
    heartbeat is installed (so the registered cron tick safely no-ops).
    """
    global settings
    if settings is None:
        settings = _load_settings()
    wiring = _load_github_client()
    if wiring is None:
        # No GitHub App auth wired yet (the concrete InstallationAuth is another
        # layer's seam): fall back to the safe not-opted-in default so nothing is
        # processed without an explicit config, rather than crashing the worker. With
        # no auth there is also no pipeline, so an accepted PRD stops after the gate.
        ctx["fetch_config"] = _no_config_fetcher
    else:
        auth, client = wiring
        fetch_claude_md = _github_claude_md_fetcher(auth, client)
        pipeline_factory = build_pipeline_factory(
            settings, auth, fetch_claude_md=fetch_claude_md
        )
        ctx["github_client"] = client
        ctx["fetch_config"] = github_config_fetcher(auth, client)
        ctx["fetch_prd_body"] = _github_issue_body_fetcher(auth, client)
        ctx["pipeline_factory"] = pipeline_factory
        # The webhook's low-latency ad-hoc kick (run_adhoc_drain_job) reads ``adhoc_drain``
        # from ctx; bind it here so a kick on a deployed worker actually drains. The
        # heartbeat's safety-net sweep fires this *same* bound drain (below) so a kick and a
        # sweep are identical work under one single-run lock.
        adhoc_drain = _bind_adhoc_drain(
            settings,
            auth,
            pipeline_factory=pipeline_factory,
            fetch_claude_md=fetch_claude_md,
        )
        ctx["adhoc_drain"] = adhoc_drain
        # The worker-global heartbeat (heartbeat_tick) reads its four collaborators from ctx;
        # bind them here so the registered cron tick actually sweeps in production instead of
        # hitting the not-wired skip. The clock is the real wall-clock, the enumerator lists
        # the App's opted-in repos, the drain is the *same* bound ad-hoc drain the kick fires,
        # and the cron tick is a bound run_cron_tick over the backlog lane.
        ctx["heartbeat_clock"] = SystemClock()
        ctx["heartbeat_enumerate_repos"] = _bind_heartbeat_enumerate(
            auth, fetch_config=ctx["fetch_config"]
        )
        ctx["heartbeat_drain"] = adhoc_drain
        ctx["heartbeat_cron_tick"] = _bind_heartbeat_cron_tick(
            settings,
            auth,
            pipeline_factory=pipeline_factory,
            fetch_claude_md=fetch_claude_md,
        )
        # Crash-resume: install the run-state store the sweep enumerates (the same file
        # every factory-built pipeline records into) and kick the sweep as a normal Arq
        # job so resuming stranded rounds never blocks worker boot.
        ctx["run_state"] = RunStateStore(run_state_store_path(settings))
        redis = ctx.get("redis")
        if redis is not None:
            # A pinned job id + the keep_result=0 registration make the kick reliable:
            # without them a restart inside arq's default 1h result window is silently
            # deduped against the previous boot's completed sweep and never runs.
            await redis.enqueue_job(RESUME_ROUNDS_TASK, _job_id="resume-rounds")
        else:
            logger.warning(
                "No redis on the worker context; the resume sweep was not enqueued"
            )
    ctx["dedupe"] = PrdDedupeStore(settings.dedupe_db_path)


def _bind_adhoc_drain(
    settings: Any,
    auth: InstallationAuth,
    *,
    pipeline_factory: PipelineFactory,
    fetch_claude_md: ClaudeMdFetcher,
) -> HeartbeatDrain:
    """Bind the production ad-hoc drain to a ``(*, repo_full_name, config)`` callable.

    The webhook's kick (:func:`run_adhoc_drain_job`) and — once issue #43 wires it — the
    heartbeat's sweep both fire the drain through this one seam, so they are the *same* work
    under one single-run lock. The shared collaborators are built once: the service-level
    :class:`~retinue.budget.BudgetGovernor` over ``settings.budget_db_path`` (the *same*
    durable rolling-24h ledger the PRD and cron lanes meter, so the budget is one window) and
    a per-repo single-run lock registry, so two repos drain concurrently while a repo's own
    kicked and swept drains serialize.

    Each call mints a per-repo installation token, then constructs the per-repo gh seam
    (:class:`retinue.adhoc_drain.GhCli`), reuses the worker's ``pipeline_factory`` to build
    the repo's pipeline (its ``process_adhoc_pr`` opens the PR), binds the real ad-hoc
    build+PR primitive (:func:`retinue.pipeline.bind_adhoc_build`) and the PR-open-only
    stranded-branch recovery (:func:`retinue.pipeline.bind_adhoc_pr_open`), and drives
    :func:`retinue.adhoc_drain.run_adhoc_drain` via :func:`retinue.wiring.bind_adhoc_drain`.

    Args:
        settings: The runtime settings carrying the budget + Anthropic config.
        auth: The GitHub App installation auth used to mint per-repo tokens.
        pipeline_factory: The worker's pipeline factory, reused so the ad-hoc PR step rides
            the same per-repo pipeline the PRD lane builds.
        fetch_claude_md: Reads each repo's ``CLAUDE.md`` (the done-check command source).

    Returns:
        The bound drain — an async ``(*, repo_full_name, config) -> None``.
    """
    from retinue.adhoc_drain import AdhocDrainLock, GhCli
    from retinue.budget import AuthMode, BudgetGovernor, BudgetLedger, SystemClock
    from retinue.pipeline import bind_adhoc_build, bind_adhoc_pr_open
    from retinue.wiring import bind_adhoc_drain

    governor = BudgetGovernor(
        BudgetLedger(
            settings.budget_db_path,
            clock=SystemClock(),
            auth_mode=AuthMode.from_config(settings.auth_mode),
            weekly_budget=settings.weekly_budget,
            daily_cap_fraction=settings.budget_daily_cap_fraction,
        )
    )
    locks: dict[str, AdhocDrainLock] = {}

    async def drain(*, repo_full_name: str, config: RepoConfig) -> None:
        token = (await auth.installation_token(repo_full_name)).token
        gh = GhCli(token=token)
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
        bound = bind_adhoc_drain(
            gh=gh,
            build=build,
            open_pr=bind_adhoc_pr_open(pipeline),
            governor=governor,
            estimated_amount=_ADHOC_DRAIN_ESTIMATED_AMOUNT,
            lock=locks.setdefault(repo_full_name, AdhocDrainLock()),
        )
        await bound(repo_full_name=repo_full_name, config=config)

    return drain


def _bind_heartbeat_enumerate(
    auth: InstallationAuth, *, fetch_config: ConfigFetcher
) -> RepoEnumerator:
    """Bind the heartbeat's opted-in repo enumerator over the GitHub-App installed set.

    Returns an async ``() -> list[DueRepo]`` that lists the App's installed repositories
    (:meth:`retinue.github_app.InstalledRepos.installed_repositories`), fetches each repo's
    ``.github/retinue.yml`` through the *same* opt-in ``fetch_config`` seam the PRD gate
    uses, and yields a :class:`~retinue.heartbeat.DueRepo` for each repo with an accepted
    :class:`~retinue.repo_config.RepoConfig`. A repo not opted in (no fetchable config) or
    with a malformed config is dropped, so the sweep only touches opted-in repos — the same
    opt-in resolution the webhook/PRD path applies, just enumerated rather than event-driven.

    The enumeration seam is the production :class:`retinue.github_app.GitHubInstallationAuth`,
    which also satisfies :class:`~retinue.github_app.InstalledRepos`; a bare auth without it
    yields an empty sweep rather than crashing the tick.
    """

    async def enumerate_repos() -> list[DueRepo]:
        list_repos = getattr(auth, "installed_repositories", None)
        if list_repos is None:
            return []
        due: list[DueRepo] = []
        for repo_full_name in await list_repos():
            raw = await fetch_config(repo_full_name)
            if raw is None:
                continue
            config = load_repo_config(raw)
            if config is None:
                continue
            due.append(DueRepo(repo_full_name=repo_full_name, config=config))
        return due

    return enumerate_repos


def _bind_heartbeat_cron_tick(
    settings: Any,
    auth: InstallationAuth,
    *,
    pipeline_factory: PipelineFactory,
    fetch_claude_md: ClaudeMdFetcher,
) -> HeartbeatCronTick:
    """Bind the heartbeat's backlog cron tick to ``run_cron_tick`` over the real adapters.

    Returns an async ``(*, repo_full_name, tick_number) -> CronTickResult`` — the
    :data:`retinue.heartbeat.HeartbeatCronTick` shape — that drives one repo's backlog lane
    through :func:`retinue.wiring.bind_cron_tick`. The shared collaborators are built once:
    the service-level :class:`~retinue.budget.BudgetGovernor` over ``settings.budget_db_path``
    (the *same* durable rolling-24h ledger the PRD and ad-hoc lanes meter, so the budget is
    one window) and a per-repo single-run lock registry, so two repos tick concurrently while
    a repo's own ticks serialize.

    Each call mints a per-repo installation token, constructs the per-repo backlog gh seam
    (:class:`retinue.cron.GhCli`) and the cron build (:class:`retinue.cron.SliceBuilder`
    over the same orchestrator adapters the PRD lane builds), binds them through
    :func:`retinue.wiring.bind_cron_tick`, and runs one tick. The heartbeat owns the
    per-tick estimate (the flat per-build charge), so the bound callable supplies it.

    Args:
        settings: The runtime settings carrying the budget + Anthropic config.
        auth: The GitHub App installation auth used to mint per-repo tokens.
        pipeline_factory: The worker's pipeline factory (accepted for symmetry with the
            other binders; the cron build assembles its own orchestrator adapters).
        fetch_claude_md: Reads each repo's ``CLAUDE.md`` (the done-check command source).

    Returns:
        The bound cron tick — an async ``(*, repo_full_name, tick_number) -> CronTickResult``.
    """
    from contextlib import AbstractAsyncContextManager

    from retinue.budget import AuthMode, BudgetGovernor, BudgetLedger
    from retinue.cron import CronLock, GhCli
    from retinue.pipeline import build_cron_slice_builder
    from retinue.wiring import bind_cron_tick

    governor = BudgetGovernor(
        BudgetLedger(
            settings.budget_db_path,
            clock=SystemClock(),
            auth_mode=AuthMode.from_config(settings.auth_mode),
            weekly_budget=settings.weekly_budget,
            daily_cap_fraction=settings.budget_daily_cap_fraction,
        )
    )
    locks: dict[str, AbstractAsyncContextManager[object]] = {}

    async def cron_tick(
        *, repo_full_name: str, tick_number: int
    ) -> CronTickResult:
        token = (await auth.installation_token(repo_full_name)).token
        gh = GhCli(token=token)
        build = await build_cron_slice_builder(
            settings,
            auth,
            repo_full_name=repo_full_name,
            token=token,
            fetch_claude_md=fetch_claude_md,
        )
        bound = bind_cron_tick(
            gh=gh,
            governor=governor,
            clock=SystemClock(),
            build=build,
            lock=locks.setdefault(repo_full_name, CronLock()),
        )
        return await bound(
            repo_full_name=repo_full_name,
            tick_number=tick_number,
            estimated_amount=_ADHOC_DRAIN_ESTIMATED_AMOUNT,
        )

    return cron_tick


# The flat per-build charge the ad-hoc drain meters against the shared rolling-24h cap,
# matching the PRD lane's estimate (:data:`retinue.pipeline._BUILD_ESTIMATED_AMOUNT`); a
# build that would cross the cap is skipped so the one shared budget is never overshot.
_ADHOC_DRAIN_ESTIMATED_AMOUNT = 1.0


async def on_shutdown(ctx: dict[str, Any]) -> None:
    """Close what :func:`on_startup` opened: the GitHub HTTP client and dedupe store.

    The dedupe store's SQLite connection rides an aiosqlite worker thread; skipping
    its close would strand that thread past shutdown.
    """
    client: httpx.AsyncClient | None = ctx.get("github_client")
    if client is not None:
        await client.aclose()
    dedupe: PrdDedupeStore | None = ctx.get("dedupe")
    if dedupe is not None:
        await dedupe.close()


async def _no_config_fetcher(repo_full_name: str) -> str | None:
    """Fallback fetcher used when no GitHub App auth is wired: treat repo as opted out.

    The safe default — nothing is processed without an explicit, fetchable config.
    """
    logger.debug("No GitHub auth wired; treating %s as not opted in", repo_full_name)
    return None


def _load_github_client() -> tuple[InstallationAuth, httpx.AsyncClient] | None:
    """Construct the production GitHub installation auth and HTTP client, if available.

    Returns ``None`` in two cases, both of which make the worker fall back to the safe
    not-opted-in default rather than crashing: when no concrete ``build_installation_auth``
    is wired at all, and — the fresh-deploy case — when the builder exists but raises
    :class:`~retinue.github_app.InstallationAuthError` because the GitHub App is not yet
    configured (no ``github_app_id`` / private-key path). A deploy with only
    ``WEBHOOK_SECRET`` set must boot and log SKIPs (see DEPLOY.md), so an unconfigured App
    is graceful degradation, not a startup failure. Resolved lazily so registering the
    task (e.g. in tests) needs no GitHub App credentials and opens no network client at
    import time.
    """
    import retinue.github_app as github_app

    builder = getattr(github_app, "build_installation_auth", None)
    if builder is None:
        return None
    try:
        auth = builder()
    except github_app.InstallationAuthError:
        return None
    return auth, httpx.AsyncClient(timeout=30.0)


# Module-level settings, loaded lazily in main() so importing this module does
# not require the env vars to be present (e.g. when registering the task in tests).
settings: Any = None


def _load_settings() -> Any:
    from retinue.config import Settings

    return Settings()  # type: ignore[call-arg]


def main() -> None:
    """Console-script entrypoint: start the Arq worker with WorkerSettings.

    Resolves ``WorkerSettings.redis_settings`` from the configured ``REDIS_URL``
    here, at process start: arq reads that class attribute when it constructs the
    Worker, before ``on_startup`` runs, so the override must be applied now.
    """
    from arq.worker import run_worker

    _configure_logging()

    global settings
    if settings is None:
        settings = _load_settings()
    WorkerSettings.redis_settings = RedisSettings.from_dsn(settings.redis_url)
    # arq reads job_timeout off the class before on_startup; its 300s default cancels a
    # real build mid-implement, so override it here at process start (issue: dogfood).
    WorkerSettings.job_timeout = settings.job_timeout_seconds

    run_worker(WorkerSettings)  # type: ignore[arg-type]


class WorkerSettings:
    """Arq WorkerSettings: registers process_prd and the Redis connection.

    Launch the worker process with:
        arq retinue.worker.WorkerSettings
    """

    # The re-kicked tasks register with ``keep_result=0``: arq's enqueue dedups on the
    # completed job's lingering *result* key too (default 1h), so keeping results would
    # silently swallow a restart's sweep kick, a post-drain webhook kick, and a deferred
    # round's re-enqueue. ``resume_round_job`` also self-retries via ``Retry`` while the
    # budget stays deferred, so it carries a bounded ``max_tries``.
    functions = [
        process_prd,
        process_review_job,
        reap_pr_job,
        arq_func(run_adhoc_drain_job, keep_result=0),
        arq_func(resume_rounds_job, keep_result=0),
        arq_func(resume_round_job, keep_result=0, max_tries=_RESUME_ROUND_MAX_TRIES),
    ]
    # The worker-global cron heartbeat: fires every Nth minute as the safety-net sweep for
    # issues labeled while the webhook was missed (firing the ad-hoc drain for each due repo)
    # and drives the backlog cron lane (``run_cron_tick``) — the first runtime caller of the
    # previously-dead lane. Its collaborators are read from ``ctx`` in :func:`heartbeat_tick`;
    # a bare worker with none wired ticks harmlessly.
    cron_jobs = [cron(heartbeat_tick, minute=set(HEARTBEAT_MINUTES))]
    on_startup = on_startup
    on_shutdown = on_shutdown
    # Overridden from JOB_TIMEOUT_SECONDS in main() at process start; the default here must
    # already outlast a real build, since arq reads it before on_startup and its own 300s
    # default would cancel the drain mid-implement.
    job_timeout: int = 1800
    # Overridden from the configured REDIS_URL in main() at process start (arq
    # reads this attribute before on_startup runs); the localhost default keeps it
    # a valid RedisSettings for the ``arq retinue.worker.WorkerSettings`` launch.
    redis_settings: RedisSettings = RedisSettings()
