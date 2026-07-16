"""Tests for the worker's pipeline wiring: process_prd + the review/reap tasks.

The worker tasks read their collaborators from the Arq ``ctx`` (populated by
``on_startup``). These tests inject fakes into ``ctx`` — a recording pipeline, a config
fetcher, a PRD-body fetcher — so the dispatch and parsing are exercised with no real gh,
Anthropic, Docker, or network.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

import retinue.github_app as github_app
import retinue.worker as worker
from retinue.dedupe import PrdDedupeStore
from retinue.github_app import InstallationAuthError, InstallationToken
from retinue.handoff import MergedPullRequest, ReapOutcome, ReapResult
from retinue.loopback import (
    HeimdallReview,
    ReviewState,
    VerdictOutcome,
    VerdictResult,
    parse_heimdall_review,
)
from retinue.pipeline import PrdJobResult, ResumeRoundOutcome
from retinue.queue import RESUME_ROUND_TASK, RESUME_ROUNDS_TASK, RUN_ADHOC_DRAIN_TASK
from retinue.reconcile import ReconcileResult, ResumePhase, RunStateStore
from retinue.repo_config import RepoConfig
from retinue.vocab import Severity
from retinue.worker import (
    WorkerSettings,
    on_shutdown,
    on_startup,
    process_prd,
    process_review_job,
    reap_pr_job,
    resume_round_job,
    resume_rounds_job,
    run_adhoc_drain_job,
)

_CONFIG_YAML = "staging_branch: staging\nretry_cap: 2\n"


@dataclass
class _RecordingPipeline:
    """A fake Pipeline recording every call the worker tasks make against it."""

    prd_calls: list[dict[str, Any]] = field(default_factory=list)
    reviews: list[HeimdallReview] = field(default_factory=list)
    reaps: list[MergedPullRequest] = field(default_factory=list)
    pr_round: tuple[int, list[int]] | None = (7, [100, 101])
    resume_calls: list[tuple[str, int]] = field(default_factory=list)
    resume_failures: set[int] = field(default_factory=set)
    prd_result: PrdJobResult | None = None
    prd_exception: Exception | None = None
    rounds: set[tuple[str, int]] = field(default_factory=set)
    resume_deferred: bool = False

    async def process_prd_job(
        self, *, repo_full_name: str, prd_number: int, prd_body: str
    ) -> PrdJobResult:
        self.prd_calls.append(
            {"repo": repo_full_name, "prd": prd_number, "body": prd_body}
        )
        if self.prd_exception is not None:
            raise self.prd_exception
        return self.prd_result or PrdJobResult(sliced=True, pr_opened=True)

    async def process_review(self, review: HeimdallReview) -> VerdictResult:
        self.reviews.append(review)
        return VerdictResult(outcome=VerdictOutcome.CONVERGED)

    async def reap_pr(self, merged: MergedPullRequest) -> ReapResult:
        self.reaps.append(merged)
        return ReapResult(outcome=ReapOutcome.REAPED, prd_closed=True)

    async def round_for_pr(
        self, *, repo_full_name: str, pr_number: int
    ) -> tuple[int, list[int]] | None:
        return self.pr_round

    async def has_round(self, *, repo_full_name: str, prd_number: int) -> bool:
        return (repo_full_name, prd_number) in self.rounds

    async def resume_round(
        self, *, repo_full_name: str, prd_number: int
    ) -> ResumeRoundOutcome:
        self.resume_calls.append((repo_full_name, prd_number))
        if prd_number in self.resume_failures:
            raise RuntimeError(f"resume of PRD #{prd_number} exploded")
        return ResumeRoundOutcome(
            reconcile=ReconcileResult(
                phase=ResumePhase.DONE,
                integration_branch=f"retinue/prd-{prd_number}",
            ),
            deferred=self.resume_deferred,
        )


def _ctx(tmp_path: Path, pipeline: _RecordingPipeline, *, body: str = "") -> dict[str, Any]:
    async def fetch_config(repo_full_name: str) -> str | None:
        return _CONFIG_YAML

    async def fetch_body(repo_full_name: str, issue_number: int) -> str:
        return body

    async def factory(repo_full_name: str, config: RepoConfig) -> _RecordingPipeline:
        return pipeline

    return {
        "fetch_config": fetch_config,
        "fetch_prd_body": fetch_body,
        "pipeline_factory": factory,
        "dedupe": PrdDedupeStore(tmp_path / "dedupe.sqlite3"),
    }


CtxFactory = Callable[..., dict[str, Any]]


@pytest_asyncio.fixture()
async def make_ctx(tmp_path: Path) -> AsyncIterator[CtxFactory]:
    """A :func:`_ctx` factory that closes each ctx's dedupe store at teardown.

    A leaked store's aiosqlite thread races the test's already-closed event loop at
    GC time; closing here keeps teardown deterministic.
    """
    stores: list[PrdDedupeStore] = []

    def _make(pipeline: _RecordingPipeline, *, body: str = "") -> dict[str, Any]:
        ctx = _ctx(tmp_path, pipeline, body=body)
        stores.append(ctx["dedupe"])
        return ctx

    yield _make
    for store in stores:
        await store.close()


# --- process_prd ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_prd_drives_pipeline_with_fetched_body(
    make_ctx: CtxFactory,
) -> None:
    """An accepted PRD reaches the pipeline with its fetched issue body."""
    pipeline = _RecordingPipeline()
    body = "Implement the thing with enough body text to slice it responsibly here."
    ctx = make_ctx(pipeline, body=body)

    await process_prd(ctx, repo_full_name="owner/repo", issue_number=7, action="opened")

    assert pipeline.prd_calls == [{"repo": "owner/repo", "prd": 7, "body": body}]


@pytest.mark.asyncio
async def test_process_prd_without_pipeline_is_a_noop(make_ctx: CtxFactory) -> None:
    """With no pipeline_factory wired the accepted PRD stops after the gate."""
    pipeline = _RecordingPipeline()
    ctx = make_ctx(pipeline)
    del ctx["pipeline_factory"]

    await process_prd(ctx, repo_full_name="owner/repo", issue_number=7, action="opened")

    assert pipeline.prd_calls == []


@pytest.mark.asyncio
async def test_process_prd_releases_dedupe_on_a_pre_slice_crash(
    make_ctx: CtxFactory,
) -> None:
    """A pipeline crash before any durable slice releases the PRD's dedupe claim.

    The claim is recorded at the gate, before any run state persists; without the
    release a pre-slice crash burns the PRD forever — every redelivery reads as a
    duplicate and the PRD is never processed.
    """
    pipeline = _RecordingPipeline(prd_exception=RuntimeError("boom before slicing"))
    ctx = make_ctx(pipeline)

    with pytest.raises(RuntimeError, match="boom before slicing"):
        await process_prd(
            ctx, repo_full_name="owner/repo", issue_number=7, action="opened"
        )

    # The claim was released: a redelivery claims the key afresh.
    assert await ctx["dedupe"].claim("owner/repo#7") is True


@pytest.mark.asyncio
async def test_process_prd_keeps_the_claim_when_the_round_persisted(
    make_ctx: CtxFactory,
) -> None:
    """A crash *after* slices persisted keeps the claim — the resume sweep owns it.

    Once the round's slice set is durable, the startup sweep will resume it; releasing
    the claim would let a redelivery re-slice the same PRD into duplicate issues.
    """
    pipeline = _RecordingPipeline(prd_exception=RuntimeError("boom after slicing"))
    pipeline.rounds.add(("owner/repo", 7))
    ctx = make_ctx(pipeline)

    with pytest.raises(RuntimeError, match="boom after slicing"):
        await process_prd(
            ctx, repo_full_name="owner/repo", issue_number=7, action="opened"
        )

    assert await ctx["dedupe"].claim("owner/repo#7") is False  # still claimed


# --- budget-deferred PRD re-enqueue ----------------------------------------------


class _RecordingRedis:
    """A fake Arq pool recording every enqueue_job call's task name and kwargs."""

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, dict[str, Any]]] = []

    async def enqueue_job(self, task_name: str, *args: Any, **kwargs: Any) -> object:
        self.enqueued.append((task_name, kwargs))
        return object()


@pytest.mark.asyncio
async def test_process_prd_enqueues_a_resume_for_a_deferred_build(
    make_ctx: CtxFactory,
) -> None:
    """A budget-deferred PRD enqueues resume_round_job for when the window frees.

    Re-enqueueing PROCESS_PRD_TASK instead would re-slice the PRD into duplicate
    issues; the resume task reconciles and re-drives only the unbuilt phase.
    """
    from datetime import UTC, datetime

    when = datetime(2026, 6, 23, tzinfo=UTC)
    pipeline = _RecordingPipeline(
        prd_result=PrdJobResult(
            sliced=True, pr_opened=False, deferred=True, defer_until=when
        )
    )
    ctx = make_ctx(pipeline)
    redis = _RecordingRedis()
    ctx["redis"] = redis

    await process_prd(ctx, repo_full_name="owner/repo", issue_number=7, action="opened")

    assert redis.enqueued == [
        (
            RESUME_ROUND_TASK,
            {
                "repo_full_name": "owner/repo",
                "prd_number": 7,
                "_job_id": "resume-round:owner/repo#7",
                "_defer_until": when,
            },
        )
    ]


@pytest.mark.asyncio
async def test_process_prd_deferral_without_a_window_falls_back_an_hour(
    make_ctx: CtxFactory,
) -> None:
    """A deferral with no ``defer_until`` re-enqueues on the one-hour fallback."""
    pipeline = _RecordingPipeline(
        prd_result=PrdJobResult(sliced=True, pr_opened=False, deferred=True)
    )
    ctx = make_ctx(pipeline)
    redis = _RecordingRedis()
    ctx["redis"] = redis

    await process_prd(ctx, repo_full_name="owner/repo", issue_number=7, action="opened")

    assert len(redis.enqueued) == 1
    task, kwargs = redis.enqueued[0]
    assert task == RESUME_ROUND_TASK
    assert kwargs["_defer_by"] == 3600
    assert "_defer_until" not in kwargs


@pytest.mark.asyncio
async def test_process_prd_without_deferral_enqueues_no_resume(
    make_ctx: CtxFactory,
) -> None:
    """An undeferred PRD run enqueues nothing extra."""
    pipeline = _RecordingPipeline()
    ctx = make_ctx(pipeline)
    redis = _RecordingRedis()
    ctx["redis"] = redis

    await process_prd(ctx, repo_full_name="owner/repo", issue_number=7, action="opened")

    assert redis.enqueued == []


@pytest.mark.asyncio
async def test_resume_round_job_resumes_the_round(make_ctx: CtxFactory) -> None:
    """The deferred-resume task drives the pipeline's resume for its one round."""
    pipeline = _RecordingPipeline()
    ctx = make_ctx(pipeline)

    await resume_round_job(ctx, repo_full_name="owner/repo", prd_number=7)

    assert pipeline.resume_calls == [("owner/repo", 7)]


@pytest.mark.asyncio
async def test_resume_round_job_retries_while_still_deferred(
    make_ctx: CtxFactory,
) -> None:
    """A resume whose re-gated build is still over budget retries via arq's Retry.

    Re-enqueueing itself under the same ``_job_id`` would be silently dropped (the
    running job's key still exists), so the still-deferred resume must raise
    ``arq.worker.Retry`` and let arq re-schedule the same job.
    """
    from arq.worker import Retry

    pipeline = _RecordingPipeline(resume_deferred=True)
    ctx = make_ctx(pipeline)

    with pytest.raises(Retry):
        await resume_round_job(ctx, repo_full_name="owner/repo", prd_number=7)

    assert pipeline.resume_calls == [("owner/repo", 7)]


@pytest.mark.asyncio
async def test_resume_round_job_skips_a_deopted_repo(make_ctx: CtxFactory) -> None:
    """A deferred resume for a repo no longer opted in is a skip, not a crash."""
    pipeline = _RecordingPipeline()
    ctx = make_ctx(pipeline)

    async def no_config(repo_full_name: str) -> str | None:
        return None

    ctx["fetch_config"] = no_config

    await resume_round_job(ctx, repo_full_name="owner/repo", prd_number=7)

    assert pipeline.resume_calls == []


# --- review loopback ------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_review_job_parses_and_drives_loopback(
    make_ctx: CtxFactory,
) -> None:
    """A review job resolves the PRD, parses findings, and drives the loopback."""
    pipeline = _RecordingPipeline(pr_round=(7, [100]))
    ctx = make_ctx(pipeline)

    await process_review_job(
        ctx,
        repo_full_name="owner/repo",
        pr_number=99,
        review_state="changes_requested",
        review_body="high: a blocking problem\nlow: a nit",
    )

    assert len(pipeline.reviews) == 1
    review = pipeline.reviews[0]
    assert review.pr_number == 99
    assert review.prd_number == 7
    assert review.integration_branch == "retinue/prd-7"
    assert review.state is ReviewState.REQUEST_CHANGES
    assert [f.severity for f in review.findings] == [Severity.HIGH, Severity.LOW]


@pytest.mark.asyncio
async def test_process_review_job_skips_unknown_pr(make_ctx: CtxFactory) -> None:
    """A review on a PR not in run-state is skipped (not the retinue's PR)."""
    pipeline = _RecordingPipeline(pr_round=None)
    ctx = make_ctx(pipeline)

    await process_review_job(
        ctx, repo_full_name="owner/repo", pr_number=5, review_state="approved"
    )

    assert pipeline.reviews == []


@pytest.mark.asyncio
async def test_process_review_job_serializes_per_pr(make_ctx: CtxFactory) -> None:
    """Two review jobs for the same PR run one at a time, never interleaved.

    GitHub redelivers ``pull_request_review`` webhooks; without a per-PR lock two
    deliveries both read a stale round count and double-consume the rebuild budget.
    """

    class _SlowPipeline(_RecordingPipeline):
        def __init__(self) -> None:
            super().__init__(pr_round=(7, [100]))
            self.active = 0
            self.max_active = 0

        async def process_review(self, review: HeimdallReview) -> VerdictResult:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0)
            self.active -= 1
            return await super().process_review(review)

    pipeline = _SlowPipeline()
    ctx = make_ctx(pipeline)

    def job() -> Any:
        return process_review_job(
            ctx,
            repo_full_name="owner/repo",
            pr_number=99,
            review_state="changes_requested",
            review_body="high: a blocking problem",
        )

    await asyncio.gather(job(), job())

    assert len(pipeline.reviews) == 2
    assert pipeline.max_active == 1


# --- reap -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reap_pr_job_resolves_slices_and_reaps(make_ctx: CtxFactory) -> None:
    """A merge job resolves the PRD + slice issues from run-state and reaps."""
    pipeline = _RecordingPipeline(pr_round=(7, [100, 101]))
    ctx = make_ctx(pipeline)

    await reap_pr_job(ctx, repo_full_name="owner/repo", pr_number=99)

    assert pipeline.reaps == [
        MergedPullRequest(
            repo_full_name="owner/repo",
            pr_number=99,
            prd_number=7,
            slice_issues=[100, 101],
        )
    ]


@pytest.mark.asyncio
async def test_reap_pr_job_skips_unknown_pr(make_ctx: CtxFactory) -> None:
    """A merge of a PR the retinue never opened is skipped, not reaped."""
    pipeline = _RecordingPipeline(pr_round=None)
    ctx = make_ctx(pipeline)

    await reap_pr_job(ctx, repo_full_name="owner/repo", pr_number=5)

    assert pipeline.reaps == []


# --- resume sweep (crash-resume on restart) ---------------------------------------


async def _seeded_run_state(tmp_path: Path) -> RunStateStore:
    """A run-state store persisting two in-flight rounds, as a crash left them."""
    store = RunStateStore(tmp_path / "run-state.sqlite3")
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=7, issue_numbers=[100, 101]
    )
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=9, issue_numbers=[200]
    )
    return store


@pytest.mark.asyncio
async def test_resume_rounds_job_resumes_every_persisted_round(
    tmp_path: Path,
    make_ctx: CtxFactory,
) -> None:
    """The sweep resumes each persisted round through the repo's pipeline."""
    pipeline = _RecordingPipeline()
    ctx = make_ctx(pipeline)
    ctx["run_state"] = await _seeded_run_state(tmp_path)

    await resume_rounds_job(ctx)

    assert pipeline.resume_calls == [("owner/repo", 7), ("owner/repo", 9)]


@pytest.mark.asyncio
async def test_resume_rounds_job_skips_a_failing_round_and_continues(
    tmp_path: Path,
    make_ctx: CtxFactory,
) -> None:
    """One round's resume crashing is logged and skipped; the sweep still finishes."""
    pipeline = _RecordingPipeline(resume_failures={7})
    ctx = make_ctx(pipeline)
    ctx["run_state"] = await _seeded_run_state(tmp_path)

    await resume_rounds_job(ctx)  # must not raise

    assert pipeline.resume_calls == [("owner/repo", 7), ("owner/repo", 9)]


@pytest.mark.asyncio
async def test_resume_rounds_job_without_run_state_is_a_noop(
    make_ctx: CtxFactory,
) -> None:
    """A bare worker (no run-state/pipeline wired) logs and returns, never crashing."""
    pipeline = _RecordingPipeline()
    ctx = make_ctx(pipeline)
    assert "run_state" not in ctx

    await resume_rounds_job(ctx)

    assert pipeline.resume_calls == []


@pytest.mark.asyncio
async def test_resume_rounds_job_skips_a_deopted_repo(
    tmp_path: Path,
    make_ctx: CtxFactory,
) -> None:
    """A persisted round of a repo no longer opted in is skipped, its row left alone."""
    pipeline = _RecordingPipeline()
    ctx = make_ctx(pipeline)
    store = await _seeded_run_state(tmp_path)
    ctx["run_state"] = store

    async def no_config(repo_full_name: str) -> str | None:
        return None

    ctx["fetch_config"] = no_config

    await resume_rounds_job(ctx)

    assert pipeline.resume_calls == []
    assert len(await store.all_rounds()) == 2  # the rows survive for a later sweep


def _registered_names() -> set[str]:
    """The registered task names; ``arq.worker.func`` wrappers carry ``.name``."""
    return {
        getattr(fn, "name", None) or getattr(fn, "__name__")  # noqa: B009
        for fn in WorkerSettings.functions
    }


def test_worker_registers_the_resume_task() -> None:
    """WorkerSettings registers ``resume_rounds_job`` under the enqueue-side task name."""
    assert RESUME_ROUNDS_TASK in _registered_names()
    assert resume_rounds_job.__name__ == RESUME_ROUNDS_TASK


def test_worker_registers_the_deferred_resume_task() -> None:
    """WorkerSettings registers ``resume_round_job`` under the enqueue-side task name."""
    assert RESUME_ROUND_TASK in _registered_names()
    assert resume_round_job.__name__ == RESUME_ROUND_TASK


def test_re_enqueued_tasks_keep_no_result() -> None:
    """The self-re-enqueued tasks register with ``keep_result=0``.

    arq's enqueue dedups on the completed job's *result* key too; a lingering result
    (default 1h) would silently drop the next kick — a restart inside that window
    would skip the resume sweep, and a deferred resume could never re-enqueue.
    """
    from arq.worker import Function

    by_name = {
        fn.name: fn for fn in WorkerSettings.functions if isinstance(fn, Function)
    }
    for task in (RESUME_ROUNDS_TASK, RESUME_ROUND_TASK, RUN_ADHOC_DRAIN_TASK):
        assert task in by_name, f"{task} is not registered via arq func()"
        assert by_name[task].keep_result_s == 0, f"{task} keeps its result"


@pytest.mark.asyncio
async def test_resume_rounds_job_re_enqueues_a_deferred_round(
    tmp_path: Path,
    make_ctx: CtxFactory,
) -> None:
    """The startup sweep re-enqueues a round whose resumed build was budget-deferred.

    Without this the deferred round's row just sits until the *next* restart happens
    to sweep it — the deferral would only ever resolve by accident.
    """
    pipeline = _RecordingPipeline(resume_deferred=True)
    ctx = make_ctx(pipeline)
    ctx["run_state"] = await _seeded_run_state(tmp_path)
    redis = _RecordingRedis()
    ctx["redis"] = redis

    await resume_rounds_job(ctx)

    assert [task for task, _ in redis.enqueued] == [RESUME_ROUND_TASK] * 2
    assert redis.enqueued[0][1]["_job_id"] == "resume-round:owner/repo#7"
    assert redis.enqueued[1][1]["_job_id"] == "resume-round:owner/repo#9"


# --- ad-hoc drain kick ----------------------------------------------------------


@pytest.mark.asyncio
async def test_run_adhoc_drain_job_drives_the_bound_drain(make_ctx: CtxFactory) -> None:
    """A kicked drain job calls the bound drain from ctx with the repo (and its config)."""
    calls: list[dict[str, Any]] = []

    async def drain(*, repo_full_name: str, config: RepoConfig) -> None:
        calls.append({"repo": repo_full_name, "config": config})

    ctx = make_ctx(_RecordingPipeline())
    ctx["adhoc_drain"] = drain

    await run_adhoc_drain_job(ctx, repo_full_name="owner/repo")

    assert len(calls) == 1
    assert calls[0]["repo"] == "owner/repo"
    assert calls[0]["config"].staging_branch == "staging"


@pytest.mark.asyncio
async def test_run_adhoc_drain_job_without_drain_is_a_noop(
    make_ctx: CtxFactory,
) -> None:
    """With no drain wired (bare worker) the kick logs and returns, never crashing."""
    ctx = make_ctx(_RecordingPipeline())
    assert "adhoc_drain" not in ctx

    await run_adhoc_drain_job(ctx, repo_full_name="owner/repo")  # must not raise


@pytest.mark.asyncio
async def test_run_adhoc_drain_job_skips_a_deopted_repo(make_ctx: CtxFactory) -> None:
    """A repo no longer opted in (no config) is a skip — the drain is never fired."""
    calls: list[str] = []

    async def drain(*, repo_full_name: str, config: RepoConfig) -> None:
        calls.append(repo_full_name)

    ctx = make_ctx(_RecordingPipeline())
    ctx["adhoc_drain"] = drain

    async def no_config(repo_full_name: str) -> str | None:
        return None

    ctx["fetch_config"] = no_config

    await run_adhoc_drain_job(ctx, repo_full_name="owner/repo")

    assert calls == []


def test_worker_registers_the_adhoc_drain_task() -> None:
    """WorkerSettings registers a function named ``run_adhoc_drain_job`` (the kick task).

    Mirrors the cron-job registration test: the webhook enqueues
    ``RUN_ADHOC_DRAIN_TASK`` and Arq dequeues by ``__name__``, so a function with that
    exact name must be in ``WorkerSettings.functions`` or the kick is dropped.
    """
    assert RUN_ADHOC_DRAIN_TASK in _registered_names()
    assert run_adhoc_drain_job.__name__ == RUN_ADHOC_DRAIN_TASK


def test_main_drives_job_timeout_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main`` overrides ``WorkerSettings.job_timeout`` from the configured setting.

    Arq reads ``job_timeout`` off the class before ``on_startup`` runs, so — like
    ``redis_settings`` — it is applied at process start. The arq default (300s) cancels a
    real claude build mid-implement; this override is what keeps the build alive.
    """

    class _FakeSettings:
        redis_url = "redis://localhost:6379"
        job_timeout_seconds = 1234

    monkeypatch.setattr(worker, "settings", _FakeSettings())
    monkeypatch.setattr("arq.worker.run_worker", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_configure_logging", lambda: None)

    worker.main()

    assert WorkerSettings.job_timeout == 1234


# --- parse_heimdall_review ------------------------------------------------------


def test_parse_heimdall_review_maps_state_and_findings() -> None:
    """The review parser maps gh state and reads severity:summary finding lines."""
    review = parse_heimdall_review(
        repo_full_name="owner/repo",
        pr_number=99,
        prd_number=7,
        review_state="approved",
        review_body="critical: data loss\nnot a finding line\nmedium: slow path",
    )
    assert review.state is ReviewState.APPROVED
    assert review.integration_branch == "retinue/prd-7"
    assert [(f.severity, f.summary) for f in review.findings] == [
        (Severity.CRITICAL, "data loss"),
        (Severity.MEDIUM, "slow path"),
    ]


def test_parse_heimdall_review_unknown_state_is_commented() -> None:
    """An unrecognised gh review state reads as a plain comment (no verdict)."""
    review = parse_heimdall_review(
        repo_full_name="owner/repo",
        pr_number=1,
        prd_number=2,
        review_state="dismissed",
        review_body="",
    )
    assert review.state is ReviewState.COMMENTED
    assert review.findings == []
    assert not review.carries_verdict


def test_parse_heimdall_review_clean_pass_body_carries_verdict() -> None:
    """Heimdall's clean pass — the COMMENTED "no concerns" body — reads as a verdict.

    Heimdall never submits APPROVED; its clean verdict is a COMMENT with the
    no-concerns body, so the parser must flag it verdict-carrying or the loopback
    would never converge a clean PR.
    """
    review = parse_heimdall_review(
        repo_full_name="owner/repo",
        pr_number=75,
        prd_number=51,
        review_state="commented",
        review_body="Heimdall review: no concerns found across any lens.",
    )
    assert review.state is ReviewState.COMMENTED
    assert review.findings == []
    assert review.clean_pass
    assert review.carries_verdict


def test_parse_heimdall_review_failed_note_carries_no_verdict() -> None:
    """Heimdall's "review failed" COMMENT note must not read as a verdict.

    Converging on it would hand off a PR heimdall never actually reviewed.
    """
    review = parse_heimdall_review(
        repo_full_name="owner/repo",
        pr_number=75,
        prd_number=51,
        review_state="commented",
        review_body=(
            "Heimdall review failed: the automated review could not complete after "
            "a retry. No verdict was produced for this commit."
        ),
    )
    assert review.state is ReviewState.COMMENTED
    assert not review.clean_pass
    assert not review.carries_verdict


def test_parse_heimdall_review_findings_carry_verdict() -> None:
    """A COMMENTED review with parsed finding lines is a (nits-only) verdict."""
    review = parse_heimdall_review(
        repo_full_name="owner/repo",
        pr_number=75,
        prd_number=51,
        review_state="commented",
        review_body="- low: rename this",
    )
    assert not review.clean_pass
    assert review.carries_verdict


def test_parse_heimdall_review_tolerates_markdown_finding_lines() -> None:
    """Finding lines survive common markdown dressing: bullets, bold, numbering.

    Heimdall writes prose reviews; a bullet like ``- **high**: broken auth`` must
    parse as a HIGH finding rather than silently reading as zero findings (which
    would previously converge a rejected PR).
    """
    body = (
        "- **high**: broken auth check\n"
        "* medium: slow query\n"
        "1. low: rename this\n"
        "2. **critical**: drops the table\n"
    )
    review = parse_heimdall_review(
        repo_full_name="owner/repo",
        pr_number=99,
        prd_number=7,
        review_state="changes_requested",
        review_body=body,
    )
    assert [(f.severity, f.summary) for f in review.findings] == [
        (Severity.HIGH, "broken auth check"),
        (Severity.MEDIUM, "slow query"),
        (Severity.LOW, "rename this"),
        (Severity.CRITICAL, "drops the table"),
    ]


# --- on_startup: the production wiring path -------------------------------------


class _FakeAuth:
    """A stand-in :class:`InstallationAuth` that mints a canned token without network.

    Also satisfies :class:`~retinue.github_app.InstalledRepos`: ``installed_repos`` is the
    fixed set the heartbeat enumerator lists (the App's installed repos), so the sweep is
    exercised without a live GitHub listing.
    """

    installed_repos: list[str] = []

    async def installation_token(self, repo_full_name: str) -> InstallationToken:
        return InstallationToken(token="ghs_x", clone_url="https://x/y.git")

    async def installed_repositories(self) -> list[str]:
        return list(self.installed_repos)


async def _fake_claude_md(repo_full_name: str) -> str:
    """Canned CLAUDE.md text standing in for the contents-API fetch (no network)."""
    return "## Definition of done\n```\nuv run pytest\n```\n"


def _worker_settings(tmp_path: Path) -> object:
    from retinue.config import Settings

    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        webhook_secret="s",
        dedupe_db_path=str(tmp_path / "dedupe.sqlite3"),
        budget_db_path=str(tmp_path / "budget.sqlite3"),
        weekly_budget=1000.0,
        ntfy_topic="alerts",
    )


@pytest.mark.asyncio
async def test_on_startup_wires_a_live_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """on_startup takes the auth branch and produces a pipeline with a live build lane.

    With GitHub App auth resolvable, on_startup must install the config fetcher, the
    PRD-body fetcher, and the pipeline_factory — and the factory must yield a Pipeline
    whose build lane is bound (not the dead ``build_prd is None`` of the unwired path).
    No network, Docker, or model: the auth is faked and adapter construction is pure.
    """
    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.setattr(github_app, "build_installation_auth", _FakeAuth)
    # The factory sources CLAUDE.md per repo over the contents API; stub that fetcher so
    # the wiring is exercised without a live GitHub read.
    monkeypatch.setattr(
        worker,
        "github_claude_md_fetcher",
        lambda auth, client: _fake_claude_md,
    )

    ctx: dict[str, Any] = {}
    try:
        await on_startup(ctx)

        # Took the auth branch: all three downstream seams are installed.
        assert ctx["github_client"] is not None
        assert callable(ctx["fetch_config"])
        assert callable(ctx["fetch_prd_body"])
        assert callable(ctx["pipeline_factory"])
        # The webhook's ad-hoc kick task reads ``adhoc_drain`` from ctx; a deployed
        # worker under live auth must have it bound so the kick actually drains.
        assert callable(ctx["adhoc_drain"])

        # The produced pipeline has a live build lane (the wiring blocker is closed).
        pipeline = await ctx["pipeline_factory"](
            "owner/repo", RepoConfig(staging_branch="staging", retry_cap=2)
        )
        assert pipeline.build_prd is not None
    finally:
        await on_shutdown(ctx)


@pytest.mark.asyncio
async def test_on_startup_kicks_the_resume_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under live auth, on_startup installs the run-state store and enqueues the sweep.

    The sweep runs as a normal Arq task (``RESUME_ROUNDS_TASK``) so a crash-resume never
    blocks worker startup; on_startup's job is to install ``ctx['run_state']`` (the store
    the sweep enumerates) and kick the job on the worker's own queue.
    """
    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.setattr(github_app, "build_installation_auth", _FakeAuth)
    monkeypatch.setattr(
        worker, "github_claude_md_fetcher", lambda auth, client: _fake_claude_md
    )

    redis = _RecordingRedis()
    ctx: dict[str, Any] = {"redis": redis}
    try:
        await on_startup(ctx)

        assert isinstance(ctx["run_state"], RunStateStore)
        assert [task for task, _ in redis.enqueued] == [RESUME_ROUNDS_TASK]
        # A pinned job id: with keep_result=0 a restart always re-kicks the sweep
        # instead of racing a lingering result key from the previous boot.
        assert redis.enqueued[0][1]["_job_id"] == "resume-rounds"
    finally:
        await on_shutdown(ctx)


@pytest.mark.asyncio
async def test_on_shutdown_closes_the_dedupe_store(tmp_path: Path) -> None:
    """on_shutdown releases the dedupe store's SQLite connection.

    The connection rides a non-daemon aiosqlite worker thread; a shutdown that skips
    the close strands that thread and blocks clean process exit.
    """
    store = PrdDedupeStore(tmp_path / "dedupe.sqlite3")
    await store.claim("owner/repo#1")
    assert store._db is not None

    await on_shutdown({"dedupe": store})

    assert store._db is None


@pytest.mark.asyncio
async def test_on_shutdown_without_startup_is_a_noop() -> None:
    """on_shutdown on a ctx on_startup never populated must not raise."""
    await on_shutdown({})


@pytest.mark.asyncio
async def test_on_startup_adhoc_drain_drives_one_issue_to_the_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound ``ctx['adhoc_drain']`` actually drains: one listed issue reaches the build.

    Extends the live-wiring path by *invoking* the bound drain (not merely asserting it is
    callable): a fake gh seam lists one ready issue, and the ad-hoc build (faked to avoid a
    container) drives the per-repo pipeline's ``process_adhoc_pr``. This proves
    ``_bind_adhoc_drain`` wires the per-repo lock registry + shared governor and threads the
    factory-built pipeline into ``bind_adhoc_build`` — the whole assembly runs end to end
    with no Docker, gh, model, or network.
    """
    import retinue.adhoc_drain as adhoc_drain_mod
    import retinue.pipeline as pipeline_mod
    from retinue.adhoc_build import AdhocBuildResult, AdhocIssue
    from retinue.adhoc_drain import FlightState, ReadyIssue

    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.setattr(github_app, "build_installation_auth", _FakeAuth)
    monkeypatch.setattr(
        worker, "github_claude_md_fetcher", lambda auth, client: _fake_claude_md
    )

    # Fake the gh seam the drain constructs: list one ready ad-hoc issue, none in flight.
    class _FakeGhCli:
        def __init__(self, *, token: str) -> None:
            self.token = token

        async def list_ready(self, *, repo_full_name: str) -> list[ReadyIssue]:
            return [ReadyIssue(number=31, labels=["ready-for-agent"], body="")]

        async def flight_state(
            self, *, repo_full_name: str, issue_number: int
        ) -> FlightState:
            return FlightState.ABSENT

    monkeypatch.setattr(adhoc_drain_mod, "GhCli", _FakeGhCli)

    # Fake the ad-hoc build so no container spawns, but drive the *real per-repo pipeline*
    # the factory built — proving bind_adhoc_build is handed that pipeline and its
    # process_adhoc_pr runs as the build's PR step.
    pr_calls: list[tuple[int, bool]] = []
    pipelines_seen: list[object] = []

    def _fake_bind_adhoc_build(settings: object, auth: object, **kwargs: object) -> object:
        pipeline = kwargs["pipeline"]
        pipelines_seen.append(pipeline)

        async def build(issue: AdhocIssue, *, repo_full_name: str) -> None:
            # A red result drives the *real* factory-built pipeline's process_adhoc_pr
            # without opening a network PR (a red build skips the PR step, returning None),
            # so the per-repo pipeline is exercised end to end with no gh/network.
            result = AdhocBuildResult(branch=issue.branch, passed=False)
            pr_result = await pipeline.process_adhoc_pr(issue, result)  # type: ignore[attr-defined]
            pr_calls.append((issue.issue_number, pr_result is None))

        return build

    monkeypatch.setattr(pipeline_mod, "bind_adhoc_build", _fake_bind_adhoc_build)

    ctx: dict[str, Any] = {}
    try:
        await on_startup(ctx)
        drain = ctx["adhoc_drain"]
        assert callable(drain)

        config = RepoConfig(staging_branch="staging", retry_cap=2)
        await drain(repo_full_name="owner/repo", config=config)

        # The listed issue (#31) drove the build, which ran the factory-built pipeline's
        # process_adhoc_pr (a red build skips, returning None) — the per-repo pipeline is
        # threaded through, the shared governor metered it (default budget has room), and
        # the per-repo lock serialized the run.
        assert pr_calls == [(31, True)]  # (issue_number, process_adhoc_pr returned None)
        assert pipelines_seen  # bind_adhoc_build was handed the per-repo pipeline
    finally:
        await on_shutdown(ctx)


@pytest.mark.asyncio
async def test_on_startup_without_auth_installs_no_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no GitHub App auth builder, on_startup falls back to the safe not-opted path."""
    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.delattr(github_app, "build_installation_auth", raising=False)

    ctx: dict[str, Any] = {}
    await on_startup(ctx)

    # No auth -> no pipeline; the fetcher is the not-opted-in fallback.
    assert "pipeline_factory" not in ctx
    assert await ctx["fetch_config"]("owner/repo") is None


@pytest.mark.asyncio
async def test_on_startup_with_unconfigured_auth_falls_back_to_not_opted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present-but-unconfigured auth builder must degrade, not crash the worker.

    Production wires a concrete ``build_installation_auth`` that raises
    ``InstallationAuthError`` when ``github_app_id``/key path are unset — a fresh deploy
    with no GitHub App registered yet. ``on_startup`` must catch that and install the
    safe not-opted-in fetcher so the worker boots and logs SKIPs (the graceful fallback
    DEPLOY.md promises), rather than the builder's exception killing worker startup.
    """
    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))

    def _raise_unconfigured() -> object:
        raise InstallationAuthError(
            "GitHub App auth is unconfigured: set github_app_id and "
            "github_app_private_key_path"
        )

    monkeypatch.setattr(github_app, "build_installation_auth", _raise_unconfigured)

    ctx: dict[str, Any] = {}
    await on_startup(ctx)

    assert "pipeline_factory" not in ctx
    assert await ctx["fetch_config"]("owner/repo") is None


# --- on_startup: the heartbeat collaborators (issue #43) ------------------------


def _fake_config_fetcher(auth: object, client: object) -> Any:
    """Stand in for the contents-API config fetcher: every repo is opted in (no network)."""

    async def fetch(repo_full_name: str) -> str | None:
        return _CONFIG_YAML

    return fetch


@pytest.mark.asyncio
async def test_on_startup_wires_the_heartbeat_collaborators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under live auth, on_startup installs all four heartbeat collaborators on ctx.

    The registered ``heartbeat_tick`` reads ``heartbeat_enumerate_repos``,
    ``heartbeat_clock``, ``heartbeat_drain``, and ``heartbeat_cron_tick`` from ctx; without
    these it no-ops every tick. The drain must be the *same* object the webhook kick fires
    (``ctx['adhoc_drain']``) so a kick and a sweep are one drain.
    """
    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.setattr(github_app, "build_installation_auth", _FakeAuth)
    monkeypatch.setattr(
        worker, "github_claude_md_fetcher", lambda auth, client: _fake_claude_md
    )

    ctx: dict[str, Any] = {}
    try:
        await on_startup(ctx)

        assert callable(ctx["heartbeat_enumerate_repos"])
        assert ctx["heartbeat_clock"].now() is not None  # the real wall-clock seam
        assert callable(ctx["heartbeat_cron_tick"])
        # The heartbeat sweep fires the SAME bound drain the webhook kick fires.
        assert ctx["heartbeat_drain"] is ctx["adhoc_drain"]
    finally:
        await on_shutdown(ctx)


@pytest.mark.asyncio
async def test_on_startup_without_auth_leaves_the_heartbeat_unwired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No auth -> no heartbeat collaborators, so the registered tick stays a safe no-op."""
    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.delattr(github_app, "build_installation_auth", raising=False)

    ctx: dict[str, Any] = {}
    await on_startup(ctx)

    assert "heartbeat_enumerate_repos" not in ctx
    assert "heartbeat_clock" not in ctx
    assert "heartbeat_drain" not in ctx
    assert "heartbeat_cron_tick" not in ctx


@pytest.mark.asyncio
async def test_on_startup_heartbeat_enumerate_yields_opted_in_due_repos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound enumerator lists the App's installed, opted-in repos as DueRepos."""
    from retinue.heartbeat import DueRepo

    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.setattr(github_app, "build_installation_auth", _FakeAuth)
    monkeypatch.setattr(
        worker, "github_claude_md_fetcher", lambda auth, client: _fake_claude_md
    )
    # The App is installed on these repos; the fetcher reports each as opted in.
    monkeypatch.setattr(_FakeAuth, "installed_repos", ["owner/a", "owner/b"])
    monkeypatch.setattr(worker, "github_config_fetcher", _fake_config_fetcher)

    ctx: dict[str, Any] = {}
    try:
        await on_startup(ctx)
        due = await ctx["heartbeat_enumerate_repos"]()
    finally:
        await on_shutdown(ctx)

    assert [r.repo_full_name for r in due] == ["owner/a", "owner/b"]
    assert all(isinstance(r, DueRepo) for r in due)
    assert all(r.config.staging_branch == "staging" for r in due)


@pytest.mark.asyncio
async def test_on_startup_heartbeat_tick_drives_run_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the collaborators wired, ``heartbeat_tick(ctx)`` drives a real sweep, not the skip.

    Proves criterion 2: one opted-in, cron-due repo is enumerated, its safety-net ad-hoc
    drain fires (one ready issue reaches the build), and its backlog cron lane ticks — the
    not-wired ``is None`` guard is never hit. The leaf gh/build seams are faked so no
    container, gh, model, or network runs.
    """
    import retinue.adhoc_drain as adhoc_drain_mod
    import retinue.cron as cron_mod
    import retinue.pipeline as pipeline_mod
    from retinue.adhoc_build import AdhocIssue
    from retinue.adhoc_drain import FlightState, ReadyIssue
    from retinue.heartbeat import heartbeat_tick

    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.setattr(github_app, "build_installation_auth", _FakeAuth)
    monkeypatch.setattr(
        worker, "github_claude_md_fetcher", lambda auth, client: _fake_claude_md
    )
    # One installed repo whose cron is due on every tick, opted in via the fake fetcher.
    cron_yaml = "staging_branch: staging\ncron: '* * * * *'\n"

    def _due_config_fetcher(auth: object, client: object) -> Any:
        async def fetch(repo_full_name: str) -> str | None:
            return cron_yaml

        return fetch

    monkeypatch.setattr(_FakeAuth, "installed_repos", ["owner/due"])
    monkeypatch.setattr(worker, "github_config_fetcher", _due_config_fetcher)

    # Fake the ad-hoc drain's gh seam: one ready issue, none in flight.
    class _FakeAdhocGh:
        def __init__(self, *, token: str) -> None:
            self.token = token

        async def list_ready(self, *, repo_full_name: str) -> list[ReadyIssue]:
            return [ReadyIssue(number=31, labels=["ready-for-agent"], body="")]

        async def flight_state(
            self, *, repo_full_name: str, issue_number: int
        ) -> FlightState:
            return FlightState.ABSENT

    monkeypatch.setattr(adhoc_drain_mod, "GhCli", _FakeAdhocGh)

    drained: list[int] = []

    def _fake_bind_adhoc_build(settings: object, auth: object, **kwargs: object) -> object:
        async def build(issue: AdhocIssue, *, repo_full_name: str) -> None:
            drained.append(issue.issue_number)

        return build

    monkeypatch.setattr(pipeline_mod, "bind_adhoc_build", _fake_bind_adhoc_build)

    # Fake the cron backlog gh seam (empty backlog) so the cron lane ticks idle, no build.
    class _FakeCronGh:
        def __init__(self, *, token: str) -> None:
            self.token = token

        async def list_backlog(self, *, repo_full_name: str) -> list[object]:
            return []

    monkeypatch.setattr(cron_mod, "GhCli", _FakeCronGh)

    ctx: dict[str, Any] = {}
    try:
        await on_startup(ctx)
        await heartbeat_tick(ctx)
    finally:
        await on_shutdown(ctx)

    # The due repo's safety-net drain fired (issue #31 reached the build) — not the no-op.
    assert drained == [31]
    # The tick counter advanced, proving run_heartbeat ran rather than the not-wired skip.
    assert ctx["heartbeat_tick_number"] == 1
