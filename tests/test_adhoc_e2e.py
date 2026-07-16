"""End-to-end ad-hoc lane test (issue #36 AC1): kick -> drain -> build -> PR -> handoff -> reap.

Drives a single ``ready-for-agent`` issue through the *whole* ad-hoc lane with only the leaf
I/O faked — the Docker container exec, the in-container ``claude`` CLI exec, the Messages-API
POST, the ``gh`` queries, and Redis (via ``fakeredis``). Every decision and dispatch function
runs for real: the webhook kick over the real Arq spine, ``run_adhoc_drain`` (classify / rank /
dedup / lock / budget), ``build_adhoc_issue`` over the *real* ``ContainerPlanner`` /
``ContainerImplementer`` / ``ContainerAdhocReviewer`` adapters, the real ``Pipeline``
``process_adhoc_pr`` / ``process_review`` / ``reap_pr`` shared decision code, and the
``AdhocIssue.from_fetched_issue`` chain-depth materialization the #39/#40 review-fix bound
rides on.

The per-file fakes are reused by import (the repo's established convention — no shared
conftest/mock module): the container/auth/secret/report leaves from ``tests.test_done_check``,
the gh drain seam from ``tests.test_adhoc_drain``, and the pipeline collaborators from
``tests.test_pipeline``.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import fakeredis
import pytest
import pytest_asyncio
from arq import ArqRedis
from arq.worker import Worker

from retinue.adhoc_build import (
    AdhocIssue,
    ContainerAdhocReviewer,
    ContainerPlanner,
    build_adhoc_issue,
    render_chain_depth,
)
from retinue.adhoc_drain import AdhocDrainLock, ReadyIssue, run_adhoc_drain
from retinue.done_check import DoneCheckReport
from retinue.handoff import MergedPullRequest, ReapOutcome
from retinue.loopback import HeimdallReview, ReviewState, VerdictOutcome
from retinue.messages_api import HttpResponse
from retinue.orchestrator import ContainerImplementer
from retinue.pipeline import Pipeline, bind_adhoc_pr_open
from retinue.queue import AdhocDrainJob, enqueue_adhoc_drain
from retinue.repo_config import RepoConfig
from retinue.reviewer import AgentSdkReviewGenerator
from retinue.slicer import SlicePlan
from retinue.worker import run_adhoc_drain_job
from tests.fakes import (
    CLAUDE_MD,
    FakeAdhocGh,
    FakeAuth,
    FakeRuntime,
    _created,
    _FakePrOps,
    _FakeReapGh,
    _governor,
    _noop_rebuild,
    _RecordingNotifier,
    _resolver,
    _sink,
)


@pytest_asyncio.fixture()
async def arq_pool() -> AsyncIterator[ArqRedis]:
    """An ArqRedis backed by an isolated in-process fakeredis server (the real spine)."""
    server = fakeredis.FakeServer()
    fake = fakeredis.FakeAsyncRedis(server=server)
    pool = ArqRedis(pool_or_conn=fake.connection_pool)
    try:
        yield pool
    finally:
        # Idempotent: the worker may have already closed the shared pool.
        with contextlib.suppress(Exception):
            await pool.aclose()


class _CleanReviewTransport:
    """Fake Messages-API transport: the advisory reviewer always sees a clean diff.

    Returns the one 200 response the real :meth:`AgentSdkReviewGenerator._parse` reads as an
    empty :class:`~retinue.reviewer.ReviewPlan` — so the real review pass runs end to end and
    files no review-fix follow-up.
    """

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any]
    ) -> HttpResponse:
        return HttpResponse(
            status_code=200,
            body={"content": [{"type": "text", "text": '{"findings": []}'}]},
        )


async def _no_slices(prd_body: str) -> SlicePlan:
    """Slicer stub: the ad-hoc lane never slices, so this is wired but never called."""
    return SlicePlan(slices=[])


@pytest.mark.asyncio
async def test_adhoc_lane_runs_kick_to_reap_end_to_end(
    arq_pool: ArqRedis, tmp_path: Path
) -> None:
    """One ready issue flows kick -> classify -> drain -> build -> PR -> handoff -> reap.

    Only the leaves are faked; every decision/dispatch runs for real. The fake container exits
    0 for every command, so the planner captures a plan, the implementer commits, the
    done-check is green and the ``issue-31`` branch is "pushed"; the faked Messages-API POST
    yields a clean review; and the real ``Pipeline`` opens the PR, converges the shared
    loopback into a handoff, and reaps the single ad-hoc issue on the simulated merge.
    """
    config = RepoConfig(staging_branch="staging", retry_cap=2)
    governor = _governor(tmp_path)  # one shared service-level governor (PRD lane + ad-hoc)

    handed_off: list[int] = []

    async def handoff(*, repo_full_name: str, pr_number: int) -> None:
        handed_off.append(pr_number)

    pr_ops = _FakePrOps()
    reap_gh = _FakeReapGh(children=[])
    pipeline = Pipeline(
        config=config,
        claude_md=CLAUDE_MD,
        governor=governor,
        notifier=_RecordingNotifier(),  # type: ignore[arg-type]  # recording fake
        create_issue=_created,
        slice_generate=_no_slices,
        pr_ops=pr_ops,
        reap_gh=reap_gh,
        round_store_path=tmp_path / "rounds.sqlite3",
        retry_store_path=tmp_path / "retries.sqlite3",
        run_state_path=tmp_path / "runstate.sqlite3",
        handoff=handoff,
        rebuild=_noop_rebuild,  # resolved eagerly by process_review even on the converge path
    )

    # Real role adapters over fake leaves: the planner/implementer exec the fake container
    # (every command exits 0, so the done-check is green), and the reviewer's only network
    # edge — the Messages-API POST — is faked to a clean (no-findings) response.
    planner = ContainerPlanner(credential="k")
    implementer = ContainerImplementer(credential="k")
    reviewer = ContainerAdhocReviewer(
        repo_full_name="owner/repo",
        config=config,
        generate=AgentSdkReviewGenerator(
            credential="k", transport=_CleanReviewTransport()
        ),
        create_issue=pipeline.create_issue,
        credential="k",
    )
    auth = FakeAuth()
    runtime = FakeRuntime()
    reports: list[DoneCheckReport] = []
    report = _sink(reports)
    resolve_secret = _resolver({})

    built_issues: list[AdhocIssue] = []

    async def adhoc_build(issue: AdhocIssue, *, repo_full_name: str) -> None:
        # The drain's downstream: the real build primitive over the leaf fakes, then the real
        # pipeline PR step — exactly what ``bind_adhoc_build`` chains, but injectable here.
        built_issues.append(issue)
        result = await build_adhoc_issue(
            issue,
            config,
            CLAUDE_MD,
            planner=planner,
            implementer=implementer,
            auth=auth,
            runtime=runtime,
            resolve_secret=resolve_secret,
            report=report,
            reviewer=reviewer,
        )
        await pipeline.process_adhoc_pr(issue, result)

    # A ``Chain-depth: 1`` marker in the body proves the drain materializes the issue through
    # ``from_fetched_issue`` (the #40 gotcha): a bare ``AdhocIssue(...)`` would default to 0.
    gh = FakeAdhocGh(
        [
            ReadyIssue(
                number=31,
                labels=["ready-for-agent"],
                body=f"Fix the flaky test.\n\n{render_chain_depth(1)}",
            )
        ]
    )
    lock = AdhocDrainLock()

    async def bound_drain(*, repo_full_name: str, config: RepoConfig) -> None:
        # The heartbeat-drain shape ``wiring.bind_adhoc_drain`` produces, with the leaf
        # fakes injected at run_adhoc_drain's seams instead of the production adapters
        # the merged bind constructs itself.
        await run_adhoc_drain(
            repo_full_name=repo_full_name,
            gh=gh,
            build=adhoc_build,
            open_pr=bind_adhoc_pr_open(pipeline),
            config=config,
            governor=governor,
            estimated_amount=1.0,
            lock=lock,
        )

    seen_ctx: dict[str, object] = {}

    async def fetch_config(repo_full_name: str) -> str:
        return "staging_branch: staging\n"

    async def on_startup(ctx: dict[str, Any]) -> None:
        # Mirror worker.on_startup: the webhook kick and the heartbeat sweep fire the SAME
        # bound drain (the #43 single-lock invariant the e2e must not silently break).
        ctx["adhoc_drain"] = bound_drain
        ctx["heartbeat_drain"] = bound_drain
        ctx["fetch_config"] = fetch_config
        seen_ctx.update(ctx)

    # --- kick: the real Arq spine drives run_adhoc_drain_job over fakeredis ---------------
    await enqueue_adhoc_drain(arq_pool, AdhocDrainJob(repo_full_name="owner/repo"))
    worker = Worker(
        functions=[run_adhoc_drain_job],
        redis_pool=arq_pool,
        burst=True,
        poll_delay=0.01,
        handle_signals=False,
        on_startup=on_startup,
    )
    # log_redis_info issues a Server command in a pipeline fakeredis doesn't support; it only
    # prints a startup banner and has no functional role (mirrors tests/test_roundtrip.py).
    with patch("arq.worker.log_redis_info", new=AsyncMock()):
        await worker.main()
    await worker.close()

    assert worker.jobs_complete == 1
    assert worker.jobs_failed == 0

    # The issue was materialized through ``from_fetched_issue`` — its chain_depth was read
    # back from the body (1), not defaulted — and built on the ``issue-31`` branch.
    assert built_issues == [
        AdhocIssue(repo_full_name="owner/repo", issue_number=31, chain_depth=1)
    ]
    assert reports[0].passed is True  # the green done-check pushed the branch

    # One PR opened ``issue-31`` -> staging: the head is the issue branch itself (no
    # integration branch), and the PR<->issue mapping is recorded for the loopback and reap.
    assert len(pr_ops.opened) == 1
    assert pr_ops.opened[0].head == "issue-31"
    assert pr_ops.opened[0].base == "staging"
    mapping = await pipeline.round_for_pr(repo_full_name="owner/repo", pr_number=99)
    assert mapping == (31, [])  # the single ad-hoc issue, no PRD parent, no slices

    # --- shared loopback -> handoff: a clean heimdall review converges on the issue-31 PR ---
    review = HeimdallReview(
        repo_full_name="owner/repo",
        pr_number=99,
        prd_number=31,
        prd_issue_number=31,
        integration_branch="issue-31",
        state=ReviewState.APPROVED,
        findings=[],
    )
    verdict = await pipeline.process_review(review)
    assert verdict.outcome is VerdictOutcome.CONVERGED
    assert handed_off == [99]

    # --- reap: the simulated human merge closes the single ad-hoc issue --------------------
    assert mapping is not None
    reap = await pipeline.reap_pr(
        MergedPullRequest(
            repo_full_name="owner/repo",
            pr_number=99,
            prd_number=mapping[0],
            slice_issues=mapping[1],
        )
    )
    assert reap.outcome is ReapOutcome.REAPED
    assert reap_gh.closed == [31]  # the single ad-hoc issue, no PRD parent to reap

    # --- drain-identity guard: the kick and heartbeat sweep fire the SAME bound drain (#43) -
    assert seen_ctx["heartbeat_drain"] is seen_ctx["adhoc_drain"]
