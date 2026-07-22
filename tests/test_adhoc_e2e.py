"""End-to-end ad-hoc lane test (issue #36 AC1): kick -> drain -> build -> PR -> reap.

Drives a single ``ready-for-agent`` issue through the surviving ad-hoc lane with only the
leaf I/O faked — the ``gh`` queries (via ``FakeAdhocGh``), Redis (via ``fakeredis``), and
the per-issue build primitive. Every *decision and dispatch* function runs for real: the
webhook kick over the real Arq spine, ``run_adhoc_drain`` (admit / gate on readiness /
classify flight-state / rank / dedup / lock / budget), the real ``Pipeline``
``process_adhoc_pr`` / ``reap_pr`` shared decision code, and the
``AdhocIssue.from_fetched_issue`` chain-depth materialization the #39/#40 review-fix bound
rides on (done inside the drain, before the build primitive is dispatched).

The heimdall review loopback -> handoff convergence stage was deleted wholesale with the
PRD build lane (PRD #80): the ad-hoc lane opens an ``issue-<N>`` -> target-branch PR that a
human merges directly, and the merge webhook reaps it — there is no review round in between.

The build primitive (``build_adhoc_issue``) is faked to a green result rather than run over
fake container/Agent-SDK leaves: its own real coverage lives in ``tests/test_adhoc_build.py``,
and faking it at the dispatch boundary keeps this e2e focused on the drain -> PR-open ->
reap wiring that survives the deletion.

The shared fakes are imported from ``tests.fakes``.
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
    AdhocBuildResult,
    AdhocIssue,
    ReviewGateOutcome,
    render_chain_depth,
)
from retinue.adhoc_drain import AdhocDrainLock, ReadyIssue, run_adhoc_drain
from retinue.handoff import MergedPullRequest, ReapOutcome
from retinue.pipeline import Pipeline, bind_adhoc_pr_open
from retinue.queue import AdhocDrainJob, enqueue_adhoc_drain
from retinue.repo_config import RepoConfig
from retinue.run_ledger import RunLedgerStore
from retinue.worker import run_adhoc_drain_job
from tests.fakes import (
    CLAUDE_MD,
    FakeAdhocGh,
    _created,
    _FakePrOps,
    _FakeReapGh,
    _governor,
    _RecordingNotifier,
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


@pytest.mark.asyncio
async def test_adhoc_lane_runs_kick_to_reap_end_to_end(
    arq_pool: ArqRedis, tmp_path: Path
) -> None:
    """One ready issue flows kick -> classify -> drain -> build -> PR -> reap.

    Only the leaves are faked; every decision/dispatch runs for real. The drain materializes
    the issue through ``from_fetched_issue`` and dispatches a (faked green) build; the real
    ``Pipeline`` then opens the ``issue-31`` -> staging PR, records the PR<->issue mapping,
    and reaps the single ad-hoc issue on the simulated human merge.
    """
    config = RepoConfig(target_branch="staging")
    governor = _governor(tmp_path)  # one shared service-level governor

    pr_ops = _FakePrOps()
    reap_gh = _FakeReapGh(children=[])
    # The cross-process run-ledger file: one store instance shared by the pipeline (writes
    # its terminal states) and the drain (writes queued/building), mirroring the one file
    # the worker and web processes share in production.
    run_ledger = RunLedgerStore(tmp_path / "run-ledger.sqlite3")
    pipeline = Pipeline(
        config=config,
        claude_md=CLAUDE_MD,
        governor=governor,
        notifier=_RecordingNotifier(),  # type: ignore[arg-type]  # recording fake
        create_issue=_created,
        pr_ops=pr_ops,
        reap_gh=reap_gh,
        retry_store_path=tmp_path / "retries.sqlite3",
        run_state_path=tmp_path / "runstate.sqlite3",
        run_ledger=run_ledger,
    )

    built_issues: list[AdhocIssue] = []

    async def adhoc_build(issue: AdhocIssue, *, repo_full_name: str) -> None:
        # The drain's downstream: the build primitive (faked green here — its real coverage
        # lives in tests/test_adhoc_build.py) followed by the *real* pipeline PR step, exactly
        # what ``bind_adhoc_build`` chains but with the leaf build injected.
        built_issues.append(issue)
        # A green build carries a review-gate outcome (clean here), exactly as production
        # does; the pipeline consumes it and — nothing blocking, nothing backlog — opens
        # the PR. The gate's own build-time coverage lives in tests/test_adhoc_build.py.
        result = AdhocBuildResult(
            branch=issue.branch,
            passed=True,
            gate=ReviewGateOutcome(blocking=[], backlog=[]),
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
            readiness_gh=gh,
            build=adhoc_build,
            open_pr=bind_adhoc_pr_open(pipeline),
            config=config,
            governor=governor,
            ledger=run_ledger,
            estimated_amount=1.0,
            lock=lock,
        )

    seen_ctx: dict[str, object] = {}

    async def fetch_config(repo_full_name: str) -> str:
        return "target_branch: staging\n"

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
    # back from the body (1), not defaulted — and dispatched on the ``issue-31`` branch.
    assert built_issues == [
        AdhocIssue(repo_full_name="owner/repo", issue_number=31, chain_depth=1)
    ]

    # One PR opened ``issue-31`` -> staging: the head is the issue branch itself (no
    # integration branch), and the PR<->issue mapping is recorded for the reap.
    assert len(pr_ops.opened) == 1
    assert pr_ops.opened[0].head == "issue-31"
    assert pr_ops.opened[0].base == "staging"
    mapping = await pipeline.round_for_pr(repo_full_name="owner/repo", pr_number=99)
    assert mapping == (31, [])  # the single ad-hoc issue, no PRD parent, no slices

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

    # The end-to-end run-ledger trail on the one shared file: queued/building (the drain),
    # then pr_opened (the pipeline's PR step), and finally merged (the reap) — the terminal
    # state overwrites pr_opened, so the ledger's one current row reads merged.
    ledger_rows = await run_ledger.rows()
    assert len(ledger_rows) == 1
    assert ledger_rows[0].issue == 31
    assert ledger_rows[0].state == "merged"
