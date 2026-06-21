"""End-to-end enqueue -> dequeue round-trip over a faked Redis.

Proves the transport spine: a job enqueued through :func:`enqueue_prd` is picked
up by a real Arq worker running :func:`process_prd`, with the repo, issue number,
and action carried across the queue intact. Redis is faked with ``fakeredis`` so
the test is hermetic; Arq's enqueue, queue storage, dequeue, and deserialization
are all exercised for real.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import fakeredis
import pytest
import pytest_asyncio
from arq import ArqRedis
from arq.worker import Worker

from retinue.queue import PrdJob, enqueue_prd
from retinue.worker import process_prd


@pytest_asyncio.fixture()
async def arq_pool() -> AsyncIterator[ArqRedis]:
    """An ArqRedis backed by an isolated in-process fakeredis server."""
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
async def test_enqueue_dequeue_roundtrip(
    arq_pool: ArqRedis, caplog: pytest.LogCaptureFixture
) -> None:
    """A job enqueued via enqueue_prd is dequeued and run by the real process_prd.

    The real task logs the repo, issue number, and action; asserting on that log
    line proves the enqueue -> dequeue round-trip carried the PRD fields intact.
    """
    job = PrdJob(repo_full_name="owner/repo", issue_number=42, action="opened")
    job_id = await enqueue_prd(arq_pool, job)
    assert job_id  # a real job id was assigned

    worker = Worker(
        functions=[process_prd],
        redis_pool=arq_pool,
        burst=True,
        poll_delay=0.01,
        handle_signals=False,
    )
    # log_redis_info issues an INFO Server command in a pipeline that fakeredis
    # doesn't support; it only prints a startup banner and has no functional role.
    with (
        patch("arq.worker.log_redis_info", new=AsyncMock()),
        caplog.at_level(logging.INFO, logger="retinue.worker"),
    ):
        await worker.main()
    await worker.close()

    # Exactly one job ran to completion over the round-trip.
    assert worker.jobs_complete == 1
    assert worker.jobs_failed == 0
    # The dequeued task logged the repo, issue number, and action it received.
    assert "Processing PRD for owner/repo#42 action=opened" in caplog.text
