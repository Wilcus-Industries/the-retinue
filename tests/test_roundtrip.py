"""End-to-end enqueue -> dequeue round-trip over a faked Redis.

Proves the transport spine: a kick enqueued through :func:`enqueue_adhoc_drain` is picked
up by a real Arq worker running :func:`run_adhoc_drain_job`, with the repo carried across
the queue intact. Redis is faked with ``fakeredis`` so the test is hermetic; Arq's enqueue,
queue storage, dequeue, and deserialization are all exercised for real.
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

from retinue.queue import AdhocDrainJob, enqueue_adhoc_drain
from retinue.worker import run_adhoc_drain_job


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
    """A kick enqueued via enqueue_adhoc_drain is dequeued and run by run_adhoc_drain_job.

    With no drain wired into the bare worker ctx the task logs which repo it received;
    asserting on that log line proves the enqueue -> dequeue round-trip carried the repo
    intact.
    """
    job = AdhocDrainJob(repo_full_name="owner/repo")
    job_id = await enqueue_adhoc_drain(arq_pool, job)
    assert job_id  # a real job id was assigned

    worker = Worker(
        functions=[run_adhoc_drain_job],
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
    # The dequeued task logged the repo it received.
    assert "No ad-hoc drain wired; dropping kick for owner/repo" in caplog.text
