"""Tests for the Arq queue module: enqueue pushes the right task and kwargs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from retinue.queue import PROCESS_PRD_TASK, PrdJob, enqueue_prd


@pytest.fixture()
def job() -> PrdJob:
    return PrdJob(repo_full_name="owner/repo", issue_number=7, action="opened")


@pytest.mark.asyncio
async def test_enqueue_prd_calls_arq(job: PrdJob) -> None:
    """enqueue_prd pushes a job with the process_prd task name and PRD kwargs."""
    mock_pool = AsyncMock()
    mock_pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="jid-1"))

    job_id = await enqueue_prd(mock_pool, job)

    assert job_id == "jid-1"
    mock_pool.enqueue_job.assert_awaited_once()
    call_args = mock_pool.enqueue_job.call_args
    assert call_args[0][0] == PROCESS_PRD_TASK
    assert call_args[1]["repo_full_name"] == job.repo_full_name
    assert call_args[1]["issue_number"] == job.issue_number
    assert call_args[1]["action"] == job.action


@pytest.mark.asyncio
async def test_enqueue_prd_deduplicated_returns_empty(job: PrdJob) -> None:
    """When Arq deduplicates the job (enqueue_job returns None), the id is empty."""
    mock_pool = AsyncMock()
    mock_pool.enqueue_job = AsyncMock(return_value=None)

    assert await enqueue_prd(mock_pool, job) == ""
