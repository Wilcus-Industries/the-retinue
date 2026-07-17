"""Tests for the Arq queue module: enqueue pushes the right task and kwargs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from retinue.queue import (
    REAP_PR_TASK,
    RUN_ADHOC_DRAIN_TASK,
    AdhocDrainJob,
    MergedPrJob,
    enqueue_adhoc_drain,
    enqueue_merged_pr,
)


@pytest.mark.asyncio
async def test_enqueue_merged_pr_calls_arq() -> None:
    """enqueue_merged_pr pushes the reap task with the repo + PR number."""
    mock_pool = AsyncMock()
    mock_pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="jid-m"))
    job = MergedPrJob(repo_full_name="owner/repo", pr_number=42)

    assert await enqueue_merged_pr(mock_pool, job) == "jid-m"
    call_args = mock_pool.enqueue_job.call_args
    assert call_args[0][0] == REAP_PR_TASK
    assert call_args[1]["repo_full_name"] == "owner/repo"
    assert call_args[1]["pr_number"] == 42


@pytest.mark.asyncio
async def test_enqueue_adhoc_drain_calls_arq() -> None:
    """enqueue_adhoc_drain pushes the drain task with the repo and a per-repo job id."""
    mock_pool = AsyncMock()
    mock_pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="jid-d"))
    job = AdhocDrainJob(repo_full_name="owner/repo")

    assert await enqueue_adhoc_drain(mock_pool, job) == "jid-d"
    call_args = mock_pool.enqueue_job.call_args
    assert call_args[0][0] == RUN_ADHOC_DRAIN_TASK
    assert call_args[1]["repo_full_name"] == "owner/repo"
    # A burst of ready-for-agent events for one repo must collapse to a single
    # in-flight drain, so the job id is keyed on the repo (Arq dedups on it).
    assert call_args[1]["_job_id"] == "adhoc-drain:owner/repo"


@pytest.mark.asyncio
async def test_enqueue_adhoc_drain_deduplicated_returns_empty() -> None:
    """A coalesced drain enqueue (enqueue_job None) yields an empty id."""
    mock_pool = AsyncMock()
    mock_pool.enqueue_job = AsyncMock(return_value=None)
    assert await enqueue_adhoc_drain(mock_pool, AdhocDrainJob("owner/repo")) == ""


@pytest.mark.asyncio
async def test_enqueue_adhoc_drain_clears_stale_result_before_enqueue() -> None:
    """A kick clears any completed drain's result key so it coalesces only in-flight.

    Arq's enqueue dedups on both the queued/running job key AND the *result* key, which
    lingers for ``keep_result`` seconds after a drain finishes — so without clearing it
    a post-completion kick is silently dropped for up to an hour. The kick deletes the
    stale result key first (a no-op while queued/running, where the job key still
    coalesces), so coalescing is bounded to the actually-in-flight window.
    """
    parent = MagicMock()
    parent.delete = AsyncMock()
    parent.enqueue_job = AsyncMock(return_value=MagicMock(job_id="jid-d"))

    assert await enqueue_adhoc_drain(parent, AdhocDrainJob("owner/repo")) == "jid-d"

    parent.delete.assert_awaited_once_with("arq:result:adhoc-drain:owner/repo")
    # The result key must be cleared BEFORE the enqueue, never after.
    assert [c[0] for c in parent.mock_calls].index("delete") < [
        c[0] for c in parent.mock_calls
    ].index("enqueue_job")
