"""Arq queue helpers: the PRD job model and its enqueue function.

The job carries just enough to identify the issue the worker should act on:
the repository, the issue number, and the action that triggered the delivery.
"""

from __future__ import annotations

from dataclasses import dataclass

from arq import ArqRedis

# The Arq task name the worker registers; the enqueue side and the worker must
# agree on this string, so it lives here as the single source of truth.
PROCESS_PRD_TASK = "process_prd"


@dataclass(frozen=True)
class PrdJob:
    """Serialisable description of a single PRD-processing task.

    Attributes:
        repo_full_name: e.g. "owner/repo".
        issue_number: The GitHub issue number.
        action: The webhook action that triggered the job (e.g. "opened").
    """

    repo_full_name: str
    issue_number: int
    action: str


async def enqueue_prd(pool: ArqRedis, job: PrdJob) -> str:
    """Enqueue a PRD-processing job onto the Arq queue.

    Passes the :class:`PrdJob` fields as keyword arguments so the worker receives
    them by name.

    Args:
        pool: The connected Arq Redis pool.
        job: The PRD job to enqueue.

    Returns:
        The Arq job ID of the newly enqueued job (empty string if Arq
        deduplicated it against an identical in-flight job ID).
    """
    arq_job = await pool.enqueue_job(
        PROCESS_PRD_TASK,
        repo_full_name=job.repo_full_name,
        issue_number=job.issue_number,
        action=job.action,
    )
    # enqueue_job returns None when the job_id already exists (idempotent re-submit).
    return arq_job.job_id if arq_job is not None else ""
