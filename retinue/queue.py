"""Arq queue helpers: the merge-reap and scheduler-drain job models and enqueues.

Each job carries just enough to identify the work the worker should do: the repository
(and, for a merge, the PR number).
"""

from __future__ import annotations

from dataclasses import dataclass

from arq import ArqRedis
from arq.constants import result_key_prefix

# The Arq task names the worker registers; the enqueue side and the worker must agree on
# these strings, so they live here as the single source of truth.
REAP_PR_TASK = "reap_pr_job"
RUN_ADHOC_DRAIN_TASK = "run_adhoc_drain_job"


@dataclass(frozen=True)
class MergedPrJob:
    """A ``pull_request`` closed+merged event routed to the merge reap.

    Attributes:
        repo_full_name: e.g. "owner/repo".
        pr_number: The merged pull request number.
    """

    repo_full_name: str
    pr_number: int


@dataclass(frozen=True)
class AdhocDrainJob:
    """A low-latency kick that runs one ad-hoc drain for a repo.

    The webhook enqueues this when a ``ready-for-agent`` (non-``prd``) issue event
    arrives, so the repo's ad-hoc backlog is drained without waiting for the cron tick.
    The job carries only the repo because :func:`retinue.adhoc_drain.run_adhoc_drain`
    re-lists the repo's ready-for-agent issues itself — the event is a *kick*, not a
    per-issue task, so a burst of events for one repo collapses to a single drain.

    Attributes:
        repo_full_name: e.g. "owner/repo".
    """

    repo_full_name: str


async def enqueue_merged_pr(pool: ArqRedis, job: MergedPrJob) -> str:
    """Enqueue a merge-reap job onto the Arq queue.

    The :class:`MergedPrJob` fields ride as keyword arguments so the worker receives them
    by name.

    Returns:
        The Arq job ID, or an empty string when Arq deduplicated the submission.
    """
    arq_job = await pool.enqueue_job(
        REAP_PR_TASK,
        repo_full_name=job.repo_full_name,
        pr_number=job.pr_number,
    )
    return arq_job.job_id if arq_job is not None else ""


def _adhoc_drain_job_id(repo_full_name: str) -> str:
    """The per-repo Arq job id that coalesces a burst of drain kicks into one.

    Arq dedups on the ``_job_id`` of an in-flight job, so keying it on the repo makes
    a flurry of ``ready-for-agent`` events for the same repo enqueue *at most one*
    ad-hoc drain at a time — the low-latency kick stays single-flight per repo, exactly
    as the drain itself runs at most once per repo (:func:`retinue.adhoc_drain.run_adhoc_drain`).
    """
    return f"adhoc-drain:{repo_full_name}"


async def enqueue_adhoc_drain(pool: ArqRedis, job: AdhocDrainJob) -> str:
    """Enqueue a single ad-hoc drain kick for a repo onto the Arq queue.

    Pins a per-repo ``_job_id`` so concurrent kicks for the same repo collapse to one
    in-flight drain (Arq dedups on the id).

    Arq's enqueue dedups on both the queued/running job key *and* the completed-job
    *result* key, which lingers for the worker's ``keep_result`` window (1h by default).
    Without intervention a kick arriving in that post-completion window is silently
    dropped, so a repo could wait up to an hour for its next drain. We delete any stale
    result key for this job id first: while a drain is actually queued or running the
    result key does not yet exist (the live job key still coalesces the kick), so this
    narrows coalescing to the genuinely-in-flight window without letting a burst enqueue
    duplicate drains.

    Args:
        pool: The connected Arq Redis pool.
        job: The ad-hoc drain kick to enqueue.

    Returns:
        The Arq job ID, or an empty string when Arq coalesced this kick against an
        already-queued drain for the same repo.
    """
    job_id = _adhoc_drain_job_id(job.repo_full_name)
    await pool.delete(result_key_prefix + job_id)
    arq_job = await pool.enqueue_job(
        RUN_ADHOC_DRAIN_TASK,
        repo_full_name=job.repo_full_name,
        _job_id=job_id,
    )
    return arq_job.job_id if arq_job is not None else ""
