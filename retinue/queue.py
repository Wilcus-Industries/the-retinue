"""Arq queue helpers: the PRD job model and its enqueue function.

The job carries just enough to identify the issue the worker should act on:
the repository, the issue number, and the action that triggered the delivery.
"""

from __future__ import annotations

from dataclasses import dataclass

from arq import ArqRedis

# The Arq task names the worker registers; the enqueue side and the worker must agree on
# these strings, so they live here as the single source of truth.
PROCESS_PRD_TASK = "process_prd"
PROCESS_REVIEW_TASK = "process_review_job"
REAP_PR_TASK = "reap_pr_job"
RUN_ADHOC_DRAIN_TASK = "run_adhoc_drain_job"
RESUME_ROUNDS_TASK = "resume_rounds_job"


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


@dataclass(frozen=True)
class ReviewJob:
    """A ``pull_request_review`` event routed to the heimdall loopback.

    Carries the minimal routing identity GitHub hands us on a review; the worker task
    rebuilds the full :class:`retinue.loopback.HeimdallReview` (parsing the review body
    into findings) before driving :func:`retinue.loopback.process_review`.

    Attributes:
        repo_full_name: e.g. "owner/repo".
        pr_number: The reviewed pull request number.
        review_state: The gh review state (``approved`` / ``commented`` /
            ``changes_requested``).
        review_body: The review body the worker parses heimdall findings out of.
    """

    repo_full_name: str
    pr_number: int
    review_state: str
    review_body: str = ""


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


async def enqueue_review(pool: ArqRedis, job: ReviewJob) -> str:
    """Enqueue a heimdall-review-loopback job onto the Arq queue.

    Mirrors :func:`enqueue_prd`: the :class:`ReviewJob` fields ride as keyword
    arguments so the worker receives them by name.

    Returns:
        The Arq job ID, or an empty string when Arq deduplicated the submission.
    """
    arq_job = await pool.enqueue_job(
        PROCESS_REVIEW_TASK,
        repo_full_name=job.repo_full_name,
        pr_number=job.pr_number,
        review_state=job.review_state,
        review_body=job.review_body,
    )
    return arq_job.job_id if arq_job is not None else ""


async def enqueue_merged_pr(pool: ArqRedis, job: MergedPrJob) -> str:
    """Enqueue a merge-reap job onto the Arq queue.

    Mirrors :func:`enqueue_prd`: the :class:`MergedPrJob` fields ride as keyword
    arguments so the worker receives them by name.

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

    Mirrors :func:`enqueue_prd` but pins a per-repo ``_job_id`` so concurrent kicks for
    the same repo collapse to one in-flight drain (Arq dedups on the id).

    Args:
        pool: The connected Arq Redis pool.
        job: The ad-hoc drain kick to enqueue.

    Returns:
        The Arq job ID, or an empty string when Arq coalesced this kick against an
        already-queued drain for the same repo.
    """
    arq_job = await pool.enqueue_job(
        RUN_ADHOC_DRAIN_TASK,
        repo_full_name=job.repo_full_name,
        _job_id=_adhoc_drain_job_id(job.repo_full_name),
    )
    return arq_job.job_id if arq_job is not None else ""


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
