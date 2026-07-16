"""FastAPI webhook router: signature verification and issues-event dispatch.

Verifies the HMAC-SHA256 signature on every incoming webhook, then routes an
``issues`` event (action opened/reopened/edited/labeled) by label: a ``prd`` issue
enqueues a PRD job onto the Arq queue, a ``ready-for-agent`` non-``prd`` issue
enqueues a single ad-hoc drain kick (the low-latency admission of ad-hoc work), and
``prd`` wins when both labels are present. An issue with neither relevant label, or a
non-relevant action, is acked 204 and enqueues nothing — so the slicer only sees PRDs
and the drain only fires for ready-for-agent work. The enqueue is awaited inline
before the ack so a failed enqueue surfaces as a 5xx (GitHub redelivers) rather than
vanishing after a 202.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any

from fastapi import APIRouter, Header, Request, Response

from retinue.queue import (
    AdhocDrainJob,
    MergedPrJob,
    PrdJob,
    ReviewJob,
    enqueue_adhoc_drain,
    enqueue_merged_pr,
    enqueue_prd,
    enqueue_review,
)
from retinue.vocab import PRD_LABEL, READY_LABEL

logger = logging.getLogger(__name__)

# The PRD label gate. Per the PRD, the retinue acts only on issues carrying the ``prd``
# label, and only on the actions that can newly mark an issue as a PRD or change its body
# — opened/reopened/edited/labeled. Any other action (closed, assigned, unlabeled, …) or
# an unlabeled issue is acked 204 and enqueues nothing, so non-PRD issue activity never
# reaches the slicer. The ad-hoc kick shares the same relevant-action set: a
# ``ready-for-agent`` issue admits work only on the actions that can newly add the label
# or change the issue.
_ISSUE_ACTIONS = frozenset({"opened", "reopened", "edited", "labeled"})


def _issue_label_names(body: dict[str, Any]) -> list[str]:
    """The label names on an ``issues`` payload (empty when none / malformed)."""
    labels = body.get("issue", {}).get("labels", [])
    return [label.get("name") for label in labels]


def _is_relevant_issue_action(body: dict[str, Any]) -> bool:
    """Whether the issue action can newly mark or change the issue (see :data:`_ISSUE_ACTIONS`)."""
    return body.get("action") in _ISSUE_ACTIONS


def _is_prd_issue_event(body: dict[str, Any]) -> bool:
    """Whether an ``issues`` payload should drive a PRD job.

    True only when the issue carries the ``prd`` label *and* the action is one the
    gate accepts (see :data:`_ISSUE_ACTIONS`).
    """
    if not _is_relevant_issue_action(body):
        return False
    return PRD_LABEL in _issue_label_names(body)


def _is_adhoc_issue_event(body: dict[str, Any]) -> bool:
    """Whether an ``issues`` payload should kick an ad-hoc drain.

    True only when the issue carries the ``ready-for-agent`` label but **not** ``prd``
    (prd wins when both are present, routing to the slicer) and the action is one the
    gate accepts. The caller checks the PRD gate first, so this need not re-exclude
    ``prd`` for routing — but it does so explicitly to stay correct in isolation.
    """
    if not _is_relevant_issue_action(body):
        return False
    labels = _issue_label_names(body)
    return READY_LABEL in labels and PRD_LABEL not in labels


def compute_signature(payload: bytes, secret: str) -> str:
    """Return the ``sha256=<hex>`` HMAC header GitHub puts in ``X-Hub-Signature-256``.

    The single source of truth for the webhook HMAC, so the verify path and any
    signer (e.g. test helpers) cannot drift.

    Args:
        payload: The raw request body bytes.
        secret: The webhook secret configured on the GitHub App.

    Returns:
        ``sha256=`` followed by the hex HMAC-SHA256 digest.
    """
    return "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def verify_signature(payload: bytes, secret: str, signature_header: str | None) -> bool:
    """Return True only if the webhook signature is present and valid.

    GitHub signs every webhook with HMAC-SHA256 using the configured secret and
    puts the result in ``X-Hub-Signature-256`` as ``sha256=<hex>``.

    Args:
        payload: The raw request body bytes.
        secret: The webhook secret configured on the GitHub App.
        signature_header: The value of ``X-Hub-Signature-256``.

    Returns:
        True when the header is present and matches; False otherwise.
    """
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = compute_signature(payload, secret)
    return hmac.compare_digest(expected, signature_header)


def _build_job(body: dict[str, Any]) -> PrdJob:
    return PrdJob(
        repo_full_name=body["repository"]["full_name"],
        issue_number=body["issue"]["number"],
        action=body.get("action", ""),
    )


def _review_author_login(body: dict[str, Any]) -> str:
    """The login of the account that submitted a ``pull_request_review``.

    GitHub puts the reviewing account under ``review.user.login``; an empty string is
    returned when the shape is missing so a malformed payload reads as "not heimdall".
    """
    return str(body.get("review", {}).get("user", {}).get("login") or "")


def _build_review_job(body: dict[str, Any]) -> ReviewJob:
    """Build a :class:`ReviewJob` from a ``pull_request_review`` payload.

    GitHub puts the review under ``review`` and the PR under ``pull_request``; the
    review's ``state`` and ``body`` are what the loopback worker parses heimdall's
    verdict and findings out of.
    """
    review = body.get("review", {})
    return ReviewJob(
        repo_full_name=body["repository"]["full_name"],
        pr_number=body["pull_request"]["number"],
        review_state=str(review.get("state", "")),
        review_body=str(review.get("body") or ""),
    )


def _build_merged_pr_job(body: dict[str, Any]) -> MergedPrJob:
    """Build a :class:`MergedPrJob` from a ``pull_request`` closed+merged payload."""
    return MergedPrJob(
        repo_full_name=body["repository"]["full_name"],
        pr_number=body["pull_request"]["number"],
    )


def _is_merge_event(body: dict[str, Any]) -> bool:
    """Whether a ``pull_request`` payload is the human-merge signal (closed + merged).

    GitHub fires ``pull_request`` ``closed`` for both a merge and a plain close; only a
    merge (``merged == true``) drives the reap, so a close-without-merge is acked and
    ignored.
    """
    return body.get("action") == "closed" and bool(
        body.get("pull_request", {}).get("merged")
    )


def make_webhook_router(*, webhook_secret: str, heimdall_bot_login: str) -> APIRouter:
    """Return a configured APIRouter with the ``/webhook`` POST endpoint.

    The handler reads the Arq pool from ``request.app.state.arq_pool`` at request
    time so it picks up the pool created by the app's lifespan hook, even though
    the router is built before the lifespan runs.

    Args:
        webhook_secret: The HMAC secret to verify incoming requests against.
        heimdall_bot_login: The bot login a ``pull_request_review`` must be authored by
            for the loopback to be enqueued; any other reviewer is acked 204 and dropped,
            so only heimdall's verdict drives a rebuild/converge round.
    """
    router = APIRouter()

    @router.post("/webhook")
    async def handle_webhook(
        request: Request,
        x_github_event: str | None = Header(default=None),
        x_hub_signature_256: str | None = Header(default=None),
    ) -> Response:
        """Receive a GitHub webhook, verify it, and enqueue a PRD job if relevant."""
        payload = await request.body()
        if not verify_signature(payload, webhook_secret, x_hub_signature_256):
            # 401 on a missing or mismatched signature; nothing is enqueued.
            return Response(status_code=401, content="Invalid webhook signature")

        # Three event types route to work; everything else is acked (204) without
        # enqueuing. Resolve the pool at request time from app.state so the
        # lifespan-created pool is always used, and enqueue inline before acking: a
        # failure raises and the handler returns 5xx (GitHub redelivers) rather than
        # dropping the job after a 202 has already been sent.
        if x_github_event == "issues":
            return await _dispatch_issue(request)
        if x_github_event == "pull_request":
            return await _dispatch_pull_request(request)
        if x_github_event == "pull_request_review":
            return await _dispatch_review(request)
        return Response(status_code=204)

    async def _dispatch_issue(request: Request) -> Response:
        """Route an ``issues`` event: PRD job, ad-hoc drain kick, or ack-and-drop.

        PRD wins when both labels are present — a ``prd`` issue drives the slicer even
        if it also carries ``ready-for-agent``. A ``ready-for-agent`` non-``prd`` issue
        kicks a single ad-hoc drain (the low-latency admission). An issue with neither
        relevant label (or a non-relevant action) is acked 204 and enqueues nothing.
        """
        body: dict[str, Any] = await request.json()
        if _is_prd_issue_event(body):
            return await _enqueue_prd_job(request, body)
        if _is_adhoc_issue_event(body):
            return await _enqueue_adhoc_kick(request, body)
        # Neither a PRD nor a ready-for-agent issue on a relevant action: ack and enqueue
        # nothing so the slicer and the ad-hoc drain only ever see work meant for them.
        return Response(status_code=204)

    async def _enqueue_prd_job(request: Request, body: dict[str, Any]) -> Response:
        job = _build_job(body)
        await enqueue_prd(request.app.state.arq_pool, job)
        logger.info(
            "Enqueued PRD job for %s#%d action=%s",
            job.repo_full_name,
            job.issue_number,
            job.action,
        )
        return Response(status_code=202)

    async def _enqueue_adhoc_kick(request: Request, body: dict[str, Any]) -> Response:
        job = AdhocDrainJob(repo_full_name=body["repository"]["full_name"])
        await enqueue_adhoc_drain(request.app.state.arq_pool, job)
        logger.info("Enqueued ad-hoc drain kick for %s", job.repo_full_name)
        return Response(status_code=202)

    async def _dispatch_pull_request(request: Request) -> Response:
        """Route a ``pull_request`` event: reap only on the human merge (closed+merged)."""
        body: dict[str, Any] = await request.json()
        if not _is_merge_event(body):
            # A non-merge pull_request action (opened, synchronize, plain close) is acked
            # and ignored — only a merge drives the reap.
            return Response(status_code=204)
        job = _build_merged_pr_job(body)
        await enqueue_merged_pr(request.app.state.arq_pool, job)
        logger.info(
            "Enqueued merge-reap job for %s PR #%d",
            job.repo_full_name,
            job.pr_number,
        )
        return Response(status_code=202)

    async def _dispatch_review(request: Request) -> Response:
        """Route a ``pull_request_review`` event into the heimdall loopback.

        Only heimdall's bot verdict drives the loopback: a review by anyone else (a human
        ``high:`` line, an approving other-bot review) is acked 204 and enqueues nothing,
        so it can never burn a rebuild round or spuriously trigger convergence/handoff.
        """
        body: dict[str, Any] = await request.json()
        if _review_author_login(body) != heimdall_bot_login:
            return Response(status_code=204)
        job = _build_review_job(body)
        await enqueue_review(request.app.state.arq_pool, job)
        logger.info(
            "Enqueued review-loopback job for %s PR #%d state=%s",
            job.repo_full_name,
            job.pr_number,
            job.review_state,
        )
        return Response(status_code=202)

    return router
