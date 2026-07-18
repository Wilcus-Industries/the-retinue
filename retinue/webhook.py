"""FastAPI webhook router: signature verification and event dispatch.

Verifies the HMAC-SHA256 signature on every incoming webhook, then routes two event types
to work: an ``issues`` event on a relevant action (one that can newly ready the issue or
change it) kicks a single scheduler drain — the low-latency admission of ready work — and a
``pull_request`` closed+merged event enqueues a merge reap. Everything else is acked 204 and
enqueues nothing. The enqueue is awaited inline before the ack so a failed enqueue surfaces
as a 5xx (GitHub redelivers) rather than vanishing after a 202.
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
    enqueue_adhoc_drain,
    enqueue_merged_pr,
)

logger = logging.getLogger(__name__)

# The relevant issue actions: only the ones that can newly add a trigger label or change
# the issue drive a drain kick. Any other action (closed, assigned, unlabeled, …) is acked
# 204 and enqueues nothing.
_ISSUE_ACTIONS = frozenset({"opened", "reopened", "edited", "labeled"})


def _is_relevant_issue_action(body: dict[str, Any]) -> bool:
    """Whether the issue action can newly mark or change the issue (see :data:`_ISSUE_ACTIONS`)."""
    return body.get("action") in _ISSUE_ACTIONS


def _is_adhoc_issue_event(body: dict[str, Any]) -> bool:
    """Whether an ``issues`` payload should kick a scheduler drain.

    True whenever the action is one the gate accepts (see :data:`_ISSUE_ACTIONS`) — the
    kick is only a per-repo "drain this repo" signal, not a per-issue task, so it does not
    gate on any specific label. The drain re-lists and re-filters the repo's ready issues by
    its own configured ``trigger_label`` (which varies per repo, e.g. a BYOK repo's custom
    label), so hardcoding a label here would silently starve repos that don't use the
    default ``ready-for-agent``.
    """
    return _is_relevant_issue_action(body)


def compute_signature(payload: bytes, secret: str) -> str:
    """Return the ``sha256=<hex>`` HMAC header GitHub puts in ``X-Hub-Signature-256``.

    The single source of truth for the webhook HMAC, so the verify path and any signer
    (e.g. test helpers) cannot drift.

    Args:
        payload: The raw request body bytes.
        secret: The webhook secret configured on the GitHub App.

    Returns:
        ``sha256=`` followed by the hex HMAC-SHA256 digest.
    """
    return "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def verify_signature(payload: bytes, secret: str, signature_header: str | None) -> bool:
    """Return True only if the webhook signature is present and valid.

    GitHub signs every webhook with HMAC-SHA256 using the configured secret and puts the
    result in ``X-Hub-Signature-256`` as ``sha256=<hex>``.

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


def make_webhook_router(*, webhook_secret: str) -> APIRouter:
    """Return a configured APIRouter with the ``/webhook`` POST endpoint.

    The handler reads the Arq pool from ``request.app.state.arq_pool`` at request time so it
    picks up the pool created by the app's lifespan hook, even though the router is built
    before the lifespan runs.

    Args:
        webhook_secret: The HMAC secret to verify incoming requests against.
    """
    router = APIRouter()

    @router.post("/webhook")
    async def handle_webhook(
        request: Request,
        x_github_event: str | None = Header(default=None),
        x_hub_signature_256: str | None = Header(default=None),
    ) -> Response:
        """Receive a GitHub webhook, verify it, and enqueue work if relevant."""
        payload = await request.body()
        if not verify_signature(payload, webhook_secret, x_hub_signature_256):
            # 401 on a missing or mismatched signature; nothing is enqueued.
            return Response(status_code=401, content="Invalid webhook signature")

        # Two event types route to work; everything else is acked (204) without enqueuing.
        # Enqueue inline before acking: a failure raises and the handler returns 5xx (GitHub
        # redelivers) rather than dropping the job after a 202 has already been sent.
        if x_github_event == "issues":
            return await _dispatch_issue(request)
        if x_github_event == "pull_request":
            return await _dispatch_pull_request(request)
        return Response(status_code=204)

    async def _dispatch_issue(request: Request) -> Response:
        """Route an ``issues`` event: scheduler-drain kick, or ack-and-drop.

        A relevant action (see :data:`_ISSUE_ACTIONS`) kicks a single scheduler drain (the
        low-latency admission), regardless of the issue's labels — the drain re-filters by
        the repo's own configured trigger label. A non-relevant action is acked 204 and
        enqueues nothing.
        """
        body: dict[str, Any] = await request.json()
        if _is_adhoc_issue_event(body):
            return await _enqueue_adhoc_kick(request, body)
        return Response(status_code=204)

    async def _enqueue_adhoc_kick(request: Request, body: dict[str, Any]) -> Response:
        job = AdhocDrainJob(repo_full_name=body["repository"]["full_name"])
        await enqueue_adhoc_drain(request.app.state.arq_pool, job)
        logger.info("Enqueued scheduler-drain kick for %s", job.repo_full_name)
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

    return router
