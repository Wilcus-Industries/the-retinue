"""FastAPI webhook router: signature verification and issues-event dispatch.

Verifies the HMAC-SHA256 signature on every incoming webhook, acts only on
``issues`` events, and enqueues a PRD job onto the Arq queue before acking 202.
The enqueue is awaited inline before the ack so a failed enqueue surfaces as a
5xx (GitHub redelivers) rather than vanishing after a 202 was already sent.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any

from fastapi import APIRouter, Header, Request, Response

from retinue.queue import PrdJob, enqueue_prd

logger = logging.getLogger(__name__)


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


def make_webhook_router(*, webhook_secret: str) -> APIRouter:
    """Return a configured APIRouter with the ``/webhook`` POST endpoint.

    The handler reads the Arq pool from ``request.app.state.arq_pool`` at request
    time so it picks up the pool created by the app's lifespan hook, even though
    the router is built before the lifespan runs.

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
        """Receive a GitHub webhook, verify it, and enqueue a PRD job if relevant."""
        payload = await request.body()
        if not verify_signature(payload, webhook_secret, x_hub_signature_256):
            # 401 on a missing or mismatched signature; nothing is enqueued.
            return Response(status_code=401, content="Invalid webhook signature")

        # Act only on issues events; everything else is acked without enqueuing.
        if x_github_event != "issues":
            return Response(status_code=204)

        body: dict[str, Any] = await request.json()
        job = _build_job(body)

        # Resolve the pool at request time from app.state so the lifespan-created
        # pool is always used. Enqueue inline before acking: a failure raises and
        # the handler returns 5xx (GitHub redelivers) rather than dropping the job
        # after a 202 has already been sent.
        pool = request.app.state.arq_pool
        await enqueue_prd(pool, job)

        logger.info(
            "Enqueued PRD job for %s#%d action=%s",
            job.repo_full_name,
            job.issue_number,
            job.action,
        )
        return Response(status_code=202)

    return router
