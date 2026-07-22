"""FastAPI API router: the authed read/control REST surface (``/api/*``).

Every route on this router requires ``Authorization: Bearer <api_service_token>``,
checked with the same constant-time-compare idiom the webhook uses for its HMAC
signature (:func:`retinue.webhook.verify_signature`) — a bearer mismatch never leaks
timing information about how much of the token matched. The check is wired as a
router-level dependency so every route added to this router is authed automatically,
with no per-route opt-in to forget.

First endpoint: ``POST /api/drain``, which enqueues a real ad-hoc scheduler drain via
the same :func:`retinue.queue.enqueue_adhoc_drain` path the webhook uses.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from retinue.queue import AdhocDrainJob, enqueue_adhoc_drain
from retinue.run_ledger import RunLedgerStore

logger = logging.getLogger(__name__)


class DrainRequest(BaseModel):
    """Body of a ``POST /api/drain`` request."""

    repo_full_name: str


def verify_bearer_token(authorization: str | None, expected_token: str) -> bool:
    """Return True only if ``authorization`` is a valid ``Bearer <expected_token>``.

    Args:
        authorization: The raw ``Authorization`` header value, if present.
        expected_token: The configured ``api_service_token`` to compare against.

    Returns:
        True when the header is present, prefixed ``Bearer ``, and matches the
        expected token via a constant-time compare; False otherwise.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return False
    token = authorization.removeprefix("Bearer ")
    return hmac.compare_digest(token, expected_token)


def make_api_router(
    *, api_service_token: str, run_ledger: RunLedgerStore
) -> APIRouter:
    """Return a configured APIRouter with the authed ``/api/*`` surface.

    The handler reads the Arq pool from ``request.app.state.arq_pool`` at request
    time, mirroring :func:`retinue.webhook.make_webhook_router` so it picks up the
    pool created by the app's lifespan hook.

    Args:
        api_service_token: The bearer token every ``/api/*`` request must present.
        run_ledger: The run-ledger store ``GET /api/runs`` reads the recorded run-state
            rows back from (the reader side of the cross-process ledger the worker writes).
    """

    async def _require_bearer_token(
        authorization: str | None = Header(default=None),
    ) -> None:
        """Router-level dependency: 401s before any handler body runs on a bad token."""
        if not verify_bearer_token(authorization, api_service_token):
            raise HTTPException(
                status_code=401, detail="Invalid or missing bearer token"
            )

    router = APIRouter(prefix="/api", dependencies=[Depends(_require_bearer_token)])

    @router.post("/drain", status_code=202)
    async def drain(request: Request, body: DrainRequest) -> dict[str, str]:
        """Enqueue a real ad-hoc scheduler drain for ``body.repo_full_name``."""
        job = AdhocDrainJob(repo_full_name=body.repo_full_name)
        job_id = await enqueue_adhoc_drain(request.app.state.arq_pool, job)
        logger.info("Enqueued API-triggered drain for %s", job.repo_full_name)
        return {"job_id": job_id}

    @router.get("/runs")
    async def runs() -> list[dict[str, object]]:
        """Return every recorded run-ledger row as JSON (authed)."""
        rows = await run_ledger.rows()
        return [
            {
                "repo": r.repo,
                "issue": r.issue,
                "state": r.state,
                "url": r.url,
                "updated_at": r.updated_at,
            }
            for r in rows
        ]

    return router
