"""Authed read/control REST surface for the Retinue API.

Every route on this router requires ``Authorization: Bearer <api_service_token>``
verified with a constant-time compare (same idiom as ``verify_signature``).
"""

from __future__ import annotations

import hmac
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from retinue.queue import AdhocDrainJob, enqueue_adhoc_drain

logger = logging.getLogger(__name__)


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization[len("Bearer "):]


def verify_bearer_token(presented: str | None, configured: str) -> bool:
    """Return True only when ``presented`` matches the configured token.

    An empty configured token is never valid (fail closed); ``compare_digest``
    guards against timing attacks.
    """
    if not configured or not presented:
        return False
    return hmac.compare_digest(presented, configured)


class DrainRequest(BaseModel):
    repo_full_name: str


def make_api_router(*, api_service_token: str) -> APIRouter:
    """Return a configured APIRouter with the authed ``/api`` endpoints.

    Every route requires ``Authorization: Bearer <api_service_token>`` verified
    with a constant-time compare. Reads the Arq pool from
    ``request.app.state.arq_pool`` at request time.

    Args:
        api_service_token: The bearer token required on every request. When empty
            the router rejects every request (fail closed).
    """

    def require_bearer(authorization: str | None = Header(default=None)) -> None:
        if not verify_bearer_token(_extract_bearer(authorization), api_service_token):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    router = APIRouter(prefix="/api", dependencies=[Depends(require_bearer)])

    @router.post("/drain", status_code=202)
    async def drain(request: Request, body: DrainRequest) -> dict[str, str]:
        """Enqueue a real ad-hoc drain for the given repo.

        Args:
            request: The incoming FastAPI request (pool read from ``app.state``).
            body: ``{"repo_full_name": "owner/repo"}`` — the target repo.

        Returns:
            ``{"job_id": "<arq-job-id>"}`` on success (202 Accepted).
        """
        job = AdhocDrainJob(repo_full_name=body.repo_full_name)
        job_id = await enqueue_adhoc_drain(request.app.state.arq_pool, job)
        logger.info("API-kicked ad-hoc drain for %s", job.repo_full_name)
        return {"job_id": job_id}

    return router
