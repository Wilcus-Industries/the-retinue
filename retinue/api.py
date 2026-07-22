"""FastAPI API router: the authed read/control REST surface (``/api/*``).

Every route on this router requires ``Authorization: Bearer <api_service_token>``,
checked with the same constant-time-compare idiom the webhook uses for its HMAC
signature (:func:`retinue.webhook.verify_signature`) — a bearer mismatch never leaks
timing information about how much of the token matched. The check is wired as a
router-level dependency so every route added to this router is authed automatically,
with no per-route opt-in to forget.

First endpoint: ``POST /api/drain``, which enqueues a real ad-hoc scheduler drain via
the same :func:`retinue.queue.enqueue_adhoc_drain` path the webhook uses.

``GET /api/runs`` returns every recorded run-ledger row (the reader side of the
cross-process run-state ledger the worker writes).

``GET /api/budget`` reads the shared budget SQLite ledger read-only from the API
process (issue #90) — it never opens the worker's in-memory governor, only a
:class:`retinue.budget.BudgetLedger` bound to the same ``budget_db_path`` the worker
writes.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from retinue.budget import BudgetLedger
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
    *,
    api_service_token: str,
    run_ledger: RunLedgerStore,
    budget_ledger: BudgetLedger,
) -> APIRouter:
    """Return a configured APIRouter with the authed ``/api/*`` surface.

    The handler reads the Arq pool from ``request.app.state.arq_pool`` at request
    time, mirroring :func:`retinue.webhook.make_webhook_router` so it picks up the
    pool created by the app's lifespan hook.

    Args:
        api_service_token: The bearer token every ``/api/*`` request must present.
        run_ledger: The run-ledger store ``GET /api/runs`` reads the recorded run-state
            rows back from (the reader side of the cross-process ledger the worker writes).
        budget_ledger: The API process's own read-only handle onto the shared budget
            SQLite ledger (see :func:`retinue.app.create_app`), backing
            ``GET /api/budget``.
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

    @router.get("/budget")
    async def budget() -> dict[str, float]:
        """Return the trailing-24h spend and the rolling-24h cap, read from the ledger.

        Reads only — never records a charge. ``cap()`` is computed from the ledger's
        own ``weekly_budget``/``daily_cap_fraction`` (the API process's ``Settings``),
        so it is available with no dependency on the worker's governor.
        """
        return {
            "trailing_24h_spend": await budget_ledger.trailing_24h_spend(),
            "cap": budget_ledger.cap(),
        }

    return router
