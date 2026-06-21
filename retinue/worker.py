"""Arq worker: the process_prd task and WorkerSettings.

The walking-skeleton worker dequeues a PRD job and logs the repository, issue
number, and action. Later issues replace the body of :func:`process_prd` with the
real PRD pipeline; the queue contract (task name and kwargs) stays the same.

Launch the worker with:
    arq retinue.worker.WorkerSettings
or via the ``retinue-worker`` console script.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from arq.connections import RedisSettings

logger = logging.getLogger(__name__)


async def process_prd(
    ctx: dict[str, Any],
    *,
    repo_full_name: str,
    issue_number: int,
    action: str,
) -> None:
    """Arq task: log the dequeued PRD job (repo, issue number, action).

    This is the walking-skeleton terminus of the transport spine — it proves the
    enqueue -> dequeue round-trip end to end. Later issues extend this with the
    real PRD-processing pipeline.

    Args:
        ctx: Arq worker context (unused in the skeleton).
        repo_full_name: e.g. "owner/repo".
        issue_number: The GitHub issue number.
        action: The webhook action that triggered the job.
    """
    logger.info(
        "Processing PRD for %s#%d action=%s",
        repo_full_name,
        issue_number,
        action,
    )


def _configure_logging() -> None:
    """Send INFO-level logs to stdout so the worker's progress is observable.

    ``run_worker`` does not configure logging (only arq's CLI does), so without
    this the root logger sits at WARNING with no handler and every ``logger.info``
    line is dropped. Install a single stdout handler at INFO; ``force=True``
    rebinds any pre-existing root config to the current stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


# Module-level settings, loaded lazily in main() so importing this module does
# not require the env vars to be present (e.g. when registering the task in tests).
settings: Any = None


def _load_settings() -> Any:
    from retinue.config import Settings

    return Settings()  # type: ignore[call-arg]


def main() -> None:
    """Console-script entrypoint: start the Arq worker with WorkerSettings.

    Resolves ``WorkerSettings.redis_settings`` from the configured ``REDIS_URL``
    here, at process start: arq reads that class attribute when it constructs the
    Worker, before ``on_startup`` runs, so the override must be applied now.
    """
    from arq.worker import run_worker

    _configure_logging()

    global settings
    if settings is None:
        settings = _load_settings()
    WorkerSettings.redis_settings = RedisSettings.from_dsn(settings.redis_url)

    run_worker(WorkerSettings)  # type: ignore[arg-type]


class WorkerSettings:
    """Arq WorkerSettings: registers process_prd and the Redis connection.

    Launch the worker process with:
        arq retinue.worker.WorkerSettings
    """

    functions = [process_prd]
    # Overridden from the configured REDIS_URL in main() at process start (arq
    # reads this attribute before on_startup runs); the localhost default keeps it
    # a valid RedisSettings for the ``arq retinue.worker.WorkerSettings`` launch.
    redis_settings: RedisSettings = RedisSettings()
