"""SQLite-backed implementer-retry counter for the triage loop.

When an implementer fails, the orchestrator's triage decides whether to retry —
bounded by a cap — or to give up and reslice/escalate. That bound must be a
*persisted* count: a retry budget has to survive a worker restart and must not be
reset just by re-running the orchestrator, otherwise a doomed slice could retry
forever. :class:`ImplRetryStore` records one attempt counter per slice, keyed by
repo + issue, mirroring the durable-SQLite style of
:class:`retinue.dedupe.PrdDedupeStore`.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from retinue.orchestrator import Slice

_SCHEMA = """
CREATE TABLE IF NOT EXISTS impl_retries (
    slice_key TEXT PRIMARY KEY,
    attempts  INTEGER NOT NULL DEFAULT 0
)
"""


def impl_retry_key(slice_: Slice) -> str:
    """Return the retry identity of a slice: its repo and issue number.

    Args:
        slice_: The slice whose retry identity to compute.

    Returns:
        A stable ``"owner/repo#<issue>"`` key.
    """
    return f"{slice_.repo_full_name}#{slice_.issue_number}"


class ImplRetryStore:
    """Durable per-slice implementer-attempt counter over a SQLite file.

    The count is the number of implementer attempts recorded for a slice so far.
    Triage reads it to decide whether another retry is within the cap, and records
    a new attempt before each retry so the budget is consumed even across restarts.

    Args:
        db_path: Path to the SQLite database file. Created on first use; parent
            directories are created if missing.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)

    async def count(self, key: str) -> int:
        """Return the number of attempts recorded for ``key`` (zero if unseen).

        Args:
            key: The slice retry key (see :func:`impl_retry_key`).

        Returns:
            The persisted attempt count, or ``0`` for a slice never recorded.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_SCHEMA)
            async with db.execute(
                "SELECT attempts FROM impl_retries WHERE slice_key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    async def record_attempt(self, key: str) -> int:
        """Atomically increment ``key``'s attempt count and return the new value.

        The upsert is atomic on the primary key, so concurrent orchestrator runs
        cannot lose an increment.

        Args:
            key: The slice retry key (see :func:`impl_retry_key`).

        Returns:
            The attempt count after this increment.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_SCHEMA)
            await db.execute(
                """
                INSERT INTO impl_retries (slice_key, attempts) VALUES (?, 1)
                ON CONFLICT(slice_key) DO UPDATE SET attempts = attempts + 1
                """,
                (key,),
            )
            await db.commit()
            async with db.execute(
                "SELECT attempts FROM impl_retries WHERE slice_key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0
