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

import asyncio
from pathlib import Path

import aiosqlite

from retinue.orchestrator import Slice

_SCHEMA = """
CREATE TABLE IF NOT EXISTS impl_retries (
    slice_key TEXT PRIMARY KEY,
    attempts  INTEGER NOT NULL DEFAULT 0
)
"""

# A fresh ImplRetryStore is constructed per build binding, so it is NOT long-lived and
# must not hold a persistent connection (that would leak a connection/thread per event).
# The one-time cost — mkdir, schema, and the persistent WAL mode — is instead memoised
# per db-path so it runs once per process per file, not on every call. WAL is a durable
# property of the db file, so setting it once is enough.
_initialized_paths: set[Path] = set()
# Serialises only the one-time WAL switch + schema create per path: concurrent
# first-calls must not race the journal-mode transition (which errors "database is
# locked"). Steady state hits the in-set fast path and never touches this lock.
_init_lock = asyncio.Lock()


async def _ensure_initialized(db_path: Path) -> None:
    """Create the parent dir, schema, and WAL mode once per db-path per process."""
    if db_path in _initialized_paths:
        return
    async with _init_lock:
        if db_path in _initialized_paths:
            return
        db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(_SCHEMA)
            await db.commit()
        _initialized_paths.add(db_path)


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
        await _ensure_initialized(self._db_path)
        async with (
            aiosqlite.connect(self._db_path) as db,
            db.execute(
                "SELECT attempts FROM impl_retries WHERE slice_key = ?", (key,)
            ) as cursor,
        ):
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
        await _ensure_initialized(self._db_path)
        async with aiosqlite.connect(self._db_path) as db:
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
