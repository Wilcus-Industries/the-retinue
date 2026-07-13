"""SQLite-backed deduplication of PRD events.

GitHub redelivers webhooks and fires multiple ``issues`` actions for one issue, so
the same PRD can reach the worker more than once. :class:`PrdDedupeStore` records
each PRD the worker has accepted, keyed by repo + issue, and lets exactly the first
claim through. The store is durable (an on-disk SQLite file) so dedupe survives a
worker restart, and the claim is atomic so concurrent workers cannot both win.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite

from retinue.queue import PrdJob

_SCHEMA = """
CREATE TABLE IF NOT EXISTS processed_prds (
    prd_key    TEXT PRIMARY KEY,
    claimed_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def prd_dedupe_key(job: PrdJob) -> str:
    """Return the dedupe identity of a PRD: its repo and issue number.

    Deliberately excludes ``action`` — ``opened``, ``labeled``, and a redelivery
    all refer to the same PRD and must collapse to one key.

    Args:
        job: The PRD job whose identity to compute.

    Returns:
        A stable ``"owner/repo#<issue>"`` key.
    """
    return f"{job.repo_full_name}#{job.issue_number}"


class PrdDedupeStore:
    """Durable, atomic first-claim-wins dedupe over a SQLite file.

    Constructed once at worker startup and held for the process lifetime, so it keeps a
    single lazily-opened connection (schema + WAL pragmas applied once) rather than
    reconnecting per call. A per-store :class:`asyncio.Lock` serialises access, since a
    shared aiosqlite connection cannot have statements from concurrent tasks interleave.

    Args:
        db_path: Path to the SQLite database file. Created on first use; parent
            directories are created if missing.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def _connection(self) -> aiosqlite.Connection:
        """Return the store's connection, opening + preparing it on first use.

        Callers must hold ``self._lock``. WAL + ``synchronous=NORMAL`` and the schema
        are applied exactly once, when the connection is first opened.
        """
        if self._db is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            db = await aiosqlite.connect(self._db_path)
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute(_SCHEMA)
            await db.commit()
            self._db = db
        return self._db

    async def claim(self, key: str) -> bool:
        """Atomically claim a PRD key; return True only for the first claimant.

        The PRIMARY KEY makes the insert atomic: a duplicate raises
        ``IntegrityError``, which we translate into a ``False`` (already processed)
        rather than an error, so a redelivered PRD is a quiet skip.

        Args:
            key: The PRD dedupe key (see :func:`prd_dedupe_key`).

        Returns:
            True if this call recorded the key for the first time; False if the key
            was already present (a duplicate to ignore).
        """
        async with self._lock:
            db = await self._connection()
            try:
                await db.execute(
                    "INSERT INTO processed_prds (prd_key) VALUES (?)", (key,)
                )
            except aiosqlite.IntegrityError:
                return False
            await db.commit()
            return True

    async def release(self, key: str) -> None:
        """Delete ``key``'s claim so a re-delivery can claim it again.

        The claim is recorded before the PRD's run state is durable; a worker that dies
        in that window would otherwise burn the PRD forever (every retry sees a
        duplicate). The failure path releases the claim to undo it. Idempotent:
        releasing a key that was never claimed deletes zero rows and is a no-op.

        Args:
            key: The PRD dedupe key (see :func:`prd_dedupe_key`).
        """
        async with self._lock:
            db = await self._connection()
            await db.execute("DELETE FROM processed_prds WHERE prd_key = ?", (key,))
            await db.commit()

    async def close(self) -> None:
        """Close the underlying connection. Idempotent; safe to call at shutdown."""
        async with self._lock:
            if self._db is not None:
                await self._db.close()
                self._db = None
