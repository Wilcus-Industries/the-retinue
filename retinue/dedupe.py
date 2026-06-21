"""SQLite-backed deduplication of PRD events.

GitHub redelivers webhooks and fires multiple ``issues`` actions for one issue, so
the same PRD can reach the worker more than once. :class:`PrdDedupeStore` records
each PRD the worker has accepted, keyed by repo + issue, and lets exactly the first
claim through. The store is durable (an on-disk SQLite file) so dedupe survives a
worker restart, and the claim is atomic so concurrent workers cannot both win.
"""

from __future__ import annotations

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

    Args:
        db_path: Path to the SQLite database file. Created on first use; parent
            directories are created if missing.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)

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
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_SCHEMA)
            try:
                await db.execute(
                    "INSERT INTO processed_prds (prd_key) VALUES (?)", (key,)
                )
            except aiosqlite.IntegrityError:
                return False
            await db.commit()
            return True
