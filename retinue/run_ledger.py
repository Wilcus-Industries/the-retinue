"""The cross-process run-ledger: the coarse lifecycle state of each issue's build.

The scheduler drain is stateless per pass, but the operator-facing API still needs to
answer "what is the retinue doing right now?". This durable ledger is that seam: the
worker writes a coarse run-state at the drain's choke points (``queued`` on admit,
``building`` on build start; the terminal states land in a later slice), and the web
process reads the same SQLite file back for ``GET /api/runs``.

The two processes share one file on the mounted ``worker-data`` volume, resolved from the
same :func:`retinue.config.state_dir` the other durable stores use. The store mirrors the
durable-SQLite style of :class:`retinue.reconcile.RunStateStore`: a fresh connection per
call, the schema executed in every method, and a ``commit`` before return, so a reader in a
second process always sees a consistent, up-to-date file.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

from retinue.config import state_dir

if TYPE_CHECKING:
    from retinue.config import Settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_ledger (
    repo       TEXT NOT NULL,
    issue      INTEGER NOT NULL,
    state      TEXT NOT NULL,
    url        TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (repo, issue)
)
"""


class RunState(enum.Enum):
    """The coarse lifecycle state of one issue's build, as reported to the API.

    Only ``QUEUED`` and ``BUILDING`` are written this slice (at the drain's admit and
    build-start choke points); the terminal states — ``ESCALATED``, ``PR_OPENED``,
    ``FAILED``, ``MERGED`` — are part of the vocabulary now but land in a later slice.
    """

    QUEUED = "queued"
    BUILDING = "building"
    ESCALATED = "escalated"
    PR_OPENED = "pr_opened"
    FAILED = "failed"
    MERGED = "merged"


@dataclass(frozen=True)
class RunLedgerRow:
    """One recorded run-ledger row: an issue's latest coarse state and when it changed."""

    repo: str
    issue: int
    state: str
    url: str | None
    updated_at: str


class RunLedgerStore:
    """Durable coarse run-state, one row per ``(repo, issue)``, upserted on each change.

    Mirrors the durable-SQLite style of :class:`retinue.reconcile.RunStateStore`: a fresh
    connection per call, the schema executed in every method, and a ``commit`` before
    return, so the web reader in a second process always sees the worker's latest writes.

    Args:
        db_path: Path to the SQLite database file. Created on first use; parent
            directories are created if missing.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)

    async def record(
        self,
        *,
        repo_full_name: str,
        issue: int,
        state: RunState,
        url: str | None = None,
    ) -> None:
        """Upsert one issue's coarse run-state, keyed on ``(repo, issue)``.

        Re-recording the same key overwrites its state, url, and timestamp, so the ledger
        holds one current row per issue rather than an append-only history.
        """
        now = datetime.now(UTC).isoformat()
        async with self._connect() as db:
            await db.execute(_SCHEMA)
            await db.execute(
                """
                INSERT INTO run_ledger (repo, issue, state, url, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(repo, issue) DO UPDATE SET
                    state = excluded.state,
                    url = excluded.url,
                    updated_at = excluded.updated_at
                """,
                (repo_full_name, issue, state.value, url, now),
            )
            await db.commit()

    async def rows(self) -> list[RunLedgerRow]:
        """Return every recorded row, most-recently-updated first (empty if unseen)."""
        async with self._connect() as db:
            await db.execute(_SCHEMA)
            async with db.execute(
                "SELECT repo, issue, state, url, updated_at FROM run_ledger "
                "ORDER BY updated_at DESC, repo ASC, issue ASC"
            ) as cursor:
                fetched = await cursor.fetchall()
        return [
            RunLedgerRow(
                repo=str(r[0]),
                issue=int(r[1]),
                state=str(r[2]),
                url=None if r[3] is None else str(r[3]),
                updated_at=str(r[4]),
            )
            for r in fetched
        ]

    def _connect(self) -> aiosqlite.Connection:
        """Open a fresh DB connection, ensuring the parent dir exists first."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        return aiosqlite.connect(self._db_path)


def run_ledger_store_path(settings: Settings) -> Path:
    """The run-ledger SQLite file the worker writes and the API reads.

    Co-located with the other durable stores (see :func:`retinue.config.state_dir`) so the
    web and worker containers share it over the one mounted volume.
    """
    return state_dir(settings) / "run-ledger.sqlite3"
