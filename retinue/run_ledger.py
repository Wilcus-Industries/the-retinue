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

The worker writes ``queued``/``building`` at the drain's admit and build-start choke
points, and the terminal states (``escalated``, ``pr_opened``, ``failed``, ``merged``) at
the pipeline's own choke points — :meth:`~retinue.pipeline.Pipeline.process_adhoc_pr` and
:meth:`~retinue.pipeline.Pipeline.reap_pr` (issue #91). ``GET /api/escalations`` reads the
``escalated`` rows back for the operator.
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

    ``QUEUED`` and ``BUILDING`` are written by the drain (its admit and build-start
    choke points); the terminal states are written by the pipeline
    (:mod:`retinue.pipeline`) at its own choke points: ``ESCALATED`` on a blocking review
    gate, ``PR_OPENED`` when the ad-hoc PR opens, ``FAILED`` on a red build, and
    ``MERGED`` on the human's merge reap.
    """

    QUEUED = "queued"
    BUILDING = "building"
    ESCALATED = "escalated"
    PR_OPENED = "pr_opened"
    FAILED = "failed"
    MERGED = "merged"


# The coarse lifecycle order, low → high. :meth:`RunLedgerStore.record` refuses to move a
# row backward down this order: an issue keeps its trigger label until reap, so a later
# drain pass re-admits an in-flight or stranded issue and re-records ``queued`` — which
# must not clobber the ``building`` (or a future terminal state) the earlier pass reached.
_LIFECYCLE_RANK: dict[RunState, int] = {
    RunState.QUEUED: 0,
    RunState.BUILDING: 1,
    # Terminal states all outrank building; ordering among them is immaterial here.
    RunState.ESCALATED: 2,
    RunState.PR_OPENED: 2,
    RunState.FAILED: 2,
    RunState.MERGED: 2,
}

# The SQL rank of the *existing* row's state, built from the one rank table above so the
# two never drift. Used in the upsert's WHERE to reject a regression to a lower rank.
_EXISTING_RANK_CASE = (
    "CASE run_ledger.state\n"
    + "\n".join(
        f"    WHEN '{state.value}' THEN {rank}"
        for state, rank in _LIFECYCLE_RANK.items()
    )
    + "\n    ELSE 0\nEND"
)


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

        Re-recording the same key with a state at or above the row's current lifecycle rank
        overwrites its state, url, and timestamp, so the ledger holds one current row per
        issue rather than an append-only history. A *regression* to a lower rank (e.g. a
        re-admitted in-flight issue re-recording ``queued`` over its ``building``) is
        refused, so ``/api/runs`` never reports a progressed issue back as queued.
        """
        now = datetime.now(UTC).isoformat()
        async with self._connect() as db:
            await db.execute(_SCHEMA)
            await db.execute(
                f"""
                INSERT INTO run_ledger (repo, issue, state, url, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(repo, issue) DO UPDATE SET
                    state = excluded.state,
                    url = excluded.url,
                    updated_at = excluded.updated_at
                WHERE ? >= {_EXISTING_RANK_CASE}
                """,
                # ``_EXISTING_RANK_CASE`` is built from our own int constants, never user
                # input; the row's fields all stay bound parameters.
                (repo_full_name, issue, state.value, url, now, _LIFECYCLE_RANK[state]),
            )
            await db.commit()

    async def rows(self) -> list[RunLedgerRow]:
        """Return every recorded row, most-recently-updated first (empty if unseen)."""
        return await self._select()

    async def escalations(self) -> list[RunLedgerRow]:
        """Return every row currently in :attr:`RunState.ESCALATED`, most recent first.

        The reader side of ``GET /api/escalations``: an issue the review gate blocked and
        left for a human, each with the GitHub issue URL recorded when it escalated
        (:meth:`~retinue.pipeline.Pipeline._escalate_blocking`).
        """
        return await self._select(state=RunState.ESCALATED)

    async def _select(self, *, state: RunState | None = None) -> list[RunLedgerRow]:
        """Run the shared row-select query, optionally filtered to one ``state``."""
        query = "SELECT repo, issue, state, url, updated_at FROM run_ledger "
        params: tuple[str, ...] = ()
        if state is not None:
            query += "WHERE state = ? "
            params = (state.value,)
        query += "ORDER BY updated_at DESC, repo ASC, issue ASC"
        async with self._connect() as db:
            await db.execute(_SCHEMA)
            async with db.execute(query, params) as cursor:
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
