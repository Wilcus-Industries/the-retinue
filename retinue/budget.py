"""Budget governor: a rolling-24h spend ledger with mid-run pause/resume (issue #14).

The retinue meters agent token spend against a service-level weekly budget and enforces
a per-rolling-24h-window ceiling (``cap`` = a fraction, by default 12%, of the weekly
budget). The budget is shared across both lanes — the orchestrator's :func:`build_prd`
run and the cron lane — so the ledger is a single service-level SQLite store, mirroring
the durable-SQLite style of :class:`retinue.dedupe.PrdDedupeStore` and
:class:`retinue.impl_retry.ImplRetryStore`.

Two enforcement points:

* **gate at run start** (:meth:`BudgetGovernor.gate`) — if the run's estimated charge
  would push the trailing-24h spend over the cap, the run is *deferred* until the window
  frees (when the oldest in-window charge ages out).
* **meter mid-run** (:meth:`BudgetGovernor.meter`) — a charge that would cross the cap
  *pauses* the run and checkpoints it (the owned slice set is recorded in the reconcile
  :class:`~retinue.reconcile.RunStateStore`). :meth:`BudgetGovernor.try_resume` resumes
  it once the window frees by reusing the reconciliation machinery
  (:func:`~retinue.reconcile.reconcile_run`), so only the unfinished slices rebuild — no
  duplicate issue, branch, or PR.

Auth-aware metering: an API key meters dollars against a weekly-$ budget; subscription
OAuth meters tokens against a weekly-token budget. The math is identical; only the unit
and the weekly budget differ, both carried on the ledger.

The clock is injected (:class:`Clock`) so the rolling-24h window is deterministic in
tests rather than tied to wall-clock; the ledger lives in a SQLite file and the gh
queries flow through the injected reconcile seam, so the whole flow runs with no real
``gh`` and no network.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import aiosqlite

from retinue.orchestrator import PrdSlice
from retinue.reconcile import (
    ReconcileGh,
    ReconcileResult,
    RunStateStore,
    reconcile_run,
    run_state_key,
)

logger = logging.getLogger(__name__)

_WINDOW = timedelta(hours=24)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spend_entries (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    spent_at TEXT NOT NULL,
    amount  REAL NOT NULL
)
"""

_PAUSE_SCHEMA = """
CREATE TABLE IF NOT EXISTS budget_pauses (
    prd_key   TEXT PRIMARY KEY,
    resume_at TEXT NOT NULL
)
"""


class AuthMode(enum.Enum):
    """The metering unit a run's spend is charged in, selected by the auth mode.

    ``API_KEY`` meters dollars against a weekly-$ budget; ``SUBSCRIPTION`` meters tokens
    against a weekly-token budget. The rolling-24h math is identical for both; only the
    unit and the weekly budget differ.
    """

    API_KEY = "api_key"
    SUBSCRIPTION = "subscription"

    @classmethod
    def from_config(cls, value: str) -> AuthMode:
        """Parse the ``auth_mode`` config string into an :class:`AuthMode`.

        Args:
            value: The configured auth-mode string (e.g. ``"api_key"``).

        Returns:
            The matching :class:`AuthMode`.

        Raises:
            ValueError: ``value`` is not a known auth mode.
        """
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"unknown auth mode: {value!r}") from exc


class Clock(Protocol):
    """The time source the ledger reads the current instant from.

    Production uses a wall-clock implementation; tests inject a deterministic, advanceable
    fake so the rolling-24h window is reproducible. Always returns a timezone-aware UTC
    instant so window arithmetic is unambiguous.
    """

    def now(self) -> datetime:
        """Return the current instant as a timezone-aware UTC datetime."""
        ...


class SystemClock:
    """A :class:`Clock` backed by the real wall clock (UTC)."""

    def now(self) -> datetime:
        """Return the current UTC instant."""
        return datetime.now(UTC)


class BudgetLedger:
    """A durable rolling-24h spend ledger enforcing a fraction-of-weekly-budget cap.

    Each :meth:`record_spend` appends a timestamped charge; :meth:`trailing_24h_spend`
    sums only the charges inside the trailing 24h from ``now``. The :meth:`cap` is a
    fixed fraction (default 12%) of the weekly budget, in the ledger's unit (dollars in
    :attr:`AuthMode.API_KEY`, tokens in :attr:`AuthMode.SUBSCRIPTION`). The store is a
    single service-level SQLite file shared across the orchestrator and cron lanes, so a
    charge from either lane counts against the same window. Mirrors the durable-SQLite
    style of :class:`retinue.impl_retry.ImplRetryStore`.

    Args:
        db_path: Path to the SQLite database file. Created on first use; parent
            directories are created if missing.
        clock: The injected time source (see :class:`Clock`).
        auth_mode: The metering unit (dollars vs tokens).
        weekly_budget: The service-level weekly budget in the ledger's unit.
        daily_cap_fraction: Fraction of the weekly budget spendable per rolling-24h
            window (default 0.12, i.e. 12%).
    """

    def __init__(
        self,
        db_path: Path | str,
        *,
        clock: Clock,
        auth_mode: AuthMode,
        weekly_budget: float = 0.0,
        daily_cap_fraction: float = 0.12,
    ) -> None:
        self._db_path = Path(db_path)
        self._clock = clock
        self.auth_mode = auth_mode
        self._weekly_budget = weekly_budget
        self._daily_cap_fraction = daily_cap_fraction

    def cap(self) -> float:
        """Return the rolling-24h ceiling: ``daily_cap_fraction`` of the weekly budget."""
        return self._weekly_budget * self._daily_cap_fraction

    async def record_spend(self, *, amount: float) -> None:
        """Append a timestamped charge to the ledger.

        Args:
            amount: The charge in the ledger's unit (dollars or tokens). Stamped at the
                clock's current instant so it counts against the rolling window.
        """
        stamp = self._clock.now().isoformat()
        async with self._connect() as db:
            await db.execute(_SCHEMA)
            await db.execute(
                "INSERT INTO spend_entries (spent_at, amount) VALUES (?, ?)",
                (stamp, amount),
            )
            await db.commit()

    async def trailing_24h_spend(self) -> float:
        """Return the total charged in the trailing 24h from the clock's current instant.

        Charges older than 24h are excluded — the window rolls forward with the clock, so
        spend frees up as old charges age out.
        """
        cutoff = (self._clock.now() - _WINDOW).isoformat()
        async with self._connect() as db:
            await db.execute(_SCHEMA)
            async with db.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM spend_entries WHERE spent_at > ?",
                (cutoff,),
            ) as cursor:
                row = await cursor.fetchone()
        return float(row[0]) if row is not None else 0.0

    async def would_exceed(self, *, amount: float) -> bool:
        """Whether charging ``amount`` now would push the trailing-24h spend over the cap.

        Args:
            amount: The prospective charge in the ledger's unit.

        Returns:
            True when ``trailing_24h_spend + amount`` exceeds :meth:`cap`.
        """
        return (await self.trailing_24h_spend()) + amount > self.cap()

    async def window_frees_at(self) -> datetime | None:
        """Return when the oldest in-window charge ages out, freeing room.

        Returns:
            The instant the oldest trailing-24h charge falls out of the window (its
            timestamp + 24h), or ``None`` when no charge is currently in the window.
        """
        cutoff = (self._clock.now() - _WINDOW).isoformat()
        async with self._connect() as db:
            await db.execute(_SCHEMA)
            async with db.execute(
                "SELECT MIN(spent_at) FROM spend_entries WHERE spent_at > ?",
                (cutoff,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None or row[0] is None:
            return None
        return datetime.fromisoformat(row[0]) + _WINDOW

    def _connect(self) -> aiosqlite.Connection:
        """Open a fresh DB connection, ensuring the parent dir exists first."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        return aiosqlite.connect(self._db_path)


@dataclass(frozen=True)
class GateDecision:
    """The run-start gate verdict.

    Attributes:
        deferred: True when the run is held back because its estimated charge would start
            it over the cap.
        defer_until: When the window frees enough to admit the run; ``None`` when the run
            was admitted.
    """

    deferred: bool
    defer_until: datetime | None


@dataclass(frozen=True)
class MeterDecision:
    """The mid-run meter verdict.

    Attributes:
        paused: True when the charge would cross the cap, so the run was paused and
            checkpointed.
        resume_at: When the window frees enough to resume; ``None`` when the charge was
            metered through without a pause.
    """

    paused: bool
    resume_at: datetime | None


class BudgetGovernor:
    """Gates a run at start and meters it mid-flight against the rolling-24h cap.

    Wraps a :class:`BudgetLedger`. :meth:`gate` defers a run that would start over the
    cap. :meth:`meter` pauses and checkpoints a run whose next charge would cross the cap,
    recording the owned slice set in the reconcile :class:`~retinue.reconcile.RunStateStore`
    so :meth:`try_resume` can rebuild only the unfinished work via
    :func:`~retinue.reconcile.reconcile_run` once the window frees.

    Args:
        ledger: The rolling-24h spend ledger to enforce against.
        run_state: The reconcile run-state store the pause checkpoints into; defaults to a
            store on the ledger's DB file so the whole budget state is one service-level
            store.
    """

    def __init__(
        self, ledger: BudgetLedger, *, run_state: RunStateStore | None = None
    ) -> None:
        self._ledger = ledger
        self._run_state = run_state or RunStateStore(ledger._db_path)
        self._db_path = ledger._db_path
        # In-process cache of each paused PRD's blocked_by graph. The reconcile
        # RunStateStore persists only slice numbers, so the dependency edges are held
        # here for a same-process resume; a cross-restart resume falls back to an empty
        # graph (reconcile still resumes the unfinished slices correctly).
        self._blocked_by: dict[str, dict[int, list[int]]] = {}

    async def gate(self, *, estimated_amount: float) -> GateDecision:
        """Admit or defer a run by its estimated charge against the rolling-24h cap.

        Args:
            estimated_amount: The run's estimated total charge in the ledger's unit.

        Returns:
            A :class:`GateDecision`: ``deferred`` with a ``defer_until`` when the charge
            would start the run over the cap, else an admitted decision.
        """
        if await self._ledger.would_exceed(amount=estimated_amount):
            defer_until = await self._ledger.window_frees_at()
            logger.info(
                "Budget gate deferring run: estimated %.4g would exceed the 24h cap "
                "(%.4g); defer until %s",
                estimated_amount,
                self._ledger.cap(),
                defer_until,
            )
            return GateDecision(deferred=True, defer_until=defer_until)
        return GateDecision(deferred=False, defer_until=None)

    async def meter(
        self,
        *,
        repo_full_name: str,
        prd_number: int,
        amount: float,
        slices: list[PrdSlice],
    ) -> MeterDecision:
        """Meter a mid-run charge; pause + checkpoint if it would cross the cap.

        A charge that fits under the cap is recorded and the run continues. A charge that
        would cross the cap is *not* recorded; instead the run is checkpointed (its owned
        slice set saved to the run-state store) and a paused decision is returned, to be
        picked up later by :meth:`try_resume`.

        Args:
            repo_full_name: The target repo, e.g. "owner/repo".
            prd_number: The PRD round's tracking issue number.
            amount: The prospective charge in the ledger's unit.
            slices: The PRD round's slices, checkpointed so the resume rebuilds only the
                unfinished ones.

        Returns:
            A :class:`MeterDecision`: ``paused`` with a ``resume_at`` when the charge
            would cross the cap, else a metered-through decision.
        """
        if not await self._ledger.would_exceed(amount=amount):
            await self._ledger.record_spend(amount=amount)
            return MeterDecision(paused=False, resume_at=None)

        resume_at = await self._ledger.window_frees_at()
        await self._checkpoint(
            repo_full_name=repo_full_name,
            prd_number=prd_number,
            slices=slices,
            resume_at=resume_at,
        )
        logger.info(
            "Budget meter pausing PRD #%d (%s): charge %.4g would cross the 24h cap "
            "(%.4g); checkpointed, resume at %s",
            prd_number,
            repo_full_name,
            amount,
            self._ledger.cap(),
            resume_at,
        )
        return MeterDecision(paused=True, resume_at=resume_at)

    async def try_resume(
        self,
        *,
        repo_full_name: str,
        prd_number: int,
        gh: ReconcileGh,
    ) -> ReconcileResult | None:
        """Resume a budget-paused run once the window frees, reusing reconcile.

        Returns ``None`` while the run is still paused before its ``resume_at`` (the
        window has not freed). Once the clock passes ``resume_at`` the pause is cleared and
        the checkpointed slice set is reconciled against GitHub via
        :func:`~retinue.reconcile.reconcile_run`, so only the unfinished slices are rebuilt
        — no duplicate issue, branch, or PR.

        Args:
            repo_full_name: The target repo, e.g. "owner/repo".
            prd_number: The paused PRD round's tracking issue number.
            gh: The reconcile gh seam GitHub truth is read through.

        Returns:
            A :class:`~retinue.reconcile.ReconcileResult` to route the resumed run on, or
            ``None`` when the run is not paused or the window has not freed yet.
        """
        resume_at = await self._resume_at(repo_full_name, prd_number)
        if resume_at is None:
            return None
        if self._ledger._clock.now() < resume_at:
            return None

        slice_numbers = await self._run_state.slices_of(
            repo_full_name=repo_full_name, prd_number=prd_number
        )
        slices = await self._checkpointed_slices(
            repo_full_name, prd_number, slice_numbers
        )
        await self._clear_pause(repo_full_name, prd_number)
        logger.info(
            "Budget resume of PRD #%d (%s): window freed, reconciling %d slices",
            prd_number,
            repo_full_name,
            len(slices),
        )
        return await reconcile_run(
            repo_full_name=repo_full_name,
            prd_number=prd_number,
            slices=slices,
            gh=gh,
        )

    async def is_paused(self, *, repo_full_name: str, prd_number: int) -> bool:
        """Whether a PRD round is currently budget-paused (checkpointed, not yet resumed)."""
        return await self._resume_at(repo_full_name, prd_number) is not None

    async def _checkpoint(
        self,
        *,
        repo_full_name: str,
        prd_number: int,
        slices: list[PrdSlice],
        resume_at: datetime | None,
    ) -> None:
        """Persist the paused run's slice set and resume time for a later resume."""
        await self._run_state.record_slices(
            repo_full_name=repo_full_name,
            prd_number=prd_number,
            issue_numbers=[s.issue_number for s in slices],
        )
        self._checkpoint_blocked_by(repo_full_name, prd_number, slices)
        key = run_state_key(repo_full_name, prd_number)
        stamp = (resume_at or self._ledger._clock.now()).isoformat()
        async with self._connect() as db:
            await db.execute(_PAUSE_SCHEMA)
            await db.execute(
                """
                INSERT INTO budget_pauses (prd_key, resume_at) VALUES (?, ?)
                ON CONFLICT(prd_key) DO UPDATE SET resume_at = excluded.resume_at
                """,
                (key, stamp),
            )
            await db.commit()

    def _checkpoint_blocked_by(
        self, repo_full_name: str, prd_number: int, slices: list[PrdSlice]
    ) -> None:
        """Stash the slices' blocked_by graph so the resume preserves dependency order.

        The reconcile :class:`RunStateStore` persists only the slice issue numbers; the
        ``blocked_by`` edges are held in memory keyed by PRD so a resume in the same
        process rebuilds full :class:`PrdSlice` objects. A cross-restart resume falls back
        to an empty graph (reconcile still resumes the unfinished slices correctly).
        """
        self._blocked_by[run_state_key(repo_full_name, prd_number)] = {
            s.issue_number: list(s.blocked_by) for s in slices
        }

    async def _checkpointed_slices(
        self, repo_full_name: str, prd_number: int, slice_numbers: list[int]
    ) -> list[PrdSlice]:
        """Rebuild :class:`PrdSlice` objects from the checkpointed numbers + blocked_by."""
        graph = self._blocked_by.get(run_state_key(repo_full_name, prd_number), {})
        return [
            PrdSlice(
                repo_full_name=repo_full_name,
                issue_number=number,
                prd_number=prd_number,
                blocked_by=graph.get(number, []),
            )
            for number in slice_numbers
        ]

    async def _resume_at(self, repo_full_name: str, prd_number: int) -> datetime | None:
        """Return the recorded resume time for a paused PRD, or None when not paused."""
        key = run_state_key(repo_full_name, prd_number)
        async with self._connect() as db:
            await db.execute(_PAUSE_SCHEMA)
            async with db.execute(
                "SELECT resume_at FROM budget_pauses WHERE prd_key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
        return datetime.fromisoformat(row[0]) if row is not None else None

    async def _clear_pause(self, repo_full_name: str, prd_number: int) -> None:
        """Remove the pause record once a run resumes."""
        key = run_state_key(repo_full_name, prd_number)
        async with self._connect() as db:
            await db.execute(_PAUSE_SCHEMA)
            await db.execute("DELETE FROM budget_pauses WHERE prd_key = ?", (key,))
            await db.commit()

    def _connect(self) -> aiosqlite.Connection:
        """Open a fresh DB connection, ensuring the parent dir exists first."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        return aiosqlite.connect(self._db_path)
