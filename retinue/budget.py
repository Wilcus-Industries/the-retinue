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
  frees enough room for that estimate (the charges expire oldest-first until the trailing
  spend leaves room for the estimate under the cap). An admitted run's estimate is
  *charged* to the ledger atomically with the check, so the PRD and cron lanes' spend
  counts against the shared window.
* **meter mid-run** (:meth:`BudgetGovernor.meter`) — a charge that would cross the cap
  *pauses* the run and checkpoints it (the owned slice set is recorded in the reconcile
  :class:`~retinue.reconcile.RunStateStore`). :meth:`BudgetGovernor.try_resume` resumes
  it once the window frees by reusing the reconciliation machinery
  (:func:`~retinue.reconcile.reconcile_run`), so only the unfinished slices rebuild — no
  duplicate issue, branch, or PR. The check-and-record is atomic
  (:meth:`BudgetLedger.try_record_if_within_cap`, a ``BEGIN IMMEDIATE`` transaction), so
  two lanes metering concurrently against the shared ledger serialize on the write lock
  and the second sees the first's charge before deciding — the cap can't be overshot.

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

# How long a writer waits for the ledger's write lock before giving up. The check-and-
# record path opens its transaction with BEGIN IMMEDIATE, so a second concurrent writer
# blocks here until the first commits, then re-reads the updated trailing total. Generous
# enough that lanes serialize rather than spuriously error under contention.
_BUSY_TIMEOUT_MS = 30_000

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
    resume_at TEXT NOT NULL,
    amount    REAL NOT NULL DEFAULT 0
)
"""

# The ledger is a durable, service-level store: a DB file created before issue #21 holds
# the original two-column budget_pauses table (prd_key, resume_at). CREATE TABLE IF NOT
# EXISTS is a no-op against it, so the #21 ``amount`` column never lands and every read
# that selects it raises. Detecting the missing column and ALTER-ing it in lets the same
# schema-ensure path heal both fresh and legacy DBs (issue #23).
_PAUSE_AMOUNT_MIGRATION = (
    "ALTER TABLE budget_pauses ADD COLUMN amount REAL NOT NULL DEFAULT 0"
)


async def _ensure_pause_schema(db: aiosqlite.Connection) -> None:
    """Ensure budget_pauses exists and carries the ``amount`` column, migrating in place.

    Creates the table when absent, then adds the issue-#21 ``amount`` column to a legacy
    two-column table that predates it. Idempotent: ``CREATE TABLE IF NOT EXISTS`` plus a
    PRAGMA-guarded ALTER, so re-running against an already-migrated DB is a no-op. Must be
    called outside any started transaction (this issues DDL) so it never deadlocks the
    ``BEGIN IMMEDIATE`` check-and-record path (issue #22).
    """
    await db.execute(_PAUSE_SCHEMA)
    async with db.execute("PRAGMA table_info(budget_pauses)") as cursor:
        columns = {row[1] for row in await cursor.fetchall()}
    if "amount" not in columns:
        await db.execute(_PAUSE_AMOUNT_MIGRATION)
        await db.commit()


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
        async with self._connect() as db:
            await db.execute(_SCHEMA)
            return await self._trailing_within(db)

    async def _trailing_within(self, db: aiosqlite.Connection) -> float:
        """Sum the trailing-24h spend using an already-open connection.

        Shared by :meth:`trailing_24h_spend` and the atomic check-and-record path so the
        in-transaction re-read uses the identical window query.
        """
        cutoff = (self._clock.now() - _WINDOW).isoformat()
        async with db.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM spend_entries WHERE spent_at > ?",
            (cutoff,),
        ) as cursor:
            row = await cursor.fetchone()
        return float(row[0]) if row is not None else 0.0

    async def try_record_if_within_cap(self, *, amount: float) -> bool:
        """Atomically record ``amount`` iff it still fits under the cap; return success.

        Check-and-record is performed inside one ``BEGIN IMMEDIATE`` transaction so a
        second concurrent writer serializes behind the first: it blocks on the write lock,
        then re-reads the (now updated) trailing-24h total before deciding. This closes the
        time-of-check-to-time-of-use gap between :meth:`would_exceed` and
        :meth:`record_spend` — two lanes that would jointly cross the cap can never both
        record. The charge is stamped at the clock's current instant, like
        :meth:`record_spend`.

        Args:
            amount: The prospective charge in the ledger's unit.

        Returns:
            True when the charge fit under the cap and was recorded; False when the live
            trailing total (re-read under the write lock) leaves no room, in which case
            nothing is written.
        """
        stamp = self._clock.now().isoformat()
        async with self._connect() as db:
            await db.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            await db.execute(_SCHEMA)
            # BEGIN IMMEDIATE takes the write lock up front, so the re-read below reflects
            # any concurrent writer that committed first; aiosqlite's autocommit must be
            # off for the explicit transaction to span the read and the insert.
            await db.execute("BEGIN IMMEDIATE")
            try:
                if (await self._trailing_within(db)) + amount > self.cap():
                    await db.rollback()
                    return False
                await db.execute(
                    "INSERT INTO spend_entries (spent_at, amount) VALUES (?, ?)",
                    (stamp, amount),
                )
                await db.commit()
            except BaseException:
                await db.rollback()
                raise
        return True

    async def would_exceed(self, *, amount: float) -> bool:
        """Whether charging ``amount`` now would push the trailing-24h spend over the cap.

        Args:
            amount: The prospective charge in the ledger's unit.

        Returns:
            True when ``trailing_24h_spend + amount`` exceeds :meth:`cap`.
        """
        return (await self.trailing_24h_spend()) + amount > self.cap()

    async def window_frees_at(self, *, amount: float) -> datetime | None:
        """Return when the window frees enough room to admit ``amount`` under the cap.

        Walks the in-window charges oldest-first, accumulating the spend each one frees as
        it ages out, and returns the ``spent_at + 24h`` of the charge whose expiry first
        brings ``trailing_24h_spend - freed + amount`` to within the cap (i.e. the instant
        :meth:`would_exceed` for ``amount`` flips to False). This is later than the oldest
        charge's expiry whenever the oldest charge alone does not free enough room.

        Args:
            amount: The prospective charge the window must make room for.

        Returns:
            The instant the window frees enough for ``amount``, or ``None`` when no charge
            is in the window (nothing to wait for).
        """
        cutoff = (self._clock.now() - _WINDOW).isoformat()
        async with self._connect() as db:
            await db.execute(_SCHEMA)
            async with db.execute(
                "SELECT spent_at, amount FROM spend_entries "
                "WHERE spent_at > ? ORDER BY spent_at ASC",
                (cutoff,),
            ) as cursor:
                charges = await cursor.fetchall()
        if not charges:
            return None
        trailing = sum(float(charge) for _, charge in charges)
        freed = 0.0
        last_spent_at = ""
        for spent_at, charge in charges:
            last_spent_at = spent_at
            freed += float(charge)
            if trailing - freed + amount <= self.cap():
                return datetime.fromisoformat(spent_at) + _WINDOW
        # Even with every in-window charge aged out, ``amount`` alone exceeds the cap; the
        # last charge to expire is the earliest the window is as empty as it can get.
        return datetime.fromisoformat(last_spent_at) + _WINDOW

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
    cap and charges an admitted run's estimate to the shared ledger. :meth:`meter`
    pauses and checkpoints a run whose next charge would cross the cap,
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
        """Admit-and-charge or defer a run by its estimated charge against the 24h cap.

        An admitted run's estimate is *recorded* to the shared ledger inside the same
        write-locked transaction that verified the room
        (:meth:`BudgetLedger.try_record_if_within_cap`), so the PRD and cron lanes'
        spend counts against the rolling window — and two lanes gating concurrently
        serialize on the write lock and can never jointly overshoot the cap. A deferred
        run records nothing.

        Args:
            estimated_amount: The run's estimated total charge in the ledger's unit.
                Only call once the run will actually build on admission — the charge
                lands at the gate.

        Returns:
            A :class:`GateDecision`: ``deferred`` with a ``defer_until`` when the charge
            would start the run over the cap (nothing recorded), else an admitted
            decision with the estimate charged.
        """
        if await self._ledger.try_record_if_within_cap(amount=estimated_amount):
            return GateDecision(deferred=False, defer_until=None)
        defer_until = await self._ledger.window_frees_at(amount=estimated_amount)
        logger.info(
            "Budget gate deferring run: estimated %.4g would exceed the 24h cap "
            "(%.4g); defer until %s",
            estimated_amount,
            self._ledger.cap(),
            defer_until,
        )
        return GateDecision(deferred=True, defer_until=defer_until)

    async def meter_adhoc(self, *, amount: float) -> bool:
        """Meter one ad-hoc build's flat charge against the shared rolling-24h cap.

        The ad-hoc lane has no PRD slice set to checkpoint, so it never pauses+resumes
        the PRD way; it just charges the *same* service-level ledger the PRD lane meters,
        atomically. A charge that still fits under the cap is recorded and the build is
        admitted; one that would cross it records nothing and the build is declined. The
        check-and-record is the identical atomic primitive the PRD :meth:`meter` uses
        (:meth:`BudgetLedger.try_record_if_within_cap`), so an ad-hoc build and a PRD meter
        racing on the shared ledger serialize on the write lock and the cap is never
        overshot.

        Args:
            amount: The build's charge in the ledger's unit (dollars or tokens).

        Returns:
            True when the charge fit under the cap and was recorded (build admitted);
            False when the shared window leaves no room (build declined, nothing written).
        """
        return await self._ledger.try_record_if_within_cap(amount=amount)

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
        # Atomic check-and-record: a second lane metering concurrently serializes behind
        # this one on the ledger's write lock and re-reads the updated trailing total, so
        # two charges that would jointly cross the cap can never both record (issue #22).
        if await self._ledger.try_record_if_within_cap(amount=amount):
            return MeterDecision(paused=False, resume_at=None)

        resume_at = await self._ledger.window_frees_at(amount=amount)
        await self._checkpoint(
            repo_full_name=repo_full_name,
            prd_number=prd_number,
            slices=slices,
            resume_at=resume_at,
            amount=amount,
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
        window has not freed). It also re-verifies the cap once the clock passes
        ``resume_at``: if the window still has not freed enough for the paused charge
        (:meth:`BudgetLedger.would_exceed` is still True), it returns ``None`` and leaves
        the pause record intact rather than resuming over-cap. Only when the cap genuinely
        has room is the pause cleared and the checkpointed slice set reconciled against
        GitHub via :func:`~retinue.reconcile.reconcile_run`, so only the unfinished slices
        are rebuilt — no duplicate issue, branch, or PR.

        Args:
            repo_full_name: The target repo, e.g. "owner/repo".
            prd_number: The paused PRD round's tracking issue number.
            gh: The reconcile gh seam GitHub truth is read through.

        Returns:
            A :class:`~retinue.reconcile.ReconcileResult` to route the resumed run on, or
            ``None`` when the run is not paused or the window has not freed enough yet.
        """
        pause = await self._pause_record(repo_full_name, prd_number)
        if pause is None:
            return None
        resume_at, amount = pause
        if self._ledger._clock.now() < resume_at:
            return None
        # Re-verify the cap: a too-early resume_at (or any other in-window spend since the
        # pause) could leave the window still over-cap. Stay paused rather than resume into
        # an over-budget state.
        if await self._ledger.would_exceed(amount=self._recheck_charge(amount)):
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
        amount: float,
    ) -> None:
        """Persist the paused run's slice set, resume time, and charge for a later resume.

        The ``amount`` is stored so :meth:`try_resume` can re-verify the cap against the
        paused charge before clearing the pause, never resuming into an over-cap window.
        """
        await self._run_state.record_slices(
            repo_full_name=repo_full_name,
            prd_number=prd_number,
            issue_numbers=[s.issue_number for s in slices],
        )
        self._checkpoint_blocked_by(repo_full_name, prd_number, slices)
        key = run_state_key(repo_full_name, prd_number)
        stamp = (resume_at or self._ledger._clock.now()).isoformat()
        async with self._connect() as db:
            await _ensure_pause_schema(db)
            await db.execute(
                """
                INSERT INTO budget_pauses (prd_key, resume_at, amount) VALUES (?, ?, ?)
                ON CONFLICT(prd_key) DO UPDATE SET
                    resume_at = excluded.resume_at, amount = excluded.amount
                """,
                (key, stamp, amount),
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

    def _recheck_charge(self, amount: float) -> float:
        """The charge :meth:`try_resume` re-verifies the cap against before resuming.

        A pause recorded by issue #21 onward carries the real paused charge. A legacy row
        migrated in by issue #23 has no recorded charge (its ``amount`` defaults to 0 from
        the migration's column DEFAULT); trusting that 0 would let ``would_exceed`` pass
        the instant trailing merely dips to the cap, reintroducing the over-cap resume #21
        fixed. So a non-positive (legacy/unknown) charge is treated conservatively as a
        full ``cap()`` — the pause holds until the window has genuine headroom for a whole
        cap's worth, i.e. is essentially empty, rather than resuming over-budget.
        """
        return amount if amount > 0 else self._ledger.cap()

    async def _resume_at(self, repo_full_name: str, prd_number: int) -> datetime | None:
        """Return the recorded resume time for a paused PRD, or None when not paused."""
        pause = await self._pause_record(repo_full_name, prd_number)
        return pause[0] if pause is not None else None

    async def _pause_record(
        self, repo_full_name: str, prd_number: int
    ) -> tuple[datetime, float] | None:
        """Return the paused PRD's (resume time, charge), or None when not paused."""
        key = run_state_key(repo_full_name, prd_number)
        async with self._connect() as db:
            await _ensure_pause_schema(db)
            async with db.execute(
                "SELECT resume_at, amount FROM budget_pauses WHERE prd_key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return datetime.fromisoformat(row[0]), float(row[1])

    async def _clear_pause(self, repo_full_name: str, prd_number: int) -> None:
        """Remove the pause record once a run resumes."""
        key = run_state_key(repo_full_name, prd_number)
        async with self._connect() as db:
            await _ensure_pause_schema(db)
            await db.execute("DELETE FROM budget_pauses WHERE prd_key = ?", (key,))
            await db.commit()

    def _connect(self) -> aiosqlite.Connection:
        """Open a fresh DB connection, ensuring the parent dir exists first."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        return aiosqlite.connect(self._db_path)
