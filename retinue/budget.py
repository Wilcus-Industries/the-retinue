"""Budget governor: a rolling-24h spend ledger (issue #14).

The retinue meters agent token spend against a service-level weekly budget and enforces
a per-rolling-24h-window ceiling (``cap`` = a fraction, by default 12%, of the weekly
budget). The budget is shared across all lanes — the orchestrator's :func:`build_prd`
run, the ad-hoc lane, and the cron lane — so the ledger is a single service-level SQLite
store, mirroring the durable-SQLite style of
:class:`retinue.impl_retry.ImplRetryStore`.

Enforcement happens at admission (:meth:`BudgetGovernor.gate` for PRD/cron runs,
:meth:`BudgetGovernor.meter_adhoc` for ad-hoc builds): if the charge would push the
trailing-24h spend over the cap, the work is *deferred/declined* until the window frees
enough room (charges expire oldest-first). An admitted charge is recorded atomically
with the check (:meth:`BudgetLedger.try_record_if_within_cap`, a ``BEGIN IMMEDIATE``
transaction), so two lanes admitting concurrently against the shared ledger serialize on
the write lock and the second sees the first's charge before deciding — the cap can't be
overshot.

Auth-aware metering: an API key meters dollars against a weekly-$ budget; subscription
OAuth meters tokens against a weekly-token budget. The math is identical; only the unit
and the weekly budget differ, both carried on the ledger.

The clock is injected (:class:`Clock`) so the rolling-24h window is deterministic in
tests rather than tied to wall-clock, and the ledger lives in a SQLite file, so the
whole flow runs with no network.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import aiosqlite

from retinue.db import connect_daemon

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
        # One long-lived connection per store instance, opened lazily and guarded by a
        # per-store lock so the check-and-record transaction stays atomic under concurrent
        # asyncio tasks (a shared connection would interleave statements mid-transaction).
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def metering_disabled(self) -> bool:
        """Whether spend metering is off because the weekly budget is the 0.0 sentinel.

        The default ``weekly_budget == 0.0`` means an operator never set a budget; rather
        than cap every build at a zero ceiling (declining all work forever), metering is
        disabled and callers admit everything uncharged. Only the *exact* 0.0 sentinel
        disables — a positive budget whose fractional cap rounds to zero still enforces.
        """
        return self._weekly_budget == 0.0

    def cap(self) -> float:
        """Return the rolling-24h ceiling: ``daily_cap_fraction`` of the weekly budget."""
        return self._weekly_budget * self._daily_cap_fraction

    async def _conn(self) -> aiosqlite.Connection:
        """Return the store's long-lived connection, connecting + tuning it once.

        Lazily opens the connection (creating the parent dir), enables WAL with
        ``synchronous=NORMAL`` and the busy-timeout, and ensures the schema — all once per
        instance. ``isolation_level=None`` puts the connection in autocommit so the
        explicit ``BEGIN IMMEDIATE`` in :meth:`try_record_if_within_cap` is the only
        transaction control, never colliding with sqlite3's implicit transactions. Callers
        must hold :attr:`_lock` around this and the statements that follow.
        """
        if self._db is None:
            self._db = await connect_daemon(self._db_path, isolation_level=None)
            await self._db.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA synchronous=NORMAL")
            await self._db.execute(_SCHEMA)
        return self._db

    async def close(self) -> None:
        """Close the store's long-lived connection if opened; a later call reconnects."""
        async with self._lock:
            if self._db is not None:
                await self._db.close()
                self._db = None

    async def record_spend(self, *, amount: float) -> None:
        """Append a timestamped charge to the ledger.

        Args:
            amount: The charge in the ledger's unit (dollars or tokens). Stamped at the
                clock's current instant so it counts against the rolling window.
        """
        stamp = self._clock.now().isoformat()
        async with self._lock:
            db = await self._conn()
            await db.execute(
                "INSERT INTO spend_entries (spent_at, amount) VALUES (?, ?)",
                (stamp, amount),
            )

    async def trailing_24h_spend(self) -> float:
        """Return the total charged in the trailing 24h from the clock's current instant.

        Charges older than 24h are excluded — the window rolls forward with the clock, so
        spend frees up as old charges age out.
        """
        async with self._lock:
            db = await self._conn()
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
        async with self._lock:
            db = await self._conn()
            # BEGIN IMMEDIATE takes the write lock up front, so the re-read below reflects
            # any concurrent writer (a separate store on the same file) that committed
            # first; the per-store lock above keeps this instance's own tasks from
            # interleaving statements inside the transaction.
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
        async with self._lock:
            db = await self._conn()
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


class BudgetGovernor:
    """Gates and meters agent spend against the rolling-24h cap.

    Wraps a :class:`BudgetLedger`. :meth:`gate` defers a run that would start over the
    cap and charges an admitted run's estimate to the shared ledger; :meth:`meter_adhoc`
    admits-or-declines one ad-hoc build's flat charge; :meth:`record_charge` records a
    cheap after-the-fact side charge.

    Args:
        ledger: The rolling-24h spend ledger to enforce against.
    """

    def __init__(self, ledger: BudgetLedger) -> None:
        self._ledger = ledger
        if ledger.metering_disabled:
            logger.warning(
                "budget disabled — set WEEKLY_BUDGET to enforce spend caps"
            )

    async def close(self) -> None:
        """Release the wrapped ledger's connection."""
        await self._ledger.close()

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
        if self._ledger.metering_disabled:
            return GateDecision(deferred=False, defer_until=None)
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

    async def record_charge(self, *, amount: float) -> None:
        """Record a side charge (e.g. one classifier call) on the shared ledger.

        Unlike :meth:`gate`/:meth:`meter` this only records — the agent call has already
        happened and is cheap, so it is metered after the fact, not gated. A disabled
        budget records nothing.

        Args:
            amount: The charge in the ledger's unit (dollars or tokens).
        """
        if self._ledger.metering_disabled:
            return
        await self._ledger.record_spend(amount=amount)

    async def meter_adhoc(self, *, amount: float) -> bool:
        """Meter one ad-hoc build's flat charge against the shared rolling-24h cap.

        The ad-hoc lane charges the *same* service-level ledger the PRD lane gates
        against, atomically. A charge that still fits under the cap is recorded and the
        build is admitted; one that would cross it records nothing and the build is
        declined. The check-and-record is the identical atomic primitive the PRD
        :meth:`gate` uses (:meth:`BudgetLedger.try_record_if_within_cap`), so an ad-hoc
        build and a PRD gate racing on the shared ledger serialize on the write lock and
        the cap is never overshot.

        Args:
            amount: The build's charge in the ledger's unit (dollars or tokens).

        Returns:
            True when the charge fit under the cap and was recorded (build admitted);
            False when the shared window leaves no room (build declined, nothing written).
        """
        if self._ledger.metering_disabled:
            return True
        return await self._ledger.try_record_if_within_cap(amount=amount)


# --- the lanes' estimated charges -----------------------------------------------------
#
# Every lane meters the one shared ledger with a flat, conservative estimate; the
# constants live together here so the cross-lane relationships stay visible.

# The orchestrator build's estimated charge, gated against the rolling-24h budget cap.
# The build's true cost is only known after the implementer/done-check runs, so the gate
# uses a conservative fixed estimate; the meter (the governor's mid-run pause/resume)
# tracks the real spend once the run is underway. Kept here (not a Settings field) so the
# public config schema is unchanged.
BUILD_ESTIMATED_AMOUNT = 1.0

# The estimated charge one classifier call meters against the shared ledger. Kept small
# (a Haiku-class call) and separate from the build gate estimate, which is unchanged.
CLASSIFIER_ESTIMATED_AMOUNT = 0.01

# The flat per-build charge the ad-hoc drain meters against the shared rolling-24h cap,
# matching the PRD lane's estimate (:data:`BUILD_ESTIMATED_AMOUNT`); a
# build that would cross the cap is skipped so the one shared budget is never overshot.
ADHOC_DRAIN_ESTIMATED_AMOUNT = 1.0

# The cron backlog tick's estimated charge, gated against the rolling-24h budget cap. The
# tick's own work is pure label surgery (swapping ``backlog`` for the trigger label) with
# no model spend, so it charges nothing here; the scheduler drain meters the *real* build
# charge (:data:`ADHOC_DRAIN_ESTIMATED_AMOUNT`) separately once it later drains the
# promoted issue. Charging a nonzero amount here would double-bill every promoted issue
# and prematurely trip the rolling-24h defer on phantom spend.
CRON_PROMOTION_ESTIMATED_AMOUNT = 0.0

