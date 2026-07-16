"""Tests for the budget governor: the rolling-24h spend ledger (issue #14).

A DB-backed rolling-24h spend ledger meters agent spend and enforces a 12%/24h cap
against the service-level weekly budget, in both auth modes ($ for an API key, tokens
for subscription OAuth). The governor gates at admission (over the cap -> defer/decline)
for the PRD, cron, and ad-hoc lanes alike.

The clock is injected (no real wall-clock, so the rolling window is deterministic) and
the ledger lives in a temp SQLite file — no real ``gh``, no network.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from retinue.budget import (
    AuthMode,
    BudgetGovernor,
    BudgetLedger,
    Clock,
    GateDecision,
    SystemClock,
)
from tests.fakes import FakeClock

# The stores hold a long-lived connection for their (process-lifetime) lifespan and do
# not require callers to close them. These tests construct many short-lived stores across
# per-test event loops without closing, so a store GC'd after its loop shuts down lets
# aiosqlite's worker thread touch the closed loop — a benign teardown-only warning that
# does not occur for the single process-lifetime governor in the worker.
pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnhandledThreadExceptionWarning"
)

_T0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


# --- real SystemClock: the production adapter behind the Clock seam --------------


def test_system_clock_satisfies_the_clock_protocol() -> None:
    """The production adapter structurally implements the :class:`Clock` seam.

    The ledger reads time through the single-method ``now()`` protocol; binding the real
    impl to a ``Clock`` annotation and calling it proves it satisfies that contract (the
    Protocol is not ``@runtime_checkable``, so this is a structural, not ``isinstance``,
    check).
    """
    clock: Clock = SystemClock()
    assert callable(clock.now)
    assert isinstance(clock.now(), datetime)


def test_system_clock_now_is_timezone_aware_utc() -> None:
    """``now()`` returns a tz-aware UTC instant, so the window arithmetic is unambiguous.

    The injected ``FakeClock`` returns UTC-aware datetimes; the real impl must honour the
    identical contract or the ledger's ``spent_at`` comparisons mix naive and aware
    datetimes and raise. Pin both the tzinfo and the actual UTC offset.
    """
    now = SystemClock().now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_system_clock_tracks_wall_time_and_advances_monotonically() -> None:
    """``now()`` reads the real wall clock and never goes backwards between reads."""
    before = datetime.now(UTC)
    first = SystemClock().now()
    second = SystemClock().now()
    after = datetime.now(UTC)
    # Bracketed by two independent wall-clock reads, so it is the real clock, not a stub.
    assert before <= first <= second <= after


@pytest.mark.asyncio
async def test_ledger_window_math_is_unchanged_under_the_real_clock(
    db_path: Path,
) -> None:
    """Driven by the real :class:`SystemClock`, the trailing-24h window math is identical.

    The fake clock is only a deterministic stand-in for this same ``now()`` protocol: a
    charge recorded now sits inside the trailing 24h, and a cutoff just past it excludes
    it. Proving this against the wall clock confirms the real adapter changes none of the
    ledger's window arithmetic.
    """
    ledger = BudgetLedger(
        db_path, clock=SystemClock(), auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    await ledger.record_spend(amount=7.0)

    # A charge stamped at wall-clock now is inside the trailing 24h window.
    assert await ledger.trailing_24h_spend() == pytest.approx(7.0)
    # And it still gates against the cap exactly as the fake-clock cases do.
    assert await ledger.would_exceed(amount=4.0) is False
    assert await ledger.would_exceed(amount=6.0) is True


# --- rolling-24h window math -----------------------------------------------------


@pytest.mark.asyncio
async def test_trailing_24h_spend_sums_only_the_last_24h(db_path: Path) -> None:
    """The rolling sum counts charges in the trailing 24h and drops older ones."""
    clock = FakeClock()
    ledger = BudgetLedger(db_path, clock=clock, auth_mode=AuthMode.API_KEY)

    await ledger.record_spend(amount=10.0)
    clock.advance(timedelta(hours=23))
    await ledger.record_spend(amount=5.0)
    # Both charges are inside the trailing 24h from now.
    assert await ledger.trailing_24h_spend() == pytest.approx(15.0)

    # Move forward so the first charge (now 25h old) falls out of the window.
    clock.advance(timedelta(hours=2))
    assert await ledger.trailing_24h_spend() == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_cap_is_twelve_percent_of_weekly_budget_dollars(db_path: Path) -> None:
    """In API-key mode the cap is 12% of the weekly-$ budget."""
    ledger = BudgetLedger(
        db_path, clock=FakeClock(), auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    assert ledger.cap() == pytest.approx(12.0)


@pytest.mark.asyncio
async def test_cap_is_twelve_percent_of_weekly_budget_tokens(db_path: Path) -> None:
    """In subscription-OAuth mode the cap is 12% of the weekly-token budget."""
    ledger = BudgetLedger(
        db_path,
        clock=FakeClock(),
        auth_mode=AuthMode.SUBSCRIPTION,
        weekly_budget=1_000_000.0,
    )
    assert ledger.cap() == pytest.approx(120_000.0)


@pytest.mark.asyncio
async def test_would_exceed_true_only_when_charge_crosses_cap(db_path: Path) -> None:
    """``would_exceed`` is true exactly when trailing-24h + charge passes the cap."""
    ledger = BudgetLedger(
        db_path, clock=FakeClock(), auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    await ledger.record_spend(amount=10.0)  # cap is 12.0
    assert await ledger.would_exceed(amount=1.0) is False
    assert await ledger.would_exceed(amount=2.5) is True


@pytest.mark.asyncio
async def test_window_math_identical_in_token_mode(db_path: Path) -> None:
    """The same rolling math gates token spend against the weekly-token budget."""
    clock = FakeClock()
    ledger = BudgetLedger(
        db_path,
        clock=clock,
        auth_mode=AuthMode.SUBSCRIPTION,
        weekly_budget=1_000_000.0,
    )
    await ledger.record_spend(amount=110_000.0)  # cap is 120_000
    assert await ledger.would_exceed(amount=5_000.0) is False
    assert await ledger.would_exceed(amount=20_000.0) is True

    # After 24h the charge ages out and the window frees.
    clock.advance(timedelta(hours=25))
    assert await ledger.trailing_24h_spend() == pytest.approx(0.0)
    assert await ledger.would_exceed(amount=20_000.0) is False


@pytest.mark.asyncio
async def test_ledger_is_shared_across_lanes(db_path: Path) -> None:
    """A spend recorded by one lane is visible to a separate ledger on the same file."""
    clock = FakeClock()
    orchestrator_lane = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    await orchestrator_lane.record_spend(amount=11.0)

    # The cron lane opens its own ledger object on the same service-level DB file.
    cron_lane = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    assert await cron_lane.trailing_24h_spend() == pytest.approx(11.0)
    # The shared ledger means the cron lane sees it is already near the cap.
    assert await cron_lane.would_exceed(amount=2.0) is True


# --- budget disabled: the 0.0 weekly-budget sentinel admits everything -----------


def test_governor_warns_loudly_when_budget_is_disabled(
    db_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A weekly budget of 0.0 (the default) disables metering and logs one loud WARNING.

    A deploy that forgets WEEKLY_BUDGET must not boot silently and do nothing; the
    governor surfaces the disabled state at construction so it is visible in the logs.
    """
    ledger = BudgetLedger(
        db_path, clock=FakeClock(), auth_mode=AuthMode.API_KEY, weekly_budget=0.0
    )
    with caplog.at_level(logging.WARNING, logger="retinue.budget"):
        BudgetGovernor(ledger)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "WEEKLY_BUDGET" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_gate_admits_everything_when_budget_disabled(db_path: Path) -> None:
    """With metering disabled, gate admits every run without charging the ledger.

    The disabled sentinel must not decline work: an enormous estimate is admitted and
    nothing is recorded, so a deploy without a budget still does work.
    """
    ledger = BudgetLedger(
        db_path, clock=FakeClock(), auth_mode=AuthMode.API_KEY, weekly_budget=0.0
    )
    governor = BudgetGovernor(ledger)

    decision = await governor.gate(estimated_amount=1_000_000.0)

    assert decision == GateDecision(deferred=False, defer_until=None)
    assert await ledger.trailing_24h_spend() == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_meter_adhoc_admits_everything_when_budget_disabled(
    db_path: Path,
) -> None:
    """With metering disabled, meter_adhoc admits every build without charging."""
    ledger = BudgetLedger(
        db_path, clock=FakeClock(), auth_mode=AuthMode.API_KEY, weekly_budget=0.0
    )
    governor = BudgetGovernor(ledger)

    assert await governor.meter_adhoc(amount=1_000_000.0) is True
    assert await ledger.trailing_24h_spend() == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_record_charge_appends_to_the_ledger(db_path: Path) -> None:
    """A side charge is recorded post-hoc, adding to the trailing-24h spend.

    Unlike gate/meter this is not gated — the classifier call has already run — so the
    charge simply lands on the shared ledger and the rolling total climbs by it.
    """
    ledger = BudgetLedger(
        db_path, clock=FakeClock(), auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    governor = BudgetGovernor(ledger)

    await governor.record_charge(amount=0.01)
    await governor.record_charge(amount=0.01)

    assert await ledger.trailing_24h_spend() == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_record_charge_is_a_noop_when_budget_disabled(db_path: Path) -> None:
    """With metering disabled, record_charge writes nothing to the ledger."""
    ledger = BudgetLedger(
        db_path, clock=FakeClock(), auth_mode=AuthMode.API_KEY, weekly_budget=0.0
    )
    governor = BudgetGovernor(ledger)

    await governor.record_charge(amount=1_000_000.0)

    assert await ledger.trailing_24h_spend() == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_positive_budget_with_zero_cap_still_denies(db_path: Path) -> None:
    """Only the exact 0.0 sentinel bypasses; a positive budget whose cap rounds to 0 denies.

    weekly_budget = 100 but daily_cap_fraction = 0 gives cap() == 0. This is *not* the
    disabled sentinel, so meter_adhoc must still enforce it and decline the charge rather
    than admit everything.
    """
    ledger = BudgetLedger(
        db_path,
        clock=FakeClock(),
        auth_mode=AuthMode.API_KEY,
        weekly_budget=100.0,
        daily_cap_fraction=0.0,
    )
    governor = BudgetGovernor(ledger)

    assert ledger.cap() == pytest.approx(0.0)
    assert await governor.meter_adhoc(amount=1.0) is False


# --- concurrent check-and-record: the cap must not be overshot -------------------


@pytest.mark.asyncio
async def test_concurrent_try_record_on_one_instance_is_serialized(
    db_path: Path,
) -> None:
    """Two concurrent try_record calls on the SAME ledger must not corrupt the cap check.

    A single long-lived connection is shared across asyncio tasks, so without a per-store
    lock the two calls interleave statements inside each other's ``BEGIN IMMEDIATE`` and
    corrupt the transaction. The lock serializes them: exactly one records the last 1.0 of
    room, the other re-reads it and declines, and the cap is never overshot.
    """
    clock = FakeClock()
    ledger = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    await ledger.record_spend(amount=11.0)  # cap 12.0, 1.0 of room

    recorded = await asyncio.gather(
        ledger.try_record_if_within_cap(amount=1.0),
        ledger.try_record_if_within_cap(amount=1.0),
    )

    assert sorted(recorded) == [False, True]
    trailing = await ledger.trailing_24h_spend()
    assert trailing == pytest.approx(12.0)
    assert trailing <= ledger.cap()


@pytest.mark.asyncio
async def test_ledger_connection_thread_is_a_daemon(db_path: Path) -> None:
    """The connection's worker thread must not block interpreter exit.

    aiosqlite runs each connection on a thread; a ledger leaked without close()
    (crash paths, test teardown) would otherwise hang process shutdown forever on
    the non-daemon thread join.
    """
    clock = FakeClock()
    ledger = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    await ledger.record_spend(amount=1.0)
    assert ledger._db is not None
    assert ledger._db._thread.daemon is True
    await ledger.close()


@pytest.mark.asyncio
async def test_ledger_close_is_reusable_and_persists(db_path: Path) -> None:
    """close() releases the connection; data persists and a later call lazily reconnects."""
    clock = FakeClock()
    ledger = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    await ledger.record_spend(amount=5.0)
    await ledger.close()

    # A fresh store on the same file reads the persisted charge.
    reopened = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    assert await reopened.trailing_24h_spend() == pytest.approx(5.0)
    await reopened.close()

    # The original store lazily reconnects after close rather than raising.
    assert await ledger.trailing_24h_spend() == pytest.approx(5.0)
    await ledger.close()


@pytest.mark.asyncio
async def test_governor_close_releases_connections(db_path: Path) -> None:
    """The governor's close() tears down its own and the ledger's connections cleanly."""
    clock = FakeClock()
    ledger = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    governor = BudgetGovernor(ledger)
    await governor.meter_adhoc(amount=1.0)

    await governor.close()

    # After close the governor still works, reconnecting lazily.
    assert await governor.meter_adhoc(amount=1.0) is True
    await governor.close()


@pytest.mark.asyncio
async def test_try_record_if_within_cap_is_atomic_under_concurrency(
    db_path: Path,
) -> None:
    """The atomic primitive records only while the charge still fits, re-read live.

    Two concurrent calls for the only remaining room return [True, False] in some order:
    the second re-reads the first's committed write under the write lock and declines.
    """
    clock = FakeClock()
    ledger_a = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    ledger_b = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    await ledger_a.record_spend(amount=11.0)  # cap 12.0, 1.0 of room

    recorded = await asyncio.gather(
        ledger_a.try_record_if_within_cap(amount=1.0),
        ledger_b.try_record_if_within_cap(amount=1.0),
    )

    assert sorted(recorded) == [False, True]
    assert await ledger_a.trailing_24h_spend() == pytest.approx(12.0)


# --- ad-hoc per-build meter: atomic charge against the shared ledger -------------


@pytest.mark.asyncio
async def test_meter_adhoc_records_a_charge_that_fits(db_path: Path) -> None:
    """An ad-hoc build whose charge fits under the cap is admitted and recorded."""
    ledger = BudgetLedger(
        db_path, clock=FakeClock(), auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    governor = BudgetGovernor(ledger)

    admitted = await governor.meter_adhoc(amount=5.0)

    assert admitted is True
    assert await ledger.trailing_24h_spend() == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_meter_adhoc_declines_a_charge_over_the_cap(db_path: Path) -> None:
    """An ad-hoc build over the cap is declined and records nothing."""
    clock = FakeClock()
    ledger = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    await ledger.record_spend(amount=12.0)  # the cap is fully spent
    governor = BudgetGovernor(ledger)

    admitted = await governor.meter_adhoc(amount=1.0)

    assert admitted is False
    assert await ledger.trailing_24h_spend() == pytest.approx(12.0)


@pytest.mark.asyncio
async def test_meter_adhoc_shares_the_ledger_with_the_prd_lane(db_path: Path) -> None:
    """An ad-hoc charge counts against the same window the PRD meter charges.

    cap = 12, the PRD lane already metered 11.0 (1.0 of room). An ad-hoc build of 2.0
    cannot fit on the *shared* ledger, so meter_adhoc declines — the two lanes share one
    budget rather than each getting a fresh cap.
    """
    clock = FakeClock()
    prd_lane = BudgetGovernor(
        BudgetLedger(
            db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
        )
    )
    adhoc_lane = BudgetGovernor(
        BudgetLedger(
            db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
        )
    )
    await prd_lane._ledger.record_spend(amount=11.0)

    assert await adhoc_lane.meter_adhoc(amount=2.0) is False
    # The 1.0 of remaining room is shared: a charge that fits is still admitted.
    assert await adhoc_lane.meter_adhoc(amount=1.0) is True


# --- gate at run start -----------------------------------------------------------


@pytest.mark.asyncio
async def test_run_started_over_budget_is_deferred(db_path: Path) -> None:
    """A run that would start over the 24h cap is deferred, not built."""
    clock = FakeClock()
    ledger = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    await ledger.record_spend(amount=12.0)  # the cap is fully spent
    governor = BudgetGovernor(ledger)

    decision = await governor.gate(estimated_amount=1.0)

    assert decision.deferred is True
    assert decision.defer_until is not None


@pytest.mark.asyncio
async def test_run_under_budget_is_admitted(db_path: Path) -> None:
    """A run whose estimated charge fits under the cap is admitted (not deferred)."""
    ledger = BudgetLedger(
        db_path, clock=FakeClock(), auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    governor = BudgetGovernor(ledger)

    decision = await governor.gate(estimated_amount=5.0)

    assert decision == GateDecision(deferred=False, defer_until=None)


@pytest.mark.asyncio
async def test_defer_until_is_when_the_window_frees(db_path: Path) -> None:
    """The defer time is when the oldest in-window charge ages out and frees room."""
    clock = FakeClock()
    ledger = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    await ledger.record_spend(amount=12.0)
    governor = BudgetGovernor(ledger)

    decision = await governor.gate(estimated_amount=1.0)

    # The oldest charge frees 24h after it was recorded.
    assert decision.defer_until == _T0 + timedelta(hours=24)


@pytest.mark.asyncio
async def test_defer_until_waits_until_window_frees_enough_for_the_amount(
    db_path: Path,
) -> None:
    """When the oldest charge alone doesn't free enough, defer_until waits for more.

    cap = 12. Record 1.0 at T0, then 11.0 at T0+1h (trailing = 12, full). A 5.0 estimate
    needs more room than the 1.0 charge frees, so the defer time must be when the 11.0
    charge ages out (T0+1h+24h), not when the oldest 1.0 charge does (T0+24h).
    """
    clock = FakeClock()
    ledger = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    await ledger.record_spend(amount=1.0)
    clock.advance(timedelta(hours=1))
    await ledger.record_spend(amount=11.0)
    governor = BudgetGovernor(ledger)

    decision = await governor.gate(estimated_amount=5.0)

    assert decision.deferred is True
    assert decision.defer_until is not None
    # The 11.0 charge is the one whose expiry frees enough for a 5.0 estimate.
    assert decision.defer_until == _T0 + timedelta(hours=1) + timedelta(hours=24)

    # Re-gating at exactly defer_until is admitted: the window has genuinely freed.
    clock._now = decision.defer_until
    readmit = await governor.gate(estimated_amount=5.0)
    assert readmit.deferred is False


@pytest.mark.asyncio
async def test_gate_charges_the_admitted_estimate_to_the_ledger(db_path: Path) -> None:
    """An admitted run's estimate is recorded, so the shared window learns the spend.

    The PRD and cron lanes gate through this path and never meter separately; without
    the gate charging the ledger, their real spend would be invisible to the rolling-24h
    cap (the 12%/24h governor would be decorative for the primary lanes).
    """
    ledger = BudgetLedger(
        db_path, clock=FakeClock(), auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    governor = BudgetGovernor(ledger)

    decision = await governor.gate(estimated_amount=5.0)

    assert decision == GateDecision(deferred=False, defer_until=None)
    assert await ledger.trailing_24h_spend() == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_gate_records_nothing_when_deferred(db_path: Path) -> None:
    """A deferred run charges nothing: the estimate lands only when the run is admitted."""
    clock = FakeClock()
    ledger = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    await ledger.record_spend(amount=12.0)  # the cap is fully spent
    governor = BudgetGovernor(ledger)

    decision = await governor.gate(estimated_amount=1.0)

    assert decision.deferred is True
    assert await ledger.trailing_24h_spend() == pytest.approx(12.0)


@pytest.mark.asyncio
async def test_gate_is_atomic_across_concurrent_lanes(db_path: Path) -> None:
    """Two lanes gating concurrently for the last room admit exactly one.

    cap = 12, trailing = 11 (1.0 of room). Two governors on the shared ledger both gate
    a 1.0 estimate at once; the check-and-record serializes on the write lock, so the
    second sees the first's charge and defers — the cap is never overshot by a
    read-then-write race between the PRD and cron lanes.
    """
    clock = FakeClock()
    prd_lane = BudgetGovernor(
        BudgetLedger(
            db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
        )
    )
    cron_lane = BudgetGovernor(
        BudgetLedger(
            db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
        )
    )
    await prd_lane._ledger.record_spend(amount=11.0)

    decisions = await asyncio.gather(
        prd_lane.gate(estimated_amount=1.0),
        cron_lane.gate(estimated_amount=1.0),
    )

    assert sorted(d.deferred for d in decisions) == [False, True]
    trailing = await prd_lane._ledger.trailing_24h_spend()
    assert trailing == pytest.approx(12.0)
    assert trailing <= prd_lane._ledger.cap()


