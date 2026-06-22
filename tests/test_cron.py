"""Tests for the cron backlog drainer (issue #15).

A scheduled cron tick drains loose ``backlog`` issues one at a time:

1. **gate** on the shared :class:`retinue.budget.BudgetGovernor` — defer when the budget
   is spent (no issue is picked, no downstream runs),
2. **pick** the next backlog issue by a weighted score (priority + age), except on every
   Nth tick where a **quota floor** takes the oldest low-priority issue so low items
   provably drain,
3. **run** the same downstream the orchestrator drives (build -> PR -> loopback ->
   notify), here a single injected build callable.

At most one cron run executes at a time (an injected lock mirroring the orchestrator's
single-run guarantee), and the cron shares the orchestrator's budget governor. Every
collaborator — the gh backlog query, the clock (age-weighting + the tick counter), the
budget governor, the single-run lock, and the downstream build — is injected and faked,
so the whole flow runs with no real ``gh``, no Docker, and no wall-clock.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from retinue.budget import AuthMode, BudgetGovernor, BudgetLedger
from retinue.cron import (
    BacklogIssue,
    CronBusyError,
    CronOutcome,
    CronTickResult,
    run_cron_tick,
)


class FakeClock:
    """A deterministic, advanceable clock (the budget/cron time source)."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 6, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta


class FakeCronGh:
    """In-memory backlog query: returns the scripted backlog issues."""

    def __init__(self, issues: list[BacklogIssue]) -> None:
        self._issues = issues

    async def list_backlog(self, *, repo_full_name: str) -> list[BacklogIssue]:
        return list(self._issues)


class RecordingBuild:
    """Records each backlog issue handed to the downstream build (the mocked chain)."""

    def __init__(self) -> None:
        self.built: list[int] = []

    async def __call__(self, *, repo_full_name: str, issue_number: int) -> None:
        self.built.append(issue_number)


class OneAtATimeLock:
    """An async lock that refuses a second concurrent holder (single-cron-run guard).

    Mirrors the orchestrator's injected single-run lock: ``__aenter__`` raises
    :class:`CronBusyError` if already held rather than blocking.
    """

    def __init__(self) -> None:
        self.held = False
        self.acquisitions = 0

    async def __aenter__(self) -> OneAtATimeLock:
        if self.held:
            raise CronBusyError()
        self.held = True
        self.acquisitions += 1
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.held = False


def _issue(number: int, *, priority: str | None, age_days: float) -> BacklogIssue:
    """A backlog issue ``age_days`` old relative to the fake clock's 2026-06-01 start."""
    created = datetime(2026, 6, 1, tzinfo=UTC) - timedelta(days=age_days)
    labels = ["backlog"] + ([f"priority:{priority}"] if priority else [])
    return BacklogIssue(number=number, labels=labels, created_at=created)


def _governor(tmp_path: Path, clock: FakeClock, *, weekly_budget: float = 100.0) -> BudgetGovernor:
    ledger = BudgetLedger(
        tmp_path / "budget.sqlite3",
        clock=clock,
        auth_mode=AuthMode.API_KEY,
        weekly_budget=weekly_budget,
    )
    return BudgetGovernor(ledger)


async def _tick(
    *,
    gh: FakeCronGh,
    governor: BudgetGovernor,
    clock: FakeClock,
    build: RecordingBuild,
    tick_number: int = 1,
    estimated_amount: float = 1.0,
    lock: OneAtATimeLock | None = None,
    quota_every: int = 5,
) -> CronTickResult:
    return await run_cron_tick(
        repo_full_name="owner/repo",
        gh=gh,
        governor=governor,
        clock=clock,
        build=build,
        tick_number=tick_number,
        estimated_amount=estimated_amount,
        lock=lock or OneAtATimeLock(),
        quota_every=quota_every,
    )


# --- weighted-score selection ----------------------------------------------------


@pytest.mark.asyncio
async def test_tick_picks_highest_weighted_score(tmp_path: Path) -> None:
    """The tick picks the backlog issue with the highest priority+age weighted score."""
    clock = FakeClock()
    # A fresh critical outweighs an old low: priority dominates a small age gap.
    gh = FakeCronGh(
        [
            _issue(1, priority="low", age_days=2),
            _issue(2, priority="critical", age_days=0),
        ]
    )
    build = RecordingBuild()

    result = await _tick(gh=gh, governor=_governor(tmp_path, clock), clock=clock, build=build)

    assert result.outcome is CronOutcome.RAN
    assert result.issue_number == 2
    assert build.built == [2]


@pytest.mark.asyncio
async def test_age_breaks_ties_within_a_priority(tmp_path: Path) -> None:
    """Among same-priority issues the older one scores higher and is picked."""
    clock = FakeClock()
    gh = FakeCronGh(
        [
            _issue(1, priority="medium", age_days=1),
            _issue(2, priority="medium", age_days=30),
        ]
    )
    build = RecordingBuild()

    result = await _tick(gh=gh, governor=_governor(tmp_path, clock), clock=clock, build=build)

    assert result.issue_number == 2


# --- quota floor: low-priority items provably drain ------------------------------


@pytest.mark.asyncio
async def test_quota_floor_takes_oldest_low_priority(tmp_path: Path) -> None:
    """On a quota tick the oldest low-priority issue is taken even against a high one."""
    clock = FakeClock()
    gh = FakeCronGh(
        [
            _issue(1, priority="high", age_days=1),
            _issue(2, priority="low", age_days=10),
            _issue(3, priority="low", age_days=40),
        ]
    )
    build = RecordingBuild()

    # tick 5 with quota_every=5 -> a quota tick: the oldest low (3) is forced through.
    result = await _tick(
        gh=gh, governor=_governor(tmp_path, clock), clock=clock, build=build, tick_number=5
    )

    assert result.issue_number == 3
    assert result.outcome is CronOutcome.RAN


@pytest.mark.asyncio
async def test_low_priority_drains_over_many_ticks(tmp_path: Path) -> None:
    """Over many ticks the low-priority items provably drain via the quota floor.

    With a steady stream of high-priority arrivals, pure weighted score would starve the
    low ones forever. The every-Nth quota tick guarantees each low issue is eventually
    taken, so the low backlog drains rather than starving.
    """
    clock = FakeClock()
    governor = _governor(tmp_path, clock)
    # Two stubborn low-priority issues plus an always-present high-priority distractor.
    remaining = {
        1: _issue(1, priority="high", age_days=1),
        2: _issue(2, priority="low", age_days=20),
        3: _issue(3, priority="low", age_days=25),
    }

    picked_low: set[int] = set()
    for tick in range(1, 16):
        gh = FakeCronGh(list(remaining.values()))
        build = RecordingBuild()
        result = await _tick(
            gh=gh, governor=governor, clock=clock, build=build, tick_number=tick
        )
        chosen = result.issue_number
        assert chosen is not None
        if chosen in (2, 3):
            picked_low.add(chosen)
            remaining.pop(chosen, None)
        # The high-priority issue is "rebuilt" / re-arrives each tick (never removed).
        clock.advance(timedelta(hours=1))

    # Both low-priority issues were eventually drained despite the high distractor.
    assert picked_low == {2, 3}


# --- empty backlog ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_backlog_is_a_clean_noop(tmp_path: Path) -> None:
    """A tick with no backlog issues runs nothing and reports IDLE."""
    clock = FakeClock()
    build = RecordingBuild()

    result = await _tick(
        gh=FakeCronGh([]), governor=_governor(tmp_path, clock), clock=clock, build=build
    )

    assert result.outcome is CronOutcome.IDLE
    assert result.issue_number is None
    assert build.built == []


# --- budget gate: defer when spent -----------------------------------------------


@pytest.mark.asyncio
async def test_tick_defers_when_budget_spent(tmp_path: Path) -> None:
    """When the shared budget governor reports spent, the tick defers and builds nothing."""
    clock = FakeClock()
    governor = _governor(tmp_path, clock, weekly_budget=100.0)
    # Cap is 12% of 100 = 12. Spend it down so any estimate exceeds the window.
    await governor._ledger.record_spend(amount=12.0)
    gh = FakeCronGh([_issue(1, priority="high", age_days=1)])
    build = RecordingBuild()

    result = await _tick(
        gh=gh, governor=governor, clock=clock, build=build, estimated_amount=5.0
    )

    assert result.outcome is CronOutcome.DEFERRED
    assert result.defer_until is not None
    assert build.built == []


@pytest.mark.asyncio
async def test_gate_runs_before_the_backlog_query(tmp_path: Path) -> None:
    """A deferred tick does not even query the backlog (gate precedes selection)."""
    clock = FakeClock()
    governor = _governor(tmp_path, clock, weekly_budget=100.0)
    await governor._ledger.record_spend(amount=12.0)

    queried = False

    class TrackingGh:
        async def list_backlog(self, *, repo_full_name: str) -> list[BacklogIssue]:
            nonlocal queried
            queried = True
            return []

    result = await run_cron_tick(
        repo_full_name="owner/repo",
        gh=TrackingGh(),
        governor=governor,
        clock=clock,
        build=RecordingBuild(),
        tick_number=1,
        estimated_amount=5.0,
        lock=OneAtATimeLock(),
    )

    assert result.outcome is CronOutcome.DEFERRED
    assert queried is False


# --- single cron run -------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_run_acquires_the_lock(tmp_path: Path) -> None:
    """A normal tick acquires and releases the single-cron-run lock exactly once."""
    clock = FakeClock()
    lock = OneAtATimeLock()

    await _tick(
        gh=FakeCronGh([_issue(1, priority="high", age_days=1)]),
        governor=_governor(tmp_path, clock),
        clock=clock,
        build=RecordingBuild(),
        lock=lock,
    )

    assert lock.acquisitions == 1
    assert lock.held is False


@pytest.mark.asyncio
async def test_second_concurrent_tick_is_rejected(tmp_path: Path) -> None:
    """A second tick while one holds the lock raises CronBusyError."""
    clock = FakeClock()
    lock = OneAtATimeLock()
    lock.held = True
    lock.acquisitions = 1

    with pytest.raises(CronBusyError):
        await _tick(
            gh=FakeCronGh([_issue(1, priority="high", age_days=1)]),
            governor=_governor(tmp_path, clock),
            clock=clock,
            build=RecordingBuild(),
            lock=lock,
        )
