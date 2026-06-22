"""Tests for the budget governor: rolling-24h spend ledger + pause/resume (issue #14).

A DB-backed rolling-24h spend ledger meters agent spend and enforces a 12%/24h cap
against the service-level weekly budget, in both auth modes ($ for an API key, tokens
for subscription OAuth). The governor gates at run start (over the cap -> defer) and
meters mid-run (a charge that would cross the cap pauses + checkpoints, then resumes via
the reconcile machinery once the trailing-24h window frees).

The clock is injected (no real wall-clock, so the rolling window is deterministic), the
ledger lives in a temp SQLite file, and the resume reuses :func:`reconcile_run` through
the faked gh seam from the reconcile tests — no real ``gh``, no network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from retinue.budget import (
    AuthMode,
    BudgetGovernor,
    BudgetLedger,
    GateDecision,
    MeterDecision,
)
from retinue.orchestrator import PrdSlice
from tests.test_reconcile import FakeReconcileGh

_T0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


class FakeClock:
    """A deterministic, advanceable time source the ledger reads ``now()`` from."""

    def __init__(self, start: datetime = _T0) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """An on-disk SQLite path inside the test's tmp dir."""
    return tmp_path / "budget.sqlite3"


def _prd_slice(issue_number: int, blocked_by: list[int] | None = None) -> PrdSlice:
    return PrdSlice(
        repo_full_name="owner/repo",
        issue_number=issue_number,
        prd_number=1,
        blocked_by=blocked_by or [],
    )


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


# --- meter mid-run: pause + checkpoint, then resume ------------------------------


@pytest.mark.asyncio
async def test_charge_that_crosses_cap_pauses_and_checkpoints(db_path: Path) -> None:
    """A mid-run charge that would cross the cap pauses and checkpoints the run."""
    clock = FakeClock()
    ledger = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    await ledger.record_spend(amount=11.0)  # cap 12.0, 1.0 of room left
    governor = BudgetGovernor(ledger)
    slices = [_prd_slice(2), _prd_slice(3, blocked_by=[2])]

    decision = await governor.meter(
        repo_full_name="owner/repo",
        prd_number=1,
        amount=2.0,
        slices=slices,
    )

    assert isinstance(decision, MeterDecision)
    assert decision.paused is True
    assert decision.resume_at == _T0 + timedelta(hours=24)
    # The pause did not charge the over-cap amount.
    assert await ledger.trailing_24h_spend() == pytest.approx(11.0)


@pytest.mark.asyncio
async def test_charge_under_cap_is_metered_through(db_path: Path) -> None:
    """A mid-run charge that fits under the cap is recorded and the run continues."""
    clock = FakeClock()
    ledger = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    await ledger.record_spend(amount=8.0)
    governor = BudgetGovernor(ledger)

    decision = await governor.meter(
        repo_full_name="owner/repo",
        prd_number=1,
        amount=2.0,
        slices=[_prd_slice(2)],
    )

    assert decision.paused is False
    assert await ledger.trailing_24h_spend() == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_paused_run_resumes_via_reconcile_when_window_frees(
    db_path: Path,
) -> None:
    """A budget-paused run resumes through reconcile, building only unfinished slices."""
    clock = FakeClock()
    ledger = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    await ledger.record_spend(amount=11.0)
    governor = BudgetGovernor(ledger)
    slices = [_prd_slice(2), _prd_slice(3, blocked_by=[2])]

    paused = await governor.meter(
        repo_full_name="owner/repo",
        prd_number=1,
        amount=2.0,
        slices=slices,
    )
    assert paused.paused is True

    # The window has not freed yet: resume must still defer.
    early = await governor.try_resume(
        repo_full_name="owner/repo",
        prd_number=1,
        gh=FakeReconcileGh(closed_issues={2}, merged_branches={"issue-2"}),
    )
    assert early is None

    # Advance past the window; slice 2 landed before the pause, 3 is unfinished.
    clock.advance(timedelta(hours=25))
    result = await governor.try_resume(
        repo_full_name="owner/repo",
        prd_number=1,
        gh=FakeReconcileGh(closed_issues={2}, merged_branches={"issue-2"}),
    )

    assert result is not None
    # Resume reuses the reconcile machinery: only the unfinished slice is rebuilt.
    assert [s.issue_number for s in result.unfinished_slices] == [3]
    assert result.finished_issues == [2]


@pytest.mark.asyncio
async def test_pause_resume_is_observable_end_to_end(db_path: Path) -> None:
    """The pause->resume transition is observable: paused first, then a resume plan."""
    clock = FakeClock()
    ledger = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.SUBSCRIPTION, weekly_budget=1_000_000.0
    )
    await ledger.record_spend(amount=115_000.0)  # cap 120_000, 5_000 of room
    governor = BudgetGovernor(ledger)
    slices = [_prd_slice(2)]

    paused = await governor.meter(
        repo_full_name="owner/repo",
        prd_number=1,
        amount=10_000.0,
        slices=slices,
    )
    assert paused.paused is True
    assert await governor.is_paused(repo_full_name="owner/repo", prd_number=1) is True

    clock.advance(timedelta(hours=25))
    result = await governor.try_resume(
        repo_full_name="owner/repo",
        prd_number=1,
        gh=FakeReconcileGh(closed_issues=set(), merged_branches=set()),
    )
    assert result is not None
    # Once resumed, the run is no longer paused.
    assert await governor.is_paused(repo_full_name="owner/repo", prd_number=1) is False


@pytest.mark.asyncio
async def test_try_resume_stays_paused_when_still_over_cap(db_path: Path) -> None:
    """Past the recorded resume_at but still over cap, try_resume keeps the run paused.

    cap = 12. Record 1.0 at T0, then 11.0 at T0+1h (trailing = 12, full). A 5.0 charge
    pauses the run with resume_at = T0+1h+24h. If the clock is advanced only past the
    first (1.0) charge's expiry (T0+24h), the window has freed only 1.0 — trailing 11.0,
    so 11+5 > 12 is still over cap. try_resume must return None and leave the pause intact
    rather than resume the run over-budget.
    """
    clock = FakeClock()
    ledger = BudgetLedger(
        db_path, clock=clock, auth_mode=AuthMode.API_KEY, weekly_budget=100.0
    )
    await ledger.record_spend(amount=1.0)
    clock.advance(timedelta(hours=1))
    await ledger.record_spend(amount=11.0)
    governor = BudgetGovernor(ledger)
    slices = [_prd_slice(2)]

    paused = await governor.meter(
        repo_full_name="owner/repo",
        prd_number=1,
        amount=5.0,
        slices=slices,
    )
    assert paused.paused is True
    assert paused.resume_at == _T0 + timedelta(hours=1) + timedelta(hours=24)

    # Past T0+24h only the 1.0 charge has aged out; trailing is still 11.0, so 11+5 > 12.
    clock._now = _T0 + timedelta(hours=24, minutes=1)
    still_paused = await governor.try_resume(
        repo_full_name="owner/repo",
        prd_number=1,
        gh=FakeReconcileGh(closed_issues=set(), merged_branches=set()),
    )
    assert still_paused is None
    assert await governor.is_paused(repo_full_name="owner/repo", prd_number=1) is True

    # Once the 11.0 charge also ages out, the cap genuinely has room and resume proceeds.
    clock._now = _T0 + timedelta(hours=1) + timedelta(hours=24, minutes=1)
    result = await governor.try_resume(
        repo_full_name="owner/repo",
        prd_number=1,
        gh=FakeReconcileGh(closed_issues=set(), merged_branches=set()),
    )
    assert result is not None
    assert await governor.is_paused(repo_full_name="owner/repo", prd_number=1) is False
