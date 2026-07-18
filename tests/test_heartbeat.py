"""Tests for the worker-global cron heartbeat (issue #35).

A single arq ``cron_jobs`` tick fires on a fixed cadence and, per tick:

1. fires the ad-hoc drain (:func:`retinue.adhoc_drain.run_adhoc_drain`) as the safety-net
   sweep for issues labeled while the webhook was missed, for each repo whose
   ``repo_config.cron`` says it is **due** on this tick (the per-repo "is this repo due?"
   filter under the global tick);
2. drives :func:`retinue.cron.run_cron_tick` for the backlog lane — making the heartbeat
   the first runtime caller of the previously-dead cron lane.

Every collaborator — the clock, the opted-in repo enumeration, the per-repo drain, and the
backlog cron tick — is injected and faked, so the heartbeat runs with no real arq, Redis,
gh, Docker, or wall-clock.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from arq.cron import CronJob

from retinue.cron import CronOutcome, CronTickResult
from retinue.heartbeat import (
    DueRepo,
    HeartbeatResult,
    cron_due,
    heartbeat_tick,
    run_heartbeat,
)
from retinue.repo_config import RepoConfig
from retinue.worker import WorkerSettings
from tests.fakes import FakeClock

# --- cron_due: the per-repo "is this repo due?" filter ----------------------------


@pytest.mark.parametrize(
    ("cron_expr", "when", "due"),
    [
        # Every-6-hours cadence: due exactly at 00:00, 06:00, 12:00, 18:00.
        ("0 */6 * * *", datetime(2026, 6, 23, 6, 0, tzinfo=UTC), True),
        ("0 */6 * * *", datetime(2026, 6, 23, 7, 0, tzinfo=UTC), False),
        ("0 */6 * * *", datetime(2026, 6, 23, 6, 1, tzinfo=UTC), False),
        # Wildcard minute/hour: due on every tick on the matched fields.
        ("* * * * *", datetime(2026, 6, 23, 13, 37, tzinfo=UTC), True),
        # A specific minute and hour.
        ("30 9 * * *", datetime(2026, 6, 23, 9, 30, tzinfo=UTC), True),
        ("30 9 * * *", datetime(2026, 6, 23, 10, 30, tzinfo=UTC), False),
        # A comma list of minutes.
        ("0,30 * * * *", datetime(2026, 6, 23, 8, 30, tzinfo=UTC), True),
        ("0,30 * * * *", datetime(2026, 6, 23, 8, 15, tzinfo=UTC), False),
        # Day-of-week (2026-06-23 is a Tuesday == weekday 2).
        ("0 0 * * 2", datetime(2026, 6, 23, 0, 0, tzinfo=UTC), True),
        ("0 0 * * 1", datetime(2026, 6, 23, 0, 0, tzinfo=UTC), False),
        # A range of hours.
        ("0 9-17 * * *", datetime(2026, 6, 23, 13, 0, tzinfo=UTC), True),
        ("0 9-17 * * *", datetime(2026, 6, 23, 18, 0, tzinfo=UTC), False),
    ],
)
def test_cron_due_matches_each_field(
    cron_expr: str, when: datetime, due: bool
) -> None:
    assert cron_due(cron_expr, when) is due


@pytest.mark.parametrize(
    ("cron_expr", "when", "due"),
    [
        # A value-base step is Vixie "N-high/step": 5/15 minute == 5,20,35,50.
        ("5/15 * * * *", datetime(2026, 6, 23, 0, 5, tzinfo=UTC), True),
        ("5/15 * * * *", datetime(2026, 6, 23, 0, 20, tzinfo=UTC), True),
        ("5/15 * * * *", datetime(2026, 6, 23, 0, 35, tzinfo=UTC), True),
        ("5/15 * * * *", datetime(2026, 6, 23, 0, 50, tzinfo=UTC), True),
        # ...and not the off-step minutes, nor minute 0 below the base.
        ("5/15 * * * *", datetime(2026, 6, 23, 0, 10, tzinfo=UTC), False),
        ("5/15 * * * *", datetime(2026, 6, 23, 0, 25, tzinfo=UTC), False),
        ("5/15 * * * *", datetime(2026, 6, 23, 0, 0, tzinfo=UTC), False),
        # The same form on the hour field steps to the hour max (23): 2/6 == 2,8,14,20.
        ("0 2/6 * * *", datetime(2026, 6, 23, 2, 0, tzinfo=UTC), True),
        ("0 2/6 * * *", datetime(2026, 6, 23, 8, 0, tzinfo=UTC), True),
        ("0 2/6 * * *", datetime(2026, 6, 23, 14, 0, tzinfo=UTC), True),
        ("0 2/6 * * *", datetime(2026, 6, 23, 20, 0, tzinfo=UTC), True),
        ("0 2/6 * * *", datetime(2026, 6, 23, 0, 0, tzinfo=UTC), False),
        ("0 2/6 * * *", datetime(2026, 6, 23, 5, 0, tzinfo=UTC), False),
    ],
)
def test_cron_due_value_base_step_walks_to_field_max(
    cron_expr: str, when: datetime, due: bool
) -> None:
    """A bare-value base with a step (``N/step``) is Vixie ``N-high/step``, not point ``N``."""
    assert cron_due(cron_expr, when) is due


def test_cron_due_is_false_when_no_cadence_configured() -> None:
    """A repo with no ``cron`` set is never picked up by the scheduled sweep."""
    assert cron_due(None, datetime(2026, 6, 23, 6, 0, tzinfo=UTC)) is False


def test_cron_due_sunday_accepts_both_0_and_7() -> None:
    """Cron's Sunday is 0 or 7; 2026-06-21 is a Sunday."""
    sunday = datetime(2026, 6, 21, 0, 0, tzinfo=UTC)
    assert cron_due("0 0 * * 0", sunday) is True
    assert cron_due("0 0 * * 7", sunday) is True


# --- the fakes the heartbeat injects through --------------------------------------


class RecordingDrain:
    """Records every repo the safety-net ad-hoc drain was fired for."""

    def __init__(self) -> None:
        self.drained: list[str] = []

    async def __call__(self, *, repo_full_name: str, config: RepoConfig) -> None:
        self.drained.append(repo_full_name)


class RecordingCronTick:
    """Records every repo the backlog cron tick was driven for, the tick number, config."""

    def __init__(self) -> None:
        self.ticked: list[tuple[str, int]] = []
        self.configs: list[RepoConfig] = []

    async def __call__(
        self, *, repo_full_name: str, tick_number: int, config: RepoConfig
    ) -> CronTickResult:
        self.ticked.append((repo_full_name, tick_number))
        self.configs.append(config)
        return CronTickResult(outcome=CronOutcome.IDLE)


def _repos(*specs: tuple[str, str | None]) -> list[DueRepo]:
    """Build the opted-in repo set: ``(repo_full_name, cron_expr)`` pairs."""
    return [
        DueRepo(repo_full_name=name, config=RepoConfig(cron=cron))
        for name, cron in specs
    ]


async def _enumerate(repos: list[DueRepo]) -> list[DueRepo]:
    return list(repos)


# --- run_heartbeat: the worker-global tick driver ---------------------------------


@pytest.mark.asyncio
async def test_heartbeat_fires_the_drain_for_due_repos_only() -> None:
    """``repo_config.cron`` gates which repos the safety-net drain fires for."""
    clock = FakeClock(datetime(2026, 6, 23, 6, 0, tzinfo=UTC))  # 06:00 == */6 due
    drain = RecordingDrain()
    cron_tick = RecordingCronTick()
    repos = _repos(
        ("owner/due", "0 */6 * * *"),  # due at 06:00
        ("owner/not-due", "30 9 * * *"),  # only due at 09:30
        ("owner/no-cron", None),  # never scheduled
    )

    await run_heartbeat(
        enumerate_repos=lambda: _enumerate(repos),
        clock=clock,
        drain=drain,
        cron_tick=cron_tick,
        tick_number=1,
    )

    assert drain.drained == ["owner/due"]


@pytest.mark.asyncio
async def test_heartbeat_drives_the_backlog_cron_lane_for_every_repo() -> None:
    """The same heartbeat drives ``run_cron_tick`` for the backlog lane each repo.

    The backlog tick gates itself on the budget and picks its own work, so the heartbeat
    drives it for every opted-in repo regardless of the per-repo ad-hoc cadence — this is
    the first runtime caller of the previously-dead cron lane.
    """
    clock = FakeClock(datetime(2026, 6, 23, 7, 0, tzinfo=UTC))  # nothing ad-hoc-due
    drain = RecordingDrain()
    cron_tick = RecordingCronTick()
    repos = _repos(
        ("owner/a", "0 */6 * * *"),
        ("owner/b", None),
    )

    await run_heartbeat(
        enumerate_repos=lambda: _enumerate(repos),
        clock=clock,
        drain=drain,
        cron_tick=cron_tick,
        tick_number=4,
    )

    assert drain.drained == []  # 07:00 is not a */6 boundary
    assert cron_tick.ticked == [("owner/a", 4), ("owner/b", 4)]
    # Each repo's own config rides its tick so the trickle promotion uses that repo's
    # trigger label.
    assert [c.cron for c in cron_tick.configs] == ["0 */6 * * *", None]


@pytest.mark.asyncio
async def test_heartbeat_reports_what_it_fired() -> None:
    clock = FakeClock(datetime(2026, 6, 23, 6, 0, tzinfo=UTC))
    repos = _repos(("owner/a", "0 */6 * * *"), ("owner/b", "0 */6 * * *"))

    result = await run_heartbeat(
        enumerate_repos=lambda: _enumerate(repos),
        clock=clock,
        drain=RecordingDrain(),
        cron_tick=RecordingCronTick(),
        tick_number=2,
    )

    assert isinstance(result, HeartbeatResult)
    assert result.drained_repos == ["owner/a", "owner/b"]
    assert result.ticked_repos == ["owner/a", "owner/b"]


@pytest.mark.asyncio
async def test_heartbeat_with_no_repos_is_a_noop() -> None:
    result = await run_heartbeat(
        enumerate_repos=lambda: _enumerate([]),
        clock=FakeClock(datetime(2026, 6, 23, 6, 0, tzinfo=UTC)),
        drain=RecordingDrain(),
        cron_tick=RecordingCronTick(),
        tick_number=1,
    )
    assert result.drained_repos == []
    assert result.ticked_repos == []


@pytest.mark.asyncio
async def test_one_repo_drain_failure_does_not_abort_the_sweep() -> None:
    """A drain that raises for one repo must not starve the rest of the sweep."""
    clock = FakeClock(datetime(2026, 6, 23, 6, 0, tzinfo=UTC))
    cron_tick = RecordingCronTick()
    repos = _repos(("owner/boom", "0 */6 * * *"), ("owner/ok", "0 */6 * * *"))

    class ExplodingDrain:
        def __init__(self) -> None:
            self.drained: list[str] = []

        async def __call__(self, *, repo_full_name: str, config: RepoConfig) -> None:
            if repo_full_name == "owner/boom":
                raise RuntimeError("drain blew up")
            self.drained.append(repo_full_name)

    drain = ExplodingDrain()
    result = await run_heartbeat(
        enumerate_repos=lambda: _enumerate(repos),
        clock=clock,
        drain=drain,
        cron_tick=cron_tick,
        tick_number=1,
    )

    # The healthy repo still drained and both repos still ticked the backlog lane.
    assert drain.drained == ["owner/ok"]
    assert result.drained_repos == ["owner/ok"]
    assert [repo for repo, _ in cron_tick.ticked] == ["owner/boom", "owner/ok"]


# --- heartbeat_tick: the arq cron_jobs task reading injected ctx -------------------


@pytest.mark.asyncio
async def test_heartbeat_tick_drives_run_heartbeat_from_ctx() -> None:
    """The arq task reads its collaborators from ``ctx`` and increments the tick count."""
    clock = FakeClock(datetime(2026, 6, 23, 6, 0, tzinfo=UTC))
    drain = RecordingDrain()
    cron_tick = RecordingCronTick()
    repos = _repos(("owner/a", "0 */6 * * *"))
    ctx: dict[str, Any] = {
        "heartbeat_enumerate_repos": lambda: _enumerate(repos),
        "heartbeat_clock": clock,
        "heartbeat_drain": drain,
        "heartbeat_cron_tick": cron_tick,
    }

    await heartbeat_tick(ctx)
    await heartbeat_tick(ctx)

    # Both ticks fired; the second carried an incremented tick number for the quota floor.
    assert drain.drained == ["owner/a", "owner/a"]
    assert [tick for _, tick in cron_tick.ticked] == [1, 2]


@pytest.mark.asyncio
async def test_heartbeat_tick_is_a_noop_without_wiring() -> None:
    """A bare worker (no heartbeat collaborators in ctx) ticks harmlessly."""
    ctx: dict[str, Any] = {}
    await heartbeat_tick(ctx)  # must not raise


# --- the arq cron_jobs registration -----------------------------------------------


def test_worker_registers_the_heartbeat_as_a_cron_job() -> None:
    """A worker-global arq ``cron_jobs`` tick is registered for the heartbeat."""
    cron_jobs = getattr(WorkerSettings, "cron_jobs", None)
    assert cron_jobs, "WorkerSettings must register at least one cron job"
    assert all(isinstance(job, CronJob) for job in cron_jobs)
    coroutines = {job.coroutine for job in cron_jobs}
    assert heartbeat_tick in coroutines


def test_heartbeat_cron_job_fires_on_a_fixed_minute_cadence() -> None:
    """The heartbeat's cron schedule pins a fixed sub-hour cadence (the global tick)."""
    job = next(
        j for j in WorkerSettings.cron_jobs if j.coroutine is heartbeat_tick
    )
    # arq stores the per-field match set; the heartbeat must constrain the minute so it
    # fires on a fixed sub-hour cadence rather than every minute of every hour.
    assert isinstance(job.minute, set) and job.minute
