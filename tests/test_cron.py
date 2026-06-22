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

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from retinue.budget import AuthMode, BudgetGovernor, BudgetLedger
from retinue.cron import (
    BacklogIssue,
    CronBusyError,
    CronOutcome,
    CronTickResult,
    GhCli,
    SliceBuilder,
    run_cron_tick,
)
from retinue.loopback import Severity
from retinue.orchestrator import BuildOutcome, BuildResult, Slice, integration_branch
from retinue.repo_config import RepoConfig


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


# --- real GhCli: command assembly, auth env, payload parsing ----------------------
#
# These exercise the production CronGh's pure/parseable edges through an injected argv
# runner — no real gh, Docker, or network. The runner captures the argv + env so the
# command assembly and auth header (GH_TOKEN) are asserted, and returns canned gh JSON
# so the payload parsing into BacklogIssue is asserted.


class CapturingGhRunner:
    """Records the argv + env it was called with and returns a canned stdout payload."""

    def __init__(self, stdout: bytes = b"[]") -> None:
        self._stdout = stdout
        self.argv: Sequence[str] | None = None
        self.env: Mapping[str, str] | None = None

    async def __call__(
        self, argv: Sequence[str], env: Mapping[str, str]
    ) -> bytes:
        self.argv = argv
        self.env = env
        return self._stdout


@pytest.mark.asyncio
async def test_ghcli_assembles_the_backlog_list_command() -> None:
    """GhCli runs ``gh issue list`` scoped to the repo's open ``backlog`` issues."""
    runner = CapturingGhRunner()
    gh = GhCli(token="t0ken", runner=runner, list_limit=50)

    await gh.list_backlog(repo_full_name="owner/repo")

    argv = list(runner.argv or [])
    assert argv[:3] == ["gh", "issue", "list"]
    assert "--repo" in argv and argv[argv.index("--repo") + 1] == "owner/repo"
    assert "--label" in argv and argv[argv.index("--label") + 1] == "backlog"
    assert "--state" in argv and argv[argv.index("--state") + 1] == "open"
    assert "--limit" in argv and argv[argv.index("--limit") + 1] == "50"
    # The JSON fields requested are exactly what BacklogIssue needs.
    assert argv[argv.index("--json") + 1] == "number,labels,createdAt"


@pytest.mark.asyncio
async def test_ghcli_puts_the_token_in_the_env_not_the_argv() -> None:
    """The token authenticates via GH_TOKEN in the child env, never on the command line."""
    runner = CapturingGhRunner()
    gh = GhCli(token="s3cret", runner=runner)

    await gh.list_backlog(repo_full_name="owner/repo")

    assert (runner.env or {}).get("GH_TOKEN") == "s3cret"
    assert "s3cret" not in list(runner.argv or [])


@pytest.mark.asyncio
async def test_ghcli_omits_the_auth_env_when_no_token() -> None:
    """With no token GhCli leaves the auth env empty, deferring to gh's ambient auth."""
    runner = CapturingGhRunner()
    gh = GhCli(token=None, runner=runner)

    await gh.list_backlog(repo_full_name="owner/repo")

    assert "GH_TOKEN" not in (runner.env or {})


@pytest.mark.asyncio
async def test_ghcli_parses_the_gh_json_payload() -> None:
    """GhCli parses gh's JSON listing into BacklogIssue objects with labels + ages."""
    payload = json.dumps(
        [
            {
                "number": 7,
                "createdAt": "2026-06-01T12:00:00Z",
                "labels": [{"name": "backlog"}, {"name": "priority:high"}],
            },
            {
                "number": 9,
                "createdAt": "2026-05-20T00:00:00Z",
                "labels": [{"name": "backlog"}],
            },
        ]
    ).encode()
    gh = GhCli(runner=CapturingGhRunner(stdout=payload))

    issues = await gh.list_backlog(repo_full_name="owner/repo")

    assert [issue.number for issue in issues] == [7, 9]
    assert issues[0].labels == ["backlog", "priority:high"]
    assert issues[0].severity() is Severity.HIGH
    assert issues[0].created_at == datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    # A backlog issue with no priority label defaults to LOW (the quota-floor sweep).
    assert issues[1].severity() is Severity.LOW


@pytest.mark.asyncio
async def test_ghcli_empty_listing_is_an_empty_backlog() -> None:
    """An empty gh array parses to an empty backlog (the IDLE-tick input)."""
    gh = GhCli(runner=CapturingGhRunner(stdout=b"[]"))

    assert await gh.list_backlog(repo_full_name="owner/repo") == []


@pytest.mark.asyncio
async def test_ghcli_rejects_non_json_output() -> None:
    """Non-JSON gh output is a hard error, not a silently-empty backlog."""
    gh = GhCli(runner=CapturingGhRunner(stdout=b"not json"))

    with pytest.raises(ValueError):
        await gh.list_backlog(repo_full_name="owner/repo")


@pytest.mark.asyncio
async def test_ghcli_rejects_a_malformed_issue_entry() -> None:
    """A payload missing a required field fails loudly rather than dropping the issue."""
    payload = json.dumps([{"number": 1, "labels": [{"name": "backlog"}]}]).encode()
    gh = GhCli(runner=CapturingGhRunner(stdout=payload))

    with pytest.raises(ValueError):
        await gh.list_backlog(repo_full_name="owner/repo")


# --- real SliceBuilder (CronBuild): slice assembly + downstream wiring -------------
#
# The production CronBuild assembles the standalone Slice for the picked backlog nit and
# drives the orchestrator's build_slice downstream. These exercise that assembly + the
# decision (which integration target a loose nit drains onto) through an injected slice
# runner — no Agent SDK, Docker, gh, or network. The runner captures the assembled Slice
# and returns a canned BuildResult, so the slice it builds is asserted directly.


class CapturingSliceRunner:
    """Records the assembled Slice it was handed and returns a canned BuildResult."""

    def __init__(self, result: BuildResult | None = None) -> None:
        self._result = result or BuildResult(
            outcome=BuildOutcome.MERGED, integration_branch="retinue/prd-0"
        )
        self.slices: list[Slice] = []

    async def __call__(self, slice_: Slice) -> BuildResult:
        self.slices.append(slice_)
        return self._result


def _slice_builder(runner: CapturingSliceRunner) -> SliceBuilder:
    """A SliceBuilder whose downstream is the injected runner.

    The orchestrator collaborators are inert sentinels: with ``runner`` injected the real
    ``build_slice`` edge is never touched, so the test stays free of the Agent SDK,
    Docker, gh, and network.
    """
    sentinel = cast(Any, object())
    return SliceBuilder(
        config=RepoConfig(),
        claude_md="# CLAUDE.md",
        implementer=sentinel,
        git=sentinel,
        auth=sentinel,
        runtime=sentinel,
        resolve_secret=sentinel,
        report=sentinel,
        runner=runner,
    )


@pytest.mark.asyncio
async def test_slice_builder_assembles_a_slice_for_the_backlog_issue() -> None:
    """The builder turns the picked backlog issue into a Slice for that repo + issue."""
    runner = CapturingSliceRunner()
    build = _slice_builder(runner)

    await build(repo_full_name="owner/repo", issue_number=42)

    assert len(runner.slices) == 1
    built = runner.slices[0]
    assert built.repo_full_name == "owner/repo"
    assert built.issue_number == 42


@pytest.mark.asyncio
async def test_slice_builder_drains_a_loose_nit_onto_its_own_integration_branch() -> None:
    """A loose nit has no parent PRD, so it drains onto ``retinue/prd-<issue>``."""
    runner = CapturingSliceRunner()
    build = _slice_builder(runner)

    await build(repo_full_name="owner/repo", issue_number=42)

    built = runner.slices[0]
    # The per-issue PRD number is the issue number, giving a dedicated integration target.
    assert built.prd_number == 42
    assert integration_branch(built.prd_number) == "retinue/prd-42"
    # The implementer commits to the issue-<N> branch derived from the issue number.
    assert built.branch == "issue-42"


@pytest.mark.asyncio
async def test_slice_builder_satisfies_the_cron_build_protocol(tmp_path: Path) -> None:
    """The builder is a drop-in CronBuild: run_cron_tick drives it end to end."""
    runner = CapturingSliceRunner()
    build = _slice_builder(runner)
    clock = FakeClock()

    result = await run_cron_tick(
        repo_full_name="owner/repo",
        gh=FakeCronGh([_issue(7, priority="high", age_days=1)]),
        governor=_governor(tmp_path, clock),
        clock=clock,
        build=build,
        tick_number=1,
        estimated_amount=1.0,
        lock=OneAtATimeLock(),
    )

    assert result.outcome is CronOutcome.RAN
    assert result.issue_number == 7
    # The tick drove the real builder, which assembled and ran the slice for issue 7.
    assert [s.issue_number for s in runner.slices] == [7]
