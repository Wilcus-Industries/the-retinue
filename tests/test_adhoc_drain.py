"""Tests for the ad-hoc drain (issue #32).

The ad-hoc drain lists every open ``ready-for-agent`` non-PRD issue via the gh seam,
ranks them by ``priority:<severity>`` (no-priority lowest), and drives the ad-hoc
build+PR primitive for each up to the concurrency cap (``max_parallel``):

1. **list** — pull the repo's open ``ready-for-agent`` issues (number, labels, body),
2. **filter** — keep only the ad-hoc lane via ``ReadyIssue.is_adhoc``, which mirrors
   :func:`retinue.lane.classify`'s ad-hoc decision but does **not** call it: drop any
   PRD-labeled issue and any issue carrying a ``Part of #<prd>`` link, since those route
   to the orchestrator lane,
3. **rank** — order by ``priority:<severity>`` with no-priority lowest,
4. **drive** — materialize each ranked issue into an :class:`AdhocIssue` through
   :meth:`AdhocIssue.from_fetched_issue` (fed the fetched body, so the ``Chain-depth:``
   lineage marker is read back and the #39/#40 review-fix chain bound stays live) and
   run the injected ad-hoc build callable, bounded by ``max_parallel``.

Every collaborator — the gh issue query and the downstream build — is injected and
faked, so the whole drain runs with no real ``gh``, no Docker, and no network.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from types import TracebackType

import pytest

from retinue.adhoc_build import AdhocIssue, render_chain_depth
from retinue.adhoc_drain import (
    AdhocBuild,
    AdhocDrainBusyError,
    AdhocDrainLock,
    AdhocGh,
    AdhocPrOpen,
    FlightSnapshot,
    FlightState,
    GhCli,
    ReadyIssue,
    run_adhoc_drain,
)
from retinue.budget import AuthMode, BudgetGovernor, BudgetLedger
from retinue.loopback import Severity
from retinue.repo_config import RepoConfig
from tests.test_budget import FakeClock


def _ready(
    number: int, *, labels: list[str] | None = None, body: str = ""
) -> ReadyIssue:
    """A ``ready-for-agent`` issue as the gh seam reports it (number, labels, body)."""
    return ReadyIssue(
        number=number,
        labels=["ready-for-agent", *(labels or [])],
        body=body,
    )


def _governor(tmp_path: Path, *, weekly: float = 1000.0) -> BudgetGovernor:
    """A budget governor over a fresh temp ledger with generous headroom by default."""
    return BudgetGovernor(
        BudgetLedger(
            tmp_path / "budget.sqlite3",
            clock=FakeClock(),
            auth_mode=AuthMode.API_KEY,
            weekly_budget=weekly,
        )
    )


class _Lock:
    """A real single-run lock: the second concurrent holder raises (never blocks)."""

    def __init__(self) -> None:
        self._held = False

    async def __aenter__(self) -> _Lock:
        if self._held:
            raise AdhocDrainBusyError
        self._held = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._held = False


def _nolock() -> AbstractAsyncContextManager[object]:
    """A lock that never rejects, for tests not exercising the single-run guard."""
    from contextlib import nullcontext

    return nullcontext()


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
    """Yield the event loop until ``predicate()`` is truthy (bounded so a test fails fast).

    The drain awaits real SQLite I/O (the budget meter, on an aiosqlite executor thread)
    before each build, so a concurrency assertion can't rely on a fixed number of
    ``sleep(0)`` turns; this polls with a short real sleep so the executor threads progress.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached before the timeout")


async def _drain(
    *,
    gh: AdhocGh,
    build: AdhocBuild,
    open_pr: AdhocPrOpen | None = None,
    config: RepoConfig | None = None,
    governor: BudgetGovernor | None = None,
    tmp_path: Path | None = None,
    lock: AbstractAsyncContextManager[object] | None = None,
    prd_in_flight: bool = False,
    estimated_amount: float = 1.0,
) -> list[AdhocIssue]:
    """Invoke the drain with sensible defaults so each test sets only what it exercises.

    Exactly one of ``governor`` or ``tmp_path`` must be given: ``tmp_path`` builds a
    default generous-budget governor over a temp ledger when the test doesn't pin one.
    """
    if governor is None:
        assert tmp_path is not None, "pass a governor or a tmp_path for a default one"
        governor = _governor(tmp_path)
    try:
        return await run_adhoc_drain(
            repo_full_name="owner/repo",
            gh=gh,
            build=build,
            open_pr=open_pr or RecordingPrOpen(),
            config=config or RepoConfig(),
            governor=governor,
            estimated_amount=estimated_amount,
            lock=lock or _nolock(),
            prd_in_flight=prd_in_flight,
        )
    finally:
        # Release the governor's SQLite connection on this test's own event loop; a
        # leaked one is torn down at GC against whatever loop is current then. close()
        # reconnects lazily, so tests that meter the governor afterwards still work.
        await governor.close()


class FakeAdhocGh:
    """In-memory ready-for-agent query + whole-repo flight-state truth (the fast path).

    ``flight_snapshot`` answers the dedup + stranded-recovery question for the whole repo in
    one shot (the query the drain prefers): an issue in ``in_flight_numbers`` has a branch
    *and* an open PR (a build under way or landed -> :attr:`FlightState.IN_FLIGHT`, skip); an
    issue in ``stranded_numbers`` has a pushed ``issue-<N>`` branch but *no* open PR (a green
    build whose PR never opened -> :attr:`FlightState.STRANDED`, open its PR without
    rebuilding); every other issue is :attr:`FlightState.ABSENT` (build it). ``flight_state``
    is retained so this fake still satisfies :class:`AdhocGh`, but the drain classifies from
    the snapshot, so ``flight_state`` is not exercised on the fast path.
    """

    def __init__(
        self,
        issues: list[ReadyIssue],
        *,
        in_flight_numbers: set[int] | None = None,
        stranded_numbers: set[int] | None = None,
    ) -> None:
        self._issues = issues
        self._in_flight = in_flight_numbers or set()
        self._stranded = stranded_numbers or set()
        self.calls: list[str] = []
        self.snapshot_calls: list[str] = []
        self.flight_state_calls: list[int] = []

    async def list_ready(self, *, repo_full_name: str) -> list[ReadyIssue]:
        self.calls.append(repo_full_name)
        return list(self._issues)

    async def flight_snapshot(self, *, repo_full_name: str) -> FlightSnapshot:
        self.snapshot_calls.append(repo_full_name)
        return FlightSnapshot(
            open_pr_heads=frozenset(f"issue-{n}" for n in self._in_flight),
            issue_branches=frozenset(
                f"issue-{n}" for n in self._in_flight | self._stranded
            ),
        )

    async def flight_state(
        self, *, repo_full_name: str, issue_number: int
    ) -> FlightState:
        self.flight_state_calls.append(issue_number)
        if issue_number in self._in_flight:
            return FlightState.IN_FLIGHT
        if issue_number in self._stranded:
            return FlightState.STRANDED
        return FlightState.ABSENT


class _FlightStateOnlyGh:
    """A seam offering only per-issue ``flight_state`` (no whole-repo snapshot).

    Exercises the drain's fallback: a gh seam that does not implement
    :class:`SupportsFlightSnapshot` is classified one issue at a time via ``flight_state``.
    """

    def __init__(
        self,
        issues: list[ReadyIssue],
        *,
        in_flight_numbers: set[int] | None = None,
        stranded_numbers: set[int] | None = None,
    ) -> None:
        self._issues = issues
        self._in_flight = in_flight_numbers or set()
        self._stranded = stranded_numbers or set()
        self.flight_state_calls: list[int] = []

    async def list_ready(self, *, repo_full_name: str) -> list[ReadyIssue]:
        return list(self._issues)

    async def flight_state(
        self, *, repo_full_name: str, issue_number: int
    ) -> FlightState:
        self.flight_state_calls.append(issue_number)
        if issue_number in self._in_flight:
            return FlightState.IN_FLIGHT
        if issue_number in self._stranded:
            return FlightState.STRANDED
        return FlightState.ABSENT


class RecordingPrOpen:
    """Records each AdhocIssue whose stranded PR the drain opened (the PR-open-only seam).

    The drain drives this instead of the build for a :attr:`FlightState.STRANDED` issue —
    a green branch pushed by a prior build but never PR'd — so the PR opens with no rebuild.
    """

    def __init__(self) -> None:
        self.opened: list[AdhocIssue] = []

    async def __call__(self, issue: AdhocIssue, *, repo_full_name: str) -> None:
        self.opened.append(issue)


class RecordingAdhocBuild:
    """Records each AdhocIssue handed to the downstream build (the mocked build+PR).

    ``invoked`` is the order builds were *started* in (rank order, recorded before any
    await); ``built`` is completion order. The drain meters each build against a real
    SQLite-backed budget before calling here, so completion order can interleave — rank
    assertions read ``invoked``, concurrency assertions read ``built``.
    """

    def __init__(self, *, gate: asyncio.Event | None = None) -> None:
        self.invoked: list[AdhocIssue] = []
        self.built: list[AdhocIssue] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self._gate = gate

    async def __call__(self, issue: AdhocIssue, *, repo_full_name: str) -> None:
        self.invoked.append(issue)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self._gate is not None:
                await self._gate.wait()
            self.built.append(issue)
        finally:
            self.in_flight -= 1


# --- listing + filtering: only open ready-for-agent non-PRD issues ----------------


@pytest.mark.asyncio
async def test_drain_drives_the_build_for_each_ready_adhoc_issue(tmp_path: Path) -> None:
    """AC1/AC3: the drain drives the ad-hoc build primitive for each ready issue."""
    gh = FakeAdhocGh([_ready(7), _ready(9)])
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, tmp_path=tmp_path)

    assert {issue.issue_number for issue in build.built} == {7, 9}
    assert gh.calls == ["owner/repo"]


@pytest.mark.asyncio
async def test_prd_labeled_issues_are_excluded(tmp_path: Path) -> None:
    """AC4: a PRD-labeled (``prd``) issue is not an ad-hoc issue, so it is dropped."""
    gh = FakeAdhocGh([_ready(7), _ready(8, labels=["prd"])])
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, tmp_path=tmp_path)

    assert [issue.issue_number for issue in build.built] == [7]


@pytest.mark.asyncio
async def test_part_of_prd_issues_are_excluded(tmp_path: Path) -> None:
    """AC1/AC4: a ``Part of #<prd>`` issue routes to the orchestrator lane, not ad-hoc."""
    gh = FakeAdhocGh(
        [_ready(7), _ready(8, body="Implements the thing.\n\nPart of #42")]
    )
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, tmp_path=tmp_path)

    assert [issue.issue_number for issue in build.built] == [7]


# --- ranking: priority:<sev>, no-priority lowest ----------------------------------


@pytest.mark.asyncio
async def test_issues_are_ranked_by_priority_no_priority_lowest(tmp_path: Path) -> None:
    """AC2: issues are built highest-priority first; a no-priority issue ranks lowest."""
    gh = FakeAdhocGh(
        [
            _ready(1),  # no priority -> lowest
            _ready(2, labels=["priority:high"]),
            _ready(3, labels=["priority:critical"]),
            _ready(4, labels=["priority:low"]),
        ]
    )
    build = RecordingAdhocBuild()

    drained = await _drain(gh=gh, build=build, tmp_path=tmp_path)

    # The drain's return value is the rank-order surface (build completion can interleave).
    assert [issue.issue_number for issue in drained] == [3, 2, 4, 1]


@pytest.mark.asyncio
async def test_an_unknown_priority_label_ranks_lowest(tmp_path: Path) -> None:
    """A stray ``priority:*`` value is treated as no priority (lowest), never raises."""
    gh = FakeAdhocGh(
        [_ready(1, labels=["priority:bogus"]), _ready(2, labels=["priority:high"])]
    )
    build = RecordingAdhocBuild()

    drained = await _drain(gh=gh, build=build, tmp_path=tmp_path)

    assert [issue.issue_number for issue in drained] == [2, 1]


# --- concurrency: bounded by max_parallel -----------------------------------------


@pytest.mark.asyncio
async def test_the_drain_is_bounded_by_max_parallel(tmp_path: Path) -> None:
    """AC3: at most ``max_parallel`` ad-hoc builds run concurrently."""
    gate = asyncio.Event()
    gh = FakeAdhocGh([_ready(n) for n in range(10)])
    build = RecordingAdhocBuild(gate=gate)

    config = RepoConfig(max_parallel=3)
    drain = asyncio.create_task(
        _drain(gh=gh, build=build, config=config, tmp_path=tmp_path)
    )
    # Each build first meters against the (real, SQLite-backed) shared budget, so wait for
    # the semaphore to fill rather than pumping a fixed number of event-loop turns.
    await _wait_until(lambda: build.in_flight >= 3)
    gate.set()
    await drain

    assert build.max_in_flight == 3
    assert len(build.built) == 10


@pytest.mark.asyncio
async def test_an_unset_max_parallel_builds_all_visible_issues(tmp_path: Path) -> None:
    """An unset ``max_parallel`` does not block: every ready issue is still built."""
    gh = FakeAdhocGh([_ready(n) for n in range(5)])
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, tmp_path=tmp_path)

    assert len(build.built) == 5


@pytest.mark.asyncio
async def test_an_empty_ready_set_drives_no_build(tmp_path: Path) -> None:
    """An empty ready set drives no build and touches no downstream."""
    gh = FakeAdhocGh([])
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, tmp_path=tmp_path)

    assert build.built == []


# --- AC1 dedup: an in-flight issue (branch AND open PR) is not rebuilt --------------


@pytest.mark.asyncio
async def test_an_in_flight_issue_is_not_rebuilt(tmp_path: Path) -> None:
    """AC1: an issue whose branch AND open PR exist is skipped — no rebuild, no PR."""
    gh = FakeAdhocGh([_ready(7), _ready(9)], in_flight_numbers={9})
    build = RecordingAdhocBuild()
    open_pr = RecordingPrOpen()

    await _drain(gh=gh, build=build, open_pr=open_pr, tmp_path=tmp_path)

    assert [issue.issue_number for issue in build.built] == [7]
    # A truly in-flight issue gets neither a rebuild nor a (duplicate) PR open.
    assert open_pr.opened == []
    # The flight-state truth was read once for the whole repo, not per issue.
    assert gh.snapshot_calls == ["owner/repo"]
    assert gh.flight_state_calls == []


@pytest.mark.asyncio
async def test_flight_state_is_classified_with_one_whole_repo_query(
    tmp_path: Path,
) -> None:
    """The drain reads flight-state truth once for the whole repo — not a query per issue.

    The N+1 regression target: classification must be a single ``flight_snapshot`` call and
    then in-memory classification, not an ``flight_state`` spawn per candidate.
    """
    gh = FakeAdhocGh(
        [_ready(7), _ready(8), _ready(9)],
        stranded_numbers={8},
        in_flight_numbers={9},
    )
    build = RecordingAdhocBuild()
    open_pr = RecordingPrOpen()

    await _drain(gh=gh, build=build, open_pr=open_pr, tmp_path=tmp_path)

    assert gh.snapshot_calls == ["owner/repo"]  # exactly one whole-repo query
    assert gh.flight_state_calls == []  # never per-issue
    assert [issue.issue_number for issue in build.built] == [7]
    assert [issue.issue_number for issue in open_pr.opened] == [8]


@pytest.mark.asyncio
async def test_drain_falls_back_to_per_issue_flight_state(tmp_path: Path) -> None:
    """A seam without a whole-repo snapshot is classified per issue via ``flight_state``.

    Keeps the drain working with a minimal gh seam (e.g. the wiring test's fake) that only
    implements the per-issue ``flight_state`` — the fallback preserves the same semantics.
    """
    gh = _FlightStateOnlyGh([_ready(7), _ready(8)], stranded_numbers={8})
    build = RecordingAdhocBuild()
    open_pr = RecordingPrOpen()

    await _drain(gh=gh, build=build, open_pr=open_pr, tmp_path=tmp_path)

    assert [issue.issue_number for issue in build.built] == [7]
    assert [issue.issue_number for issue in open_pr.opened] == [8]
    assert set(gh.flight_state_calls) == {7, 8}


@pytest.mark.asyncio
async def test_all_in_flight_drives_no_build(tmp_path: Path) -> None:
    """AC1: when every candidate is already in flight, the drain builds nothing."""
    gh = FakeAdhocGh([_ready(7), _ready(9)], in_flight_numbers={7, 9})
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, tmp_path=tmp_path)

    assert build.built == []


# --- stranded recovery: a green branch with no PR gets its PR opened, not rebuilt ---


@pytest.mark.asyncio
async def test_a_stranded_branch_opens_its_pr_without_rebuilding(tmp_path: Path) -> None:
    """A pushed (green) ``issue-<N>`` branch with no open PR gets its PR opened, no rebuild.

    The regression target: a green build that pushed its branch but never opened a PR
    (e.g. the PR-open precheck failed) must not be wrongly treated as "in flight" and
    dropped forever. The drain recognizes the stranded branch and opens its PR through the
    PR-open-only seam — skipping the (wasteful, and budget-spending) rebuild.
    """
    gh = FakeAdhocGh([_ready(48)], stranded_numbers={48})
    build = RecordingAdhocBuild()
    open_pr = RecordingPrOpen()

    await _drain(gh=gh, build=build, open_pr=open_pr, tmp_path=tmp_path)

    assert build.built == []  # no rebuild
    assert [issue.issue_number for issue in open_pr.opened] == [48]  # PR opened instead


@pytest.mark.asyncio
async def test_stranded_pr_open_is_not_budget_metered(tmp_path: Path) -> None:
    """Opening a stranded branch's PR does no model work, so it spends no shared budget.

    A spent (zero-headroom) governor would decline any *build*, but opening the PR for an
    already-green branch is a free gh call — so it still happens even when the budget is
    exhausted.
    """
    governor = _governor(tmp_path, weekly=0.0)  # cap 0 -> any build is declined
    gh = FakeAdhocGh([_ready(48)], stranded_numbers={48})
    build = RecordingAdhocBuild()
    open_pr = RecordingPrOpen()

    await _drain(gh=gh, build=build, open_pr=open_pr, governor=governor)

    assert build.built == []
    assert [issue.issue_number for issue in open_pr.opened] == [48]


@pytest.mark.asyncio
async def test_absent_stranded_and_in_flight_are_handled_independently(
    tmp_path: Path,
) -> None:
    """The three flight states fan out: build absent, open-PR stranded, skip in-flight."""
    gh = FakeAdhocGh(
        [_ready(7), _ready(8), _ready(9)],
        stranded_numbers={8},
        in_flight_numbers={9},
    )
    build = RecordingAdhocBuild()
    open_pr = RecordingPrOpen()

    await _drain(gh=gh, build=build, open_pr=open_pr, tmp_path=tmp_path)

    assert [issue.issue_number for issue in build.built] == [7]  # absent -> build
    assert [issue.issue_number for issue in open_pr.opened] == [8]  # stranded -> open PR
    # #9 (in flight) is neither built nor PR'd.


@pytest.mark.asyncio
async def test_only_stranded_issues_still_open_their_prs(tmp_path: Path) -> None:
    """With no buildable issue, the drain still opens every stranded branch's PR.

    Guards the early-exit: a drain whose only ready issues are stranded must not return
    empty-handed before recovering them — their PRs still open.
    """
    gh = FakeAdhocGh([_ready(48), _ready(49)], stranded_numbers={48, 49})
    build = RecordingAdhocBuild()
    open_pr = RecordingPrOpen()

    await _drain(gh=gh, build=build, open_pr=open_pr, tmp_path=tmp_path)

    assert build.built == []
    assert {issue.issue_number for issue in open_pr.opened} == {48, 49}


@pytest.mark.asyncio
async def test_a_stranded_pr_is_opened_through_from_fetched_issue(tmp_path: Path) -> None:
    """A stranded issue is materialized via ``from_fetched_issue`` (chain depth stays live).

    The recovery path must build its :class:`AdhocIssue` the same way the build path does,
    so a recovered review-fix issue keeps its ``Chain-depth:`` lineage rather than
    defaulting to a chain origin.
    """
    body = f"a review-fix to apply.\n\n{render_chain_depth(2)}"
    gh = FakeAdhocGh([_ready(503, body=body)], stranded_numbers={503})
    build = RecordingAdhocBuild()
    open_pr = RecordingPrOpen()

    await _drain(gh=gh, build=build, open_pr=open_pr, tmp_path=tmp_path)

    assert open_pr.opened == [
        AdhocIssue(repo_full_name="owner/repo", issue_number=503, chain_depth=2)
    ]


# --- AC2 single-run lock: two concurrent drains never overlap ---------------------


@pytest.mark.asyncio
async def test_a_second_concurrent_drain_is_rejected_by_the_lock(
    tmp_path: Path,
) -> None:
    """AC2: a second drain entered while one holds the lock raises (never overlaps)."""
    gate = asyncio.Event()
    gh = FakeAdhocGh([_ready(7)])
    build = RecordingAdhocBuild(gate=gate)
    lock = _Lock()

    first = asyncio.create_task(
        _drain(gh=gh, build=build, tmp_path=tmp_path, lock=lock)
    )
    for _ in range(50):
        await asyncio.sleep(0)

    # The first drain holds the lock; a second entry is rejected, not queued.
    with pytest.raises(AdhocDrainBusyError):
        await _drain(gh=gh, build=build, tmp_path=tmp_path, lock=lock)

    gate.set()
    await first
    assert [issue.issue_number for issue in build.built] == [7]


@pytest.mark.asyncio
async def test_adhoc_drain_lock_rejects_a_second_holder_then_reenters() -> None:
    """The production lock rejects a concurrent second holder, then frees on exit.

    :class:`AdhocDrainLock` is the production single-run guard the worker binds per repo:
    the first holder enters; a second concurrent ``__aenter__`` raises (never blocks); and
    once the first exits the lock is free to be entered again (a later kick is not poisoned).
    """
    lock = AdhocDrainLock()
    async with lock:
        with pytest.raises(AdhocDrainBusyError):
            async with lock:
                pass
    # Freed on exit: a subsequent drain can take it again.
    async with lock:
        pass


# --- AC3 shared budget governor: each build meters the one shared budget -----------


@pytest.mark.asyncio
async def test_each_build_meters_the_shared_budget(tmp_path: Path) -> None:
    """AC3: every ad-hoc build charges the shared governor; an over-budget build stops.

    cap = 12 (12% of weekly 100). Two builds at 5.0 each fit (10.0 <= 12); a third
    would cross the cap, so it is not built. The shared budget is the one the PRD lane
    meters too.
    """
    governor = _governor(tmp_path, weekly=100.0)
    gh = FakeAdhocGh([_ready(1), _ready(2), _ready(3)])
    build = RecordingAdhocBuild()

    await _drain(
        gh=gh, build=build, governor=governor, estimated_amount=5.0
    )

    assert len(build.built) == 2
    assert await governor._ledger.trailing_24h_spend() == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_a_prd_charge_already_on_the_ledger_starves_the_drain(
    tmp_path: Path,
) -> None:
    """AC3: a PRD build and an ad-hoc drain share the budget — a PRD charge crowds out.

    The PRD lane has already metered 11.0 of the 12.0 cap on the *shared* ledger. The
    ad-hoc drain's 2.0 build can't fit, so it is not built — the two lanes run at once but
    share one budget governor.
    """
    governor = _governor(tmp_path, weekly=100.0)
    await governor._ledger.record_spend(amount=11.0)  # the PRD lane's prior charge
    gh = FakeAdhocGh([_ready(1)])
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, governor=governor, estimated_amount=2.0)

    assert build.built == []


# --- AC4 PRD-first ordering with priority:critical|high preemption -----------------


@pytest.mark.asyncio
async def test_prd_first_defers_ordinary_adhoc_when_a_prd_is_in_flight(
    tmp_path: Path,
) -> None:
    """AC4: with a PRD in flight, an ordinary (non-preempting) ad-hoc issue waits."""
    gh = FakeAdhocGh([_ready(1, labels=["priority:low"]), _ready(2)])
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, tmp_path=tmp_path, prd_in_flight=True)

    assert build.built == []


@pytest.mark.asyncio
async def test_a_critical_or_high_adhoc_preempts_prd_first_ordering(
    tmp_path: Path,
) -> None:
    """AC4: a ``priority:critical``/``high`` ad-hoc issue preempts a PRD in flight."""
    gh = FakeAdhocGh(
        [
            _ready(1, labels=["priority:low"]),  # ordinary -> waits behind the PRD
            _ready(2, labels=["priority:high"]),  # preempts
            _ready(3, labels=["priority:critical"]),  # preempts
        ]
    )
    build = RecordingAdhocBuild()

    drained = await _drain(gh=gh, build=build, tmp_path=tmp_path, prd_in_flight=True)

    # Only the preempting issues build, in rank order (critical before high).
    assert [issue.issue_number for issue in drained] == [3, 2]
    assert {issue.issue_number for issue in build.built} == {2, 3}


@pytest.mark.asyncio
async def test_no_prd_in_flight_builds_every_ranked_adhoc_issue(
    tmp_path: Path,
) -> None:
    """AC4: with no PRD in flight, PRD-first does not apply — every issue builds."""
    gh = FakeAdhocGh([_ready(1, labels=["priority:low"]), _ready(2)])
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, tmp_path=tmp_path, prd_in_flight=False)

    assert {issue.issue_number for issue in build.built} == {1, 2}


# --- chain-depth: built through from_fetched_issue (the #39/#40 bound stays live) --


@pytest.mark.asyncio
async def test_each_issue_is_built_through_from_fetched_issue(tmp_path: Path) -> None:
    """AC5: a fetched body carrying ``Chain-depth: <n>`` yields ``chain_depth == n``.

    The drain MUST materialize each issue via
    :meth:`AdhocIssue.from_fetched_issue` fed the fetched body — not the bare
    constructor — so the lineage marker is read back and the #39/#40 review-fix chain
    bound stays live.
    """
    body = f"a review-fix to apply.\n\n{render_chain_depth(2)}"
    gh = FakeAdhocGh([_ready(503, body=body)])
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, tmp_path=tmp_path)

    assert build.built == [
        AdhocIssue(repo_full_name="owner/repo", issue_number=503, chain_depth=2)
    ]


@pytest.mark.asyncio
async def test_a_marker_less_body_builds_a_chain_origin(tmp_path: Path) -> None:
    """A ready issue with no ``Chain-depth:`` marker is a chain origin (depth 0)."""
    gh = FakeAdhocGh([_ready(29, body="a hand-filed nit, no marker")])
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, tmp_path=tmp_path)

    assert build.built == [AdhocIssue(repo_full_name="owner/repo", issue_number=29)]


# --- real GhCli: command assembly, auth env, payload parsing ----------------------


class CapturingGhRunner:
    """Records the argv + env it was called with and returns a canned stdout payload."""

    def __init__(self, stdout: bytes = b"[]") -> None:
        self._stdout = stdout
        self.argv: Sequence[str] | None = None
        self.env: Mapping[str, str] | None = None

    async def __call__(self, argv: Sequence[str], env: Mapping[str, str]) -> bytes:
        self.argv = argv
        self.env = env
        return self._stdout


@pytest.mark.asyncio
async def test_ghcli_assembles_the_ready_list_command() -> None:
    """GhCli runs ``gh issue list`` scoped to the repo's open ``ready-for-agent`` issues."""
    runner = CapturingGhRunner()
    gh = GhCli(token="t0ken", runner=runner, list_limit=50)

    await gh.list_ready(repo_full_name="owner/repo")

    argv = list(runner.argv or [])
    assert argv[:3] == ["gh", "issue", "list"]
    assert "--repo" in argv and argv[argv.index("--repo") + 1] == "owner/repo"
    assert "--label" in argv and argv[argv.index("--label") + 1] == "ready-for-agent"
    assert "--state" in argv and argv[argv.index("--state") + 1] == "open"
    assert "--limit" in argv and argv[argv.index("--limit") + 1] == "50"
    # The body must be surfaced so the drain can feed from_fetched_issue.
    assert argv[argv.index("--json") + 1] == "number,labels,body"


@pytest.mark.asyncio
async def test_ghcli_puts_the_token_in_the_env_not_the_argv() -> None:
    """The token authenticates via GH_TOKEN in the child env, never on the command line."""
    runner = CapturingGhRunner()
    gh = GhCli(token="s3cret", runner=runner)

    await gh.list_ready(repo_full_name="owner/repo")

    assert (runner.env or {}).get("GH_TOKEN") == "s3cret"
    assert "s3cret" not in list(runner.argv or [])


@pytest.mark.asyncio
async def test_ghcli_omits_the_auth_env_when_no_token() -> None:
    """With no token GhCli leaves the auth env empty, deferring to gh's ambient auth."""
    runner = CapturingGhRunner()
    gh = GhCli(token=None, runner=runner)

    await gh.list_ready(repo_full_name="owner/repo")

    assert "GH_TOKEN" not in (runner.env or {})


@pytest.mark.asyncio
async def test_ghcli_parses_the_gh_json_payload() -> None:
    """GhCli parses gh's JSON listing into ReadyIssue objects with labels + body."""
    payload = json.dumps(
        [
            {
                "number": 7,
                "body": f"a fix.\n\n{render_chain_depth(1)}",
                "labels": [{"name": "ready-for-agent"}, {"name": "priority:high"}],
            },
            {
                "number": 9,
                "body": "",
                "labels": [{"name": "ready-for-agent"}],
            },
        ]
    ).encode()
    gh = GhCli(runner=CapturingGhRunner(stdout=payload))

    issues = await gh.list_ready(repo_full_name="owner/repo")

    assert [issue.number for issue in issues] == [7, 9]
    assert issues[0].labels == ["ready-for-agent", "priority:high"]
    assert issues[0].body == f"a fix.\n\n{render_chain_depth(1)}"
    assert issues[0].severity() is Severity.HIGH
    assert issues[1].severity() is None


@pytest.mark.asyncio
async def test_ghcli_rejects_a_non_array_payload() -> None:
    """A payload that is not a JSON array raises rather than silently dropping issues."""
    gh = GhCli(runner=CapturingGhRunner(stdout=b'{"number": 7}'))

    with pytest.raises(ValueError):
        await gh.list_ready(repo_full_name="owner/repo")


# --- real GhCli flight state: branch + open-PR truth via gh ------------------------


class SequencedGhRunner:
    """Returns a queued stdout per call, recording each argv it was handed.

    The flight-state check fans out to up to two gh calls (branch-existence, then open-PR);
    a queue of canned payloads lets a test script each leg's answer independently.
    """

    def __init__(self, stdouts: list[bytes]) -> None:
        self._stdouts = list(stdouts)
        self.argvs: list[Sequence[str]] = []

    async def __call__(self, argv: Sequence[str], env: Mapping[str, str]) -> bytes:
        self.argvs.append(argv)
        return self._stdouts.pop(0)


@pytest.mark.asyncio
async def test_ghcli_flight_state_in_flight_when_branch_and_open_pr() -> None:
    """A branch *and* an open ``issue-<N>`` PR -> IN_FLIGHT (a build is under way/landed)."""
    branch_ref = json.dumps({"ref": "refs/heads/issue-7"}).encode()
    open_pr = json.dumps([{"number": 88}]).encode()
    runner = SequencedGhRunner([branch_ref, open_pr])
    gh = GhCli(runner=runner)

    assert (
        await gh.flight_state(repo_full_name="owner/repo", issue_number=7)
        is FlightState.IN_FLIGHT
    )

    # Both legs ran: the branch ref lookup, then the open-PR list for its head.
    assert len(runner.argvs) == 2
    branch_argv = list(runner.argvs[0])
    assert branch_argv[0] == "gh" and branch_argv[1] == "api"
    assert "repos/owner/repo/git/ref/heads/issue-7" in branch_argv
    pr_argv = list(runner.argvs[1])
    assert "pr" in pr_argv and "list" in pr_argv
    assert pr_argv[pr_argv.index("--head") + 1] == "issue-7"
    assert pr_argv[pr_argv.index("--state") + 1] == "open"


@pytest.mark.asyncio
async def test_ghcli_flight_state_stranded_when_branch_but_no_open_pr() -> None:
    """A pushed ``issue-<N>`` branch with no open PR -> STRANDED (green build, PR never opened)."""
    branch_ref = json.dumps({"ref": "refs/heads/issue-7"}).encode()
    runner = SequencedGhRunner([branch_ref, b"[]"])
    gh = GhCli(runner=runner)

    assert (
        await gh.flight_state(repo_full_name="owner/repo", issue_number=7)
        is FlightState.STRANDED
    )


@pytest.mark.asyncio
async def test_ghcli_flight_state_absent_when_no_branch() -> None:
    """No ``issue-<N>`` branch -> ABSENT, and the open-PR leg is skipped.

    An open PR keeps its head branch alive, so a missing branch proves no open PR exists:
    the adapter short-circuits to ABSENT without the second gh call.
    """
    runner = _RunnerWithBranchMiss(open_pr_payload=b"[]")
    gh = GhCli(runner=runner)

    assert (
        await gh.flight_state(repo_full_name="owner/repo", issue_number=7)
        is FlightState.ABSENT
    )
    # The branch miss short-circuits: the open-PR leg was never queried.
    assert runner.pr_argv is None


class _RunnerWithBranchMiss:
    """A runner whose branch-ref leg always 404s, then answers the PR leg from a payload.

    The branch-existence leg raises :class:`GhCliError` (a missing ref is a 404 / non-zero
    exit), which the adapter reads as "no branch"; the PR leg returns the scripted payload.
    """

    def __init__(self, *, open_pr_payload: bytes) -> None:
        self._open_pr_payload = open_pr_payload
        self.pr_argv: Sequence[str] | None = None

    async def __call__(self, argv: Sequence[str], env: Mapping[str, str]) -> bytes:
        from retinue.gh import GhCliError

        if "git/ref/heads/issue-7" in " ".join(argv) or any(
            "git/ref" in part for part in argv
        ):
            raise GhCliError(argv, returncode=1, stderr="Not Found")
        self.pr_argv = argv
        return self._open_pr_payload


# --- FlightSnapshot: in-memory classification preserves FlightState semantics ------


def test_flight_snapshot_classifies_each_state() -> None:
    """The whole-repo snapshot classifies absent / stranded / in-flight exactly as before."""
    snapshot = FlightSnapshot(
        open_pr_heads=frozenset({"issue-7"}),
        issue_branches=frozenset({"issue-7", "issue-8"}),
    )

    assert snapshot.state_for(7) is FlightState.IN_FLIGHT  # branch + open PR
    assert snapshot.state_for(8) is FlightState.STRANDED  # branch, no open PR
    assert snapshot.state_for(9) is FlightState.ABSENT  # no branch


# --- real GhCli.flight_snapshot: two whole-repo queries ----------------------------


@pytest.mark.asyncio
async def test_ghcli_flight_snapshot_uses_two_whole_repo_queries() -> None:
    """One ``gh pr list`` for open-PR heads + one ``gh api`` matching-refs enumeration.

    Replaces the per-issue branch-ref + open-PR spawns with two repo-wide queries, then
    classifies in memory — the same FlightState verdicts, without the N+1 spawn fan-out.
    """
    open_prs = json.dumps([{"headRefName": "issue-7"}]).encode()
    refs = json.dumps(
        [{"ref": "refs/heads/issue-7"}, {"ref": "refs/heads/issue-48"}]
    ).encode()
    runner = SequencedGhRunner([open_prs, refs])
    gh = GhCli(runner=runner)

    snapshot = await gh.flight_snapshot(repo_full_name="owner/repo")

    # Exactly two whole-repo queries, no per-issue calls.
    assert len(runner.argvs) == 2
    pr_argv = list(runner.argvs[0])
    assert pr_argv[:3] == ["gh", "pr", "list"]
    assert pr_argv[pr_argv.index("--repo") + 1] == "owner/repo"
    assert pr_argv[pr_argv.index("--state") + 1] == "open"
    assert pr_argv[pr_argv.index("--json") + 1] == "headRefName"
    refs_argv = list(runner.argvs[1])
    assert refs_argv[:2] == ["gh", "api"]
    assert any("git/matching-refs/heads/issue-" in part for part in refs_argv)
    # And the parsed snapshot classifies each candidate with the preserved semantics.
    assert snapshot.state_for(7) is FlightState.IN_FLIGHT
    assert snapshot.state_for(48) is FlightState.STRANDED
    assert snapshot.state_for(99) is FlightState.ABSENT


@pytest.mark.asyncio
async def test_ghcli_flight_snapshot_puts_the_token_in_the_env() -> None:
    """Both whole-repo queries authenticate via GH_TOKEN in the child env, never on argv."""
    runner = SequencedGhRunner([b"[]", b"[]"])
    gh = GhCli(token="s3cret", runner=runner)

    await gh.flight_snapshot(repo_full_name="owner/repo")

    for argv in runner.argvs:
        assert "s3cret" not in list(argv)


@pytest.mark.asyncio
async def test_ghcli_flight_snapshot_handles_an_empty_repo() -> None:
    """No open PRs and no issue branches -> every issue classifies ABSENT."""
    runner = SequencedGhRunner([b"[]", b"[]"])
    gh = GhCli(runner=runner)

    snapshot = await gh.flight_snapshot(repo_full_name="owner/repo")

    assert snapshot.state_for(7) is FlightState.ABSENT
