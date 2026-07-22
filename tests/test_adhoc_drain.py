"""Tests for the scheduler drain (PRD #80).

The drain lists every open trigger-labeled issue via the gh seam, admits the ones the
scheduler acts on (trigger label present, ``hitl`` absent), gates them on blocked-by
readiness, classifies flight state, and drives the build+PR primitive for the selected
issues through the pure two-queue scheduler (priority queue first, reserved priority slot):

1. **list** — pull the repo's open trigger-labeled issues (number, labels, body),
2. **admit** — keep trigger-labeled, non-``hitl`` issues,
3. **readiness** — drop any issue with an open blocker (union of body ``## Blocked by #N``
   refs and native GitHub relations),
4. **classify** — partition by flight state (absent / stranded / in-flight),
5. **rank + select** — two-queue tier ranking with the reserved priority slot,
6. **drive** — materialize each selected issue via :meth:`AdhocIssue.from_fetched_issue`
   and run the injected build, metered against the shared budget.

Every collaborator — the gh issue query, the readiness lookups, and the downstream build —
is injected and faked, so the whole drain runs with no real ``gh``, no Docker, no network.
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
from retinue.readiness import ReadinessGh
from retinue.repo_config import RepoConfig
from retinue.run_ledger import RunLedgerStore, RunState
from tests.fakes import FakeAdhocGh, FakeClock


def _ready(
    number: int, *, labels: list[str] | None = None, body: str = ""
) -> ReadyIssue:
    """A trigger-labeled issue as the gh seam reports it (number, labels, body)."""
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
    """Yield the event loop until ``predicate()`` is truthy (bounded so a test fails fast)."""
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
    readiness_gh: ReadinessGh | None = None,
    open_pr: AdhocPrOpen | None = None,
    config: RepoConfig | None = None,
    governor: BudgetGovernor | None = None,
    ledger: RunLedgerStore | None = None,
    tmp_path: Path | None = None,
    lock: AbstractAsyncContextManager[object] | None = None,
    estimated_amount: float = 1.0,
) -> list[AdhocIssue]:
    """Invoke the drain with sensible defaults so each test sets only what it exercises.

    Exactly one of ``governor`` or ``tmp_path`` must be given (``tmp_path`` also backs the
    default run-ledger). When ``readiness_gh`` is not pinned, the same fake gh answers
    readiness (its default: no blockers, so all ready).
    """
    if governor is None:
        assert tmp_path is not None, "pass a governor or a tmp_path for a default one"
        governor = _governor(tmp_path)
    if ledger is None:
        assert tmp_path is not None, "pass a tmp_path (or ledger) so the drain can record run state"
        ledger = RunLedgerStore(tmp_path / "run-ledger.sqlite3")
    if readiness_gh is None:
        assert isinstance(gh, ReadinessGh), "gh must answer readiness or pass readiness_gh"
        readiness_gh = gh
    try:
        return await run_adhoc_drain(
            repo_full_name="owner/repo",
            gh=gh,
            readiness_gh=readiness_gh,
            build=build,
            open_pr=open_pr or RecordingPrOpen(),
            config=config or RepoConfig(),
            governor=governor,
            ledger=ledger,
            estimated_amount=estimated_amount,
            lock=lock or _nolock(),
        )
    finally:
        await governor.close()


class _FlightStateOnlyGh:
    """A seam offering only per-issue ``flight_state`` (no whole-repo snapshot).

    Exercises the drain's fallback and answers readiness (no blockers) so it can double as
    the readiness seam.
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

    async def list_ready(
        self, *, repo_full_name: str, label: str
    ) -> list[ReadyIssue]:
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

    async def native_blockers(
        self, *, repo_full_name: str, issue_number: int
    ) -> list[int]:
        return []

    async def is_closed(self, *, repo_full_name: str, issue_number: int) -> bool:
        return False


class RecordingPrOpen:
    """Records each AdhocIssue whose stranded PR the drain opened (the PR-open-only seam)."""

    def __init__(self) -> None:
        self.opened: list[AdhocIssue] = []

    async def __call__(self, issue: AdhocIssue, *, repo_full_name: str) -> None:
        self.opened.append(issue)


class RecordingAdhocBuild:
    """Records each AdhocIssue handed to the downstream build (the mocked build+PR).

    ``invoked`` is start order (rank order, recorded before any await); ``built`` is
    completion order. ``max_in_flight`` records peak concurrency.
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


# --- admission: trigger-labeled, hitl absent --------------------------------------


@pytest.mark.asyncio
async def test_drain_drives_the_build_for_each_ready_issue(tmp_path: Path) -> None:
    """The drain drives the build primitive for each admitted, ready issue."""
    gh = FakeAdhocGh([_ready(7), _ready(9)])
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, tmp_path=tmp_path)

    assert {issue.issue_number for issue in build.built} == {7, 9}
    assert gh.calls == ["owner/repo"]
    assert gh.list_labels == ["ready-for-agent"]


@pytest.mark.asyncio
async def test_the_list_query_uses_the_configured_trigger_label(tmp_path: Path) -> None:
    """A repo that renames its trigger label lists on that label, not the default."""
    gh = FakeAdhocGh([_ready(7)])
    build = RecordingAdhocBuild()
    config = RepoConfig(trigger_label="build-me")

    await _drain(gh=gh, build=build, config=config, tmp_path=tmp_path)

    assert gh.list_labels == ["build-me"]


@pytest.mark.asyncio
async def test_a_hitl_issue_is_excluded(tmp_path: Path) -> None:
    """A ``hitl``-labeled issue is escalated to a human, so the scheduler skips it."""
    gh = FakeAdhocGh([_ready(7), _ready(8, labels=["hitl"])])
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, tmp_path=tmp_path)

    assert [issue.issue_number for issue in build.built] == [7]


# --- readiness: blocked issues invisible until blockers close ----------------------


@pytest.mark.asyncio
async def test_a_body_blocked_issue_is_invisible_until_its_blocker_closes(
    tmp_path: Path,
) -> None:
    """An issue with an open ``## Blocked by #N`` blocker is not scheduled."""
    gh = FakeAdhocGh(
        [_ready(7), _ready(8, body="Do the thing.\n\n## Blocked by\n\n- #7")],
    )
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, tmp_path=tmp_path)

    # #7 is open, so #8 is blocked; only #7 builds.
    assert [issue.issue_number for issue in build.built] == [7]


@pytest.mark.asyncio
async def test_drain_records_queued_on_admit_and_building_on_build_start(
    tmp_path: Path,
) -> None:
    """The drain records ``queued`` for each admitted issue and ``building`` at build start.

    #8 is admitted but blocked by the still-open #7, so it stays ``queued`` and never
    builds; #7 is admitted (``queued``) then built, upserting its row to ``building``.
    """
    gh = FakeAdhocGh([_ready(7), _ready(8, body="Do it.\n\n## Blocked by\n\n- #7")])
    build = RecordingAdhocBuild()
    ledger = RunLedgerStore(tmp_path / "run-ledger.sqlite3")

    await _drain(gh=gh, build=build, tmp_path=tmp_path, ledger=ledger)

    states = {r.issue: r.state for r in await ledger.rows()}
    assert states[8] == RunState.QUEUED.value  # admitted but blocked -> queued, never built
    assert states[7] == RunState.BUILDING.value  # built -> queued upserted to building


@pytest.mark.asyncio
async def test_an_in_flight_issue_does_not_regress_from_building_to_queued(
    tmp_path: Path,
) -> None:
    """A built issue that goes in-flight keeps ``building`` across the next drain pass.

    An in-flight (open-PR) issue keeps its trigger label until reap, so a later drain
    still admits it. Re-recording ``queued`` for it must not clobber the ``building`` its
    build left on the ledger — else ``/api/runs`` misreports an open-PR issue as queued.
    """
    ledger = RunLedgerStore(tmp_path / "run-ledger.sqlite3")

    # Pass 1: #9 is buildable and builds -> building.
    await _drain(
        gh=FakeAdhocGh([_ready(9)]),
        build=RecordingAdhocBuild(),
        tmp_path=tmp_path,
        ledger=ledger,
    )
    # Pass 2: #9 now has an open PR (in-flight) but still carries the trigger label.
    await _drain(
        gh=FakeAdhocGh([_ready(9)], in_flight_numbers={9}),
        build=RecordingAdhocBuild(),
        tmp_path=tmp_path,
        ledger=ledger,
    )

    states = {r.issue: r.state for r in await ledger.rows()}
    assert states[9] == RunState.BUILDING.value


@pytest.mark.asyncio
async def test_a_blocked_issue_becomes_ready_when_its_blocker_closes(
    tmp_path: Path,
) -> None:
    """Once every blocker is closed, the dependent is admitted and built."""
    gh = FakeAdhocGh(
        [_ready(8, body="Do the thing.\n\n## Blocked by\n\n- #7")],
        closed_numbers={7},
    )
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, tmp_path=tmp_path)

    assert [issue.issue_number for issue in build.built] == [8]


@pytest.mark.asyncio
async def test_a_native_blocked_issue_is_invisible_until_its_blocker_closes(
    tmp_path: Path,
) -> None:
    """GitHub's native ``blocked_by`` relation gates readiness too (union source)."""
    gh = FakeAdhocGh(
        [_ready(7), _ready(8)],
        native_blockers={8: [7]},
    )
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, tmp_path=tmp_path)

    assert [issue.issue_number for issue in build.built] == [7]


# --- ranking: two queues, tier-ordered, untiered last -----------------------------


@pytest.mark.asyncio
async def test_issues_are_ranked_priority_queue_first(tmp_path: Path) -> None:
    """Ready issues drain priority-queue-first, tier-ordered, untiered last."""
    gh = FakeAdhocGh(
        [
            _ready(1),  # untiered -> lowest
            _ready(2, labels=["priority:high"]),  # priority queue
            _ready(3, labels=["priority:critical"]),  # priority queue, top
            _ready(4, labels=["priority:low"]),  # main queue
        ]
    )
    build = RecordingAdhocBuild()

    drained = await _drain(gh=gh, build=build, tmp_path=tmp_path)

    # The drain's return value is the rank-order surface (build completion can interleave).
    assert [issue.issue_number for issue in drained] == [3, 2, 4, 1]


@pytest.mark.asyncio
async def test_an_unknown_priority_label_ranks_lowest(tmp_path: Path) -> None:
    """A stray ``priority:*`` value is treated as untiered (lowest), never raises."""
    gh = FakeAdhocGh(
        [_ready(1, labels=["priority:bogus"]), _ready(2, labels=["priority:high"])]
    )
    build = RecordingAdhocBuild()

    drained = await _drain(gh=gh, build=build, tmp_path=tmp_path)

    assert [issue.issue_number for issue in drained] == [2, 1]


# --- reserved priority slot -------------------------------------------------------


@pytest.mark.asyncio
async def test_the_main_queue_holds_at_most_cap_minus_one_slots(tmp_path: Path) -> None:
    """With cap N and no priority work, the main queue fills at most N-1 slots this pass."""
    gh = FakeAdhocGh([_ready(n) for n in range(5)])  # all untiered -> main queue
    build = RecordingAdhocBuild()

    drained = await _drain(
        gh=gh, build=build, config=RepoConfig(max_parallel=3), tmp_path=tmp_path
    )

    # 3-1 = 2 main slots; the 3rd is reserved for priority work that isn't here.
    assert len(drained) == 2
    assert [issue.issue_number for issue in drained] == [0, 1]


@pytest.mark.asyncio
async def test_a_priority_arrival_takes_the_reserved_slot(tmp_path: Path) -> None:
    """A priority issue fills the reserved slot alongside cap-1 main issues."""
    gh = FakeAdhocGh(
        [
            _ready(1, labels=["priority:critical"]),
            _ready(2),
            _ready(3),
            _ready(4),
        ]
    )
    build = RecordingAdhocBuild()

    drained = await _drain(
        gh=gh, build=build, config=RepoConfig(max_parallel=3), tmp_path=tmp_path
    )

    # cap 3: the critical takes the reserved slot, then two main issues fill the rest.
    assert [issue.issue_number for issue in drained] == [1, 2, 3]


@pytest.mark.asyncio
async def test_cap_one_is_strict_priority_first(tmp_path: Path) -> None:
    """At cap 1 there is no slot to reserve: the single build is strict priority-first."""
    gh = FakeAdhocGh(
        [_ready(1), _ready(2, labels=["priority:critical"])]
    )
    build = RecordingAdhocBuild()

    drained = await _drain(
        gh=gh, build=build, config=RepoConfig(max_parallel=1), tmp_path=tmp_path
    )

    assert [issue.issue_number for issue in drained] == [2]


@pytest.mark.asyncio
async def test_an_unset_max_parallel_builds_every_ready_issue(tmp_path: Path) -> None:
    """An unset ``max_parallel`` reserves no slot: every ready issue builds this pass."""
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


# --- concurrency: bounded by max_parallel -----------------------------------------


@pytest.mark.asyncio
async def test_selected_builds_run_concurrently_up_to_the_cap(tmp_path: Path) -> None:
    """The selected builds run concurrently, peaking at the cap."""
    gate = asyncio.Event()
    # Three priority-tier issues fill all cap slots (no reserved slot left free).
    gh = FakeAdhocGh([_ready(n, labels=["priority:critical"]) for n in range(3)])
    build = RecordingAdhocBuild(gate=gate)

    config = RepoConfig(max_parallel=3)
    drain = asyncio.create_task(
        _drain(gh=gh, build=build, config=config, tmp_path=tmp_path)
    )
    await _wait_until(lambda: build.in_flight >= 3)
    gate.set()
    await drain

    assert build.max_in_flight == 3
    assert len(build.built) == 3


# --- dedup: an in-flight issue (branch AND open PR) is not rebuilt -----------------


@pytest.mark.asyncio
async def test_an_in_flight_issue_is_not_rebuilt(tmp_path: Path) -> None:
    """An issue whose branch AND open PR exist is skipped — no rebuild, no PR."""
    gh = FakeAdhocGh([_ready(7), _ready(9)], in_flight_numbers={9})
    build = RecordingAdhocBuild()
    open_pr = RecordingPrOpen()

    await _drain(gh=gh, build=build, open_pr=open_pr, tmp_path=tmp_path)

    assert [issue.issue_number for issue in build.built] == [7]
    assert open_pr.opened == []
    assert gh.snapshot_calls == ["owner/repo"]
    assert gh.flight_state_calls == []


@pytest.mark.asyncio
async def test_flight_state_is_classified_with_one_whole_repo_query(
    tmp_path: Path,
) -> None:
    """The drain reads flight-state truth once for the whole repo — not a query per issue."""
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
    """A seam without a whole-repo snapshot is classified per issue via ``flight_state``."""
    gh = _FlightStateOnlyGh([_ready(7), _ready(8)], stranded_numbers={8})
    build = RecordingAdhocBuild()
    open_pr = RecordingPrOpen()

    await _drain(gh=gh, build=build, open_pr=open_pr, tmp_path=tmp_path)

    assert [issue.issue_number for issue in build.built] == [7]
    assert [issue.issue_number for issue in open_pr.opened] == [8]
    assert set(gh.flight_state_calls) == {7, 8}


@pytest.mark.asyncio
async def test_all_in_flight_drives_no_build(tmp_path: Path) -> None:
    """When every candidate is already in flight, the drain builds nothing."""
    gh = FakeAdhocGh([_ready(7), _ready(9)], in_flight_numbers={7, 9})
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, tmp_path=tmp_path)

    assert build.built == []


# --- stranded recovery: a green branch with no PR gets its PR opened, not rebuilt ---


@pytest.mark.asyncio
async def test_a_stranded_branch_opens_its_pr_without_rebuilding(tmp_path: Path) -> None:
    """A pushed (green) ``issue-<N>`` branch with no open PR gets its PR opened, no rebuild."""
    gh = FakeAdhocGh([_ready(48)], stranded_numbers={48})
    build = RecordingAdhocBuild()
    open_pr = RecordingPrOpen()

    await _drain(gh=gh, build=build, open_pr=open_pr, tmp_path=tmp_path)

    assert build.built == []  # no rebuild
    assert [issue.issue_number for issue in open_pr.opened] == [48]  # PR opened instead


@pytest.mark.asyncio
async def test_stranded_pr_open_is_not_budget_metered(tmp_path: Path) -> None:
    """Opening a stranded branch's PR does no model work, so it spends no shared budget."""
    governor = _governor(tmp_path, weekly=0.0)  # cap 0 -> any build is declined
    gh = FakeAdhocGh([_ready(48)], stranded_numbers={48})
    build = RecordingAdhocBuild()
    open_pr = RecordingPrOpen()

    await _drain(gh=gh, build=build, open_pr=open_pr, governor=governor, tmp_path=tmp_path)

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


@pytest.mark.asyncio
async def test_only_stranded_issues_still_open_their_prs(tmp_path: Path) -> None:
    """With no buildable issue, the drain still opens every stranded branch's PR."""
    gh = FakeAdhocGh([_ready(48), _ready(49)], stranded_numbers={48, 49})
    build = RecordingAdhocBuild()
    open_pr = RecordingPrOpen()

    await _drain(gh=gh, build=build, open_pr=open_pr, tmp_path=tmp_path)

    assert build.built == []
    assert {issue.issue_number for issue in open_pr.opened} == {48, 49}


@pytest.mark.asyncio
async def test_a_stranded_pr_is_opened_through_from_fetched_issue(tmp_path: Path) -> None:
    """A stranded issue is materialized via ``from_fetched_issue`` (chain depth stays live)."""
    body = f"a review-fix to apply.\n\n{render_chain_depth(2)}"
    gh = FakeAdhocGh([_ready(503, body=body)], stranded_numbers={503})
    build = RecordingAdhocBuild()
    open_pr = RecordingPrOpen()

    await _drain(gh=gh, build=build, open_pr=open_pr, tmp_path=tmp_path)

    assert open_pr.opened == [
        AdhocIssue(repo_full_name="owner/repo", issue_number=503, chain_depth=2)
    ]


# --- single-run lock: two concurrent drains never overlap -------------------------


@pytest.mark.asyncio
async def test_a_second_concurrent_drain_is_rejected_by_the_lock(
    tmp_path: Path,
) -> None:
    """A second drain entered while one holds the lock raises (never overlaps)."""
    gate = asyncio.Event()
    gh = FakeAdhocGh([_ready(7)])
    build = RecordingAdhocBuild(gate=gate)
    lock = _Lock()

    first = asyncio.create_task(
        _drain(gh=gh, build=build, tmp_path=tmp_path, lock=lock)
    )
    for _ in range(50):
        await asyncio.sleep(0)

    with pytest.raises(AdhocDrainBusyError):
        await _drain(gh=gh, build=build, tmp_path=tmp_path, lock=lock)

    gate.set()
    await first
    assert [issue.issue_number for issue in build.built] == [7]


@pytest.mark.asyncio
async def test_adhoc_drain_lock_rejects_a_second_holder_then_reenters() -> None:
    """The production lock rejects a concurrent second holder, then frees on exit."""
    lock = AdhocDrainLock()
    async with lock:
        with pytest.raises(AdhocDrainBusyError):
            async with lock:
                pass
    async with lock:
        pass


# --- shared budget governor: each build meters the one shared budget --------------


@pytest.mark.asyncio
async def test_each_build_meters_the_shared_budget(tmp_path: Path) -> None:
    """Every build charges the shared governor; an over-budget build stops.

    cap = 12 (12% of weekly 100). Two builds at 5.0 each fit (10.0 <= 12); a third would
    cross the cap, so it is not built. All three are priority-tier so all are selected.
    """
    governor = _governor(tmp_path, weekly=100.0)
    gh = FakeAdhocGh([_ready(n, labels=["priority:critical"]) for n in (1, 2, 3)])
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, governor=governor, estimated_amount=5.0, tmp_path=tmp_path)

    assert len(build.built) == 2
    assert await governor._ledger.trailing_24h_spend() == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_a_prior_charge_on_the_ledger_starves_the_drain(
    tmp_path: Path,
) -> None:
    """A charge already on the shared ledger crowds out a drain build."""
    governor = _governor(tmp_path, weekly=100.0)
    await governor._ledger.record_spend(amount=11.0)  # a prior charge
    gh = FakeAdhocGh([_ready(1)])
    build = RecordingAdhocBuild()

    await _drain(gh=gh, build=build, governor=governor, estimated_amount=2.0, tmp_path=tmp_path)

    assert build.built == []


# --- chain-depth: built through from_fetched_issue (the review-fix bound stays live) --


@pytest.mark.asyncio
async def test_each_issue_is_built_through_from_fetched_issue(tmp_path: Path) -> None:
    """A fetched body carrying ``Chain-depth: <n>`` yields ``chain_depth == n``."""
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
    """GhCli runs ``gh issue list`` scoped to the repo's open trigger-labeled issues."""
    runner = CapturingGhRunner()
    gh = GhCli(token="t0ken", runner=runner, list_limit=50)

    await gh.list_ready(repo_full_name="owner/repo", label="ready-for-agent")

    argv = list(runner.argv or [])
    assert argv[:3] == ["gh", "issue", "list"]
    assert "--repo" in argv and argv[argv.index("--repo") + 1] == "owner/repo"
    assert "--label" in argv and argv[argv.index("--label") + 1] == "ready-for-agent"
    assert "--state" in argv and argv[argv.index("--state") + 1] == "open"
    assert "--limit" in argv and argv[argv.index("--limit") + 1] == "50"
    assert argv[argv.index("--json") + 1] == "number,labels,body"


@pytest.mark.asyncio
async def test_ghcli_lists_on_the_passed_label() -> None:
    """The trigger label is threaded from config into the list command."""
    runner = CapturingGhRunner()
    gh = GhCli(runner=runner)

    await gh.list_ready(repo_full_name="owner/repo", label="build-me")

    argv = list(runner.argv or [])
    assert argv[argv.index("--label") + 1] == "build-me"


@pytest.mark.asyncio
async def test_ghcli_puts_the_token_in_the_env_not_the_argv() -> None:
    """The token authenticates via GH_TOKEN in the child env, never on the command line."""
    runner = CapturingGhRunner()
    gh = GhCli(token="s3cret", runner=runner)

    await gh.list_ready(repo_full_name="owner/repo", label="ready-for-agent")

    assert (runner.env or {}).get("GH_TOKEN") == "s3cret"
    assert "s3cret" not in list(runner.argv or [])


@pytest.mark.asyncio
async def test_ghcli_omits_the_auth_env_when_no_token() -> None:
    """With no token GhCli leaves the auth env empty, deferring to gh's ambient auth."""
    runner = CapturingGhRunner()
    gh = GhCli(token=None, runner=runner)

    await gh.list_ready(repo_full_name="owner/repo", label="ready-for-agent")

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

    issues = await gh.list_ready(repo_full_name="owner/repo", label="ready-for-agent")

    assert [issue.number for issue in issues] == [7, 9]
    assert issues[0].labels == ["ready-for-agent", "priority:high"]
    assert issues[0].body == f"a fix.\n\n{render_chain_depth(1)}"


@pytest.mark.asyncio
async def test_ghcli_rejects_a_non_array_payload() -> None:
    """A payload that is not a JSON array raises rather than silently dropping issues."""
    gh = GhCli(runner=CapturingGhRunner(stdout=b'{"number": 7}'))

    with pytest.raises(ValueError):
        await gh.list_ready(repo_full_name="owner/repo", label="ready-for-agent")


# --- real GhCli flight state: branch + open-PR truth via gh ------------------------


class SequencedGhRunner:
    """Returns a queued stdout per call, recording each argv it was handed."""

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
    """No ``issue-<N>`` branch -> ABSENT, and the open-PR leg is skipped."""
    runner = _RunnerWithBranchMiss(open_pr_payload=b"[]")
    gh = GhCli(runner=runner)

    assert (
        await gh.flight_state(repo_full_name="owner/repo", issue_number=7)
        is FlightState.ABSENT
    )
    assert runner.pr_argv is None


class _RunnerWithBranchMiss:
    """A runner whose branch-ref leg always 404s, then answers the PR leg from a payload."""

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
    """One ``gh pr list`` for open-PR heads + one ``gh api`` matching-refs enumeration."""
    open_prs = json.dumps([{"headRefName": "issue-7"}]).encode()
    refs = json.dumps(
        [{"ref": "refs/heads/issue-7"}, {"ref": "refs/heads/issue-48"}]
    ).encode()
    runner = SequencedGhRunner([open_prs, refs])
    gh = GhCli(runner=runner)

    snapshot = await gh.flight_snapshot(repo_full_name="owner/repo")

    assert len(runner.argvs) == 2
    pr_argv = list(runner.argvs[0])
    assert pr_argv[:3] == ["gh", "pr", "list"]
    assert pr_argv[pr_argv.index("--repo") + 1] == "owner/repo"
    assert pr_argv[pr_argv.index("--state") + 1] == "open"
    assert pr_argv[pr_argv.index("--json") + 1] == "headRefName"
    refs_argv = list(runner.argvs[1])
    assert refs_argv[:2] == ["gh", "api"]
    assert any("git/matching-refs/heads/issue-" in part for part in refs_argv)
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
