"""Tests for the full-PRD orchestrator (issue #7).

The full-PRD flow wraps the single-slice ``build_slice`` primitive (issue #6):

1. **ready set** — pick every slice whose ``blocked_by`` refs are all merged/closed,
2. **fan-out** — spawn implementers in parallel, bounded by ``config.max_parallel``,
3. **topological merge** — merge the green branches in dependency order under the
   done-check (the conflict-resolving merger),
4. **loop** — repeat rounds until the ready set drains.

Every collaborator is faked, reusing the single-slice fakes: a fake implementer that
records build order, a fake git-ops that records merge order and scripts conflicts,
and the done-check container/auth seams from the done-check tests. A merge conflict is
resolved under the done-check or escalates. At most one orchestrator run executes at a
time, guarded by an injected lock. No Agent SDK, no Docker, no gh, no network.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from retinue.container import Container, RunResult
from retinue.orchestrator import (
    _DEFAULT_MAX_PARALLEL,
    ConflictResolution,
    ImplementError,
    OrchestratorBusyError,
    PrdBuildResult,
    PrdSlice,
    Slice,
    _topo_merge_order,
    build_prd,
)
from retinue.repo_config import RepoConfig
from retinue.reviewer import ReviewGenerationError
from tests.test_done_check import CLAUDE_MD, FakeAuth, FakeRuntime, _resolver, _sink
from tests.test_orchestrator import FakeGitOps, FakeImplementer


class RecordingImplementer:
    """Records build order and the live concurrency it observed.

    Each ``implement`` increments a live counter on entry, yields control once, then
    decrements on exit, recording the peak concurrency. ``started`` records issue
    numbers in the order builds began so a test can assert the ready-set ordering.
    """

    def __init__(self) -> None:
        self.started: list[int] = []
        self._live = 0
        self.peak = 0

    async def implement(self, slice_: Slice, *, container: object) -> None:
        self.started.append(slice_.issue_number)
        self._live += 1
        self.peak = max(self.peak, self._live)
        # Yield so siblings scheduled in the same round actually interleave.
        await asyncio.sleep(0)
        self._live -= 1

    def auth_env(self) -> dict[str, str]:
        return {}


class OneAtATimeLock:
    """An async lock that records contention; refuses a second concurrent holder.

    Models the single-orchestrator-run guarantee: ``__aenter__`` raises
    :class:`OrchestratorBusyError` if the lock is already held rather than blocking,
    so a second run is rejected, not silently serialized.
    """

    def __init__(self) -> None:
        self.held = False
        self.acquisitions = 0

    async def __aenter__(self) -> OneAtATimeLock:
        if self.held:
            raise OrchestratorBusyError()
        self.held = True
        self.acquisitions += 1
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.held = False


def _prd_slice(issue_number: int, blocked_by: list[int] | None = None) -> PrdSlice:
    return PrdSlice(
        repo_full_name="owner/repo",
        issue_number=issue_number,
        prd_number=1,
        blocked_by=blocked_by or [],
    )


async def _build_prd(
    slices: list[PrdSlice],
    *,
    runtime: FakeRuntime | None = None,
    git: FakeGitOps | None = None,
    implementer: object | None = None,
    config: RepoConfig | None = None,
    lock: object | None = None,
    resolve_conflict: object | None = None,
    review_round: object | None = None,
) -> PrdBuildResult:
    return await build_prd(
        slices,
        config or RepoConfig(),
        CLAUDE_MD,
        implementer=implementer or FakeImplementer(),
        git=git or FakeGitOps(),
        auth=FakeAuth(),
        runtime=runtime or FakeRuntime(),
        resolve_secret=_resolver({}),
        report=_sink([]),
        lock=lock or OneAtATimeLock(),
        resolve_conflict=resolve_conflict,
        review_round=review_round,
    )


# --- ready-set selection & topological merge order -------------------------------


@pytest.mark.asyncio
async def test_full_prd_builds_in_dependency_order() -> None:
    """A multi-slice PRD merges its branches in topological (dependency) order."""
    # 2 depends on 1; 3 depends on 2. The only valid merge order is 1, 2, 3.
    slices = [
        _prd_slice(3, blocked_by=[2]),
        _prd_slice(1),
        _prd_slice(2, blocked_by=[1]),
    ]
    git = FakeGitOps()

    result = await _build_prd(slices, git=git)

    assert result.merged_issues == [1, 2, 3]
    assert git.merges == [
        ("issue-1", "retinue/prd-1"),
        ("issue-2", "retinue/prd-1"),
        ("issue-3", "retinue/prd-1"),
    ]


@pytest.mark.asyncio
async def test_prd_slices_branch_off_the_integration_branch_not_staging() -> None:
    """Every slice's issue-<N> branch roots on retinue/prd-<n>, never on staging.

    PRD: one integration branch per PRD; implementers branch off *it* so a later-round
    slice builds on already-merged sibling work. Each container checks out issue-<N> off
    ``origin/retinue/prd-1``, and the integration branch is created off staging first.
    """
    timeline: list[str] = []
    git = FakeGitOps(timeline=timeline)
    runtime = FakeRuntime(timeline=timeline)
    config = RepoConfig(staging_branch="staging")
    slices = [_prd_slice(1), _prd_slice(2, blocked_by=[1])]

    await _build_prd(slices, runtime=runtime, git=git, config=config)

    # The integration branch is created off staging before any container is started.
    assert timeline[0] == "create:retinue/prd-1<-staging"
    create_index = timeline.index("create:retinue/prd-1<-staging")
    first_start = next(i for i, e in enumerate(timeline) if e.startswith("start:"))
    assert create_index < first_start
    # Each round's container roots issue-<N> on the integration branch, not staging.
    checkouts = [e for e in runtime.log if "checkout -B issue-" in e]
    assert checkouts  # at least one round ran
    assert all("origin/retinue/prd-1" in e for e in checkouts)
    assert not any("origin/staging" in e for e in checkouts)


@pytest.mark.asyncio
async def test_diamond_dependency_drains_in_rounds() -> None:
    """A diamond graph (1 <- {2,3} <- 4) builds and merges, draining every round."""
    slices = [
        _prd_slice(1),
        _prd_slice(2, blocked_by=[1]),
        _prd_slice(3, blocked_by=[1]),
        _prd_slice(4, blocked_by=[2, 3]),
    ]

    result = await _build_prd(slices)

    # 1 first, 4 last; 2 and 3 in between in either order.
    assert result.merged_issues[0] == 1
    assert result.merged_issues[-1] == 4
    assert set(result.merged_issues) == {1, 2, 3, 4}


def test_topo_merge_order_puts_in_round_blocker_before_dependent() -> None:
    """A round's slices merge in Kahn-topological order, not by issue number.

    When a higher-numbered slice blocks a lower-numbered one inside the same round,
    the blocker must merge first even though its issue number sorts later. Sorting by
    issue number alone (the drifted behavior) would wrongly merge #1 before #2.
    """
    blocker = _prd_slice(2)
    dependent = _prd_slice(1, blocked_by=[2])

    ordered = _topo_merge_order([dependent, blocker])

    assert [s.issue_number for s in ordered] == [2, 1]


def test_topo_merge_order_tiebreaks_independent_slices_by_issue_number() -> None:
    """Mutually independent slices stay deterministic, ordered by issue number."""
    ordered = _topo_merge_order([_prd_slice(3), _prd_slice(1), _prd_slice(2)])

    assert [s.issue_number for s in ordered] == [1, 2, 3]


@pytest.mark.asyncio
async def test_already_closed_blocker_is_satisfied() -> None:
    """A blocker that is not in the PRD's slice list counts as already merged/closed."""
    # Slice 5 is blocked by #99, which is not in the set: treated as already done.
    slices = [_prd_slice(5, blocked_by=[99])]

    result = await _build_prd(slices)

    assert result.merged_issues == [5]


# --- bounded parallel fan-out ----------------------------------------------------


@pytest.mark.asyncio
async def test_fan_out_respects_max_parallel() -> None:
    """Concurrent implementers in a round never exceed config.max_parallel."""
    # Four independent slices, all ready at once, with a cap of 2.
    slices = [_prd_slice(n) for n in range(1, 5)]
    implementer = RecordingImplementer()
    config = RepoConfig(max_parallel=2)

    result = await _build_prd(slices, implementer=implementer, config=config)

    assert implementer.peak <= 2
    assert set(result.merged_issues) == {1, 2, 3, 4}


@pytest.mark.asyncio
async def test_unset_max_parallel_runs_up_to_the_default_cap() -> None:
    """With max_parallel unset, a ready set at/under the default cap runs fully parallel."""
    slices = [_prd_slice(n) for n in range(1, _DEFAULT_MAX_PARALLEL + 1)]
    implementer = RecordingImplementer()

    await _build_prd(slices, implementer=implementer, config=RepoConfig())

    # Exactly ``_DEFAULT_MAX_PARALLEL`` slices were ready, so all ran at once.
    assert implementer.peak == _DEFAULT_MAX_PARALLEL


@pytest.mark.asyncio
async def test_unset_max_parallel_caps_fan_out_at_the_default() -> None:
    """With max_parallel unset, a large ready set is bounded by the default cap, not run all.

    A default-config repo previously sized its round semaphore to the whole ready set, so
    every slice's container started at once. The unset case now falls back to
    ``_DEFAULT_MAX_PARALLEL`` so a big ready set cannot exhaust the host.
    """
    slices = [_prd_slice(n) for n in range(1, _DEFAULT_MAX_PARALLEL + 4)]
    implementer = RecordingImplementer()

    result = await _build_prd(slices, implementer=implementer, config=RepoConfig())

    assert implementer.peak == _DEFAULT_MAX_PARALLEL
    assert set(result.merged_issues) == {s.issue_number for s in slices}


# --- a red slice blocks its dependents -------------------------------------------


@pytest.mark.asyncio
async def test_blocked_slice_stops_its_dependents() -> None:
    """A red slice is not merged, and slices that depend on it never become ready."""
    slices = [_prd_slice(1), _prd_slice(2, blocked_by=[1])]
    # The done-check fails for every slice in this run.
    runtime = FakeRuntime(results={"uv": RunResult(exit_code=1, stderr="boom")})

    result = await _build_prd(slices, runtime=runtime)

    assert result.merged_issues == []
    assert 1 in result.blocked_issues
    # 2 never ran because its blocker never merged.
    assert 2 not in result.merged_issues


@pytest.mark.asyncio
async def test_transitively_blocked_slices_are_skipped_not_dropped() -> None:
    """A subtree pruned by a red upstream slice is reported skipped, not silently lost."""
    # 1 <- 2 <- 3, with slice 1 red. 2 (direct dependent) and 3 (transitive) can
    # never become ready, so both must surface in the skipped bucket — not vanish.
    slices = [
        _prd_slice(1),
        _prd_slice(2, blocked_by=[1]),
        _prd_slice(3, blocked_by=[2]),
    ]
    runtime = FakeRuntime(results={"uv": RunResult(exit_code=1, stderr="boom")})

    result = await _build_prd(slices, runtime=runtime)

    assert result.blocked_issues == [1]
    assert result.skipped_issues == [2, 3]
    # Every input slice lands in exactly one bucket — none is dropped from all of them.
    all_buckets = (
        result.merged_issues
        + result.blocked_issues
        + result.escalated_issues
        + result.skipped_issues
    )
    assert sorted(all_buckets) == [1, 2, 3]


# --- one slice's build raising must not cancel its siblings ----------------------


class FlakyImplementer(FakeImplementer):
    """Builds every slice cleanly except one, whose ``implement`` raises.

    Models a transient per-slice build failure (a Docker/clone/implement error): the
    round must not let it cancel the sibling builds, so the raised slice escalates while
    the green siblings still merge.
    """

    def __init__(self, *, fail_for: int) -> None:
        super().__init__()
        self._fail_for = fail_for

    async def implement(
        self, slice_: Slice, *, container: Container, plan_path: str | None = None
    ) -> None:
        if slice_.issue_number == self._fail_for:
            raise ImplementError(f"boom building {slice_.branch}")
        await super().implement(slice_, container=container, plan_path=plan_path)


@pytest.mark.asyncio
async def test_one_slice_build_raising_escalates_it_but_siblings_still_merge() -> None:
    """A build that raises escalates only that slice; its green siblings still merge.

    Sibling builds run concurrently in a round; gathering them without isolating
    exceptions would let one transient failure cancel the rest and abort the round,
    discarding their work. Slice 1's build raises, but 2 and 3 must still merge.
    """
    slices = [_prd_slice(1), _prd_slice(2), _prd_slice(3)]
    git = FakeGitOps()
    implementer = FlakyImplementer(fail_for=1)

    result = await _build_prd(slices, git=git, implementer=implementer)

    assert result.escalated_issues == [1]
    assert set(result.merged_issues) == {2, 3}
    assert 1 not in result.merged_issues
    # The green siblings really merged onto the integration branch.
    assert ("issue-2", "retinue/prd-1") in git.merges
    assert ("issue-3", "retinue/prd-1") in git.merges


@pytest.mark.asyncio
async def test_raised_build_does_not_block_its_dependents_via_a_green_path() -> None:
    """A raised upstream slice escalates; its dependents are pruned to skipped, not lost."""
    slices = [_prd_slice(1), _prd_slice(2, blocked_by=[1])]
    implementer = FlakyImplementer(fail_for=1)

    result = await _build_prd(slices, implementer=implementer)

    assert result.escalated_issues == [1]
    assert result.skipped_issues == [2]
    assert result.merged_issues == []


# --- merge conflict: resolve under done-check or escalate ------------------------


@pytest.mark.asyncio
async def test_merge_conflict_resolved_under_done_check() -> None:
    """A conflict is handed to the resolver; a resolved + green slice still merges."""
    git = FakeGitOps(conflicts={"issue-1"})
    resolved: list[str] = []

    async def resolve_conflict(*, source: str, into: str) -> ConflictResolution:
        resolved.append(source)
        # Clear the scripted conflict so the retried merge succeeds.
        git._conflicts.discard(source)
        return ConflictResolution.RESOLVED

    result = await _build_prd(
        [_prd_slice(1)], git=git, resolve_conflict=resolve_conflict
    )

    assert resolved == ["issue-1"]
    assert result.merged_issues == [1]
    assert ("issue-1", "retinue/prd-1") in git.merges


@pytest.mark.asyncio
async def test_unresolvable_conflict_escalates() -> None:
    """An unresolvable conflict escalates: the slice is not merged, run stops clean."""
    git = FakeGitOps(conflicts={"issue-1"})

    async def resolve_conflict(*, source: str, into: str) -> ConflictResolution:
        return ConflictResolution.UNRESOLVED

    result = await _build_prd(
        [_prd_slice(1)], git=git, resolve_conflict=resolve_conflict
    )

    assert result.merged_issues == []
    assert 1 in result.escalated_issues


@pytest.mark.asyncio
async def test_conflict_with_no_resolver_escalates() -> None:
    """Without an injected resolver a conflict escalates rather than crashing."""
    git = FakeGitOps(conflicts={"issue-1"})

    result = await _build_prd([_prd_slice(1)], git=git)

    assert result.merged_issues == []
    assert 1 in result.escalated_issues


@pytest.mark.asyncio
async def test_resolved_conflict_that_stays_red_escalates() -> None:
    """If the done-check stays red after a resolve attempt, the slice escalates."""
    git = FakeGitOps(conflicts={"issue-1"})

    async def resolve_conflict(*, source: str, into: str) -> ConflictResolution:
        # Claims resolution but leaves the conflict in place: the retry re-conflicts.
        return ConflictResolution.RESOLVED

    result = await _build_prd(
        [_prd_slice(1)], git=git, resolve_conflict=resolve_conflict
    )

    assert result.merged_issues == []
    assert 1 in result.escalated_issues


# --- single orchestrator run -----------------------------------------------------


@pytest.mark.asyncio
async def test_single_run_acquires_the_lock() -> None:
    """A normal run acquires and releases the single-run lock exactly once."""
    lock = OneAtATimeLock()

    await _build_prd([_prd_slice(1)], lock=lock)

    assert lock.acquisitions == 1
    assert lock.held is False


@pytest.mark.asyncio
async def test_second_concurrent_run_is_rejected() -> None:
    """A second run while one holds the lock raises OrchestratorBusyError."""
    lock = OneAtATimeLock()

    async def run() -> PrdBuildResult:
        return await _build_prd([_prd_slice(1)], lock=lock)

    # First run holds the lock; while it is held, a second run must be rejected.
    lock.held = True
    lock.acquisitions = 1

    with pytest.raises(OrchestratorBusyError):
        await run()


# --- per-round internal reviewer -------------------------------------------------


class RecordingReviewer:
    """A RoundReviewer that records each round's merged set and enqueues a fix slice.

    Models the PRD's per-round review: after a round merges, ``review`` is handed the
    round's merged issue numbers. The first round's merged set carries a planted defect,
    so the reviewer files a single review-fix slice (number 201) blocked by nothing new
    and returns it to be built in a later round; subsequent rounds review clean.
    """

    def __init__(self, *, fix_for: list[int], fix_number: int = 201) -> None:
        self._fix_for = set(fix_for)
        self._fix_number = fix_number
        self.reviewed: list[list[int]] = []
        self.filed = False

    async def review(self, *, merged_issues: list[int]) -> list[PrdSlice]:
        self.reviewed.append(list(merged_issues))
        if not self.filed and self._fix_for.issubset(set(merged_issues)):
            self.filed = True
            return [_prd_slice(self._fix_number)]
        return []


@pytest.mark.asyncio
async def test_reviewer_runs_after_each_round_and_fix_builds_in_a_later_round() -> None:
    """The reviewer runs per round; a filed review-fix slice builds in a later round.

    A two-slice diamond merges in round one; the reviewer flags a planted defect and
    files a review-fix slice (#201), which must enter a *subsequent* round's ready set
    and be built and merged in the same run — proving the live per-round loop.
    """
    slices = [_prd_slice(1), _prd_slice(2, blocked_by=[1])]
    git = FakeGitOps()
    reviewer = RecordingReviewer(fix_for=[1])

    result = await _build_prd(slices, git=git, review_round=reviewer)

    # The reviewer ran after every merge round (at least the first and the fix round).
    assert reviewer.reviewed[0] == [1]
    assert reviewer.filed is True
    # The filed review-fix slice was picked up and merged in a later round, same run.
    assert 201 in result.merged_issues
    assert result.merged_issues.index(201) > result.merged_issues.index(1)
    assert ("issue-201", "retinue/prd-1") in git.merges


class RaisingReviewer:
    """A RoundReviewer whose ``review`` always raises the given error.

    Models an advisory reviewer that fails *after* the round's slices are merged and
    pushed (e.g. an HTTP 400 from the Messages API, or a leaked httpx transport error).
    """

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls = 0

    async def review(self, *, merged_issues: list[int]) -> list[PrdSlice]:
        self.calls += 1
        raise self._error


@pytest.mark.parametrize(
    "error",
    [
        ReviewGenerationError("Messages API returned 400"),
        httpx.ConnectError("connection refused"),
    ],
)
@pytest.mark.asyncio
async def test_advisory_review_failure_does_not_discard_the_merged_build(
    error: Exception,
) -> None:
    """A reviewer failure is swallowed: the already-merged round is not thrown away.

    The per-round review is advisory (it only files follow-ups, never edits code) yet runs
    after the round merges and pushes, so a raised ReviewGenerationError/httpx error must
    not fail the whole build. The reviewer is invoked, its error logged, and the build
    completes with its merged slices intact and no review-fix slices enqueued.
    """
    slices = [_prd_slice(1)]
    reviewer = RaisingReviewer(error)

    result = await _build_prd(slices, review_round=reviewer)

    assert reviewer.calls == 1
    assert result.merged_issues == [1]


@pytest.mark.asyncio
async def test_no_reviewer_seam_leaves_the_build_unchanged() -> None:
    """With no reviewer injected the build behaves exactly as before — no extra slices."""
    slices = [_prd_slice(1), _prd_slice(2, blocked_by=[1])]

    result = await _build_prd(slices)  # no review_round

    assert result.merged_issues == [1, 2]


# --- empty PRD -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_prd_is_a_clean_noop() -> None:
    """A PRD with no slices drains immediately to an empty result, touching no branch."""
    git = FakeGitOps()

    result = await _build_prd([], git=git)

    assert result.merged_issues == []
    assert result.blocked_issues == []
    assert result.escalated_issues == []
    assert result.skipped_issues == []
    # No slices means no implementers, so no integration branch is created.
    assert git.log == []
