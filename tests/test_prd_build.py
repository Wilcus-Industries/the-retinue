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

import pytest

from retinue.container import RunResult
from retinue.orchestrator import (
    ConflictResolution,
    OrchestratorBusyError,
    PrdBuildResult,
    PrdSlice,
    Slice,
    build_prd,
)
from retinue.repo_config import RepoConfig
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

    async def implement(self, slice_: Slice) -> None:
        self.started.append(slice_.issue_number)
        self._live += 1
        self.peak = max(self.peak, self._live)
        # Yield so siblings scheduled in the same round actually interleave.
        await asyncio.sleep(0)
        self._live -= 1


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
async def test_unset_max_parallel_allows_full_round_concurrency() -> None:
    """With max_parallel unset, every ready slice in a round runs concurrently."""
    slices = [_prd_slice(n) for n in range(1, 4)]
    implementer = RecordingImplementer()

    await _build_prd(slices, implementer=implementer, config=RepoConfig())

    # All three were ready together and unbounded, so all three ran at once.
    assert implementer.peak == 3


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


# --- empty PRD -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_prd_is_a_clean_noop() -> None:
    """A PRD with no slices drains immediately to an empty result."""
    result = await _build_prd([])

    assert result.merged_issues == []
    assert result.blocked_issues == []
    assert result.escalated_issues == []
    assert result.skipped_issues == []
