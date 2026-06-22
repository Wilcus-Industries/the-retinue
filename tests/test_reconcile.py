"""Tests for resume-from-GitHub reconciliation on restart (issue #13).

On worker restart the retinue reconciles in-flight run state from GitHub (the source
of truth — which slice issues are closed, which ``issue-<N>`` branches are merged, and
whether the staging PR exists) plus the persisted SQLite run-state, and resumes only
the unfinished work, producing no duplicate issues, branches, or PRs.

GitHub is the source of truth; the SQLite :class:`RunStateStore` only remembers which
slices a PRD round owns and the PR<->PRD mapping once a PR is open. Every gh query is
an injected :class:`ReconcileGh` seam faked here, and the run-state lives in a temp
SQLite file, so the whole flow runs with no real ``gh``, no network, and no concurrency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from retinue.orchestrator import PrdSlice
from retinue.reconcile import (
    ReconcileResult,
    ResumePhase,
    RunStateStore,
    reconcile_run,
)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """An on-disk SQLite path inside the test's tmp dir."""
    return tmp_path / "run-state.sqlite3"


def _prd_slice(issue_number: int, blocked_by: list[int] | None = None) -> PrdSlice:
    return PrdSlice(
        repo_full_name="owner/repo",
        issue_number=issue_number,
        prd_number=1,
        blocked_by=blocked_by or [],
    )


class FakeReconcileGh:
    """In-memory gh truth: which slice issues are closed, which branches merged, PR.

    ``closed_issues`` and ``merged_branches`` script the GitHub truth a restart reads;
    ``pr`` is the staging PR number once one exists (``None`` when no PR is open). Every
    query records its argument so a test can assert exactly which questions were asked.
    """

    def __init__(
        self,
        *,
        closed_issues: set[int] | None = None,
        merged_branches: set[str] | None = None,
        pr: int | None = None,
    ) -> None:
        self._closed_issues = closed_issues or set()
        self._merged_branches = merged_branches or set()
        self._pr = pr
        self.issue_queries: list[int] = []
        self.branch_queries: list[str] = []
        self.pr_queries: list[int] = []

    async def issue_closed(self, *, repo_full_name: str, issue_number: int) -> bool:
        self.issue_queries.append(issue_number)
        return issue_number in self._closed_issues

    async def branch_merged(self, *, repo_full_name: str, branch: str) -> bool:
        self.branch_queries.append(branch)
        return branch in self._merged_branches

    async def staging_pr(self, *, repo_full_name: str, prd_number: int) -> int | None:
        self.pr_queries.append(prd_number)
        return self._pr


async def _reconcile(
    slices: list[PrdSlice],
    *,
    gh: FakeReconcileGh,
    prd_number: int = 1,
    repo_full_name: str = "owner/repo",
) -> ReconcileResult:
    return await reconcile_run(
        repo_full_name=repo_full_name,
        prd_number=prd_number,
        slices=slices,
        gh=gh,
    )


# --- run-state store: persistence across restart ---------------------------------


@pytest.mark.asyncio
async def test_recorded_slices_survive_a_fresh_store(db_path: Path) -> None:
    """The PRD's owned slice set survives a fresh store on the same file (restart)."""
    await RunStateStore(db_path).record_slices(
        repo_full_name="owner/repo", prd_number=1, issue_numbers=[2, 3, 4]
    )
    # A brand-new store object on the same file must see the recorded slices.
    reloaded = await RunStateStore(db_path).slices_of(
        repo_full_name="owner/repo", prd_number=1
    )
    assert reloaded == [2, 3, 4]


@pytest.mark.asyncio
async def test_unseen_prd_has_no_recorded_slices(db_path: Path) -> None:
    """A PRD never recorded reports an empty slice set, not an error."""
    store = RunStateStore(db_path)
    assert await store.slices_of(repo_full_name="owner/repo", prd_number=9) == []


@pytest.mark.asyncio
async def test_recording_slices_is_idempotent(db_path: Path) -> None:
    """Re-recording the same PRD's slices does not duplicate or crash (re-run)."""
    store = RunStateStore(db_path)
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=1, issue_numbers=[2, 3]
    )
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=1, issue_numbers=[2, 3]
    )
    assert await store.slices_of(repo_full_name="owner/repo", prd_number=1) == [2, 3]


@pytest.mark.asyncio
async def test_pr_mapping_survives_a_fresh_store(db_path: Path) -> None:
    """The PR<->PRD mapping survives a fresh store on the same file (restart)."""
    await RunStateStore(db_path).record_pr(
        repo_full_name="owner/repo", prd_number=1, pr_number=101
    )
    assert (
        await RunStateStore(db_path).pr_of(repo_full_name="owner/repo", prd_number=1)
        == 101
    )


@pytest.mark.asyncio
async def test_unseen_prd_has_no_pr(db_path: Path) -> None:
    """A PRD with no recorded PR reports None, not an error."""
    store = RunStateStore(db_path)
    assert await store.pr_of(repo_full_name="owner/repo", prd_number=9) is None


@pytest.mark.asyncio
async def test_distinct_prds_track_independently(db_path: Path) -> None:
    """Different PRDs keep separate slice sets and PR mappings."""
    store = RunStateStore(db_path)
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=1, issue_numbers=[2]
    )
    await store.record_pr(repo_full_name="owner/repo", prd_number=1, pr_number=101)
    assert await store.slices_of(repo_full_name="owner/repo", prd_number=2) == []
    assert await store.pr_of(repo_full_name="owner/repo", prd_number=2) is None


# --- reconcile: PR already open -> resume at loopback, not from scratch -----------


@pytest.mark.asyncio
async def test_open_pr_resumes_at_loopback() -> None:
    """A restart after the PR is open resumes at the loopback phase, not from scratch."""
    slices = [_prd_slice(2), _prd_slice(3, blocked_by=[2])]
    gh = FakeReconcileGh(pr=101)

    result = await _reconcile(slices, gh=gh)

    assert result.phase is ResumePhase.LOOPBACK
    assert result.pr_number == 101


@pytest.mark.asyncio
async def test_open_pr_does_not_rebuild_any_slice() -> None:
    """With the PR open, no slice is resumed for build — the build phase is done."""
    slices = [_prd_slice(2), _prd_slice(3)]
    gh = FakeReconcileGh(pr=101)

    result = await _reconcile(slices, gh=gh)

    # No build work: the PR's existence proves the build round already finished.
    assert result.unfinished_slices == []


@pytest.mark.asyncio
async def test_pr_existence_short_circuits_slice_queries() -> None:
    """An open PR is checked first; per-slice issue/branch queries are skipped."""
    slices = [_prd_slice(2), _prd_slice(3)]
    gh = FakeReconcileGh(pr=101)

    await _reconcile(slices, gh=gh)

    # The PR is the source of truth that the build finished; don't re-ask per slice.
    assert gh.issue_queries == []
    assert gh.branch_queries == []


# --- reconcile: mid-build -> finish only the unfinished slices --------------------


@pytest.mark.asyncio
async def test_mid_build_resumes_only_unfinished_slices() -> None:
    """A mid-build restart resumes only slices whose issue is open and branch unmerged."""
    # Slice 2 is done (issue closed AND branch merged); slice 3 is unfinished.
    slices = [_prd_slice(2), _prd_slice(3, blocked_by=[2])]
    gh = FakeReconcileGh(
        closed_issues={2}, merged_branches={"issue-2"}, pr=None
    )

    result = await _reconcile(slices, gh=gh)

    assert result.phase is ResumePhase.BUILD
    assert [s.issue_number for s in result.unfinished_slices] == [3]


@pytest.mark.asyncio
async def test_done_slices_are_reported_and_not_rebuilt() -> None:
    """Slices already merged are reported finished, never handed back to the builder."""
    slices = [_prd_slice(2), _prd_slice(3)]
    gh = FakeReconcileGh(
        closed_issues={2}, merged_branches={"issue-2"}, pr=None
    )

    result = await _reconcile(slices, gh=gh)

    assert result.finished_issues == [2]
    assert 2 not in [s.issue_number for s in result.unfinished_slices]


@pytest.mark.asyncio
async def test_a_merged_branch_with_open_issue_is_finished() -> None:
    """A merged branch counts as done even if the issue close lagged behind (crash)."""
    # The branch is merged but the close never ran before the crash: GitHub still
    # proves the slice's work landed, so it must not be rebuilt (no duplicate branch).
    slices = [_prd_slice(2)]
    gh = FakeReconcileGh(closed_issues=set(), merged_branches={"issue-2"}, pr=None)

    result = await _reconcile(slices, gh=gh)

    assert result.finished_issues == [2]
    assert result.unfinished_slices == []


@pytest.mark.asyncio
async def test_a_closed_issue_without_merged_branch_is_finished() -> None:
    """A closed issue counts as done even if the branch query lags (no duplicate issue)."""
    slices = [_prd_slice(2)]
    gh = FakeReconcileGh(closed_issues={2}, merged_branches=set(), pr=None)

    result = await _reconcile(slices, gh=gh)

    assert result.finished_issues == [2]
    assert result.unfinished_slices == []


@pytest.mark.asyncio
async def test_fully_unbuilt_prd_resumes_the_whole_round() -> None:
    """A crash before any slice landed resumes the entire round (nothing finished)."""
    slices = [_prd_slice(2), _prd_slice(3, blocked_by=[2])]
    gh = FakeReconcileGh(closed_issues=set(), merged_branches=set(), pr=None)

    result = await _reconcile(slices, gh=gh)

    assert result.phase is ResumePhase.BUILD
    assert [s.issue_number for s in result.unfinished_slices] == [2, 3]
    assert result.finished_issues == []


# --- reconcile: every slice done but no PR -> resume at the PR-open phase ---------


@pytest.mark.asyncio
async def test_all_slices_done_no_pr_resumes_at_pr_open() -> None:
    """Every slice merged but the PR never opened resumes at the PR-open phase."""
    slices = [_prd_slice(2), _prd_slice(3)]
    gh = FakeReconcileGh(
        closed_issues={2, 3}, merged_branches={"issue-2", "issue-3"}, pr=None
    )

    result = await _reconcile(slices, gh=gh)

    assert result.phase is ResumePhase.PR_OPEN
    assert result.unfinished_slices == []
    assert result.finished_issues == [2, 3]


# --- reconcile: an empty PRD is a clean DONE -------------------------------------


@pytest.mark.asyncio
async def test_empty_prd_with_no_pr_is_done() -> None:
    """A PRD with no slices and no PR has nothing to resume — DONE."""
    gh = FakeReconcileGh(pr=None)

    result = await _reconcile([], gh=gh)

    assert result.phase is ResumePhase.DONE
    assert result.unfinished_slices == []
    assert result.pr_number is None


# --- reconcile preserves blocked_by so the resumed build keeps ordering -----------


@pytest.mark.asyncio
async def test_unfinished_slices_keep_their_blocked_by_graph() -> None:
    """Resumed slices carry their blocked_by so build_prd preserves dependency order."""
    slices = [
        _prd_slice(2),
        _prd_slice(3, blocked_by=[2]),
        _prd_slice(4, blocked_by=[3]),
    ]
    # Slice 2 landed; 3 and 4 remain, and 3 still blocks 4.
    gh = FakeReconcileGh(closed_issues={2}, merged_branches={"issue-2"}, pr=None)

    result = await _reconcile(slices, gh=gh)

    by_number = {s.issue_number: s for s in result.unfinished_slices}
    assert by_number[4].blocked_by == [3]
    assert by_number[3].blocked_by == [2]
