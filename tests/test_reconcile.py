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
    GhCliReconcile,
    PersistedRound,
    PrState,
    ReconcileResult,
    ResumePhase,
    RunStateStore,
    gh_env,
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
        pr_states: dict[int, PrState] | None = None,
    ) -> None:
        self._closed_issues = closed_issues or set()
        self._merged_branches = merged_branches or set()
        self._pr = pr
        self._pr_states = pr_states or {}
        self.issue_queries: list[int] = []
        self.branch_queries: list[str] = []
        self.pr_queries: list[int] = []

    async def issue_closed(self, *, repo_full_name: str, issue_number: int) -> bool:
        self.issue_queries.append(issue_number)
        return issue_number in self._closed_issues

    async def branch_merged(
        self, *, repo_full_name: str, branch: str, prd_number: int
    ) -> bool:
        self.branch_queries.append(branch)
        return branch in self._merged_branches

    async def staging_pr(self, *, repo_full_name: str, prd_number: int) -> int | None:
        self.pr_queries.append(prd_number)
        return self._pr

    async def pr_state(self, *, repo_full_name: str, pr_number: int) -> PrState:
        return self._pr_states.get(pr_number, PrState.OPEN)


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
async def test_round_for_pr_reverse_lookup(db_path: Path) -> None:
    """round_for_pr maps a PR number back to its PRD and owned slice set."""
    store = RunStateStore(db_path)
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=7, issue_numbers=[100, 101]
    )
    await store.record_pr(repo_full_name="owner/repo", prd_number=7, pr_number=99)

    assert await store.round_for_pr(repo_full_name="owner/repo", pr_number=99) == (
        7,
        [100, 101],
    )


@pytest.mark.asyncio
async def test_round_for_pr_unknown_pr_is_none(db_path: Path) -> None:
    """A PR number the store never recorded reverse-resolves to None."""
    store = RunStateStore(db_path)
    await store.record_pr(repo_full_name="owner/repo", prd_number=7, pr_number=99)
    assert await store.round_for_pr(repo_full_name="owner/repo", pr_number=5) is None


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


# --- run-state store: enumeration + cleanup (the startup sweep's reads) -----------


@pytest.mark.asyncio
async def test_all_rounds_lists_every_persisted_round(db_path: Path) -> None:
    """all_rounds enumerates every persisted round with its slices and PR mapping."""
    store = RunStateStore(db_path)
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=1, issue_numbers=[2, 3]
    )
    await store.record_pr(repo_full_name="owner/repo", prd_number=1, pr_number=101)
    await store.record_slices(
        repo_full_name="other/repo", prd_number=9, issue_numbers=[40]
    )

    rounds = await store.all_rounds()

    assert rounds == [
        PersistedRound(
            repo_full_name="other/repo",
            prd_number=9,
            slice_numbers=[40],
            pr_number=None,
        ),
        PersistedRound(
            repo_full_name="owner/repo",
            prd_number=1,
            slice_numbers=[2, 3],
            pr_number=101,
        ),
    ]


@pytest.mark.asyncio
async def test_all_rounds_on_a_fresh_store_is_empty(db_path: Path) -> None:
    """A store never written enumerates no rounds, not an error."""
    assert await RunStateStore(db_path).all_rounds() == []


@pytest.mark.asyncio
async def test_delete_round_removes_the_row(db_path: Path) -> None:
    """Deleting a round removes its row: no slices, no PR mapping, not enumerated."""
    store = RunStateStore(db_path)
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=1, issue_numbers=[2, 3]
    )
    await store.record_pr(repo_full_name="owner/repo", prd_number=1, pr_number=101)

    await store.delete_round(repo_full_name="owner/repo", prd_number=1)

    assert await store.all_rounds() == []
    assert await store.slices_of(repo_full_name="owner/repo", prd_number=1) == []
    assert await store.round_for_pr(repo_full_name="owner/repo", pr_number=101) is None


@pytest.mark.asyncio
async def test_delete_round_leaves_other_rounds_alone(db_path: Path) -> None:
    """Deleting one round never touches another PRD's row."""
    store = RunStateStore(db_path)
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=1, issue_numbers=[2]
    )
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=9, issue_numbers=[40]
    )

    await store.delete_round(repo_full_name="owner/repo", prd_number=1)

    assert await store.slices_of(repo_full_name="owner/repo", prd_number=9) == [40]


@pytest.mark.asyncio
async def test_delete_round_of_unknown_prd_is_a_noop(db_path: Path) -> None:
    """Deleting a round never recorded neither raises nor writes anything."""
    store = RunStateStore(db_path)
    await store.delete_round(repo_full_name="owner/repo", prd_number=9)
    assert await store.all_rounds() == []


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


# --- real gh-cli adapter: pure command-assembly + payload-parsing -----------------
#
# These exercise the production GhCliReconcile against a recording fake runner that
# returns canned `gh` JSON stdout — no subprocess, no gh, no network.


class RecordingRunner:
    """A fake :class:`GhRunner`: returns a canned stdout and records the argv it ran."""

    def __init__(self, stdout: str = "") -> None:
        self._stdout = stdout
        self.calls: list[list[str]] = []

    async def __call__(self, argv: list[str]) -> str:
        self.calls.append(argv)
        return self._stdout


def test_gh_env_carries_the_token_and_disables_prompts() -> None:
    """The gh child env authenticates via GH_TOKEN and never prompts interactively."""
    env = gh_env("ghs_secret", {"PATH": "/usr/bin", "GH_TOKEN": "stale"})

    assert env["GH_TOKEN"] == "ghs_secret"  # the auth credential gh sends as Bearer
    assert env["GH_PROMPT_DISABLED"] == "1"
    assert env["PATH"] == "/usr/bin"  # base env preserved


def test_gh_env_does_not_mutate_the_base_env() -> None:
    """Building the gh env copies rather than mutating the caller's environment."""
    base = {"PATH": "/usr/bin"}
    gh_env("tok", base)
    assert "GH_TOKEN" not in base


@pytest.mark.asyncio
async def test_issue_closed_assembles_the_issue_api_call_and_parses_state() -> None:
    """issue_closed reads the issue state via gh api and maps 'closed' to True."""
    runner = RecordingRunner('{"state": "closed"}')
    gh = GhCliReconcile(runner)

    closed = await gh.issue_closed(repo_full_name="owner/repo", issue_number=7)

    assert closed is True
    assert runner.calls == [
        ["api", "repos/owner/repo/issues/7", "--jq", "{state: .state}"]
    ]


@pytest.mark.asyncio
async def test_issue_closed_maps_open_state_to_false() -> None:
    """An issue GitHub reports 'open' is not finished."""
    gh = GhCliReconcile(RecordingRunner('{"state": "open"}'))
    assert await gh.issue_closed(repo_full_name="owner/repo", issue_number=7) is False


@pytest.mark.asyncio
async def test_branch_merged_compares_against_the_integration_branch() -> None:
    """branch_merged asks whether the round's OWN integration branch has the slice.

    The orchestrator merges slices into ``retinue/prd-<n>``, not staging — comparing
    against staging would report a merged slice as unfinished during a mid-build
    resume and rebuild it. Branch names carry a slash, so they are URL-encoded.
    """
    runner = RecordingRunner('{"ahead_by": 0, "status": "identical"}')
    gh = GhCliReconcile(runner, merge_base="staging")

    merged = await gh.branch_merged(
        repo_full_name="owner/repo", branch="issue-7", prd_number=5
    )

    # ahead_by 0 means every commit on issue-7 already landed on retinue/prd-5.
    assert merged is True
    assert runner.calls == [
        [
            "api",
            "repos/owner/repo/compare/retinue%2Fprd-5...issue-7",
            "--jq",
            "{ahead_by: .ahead_by, status: .status}",
        ]
    ]


@pytest.mark.asyncio
async def test_branch_with_unique_commits_is_not_merged() -> None:
    """A branch still ahead of the integration branch has unlanded work — not merged."""
    gh = GhCliReconcile(RecordingRunner('{"ahead_by": 3, "status": "ahead"}'))
    assert (
        await gh.branch_merged(
            repo_full_name="owner/repo", branch="issue-7", prd_number=5
        )
        is False
    )


@pytest.mark.asyncio
async def test_branch_merged_reads_a_missing_base_as_not_merged() -> None:
    """A 404 compare (integration branch not created yet) means nothing merged.

    Any other gh failure still surfaces — only a missing ref reads as truth.
    """

    class FailingRunner:
        def __init__(self, message: str) -> None:
            self._message = message

        async def __call__(self, argv: list[str]) -> str:
            raise RuntimeError(self._message)

    gh_404 = GhCliReconcile(FailingRunner("gh api ... exited 1: HTTP 404: Not Found"))
    assert (
        await gh_404.branch_merged(
            repo_full_name="owner/repo", branch="issue-7", prd_number=5
        )
        is False
    )

    gh_500 = GhCliReconcile(FailingRunner("gh api ... exited 1: HTTP 500"))
    with pytest.raises(RuntimeError, match="500"):
        await gh_500.branch_merged(
            repo_full_name="owner/repo", branch="issue-7", prd_number=5
        )


@pytest.mark.asyncio
async def test_pr_state_reads_the_pr_lifecycle_state() -> None:
    """pr_state reads gh pr view --json state and maps it to the PrState enum."""
    runner = RecordingRunner('{"state": "MERGED"}')
    gh = GhCliReconcile(runner)

    state = await gh.pr_state(repo_full_name="owner/repo", pr_number=101)

    assert state is PrState.MERGED
    assert runner.calls == [
        ["pr", "view", "101", "--repo", "owner/repo", "--json", "state"]
    ]


@pytest.mark.asyncio
async def test_staging_pr_lists_the_integration_to_staging_pr_and_parses_number() -> (
    None
):
    """staging_pr lists the open retinue/prd-<n> -> staging PR and returns its number."""
    runner = RecordingRunner('[{"number": 101}]')
    gh = GhCliReconcile(runner, merge_base="staging")

    number = await gh.staging_pr(repo_full_name="owner/repo", prd_number=5)

    assert number == 101
    assert runner.calls == [
        [
            "pr",
            "list",
            "--repo",
            "owner/repo",
            "--head",
            "retinue/prd-5",
            "--base",
            "staging",
            "--state",
            "open",
            "--json",
            "number",
        ]
    ]


@pytest.mark.asyncio
async def test_staging_pr_returns_none_when_no_pr_is_open() -> None:
    """An empty pr-list payload means no staging PR is open."""
    gh = GhCliReconcile(RecordingRunner("[]"))
    assert await gh.staging_pr(repo_full_name="owner/repo", prd_number=5) is None


@pytest.mark.asyncio
async def test_real_adapter_drives_reconcile_run_against_canned_gh() -> None:
    """End-to-end: reconcile_run routes on the real adapter reading canned gh JSON.

    A canned empty pr-list (no PR) plus a closed-issue state lands the round at PR_OPEN
    without a live gh or network — proving the adapter satisfies the ReconcileGh seam.
    """

    class ScriptedRunner:
        async def __call__(self, argv: list[str]) -> str:
            if argv[0] == "pr":
                return "[]"
            if "compare" in argv[1]:
                return '{"ahead_by": 0}'
            return '{"state": "closed"}'

    gh = GhCliReconcile(ScriptedRunner())
    result = await reconcile_run(
        repo_full_name="owner/repo",
        prd_number=5,
        slices=[_prd_slice(2), _prd_slice(3)],
        gh=gh,
    )

    assert result.phase is ResumePhase.PR_OPEN
    assert result.finished_issues == [2, 3]
