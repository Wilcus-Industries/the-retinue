"""Tests for the PRD pipeline orchestration (retinue.pipeline).

The pipeline ties the real adapters together: budget gate -> slice -> build_prd ->
open staging PR -> reconcile on resume, with triage on an implementer failure and the
heimdall loopback / reap on the webhook-driven events. Every collaborator is injected,
so these tests drive the orchestration with fakes — no Docker, gh, Agent SDK, or network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from retinue.handoff import MergedPullRequest, ReapOutcome
from retinue.loopback import (
    HeimdallFinding,
    HeimdallReview,
    ReviewState,
    VerdictOutcome,
)
from retinue.orchestrator import PrdBuildResult, PrdSlice
from retinue.pipeline import Pipeline
from retinue.reconcile import PrState, ResumePhase, RunStateStore
from retinue.repo_config import (
    ModelEffort,
    RepoConfig,
    RoutingConfig,
    RoutingLevel,
)
from retinue.slicer import (
    CreatedIssue,
    IssueDraft,
    SlicePlan,
)
from retinue.vocab import Severity
from tests.fakes import (
    FakeReconcileGh,
    _created,
    _fake_build_adhoc_issue,
    _FakePrOps,
    _FakeReapGh,
    _governor,
    _noop_rebuild,
    _RecordingAdhocPipeline,
    _RecordingNotifier,
    _settings,
)


def _config() -> RepoConfig:
    return RepoConfig(staging_branch="staging", retry_cap=2)


def _pipeline(tmp_path: Path, **overrides: object) -> Pipeline:
    """Build a Pipeline whose collaborators default to harmless recording fakes."""
    notifier = _RecordingNotifier()
    base: dict[str, object] = dict(
        config=_config(),
        claude_md="## Definition of done\n```\nuv run pytest\n```\n",
        governor=_governor(tmp_path),
        notifier=notifier,
        create_issue=_created,
        slice_generate=lambda body: _empty_plan(),
        pr_ops=_FakePrOps(),
        reap_gh=_FakeReapGh(),
        round_store_path=tmp_path / "rounds.sqlite3",
        retry_store_path=tmp_path / "retries.sqlite3",
        run_state_path=tmp_path / "runstate.sqlite3",
        rebuild=_noop_rebuild,
    )
    base.update(overrides)
    return Pipeline(**base)  # type: ignore[arg-type]


async def _empty_plan() -> SlicePlan:
    return SlicePlan(slices=[])


# --- slicing ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thin_prd_escalates_and_builds_nothing(tmp_path: Path) -> None:
    """A thin PRD escalates through the notifier and never reaches build/PR."""
    notifier = _RecordingNotifier()
    built: list[object] = []

    async def build_prd(*args: object, **kwargs: object) -> PrdBuildResult:
        built.append(kwargs)
        return PrdBuildResult("retinue/prd-7", [], [], [], [])

    pipeline = _pipeline(tmp_path, notifier=notifier, build_prd=build_prd)
    result = await pipeline.process_prd_job(
        repo_full_name="owner/repo", prd_number=7, prd_body="too short"
    )

    assert result.sliced is False
    assert built == []
    assert notifier.notes  # escalated


@pytest.mark.asyncio
async def test_substantive_prd_slices_then_builds_and_opens_pr(tmp_path: Path) -> None:
    """A real PRD slices, builds the slices, then opens the staging PR."""
    created_numbers: list[int] = []

    async def create_issue(draft: IssueDraft) -> CreatedIssue:
        number = 100 + len(created_numbers)
        created_numbers.append(number)
        return CreatedIssue(issue_number=number)

    async def generate(body: str) -> SlicePlan:
        return SlicePlan(slices=[IssueDraft(title="s1", body="build the thing")])

    pr_ops = _FakePrOps()

    async def build_prd(**kwargs: object) -> PrdBuildResult:
        return PrdBuildResult("retinue/prd-7", merged_issues=[100], blocked_issues=[],
                              escalated_issues=[], skipped_issues=[])

    pipeline = _pipeline(
        tmp_path,
        create_issue=create_issue,
        slice_generate=generate,
        pr_ops=pr_ops,
        build_prd=build_prd,
    )
    body = "Implement a real feature with enough detail to slice responsibly here."
    result = await pipeline.process_prd_job(
        repo_full_name="owner/repo", prd_number=7, prd_body=body
    )

    assert result.sliced is True
    assert result.pr_opened is True
    assert pr_ops.opened  # the staging PR was opened


@pytest.mark.asyncio
async def test_no_done_check_gate_escalates_and_skips_build(tmp_path: Path) -> None:
    """A repo with no parseable done-check escalates and skips the build, never raising.

    An opted-in repo whose CLAUDE.md carries no "Definition of done" block would make the
    build's ``parse_done_check`` raise ``DoneCheckError``, crash-looping the Arq job.
    Instead the pipeline must detect the missing gate, escalate through the notifier, and
    return a clean terminal result so the job succeeds with no build and no PR attempted.
    """
    notifier = _RecordingNotifier()
    built: list[object] = []

    async def build_prd(**kwargs: object) -> PrdBuildResult:
        built.append(kwargs)
        return PrdBuildResult("retinue/prd-7", [], [], [], [])

    async def generate(body: str) -> SlicePlan:
        return SlicePlan(slices=[IssueDraft(title="s1", body="build the thing")])

    pr_ops = _FakePrOps()
    pipeline = _pipeline(
        tmp_path,
        claude_md="# CLAUDE.md\n\nNo definition-of-done block here.\n",
        notifier=notifier,
        slice_generate=generate,
        pr_ops=pr_ops,
        build_prd=build_prd,
    )
    body = "Implement a real feature with enough detail to slice responsibly here."
    result = await pipeline.process_prd_job(
        repo_full_name="owner/repo", prd_number=7, prd_body=body
    )

    assert result.sliced is True
    assert result.done_check_missing is True
    assert result.pr_opened is False
    assert built == []  # the build was skipped
    assert pr_ops.opened == []  # no PR attempted
    assert notifier.notes  # the missing-gate escalation landed


@pytest.mark.asyncio
async def test_deferred_build_skips_pr_and_surfaces_deferral(tmp_path: Path) -> None:
    """A build that merged nothing (budget-deferred / all-blocked) opens no PR.

    ``_open_pr`` opens/syncs a PR for the ``retinue/prd-<n>`` head; a deferred or
    all-blocked build never pushed that branch, so opening a PR would 404. The pipeline
    must skip the PR step when no slices merged and still surface the no-op cleanly.
    """

    async def generate(body: str) -> SlicePlan:
        return SlicePlan(slices=[IssueDraft(title="s1", body="build the thing")])

    pr_ops = _FakePrOps()

    async def build_prd(**kwargs: object) -> PrdBuildResult:
        # No merged_issues: the budget gate deferred (or every slice was blocked).
        return PrdBuildResult("retinue/prd-7", merged_issues=[], blocked_issues=[101],
                              escalated_issues=[], skipped_issues=[])

    pipeline = _pipeline(
        tmp_path, slice_generate=generate, pr_ops=pr_ops, build_prd=build_prd
    )
    body = "Implement a real feature with enough detail to slice responsibly here."
    result = await pipeline.process_prd_job(
        repo_full_name="owner/repo", prd_number=7, prd_body=body
    )

    assert result.sliced is True
    assert result.pr_opened is False
    assert result.pr_open is None  # the PR step was never reached
    assert result.prd_build is not None
    assert result.prd_build.merged_issues == []  # the deferral is visible to the caller
    assert pr_ops.opened == []  # no PR open/sync attempted


@pytest.mark.asyncio
async def test_deferred_build_rides_the_job_result(tmp_path: Path) -> None:
    """A budget-deferred build's flag and window ride the PrdJobResult.

    The worker re-enqueues a deferred PRD's resume for when the window frees; without
    ``deferred``/``defer_until`` on the job result the gate's deferral is discarded and
    the PRD sits until a worker restart happens to sweep it.
    """
    when = datetime(2026, 6, 23, tzinfo=UTC)

    async def generate(body: str) -> SlicePlan:
        return SlicePlan(slices=[IssueDraft(title="s1", body="build the thing")])

    async def build_prd(**kwargs: object) -> PrdBuildResult:
        return PrdBuildResult("retinue/prd-7", [], [], [], [],
                              deferred=True, defer_until=when)

    pipeline = _pipeline(tmp_path, slice_generate=generate, build_prd=build_prd)
    body = "Implement a real feature with enough detail to slice responsibly here."
    result = await pipeline.process_prd_job(
        repo_full_name="owner/repo", prd_number=7, prd_body=body
    )

    assert result.deferred is True
    assert result.defer_until == when
    assert result.pr_opened is False


def test_prd_build_from_bound_threads_the_deferral() -> None:
    """The bound-build adapter carries the gate's deferral onto the PrdBuildResult."""
    from retinue.pipeline import _prd_build_from_bound
    from retinue.wiring import BoundBuildResult

    when = datetime(2026, 6, 23, tzinfo=UTC)
    build = _prd_build_from_bound(
        BoundBuildResult(deferred=True, defer_until=when), prd_number=7
    )

    assert build.deferred is True
    assert build.defer_until == when
    assert build.merged_issues == []
    assert build.integration_branch == "retinue/prd-7"


# --- review loopback -------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_review_job_converges_and_hands_off(tmp_path: Path) -> None:
    """A clean heimdall review converges: the loopback runs and hands off."""
    handed_off: list[int] = []

    async def handoff(*, repo_full_name: str, pr_number: int) -> None:
        handed_off.append(pr_number)

    pipeline = _pipeline(tmp_path, handoff=handoff)
    review = HeimdallReview(
        repo_full_name="owner/repo",
        pr_number=99,
        prd_number=7,
        prd_issue_number=7,
        integration_branch="retinue/prd-7",
        state=ReviewState.APPROVED,
        findings=[],
    )
    result = await pipeline.process_review(review)

    assert result.outcome is VerdictOutcome.CONVERGED
    assert handed_off == [99]


@pytest.mark.asyncio
async def test_process_review_job_rebuilds_on_blocking(tmp_path: Path) -> None:
    """A blocking heimdall finding files a fix-issue and rebuilds onto the same branch."""
    rebuilt: list[object] = []

    async def rebuild(request: object) -> None:
        rebuilt.append(request)

    pipeline = _pipeline(tmp_path, rebuild=rebuild)
    review = HeimdallReview(
        repo_full_name="owner/repo",
        pr_number=99,
        prd_number=7,
        prd_issue_number=7,
        integration_branch="retinue/prd-7",
        state=ReviewState.REQUEST_CHANGES,
        findings=[HeimdallFinding(summary="boom", severity=Severity.HIGH)],
    )
    result = await pipeline.process_review(review)

    assert result.outcome is VerdictOutcome.REBUILT
    assert rebuilt  # the rebuild seam fired


# --- reap ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reap_pr_job_closes_slices_and_reaps_prd(tmp_path: Path) -> None:
    """The reap closes slice issues and reaps the PRD when no non-hitl child is open."""
    reap_gh = _FakeReapGh(children=[])
    pipeline = _pipeline(tmp_path, reap_gh=reap_gh)
    merged = MergedPullRequest(
        repo_full_name="owner/repo",
        pr_number=99,
        prd_number=7,
        slice_issues=[100, 101],
    )
    result = await pipeline.reap_pr(merged)

    assert result.outcome is ReapOutcome.REAPED
    assert 100 in reap_gh.closed and 101 in reap_gh.closed
    assert 7 in reap_gh.closed  # the PRD itself


# --- crash-resume: resume_round routes a persisted round to its phase ------------


def _run_state(tmp_path: Path) -> RunStateStore:
    """The same run-state file the ``_pipeline`` helper binds, for seeding rounds."""
    return RunStateStore(tmp_path / "runstate.sqlite3")


@pytest.mark.asyncio
async def test_resume_round_mid_build_rebuilds_only_unfinished_slices(
    tmp_path: Path,
) -> None:
    """A BUILD-phase resume rebuilds only the unfinished slices, then opens the PR.

    The store persists only slice numbers, so the rebuilt slices carry an empty
    ``blocked_by`` graph (the sanctioned cross-restart fallback). Slice 100 already
    landed on GitHub; only 101 reaches the build, and the merged build opens the PR.
    """
    built: list[dict[str, Any]] = []

    async def build_prd(**kwargs: Any) -> PrdBuildResult:
        built.append(kwargs)
        return PrdBuildResult("retinue/prd-7", merged_issues=[101], blocked_issues=[],
                              escalated_issues=[], skipped_issues=[])

    gh = FakeReconcileGh(closed_issues={100}, pr=None)
    pr_ops = _FakePrOps()
    pipeline = _pipeline(tmp_path, build_prd=build_prd, reconcile_gh=gh, pr_ops=pr_ops)
    await _run_state(tmp_path).record_slices(
        repo_full_name="owner/repo", prd_number=7, issue_numbers=[100, 101]
    )

    result = await pipeline.resume_round(repo_full_name="owner/repo", prd_number=7)

    assert result.reconcile.phase is ResumePhase.BUILD
    assert len(built) == 1
    resumed: list[PrdSlice] = built[0]["slices"]
    assert [s.issue_number for s in resumed] == [101]
    assert all(s.blocked_by == [] for s in resumed)
    assert pr_ops.opened  # the merged resume-build opened the staging PR


@pytest.mark.asyncio
async def test_resume_round_at_pr_open_opens_the_pr_and_records_it(
    tmp_path: Path,
) -> None:
    """A PR_OPEN-phase resume opens the staging PR without rebuilding anything."""
    built: list[object] = []

    async def build_prd(**kwargs: object) -> PrdBuildResult:
        built.append(kwargs)
        return PrdBuildResult("retinue/prd-7", [], [], [], [])

    gh = FakeReconcileGh(closed_issues={100, 101}, pr=None)
    pr_ops = _FakePrOps()
    pipeline = _pipeline(tmp_path, build_prd=build_prd, reconcile_gh=gh, pr_ops=pr_ops)
    await _run_state(tmp_path).record_slices(
        repo_full_name="owner/repo", prd_number=7, issue_numbers=[100, 101]
    )

    result = await pipeline.resume_round(repo_full_name="owner/repo", prd_number=7)

    assert result.reconcile.phase is ResumePhase.PR_OPEN
    assert built == []  # the landed round is never rebuilt
    assert len(pr_ops.opened) == 1
    assert pr_ops.opened[0].head == "retinue/prd-7"
    # The opened PR's mapping is recorded so the loopback/reap can resolve it.
    assert await pipeline.round_for_pr(
        repo_full_name="owner/repo", pr_number=99
    ) == (7, [100, 101])


@pytest.mark.asyncio
async def test_resume_round_at_loopback_awaits_the_verdict(tmp_path: Path) -> None:
    """A LOOPBACK-phase resume re-records the PR mapping and drives nothing else.

    Heimdall verdicts arrive by webhook, so nothing is actively re-driven — but the
    PR<->PRD mapping is re-recorded (self-healing a crash between the PR open and its
    record) so the arriving verdict resolves back to the PRD.
    """
    built: list[object] = []

    async def build_prd(**kwargs: object) -> PrdBuildResult:
        built.append(kwargs)
        return PrdBuildResult("retinue/prd-7", [], [], [], [])

    gh = FakeReconcileGh(pr=55)
    pr_ops = _FakePrOps()
    pipeline = _pipeline(tmp_path, build_prd=build_prd, reconcile_gh=gh, pr_ops=pr_ops)
    # The crash hit between the PR open and record_pr: only the slices are persisted.
    await _run_state(tmp_path).record_slices(
        repo_full_name="owner/repo", prd_number=7, issue_numbers=[100, 101]
    )

    result = await pipeline.resume_round(repo_full_name="owner/repo", prd_number=7)

    assert result.reconcile.phase is ResumePhase.LOOPBACK
    assert built == []
    assert pr_ops.opened == []  # the PR already exists; nothing is re-opened
    # The mapping self-healed: the webhook's verdict for PR #55 now resolves.
    assert await pipeline.round_for_pr(
        repo_full_name="owner/repo", pr_number=55
    ) == (7, [100, 101])


@pytest.mark.asyncio
async def test_resume_round_done_cleans_up_the_run_state_row(tmp_path: Path) -> None:
    """A DONE-phase resume deletes the round's row so no sweep re-reconciles it."""
    gh = FakeReconcileGh(pr=None)
    pipeline = _pipeline(tmp_path, reconcile_gh=gh)
    store = _run_state(tmp_path)
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=7, issue_numbers=[]
    )

    result = await pipeline.resume_round(repo_full_name="owner/repo", prd_number=7)

    assert result.reconcile.phase is ResumePhase.DONE
    assert await store.all_rounds() == []


@pytest.mark.asyncio
async def test_resume_round_surfaces_a_deferred_rebuild(tmp_path: Path) -> None:
    """A BUILD-phase resume whose re-gated build was deferred surfaces the deferral.

    The resume's BUILD path re-gates through the same bound build; the caller (the
    worker's resume task) needs the deferral to re-enqueue itself, so the outcome must
    carry it rather than reading as a silently finished resume.
    """
    when = datetime(2026, 6, 23, tzinfo=UTC)

    async def build_prd(**kwargs: Any) -> PrdBuildResult:
        return PrdBuildResult("retinue/prd-7", [], [], [], [],
                              deferred=True, defer_until=when)

    gh = FakeReconcileGh(closed_issues={100}, pr=None)
    pr_ops = _FakePrOps()
    pipeline = _pipeline(tmp_path, build_prd=build_prd, reconcile_gh=gh, pr_ops=pr_ops)
    await _run_state(tmp_path).record_slices(
        repo_full_name="owner/repo", prd_number=7, issue_numbers=[100, 101]
    )

    result = await pipeline.resume_round(repo_full_name="owner/repo", prd_number=7)

    assert result.reconcile.phase is ResumePhase.BUILD
    assert result.deferred is True
    assert result.defer_until == when
    assert pr_ops.opened == []  # nothing merged, so no PR


@pytest.mark.asyncio
async def test_resume_round_reaps_a_recorded_pr_github_shows_merged(
    tmp_path: Path,
) -> None:
    """A recorded PR GitHub shows MERGED is reaped on resume (the missed-webhook reap).

    The merge webhook can be lost (worker down, tunnel rotated); without this routing
    the resume re-reconciles a landed round — all slices closed, no *open* PR — and
    tries to re-open a PR for finished work on every restart.
    """
    reap_gh = _FakeReapGh(children=[])
    gh = FakeReconcileGh(pr_states={99: PrState.MERGED})
    pipeline = _pipeline(tmp_path, reconcile_gh=gh, reap_gh=reap_gh)
    store = _run_state(tmp_path)
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=7, issue_numbers=[100, 101]
    )
    await store.record_pr(repo_full_name="owner/repo", prd_number=7, pr_number=99)

    result = await pipeline.resume_round(repo_full_name="owner/repo", prd_number=7)

    assert result.reconcile.phase is ResumePhase.DONE
    assert {100, 101, 7} <= set(reap_gh.closed)  # slices + the PRD reaped
    assert await store.all_rounds() == []  # the round's terminal event
    assert gh.pr_queries == []  # short-circuited before the staging-PR reconcile


@pytest.mark.asyncio
async def test_resume_round_drops_a_recorded_pr_a_human_closed(tmp_path: Path) -> None:
    """A recorded PR GitHub shows CLOSED (rejected, unmerged) ends the round quietly.

    A human closing the staging PR is a rejection; the resume must not reap the slices
    as done, and must not re-open a PR for the rejected work on every restart — the
    zombie-PR loop. The round's row is deleted so no later sweep resurrects it.
    """
    reap_gh = _FakeReapGh(children=[])
    gh = FakeReconcileGh(pr_states={99: PrState.CLOSED})
    pipeline = _pipeline(tmp_path, reconcile_gh=gh, reap_gh=reap_gh)
    store = _run_state(tmp_path)
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=7, issue_numbers=[100, 101]
    )
    await store.record_pr(repo_full_name="owner/repo", prd_number=7, pr_number=99)

    result = await pipeline.resume_round(repo_full_name="owner/repo", prd_number=7)

    assert result.reconcile.phase is ResumePhase.DONE
    assert reap_gh.closed == []  # rejected work is never closed as done
    assert await store.all_rounds() == []  # no sweep re-opens a PR for it


@pytest.mark.asyncio
async def test_resume_round_with_an_open_recorded_pr_proceeds(tmp_path: Path) -> None:
    """A recorded PR still OPEN routes through the normal reconcile (loopback)."""
    gh = FakeReconcileGh(pr=99, pr_states={99: PrState.OPEN})
    pipeline = _pipeline(tmp_path, reconcile_gh=gh)
    store = _run_state(tmp_path)
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=7, issue_numbers=[100, 101]
    )
    await store.record_pr(repo_full_name="owner/repo", prd_number=7, pr_number=99)

    result = await pipeline.resume_round(repo_full_name="owner/repo", prd_number=7)

    assert result.reconcile.phase is ResumePhase.LOOPBACK
    assert result.reconcile.pr_number == 99
    assert await store.all_rounds() != []  # the round still awaits its verdict


@pytest.mark.asyncio
async def test_has_round_reflects_persisted_slices(tmp_path: Path) -> None:
    """has_round is True exactly when the round persisted its slice set."""
    pipeline = _pipeline(tmp_path)

    assert await pipeline.has_round(repo_full_name="owner/repo", prd_number=7) is False

    await _run_state(tmp_path).record_slices(
        repo_full_name="owner/repo", prd_number=7, issue_numbers=[100]
    )
    assert await pipeline.has_round(repo_full_name="owner/repo", prd_number=7) is True


@pytest.mark.asyncio
async def test_reap_pr_deletes_the_round_run_state(tmp_path: Path) -> None:
    """The merge reap deletes the round's run-state row (the round's terminal event).

    Without this, every startup sweep would re-reconcile the landed round — all slices
    closed, no open PR — and try to re-open a PR for finished work.
    """
    reap_gh = _FakeReapGh(children=[])
    pipeline = _pipeline(tmp_path, reap_gh=reap_gh)
    store = _run_state(tmp_path)
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=7, issue_numbers=[100, 101]
    )
    await store.record_pr(repo_full_name="owner/repo", prd_number=7, pr_number=99)

    merged = MergedPullRequest(
        repo_full_name="owner/repo",
        pr_number=99,
        prd_number=7,
        slice_issues=[100, 101],
    )
    result = await pipeline.reap_pr(merged)

    assert result.outcome is ReapOutcome.REAPED
    assert await store.all_rounds() == []
    assert await store.round_for_pr(repo_full_name="owner/repo", pr_number=99) is None


# --- ad-hoc PR into the shared pipeline ------------------------------------------


@pytest.mark.asyncio
async def test_green_adhoc_build_opens_one_pr_into_staging(tmp_path: Path) -> None:
    """A green ad-hoc build opens exactly one PR ``issue-<N>`` -> staging, no integration.

    The ad-hoc PR head is the ``issue-<N>`` branch itself — there is no integration
    branch — and it reuses the shared PR-opener prechecks. The PR<->issue mapping is
    recorded so the loopback and reap can resolve the PR back to the single ad-hoc issue.
    """
    from retinue.adhoc_build import AdhocBuildResult, AdhocIssue

    pr_ops = _FakePrOps()
    pipeline = _pipeline(tmp_path, pr_ops=pr_ops)
    issue = AdhocIssue(repo_full_name="owner/repo", issue_number=31)

    result = await pipeline.process_adhoc_pr(
        issue, AdhocBuildResult(branch="issue-31", passed=True)
    )

    assert result is not None
    assert result.opened is True
    assert len(pr_ops.opened) == 1
    request = pr_ops.opened[0]
    assert request.head == "issue-31"  # the issue branch itself, no integration branch
    assert request.base == "staging"
    # The mapping is recorded under the single ad-hoc issue (no PRD parent, no slices).
    mapping = await pipeline.round_for_pr(repo_full_name="owner/repo", pr_number=99)
    assert mapping == (31, [])


@pytest.mark.asyncio
async def test_red_adhoc_build_opens_no_pr(tmp_path: Path) -> None:
    """A red ad-hoc build pushed nothing, so no PR is opened and no mapping recorded."""
    from retinue.adhoc_build import AdhocBuildResult, AdhocIssue

    pr_ops = _FakePrOps()
    pipeline = _pipeline(tmp_path, pr_ops=pr_ops)
    issue = AdhocIssue(repo_full_name="owner/repo", issue_number=31)

    result = await pipeline.process_adhoc_pr(
        issue, AdhocBuildResult(branch="issue-31", passed=False)
    )

    assert result is None  # the PR step was never reached
    assert pr_ops.opened == []
    assert await pipeline.round_for_pr(repo_full_name="owner/repo", pr_number=99) is None


@pytest.mark.asyncio
async def test_adhoc_pr_drives_loopback_handoff_and_reap(tmp_path: Path) -> None:
    """The ad-hoc PR enters the shared loopback/handoff, and a merge reaps its issue.

    After the green build opens the PR, a clean heimdall review converges through the
    *same* loopback and fires the test-and-merge handoff; the simulated human merge reaps
    the single ad-hoc issue closed — no PRD parent to reap.
    """
    from retinue.adhoc_build import AdhocBuildResult, AdhocIssue

    handed_off: list[int] = []

    async def handoff(*, repo_full_name: str, pr_number: int) -> None:
        handed_off.append(pr_number)

    reap_gh = _FakeReapGh(children=[])  # no Part-of children: ad-hoc has no PRD parent
    pipeline = _pipeline(tmp_path, handoff=handoff, reap_gh=reap_gh)
    issue = AdhocIssue(repo_full_name="owner/repo", issue_number=31)
    await pipeline.process_adhoc_pr(
        issue, AdhocBuildResult(branch="issue-31", passed=True)
    )

    # The PR drives the shared loopback -> handoff on a clean heimdall review.
    review = HeimdallReview(
        repo_full_name="owner/repo",
        pr_number=99,
        prd_number=31,
        prd_issue_number=31,
        integration_branch="issue-31",
        state=ReviewState.APPROVED,
        findings=[],
    )
    verdict = await pipeline.process_review(review)
    assert verdict.outcome is VerdictOutcome.CONVERGED
    assert handed_off == [99]

    # The simulated human merge reaps the single ad-hoc issue closed.
    mapping = await pipeline.round_for_pr(repo_full_name="owner/repo", pr_number=99)
    assert mapping is not None
    prd_number, slice_numbers = mapping
    merged = MergedPullRequest(
        repo_full_name="owner/repo",
        pr_number=99,
        prd_number=prd_number,
        slice_issues=slice_numbers,
    )
    reap = await pipeline.reap_pr(merged)
    assert reap.outcome is ReapOutcome.REAPED
    assert reap_gh.closed == [31]  # the single ad-hoc issue, closed exactly once


# --- production factory wiring ---------------------------------------------------


class _FakeAuth:
    async def installation_token(self, repo_full_name: str) -> object:
        from retinue.github_app import InstallationToken

        return InstallationToken(token="ghs_x", clone_url="https://x/y.git")


@pytest.mark.asyncio
async def test_build_pipeline_factory_wires_a_pipeline(tmp_path: Path) -> None:
    """The production factory mints a token and builds a fully-wired Pipeline.

    The build lane is now bound by default: the factory binds the real
    budget-gated/triaged ``build_prd`` (over the Agent-SDK implementer + container/git/
    secret/report adapters) per repo, so the produced pipeline has a live build seam.
    """
    from retinue.pipeline import build_pipeline_factory

    settings = _settings(tmp_path, ntfy_topic="alerts")
    factory = build_pipeline_factory(
        settings,  # type: ignore[arg-type]
        _FakeAuth(),  # type: ignore[arg-type]
        governor=_governor(tmp_path),
    )
    pipeline = await factory("owner/repo", _config())

    assert pipeline.rebuild is not None
    assert pipeline.reconcile_gh is not None
    # The build lane is wired: an accepted PRD reaches a real, budget-gated build.
    assert pipeline.build_prd is not None


@pytest.mark.asyncio
async def test_build_pipeline_factory_sources_claude_md(tmp_path: Path) -> None:
    """The factory sources each repo's CLAUDE.md (the done-check command) via the fetcher."""
    from retinue.pipeline import build_pipeline_factory

    fetched: list[str] = []

    async def fetch_claude_md(repo_full_name: str) -> str:
        fetched.append(repo_full_name)
        return "## Definition of done\n```\nuv run pytest\n```\n"

    settings = _settings(tmp_path, ntfy_topic="alerts")
    factory = build_pipeline_factory(
        settings,  # type: ignore[arg-type]
        _FakeAuth(),  # type: ignore[arg-type]
        governor=_governor(tmp_path),
        fetch_claude_md=fetch_claude_md,
    )
    pipeline = await factory("owner/repo", _config())

    assert fetched == ["owner/repo"]
    assert "uv run pytest" in pipeline.claude_md


@pytest.mark.asyncio
async def test_build_pipeline_factory_applies_routing_default_to_slicer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A routing-default slicer override sets the constructed slicer's model + effort.

    Drives the real production slicer construction (:class:`ClaudeSliceGenerator`), so a
    routing table's ``default:`` level override for :data:`Role.SLICER` flows end-to-end
    from the repo config through the role registry into the wired generator — proving
    both model *and* effort thread through, not just the model (the bug #63 fixes).
    """
    import retinue.pipeline as pipeline_mod
    from retinue.pipeline import build_pipeline_factory
    from retinue.roles import ROLE_REGISTRY, Role
    from retinue.slicer import ClaudeSliceGenerator

    captured: dict[str, object] = {}

    def _record(*args: object, **kwargs: object) -> ClaudeSliceGenerator:
        captured["model"] = kwargs.get("model")
        captured["effort"] = kwargs.get("effort")
        return ClaudeSliceGenerator(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pipeline_mod, "ClaudeSliceGenerator", _record)

    config = RepoConfig(
        staging_branch="staging",
        retry_cap=2,
        routing=RoutingConfig(
            default="standard",
            levels={
                "standard": RoutingLevel(
                    description="Ordinary work.",
                    roles={
                        Role.SLICER.value: ModelEffort(
                            model="slicer-custom", effort="low"
                        )
                    },
                )
            },
        ),
    )
    settings = _settings(tmp_path, ntfy_topic="alerts")
    factory = build_pipeline_factory(
        settings,  # type: ignore[arg-type]
        _FakeAuth(),  # type: ignore[arg-type]
        governor=_governor(tmp_path),
    )
    await factory("owner/repo", config)

    assert captured["model"] == "slicer-custom"
    assert captured["effort"] == "low"

    # A table-less repo config (no ``routing:`` block at all) resolves both model and
    # effort to the plain registry defaults, unaffected by the routed override above.
    captured.clear()
    await factory("owner/repo", _config())

    assert captured["model"] == ROLE_REGISTRY[Role.SLICER].model
    assert captured["effort"] == ROLE_REGISTRY[Role.SLICER].effort


def test_build_push_sink_picks_pushover_when_no_ntfy(tmp_path: Path) -> None:
    """With only Pushover configured, the push sink is the Pushover backend."""
    from retinue.notify import PushoverPushSink, build_push_sink

    settings = _settings(tmp_path, pushover_token="pk", pushover_user="uk")
    sink = build_push_sink(settings)  # type: ignore[arg-type]
    assert isinstance(sink, PushoverPushSink)


@pytest.mark.asyncio
async def test_review_reviewer_factory_diffs_round_over_integration_branch(
    tmp_path: Path,
) -> None:
    """The factory's reviewer pulls the round diff over the PRD's integration branch.

    Proves the production reviewer-wiring path without a network call: ``generate`` is
    injected as a fake, the diff source records the diff request, and a clean review files
    and enqueues nothing. The base is the PRD's integration branch (``retinue/prd-7``).
    """
    from retinue.pipeline import _build_review_reviewer_factory
    from retinue.reviewer import ReviewInput, ReviewPlan

    diffed: list[tuple[list[str], str]] = []

    class _DiffSource:
        async def round_diff(self, *, merged_branches: list[str], base: str) -> str:
            diffed.append((list(merged_branches), base))
            return "diff-body"

    reviewed: list[ReviewInput] = []

    async def generate(review_input: ReviewInput) -> ReviewPlan:
        reviewed.append(review_input)
        return ReviewPlan(findings=[])  # clean review

    settings = _settings(tmp_path, anthropic_credential="k")
    factory = _build_review_reviewer_factory(
        settings,  # type: ignore[arg-type]
        repo_full_name="owner/repo",
        token="ghs_x",
        create_issue=_created,
        diff_source=_DiffSource(),
        config=_config(),
        generate=generate,
    )
    reviewer = factory("owner/repo", 7)

    fixes = await reviewer.review(merged_issues=[2, 3])

    assert diffed == [(["issue-2", "issue-3"], "retinue/prd-7")]
    assert reviewed[0].diff == "diff-body"
    assert fixes == []  # a clean review files and enqueues nothing


def test_review_factory_applies_repo_config_model_override(tmp_path: Path) -> None:
    """A routing-default reviewer override sets the review generator's model.

    Drives the real production review-generator construction (no ``generate`` override),
    so the override flows end-to-end from the repo config's routing default level through
    the role registry into the live :class:`~retinue.reviewer.AgentSdkReviewGenerator`.
    """
    from retinue.pipeline import _build_review_reviewer_factory
    from retinue.reviewer import AgentSdkReviewGenerator
    from retinue.roles import Role
    from retinue.wiring import _BoundRoundReviewer

    class _DiffSource:
        async def round_diff(self, *, merged_branches: list[str], base: str) -> str:
            return "diff-body"

    settings = _settings(tmp_path, anthropic_credential="k")
    config = RepoConfig(
        routing=RoutingConfig(
            default="standard",
            levels={
                "standard": RoutingLevel(
                    description="Ordinary work.",
                    roles={
                        Role.REVIEWER.value: ModelEffort(model="claude-opus-4-8-custom")
                    },
                )
            },
        )
    )
    factory = _build_review_reviewer_factory(
        settings,  # type: ignore[arg-type]
        repo_full_name="owner/repo",
        token="ghs_x",
        create_issue=_created,
        diff_source=_DiffSource(),
        config=config,
    )
    reviewer = factory("owner/repo", 7)

    assert isinstance(reviewer, _BoundRoundReviewer)
    assert isinstance(reviewer.generate, AgentSdkReviewGenerator)
    assert reviewer.generate.model == "claude-opus-4-8-custom"


def test_review_factory_applies_repo_config_effort_override(tmp_path: Path) -> None:
    """A routing-default reviewer override sets the review generator's effort tier.

    Mirrors ``test_review_factory_applies_repo_config_model_override``: drives the real
    production review-generator construction so a routing-table ``effort:`` override
    flows end-to-end into the live :class:`~retinue.reviewer.AgentSdkReviewGenerator`
    (the bug #63 fixes — previously only the model threaded through).
    """
    from retinue.pipeline import _build_review_reviewer_factory
    from retinue.reviewer import AgentSdkReviewGenerator
    from retinue.roles import Role
    from retinue.wiring import _BoundRoundReviewer

    class _DiffSource:
        async def round_diff(self, *, merged_branches: list[str], base: str) -> str:
            return "diff-body"

    settings = _settings(tmp_path, anthropic_credential="k")
    config = RepoConfig(
        routing=RoutingConfig(
            default="standard",
            levels={
                "standard": RoutingLevel(
                    description="Ordinary work.",
                    roles={
                        Role.REVIEWER.value: ModelEffort(
                            model="claude-opus-4-8-custom", effort="low"
                        )
                    },
                )
            },
        )
    )
    factory = _build_review_reviewer_factory(
        settings,  # type: ignore[arg-type]
        repo_full_name="owner/repo",
        token="ghs_x",
        create_issue=_created,
        diff_source=_DiffSource(),
        config=config,
    )
    reviewer = factory("owner/repo", 7)

    assert isinstance(reviewer, _BoundRoundReviewer)
    assert isinstance(reviewer.generate, AgentSdkReviewGenerator)
    assert reviewer.generate.model == "claude-opus-4-8-custom"
    assert reviewer.generate.effort == "low"


def test_httpx_transport_is_the_default_review_transport() -> None:
    """The factory's default review transport is the real httpx-backed adapter."""
    from retinue.messages_api import HttpxTransport

    assert HttpxTransport().timeout > 0


# --- bind_adhoc_build: the drain's downstream build+PR primitive -----------------


@pytest.mark.asyncio
async def test_bind_adhoc_build_chains_process_adhoc_pr_on_a_green_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A green ad-hoc build then invokes ``process_adhoc_pr(issue, result)`` to open the PR.

    The load-bearing chain: :func:`bind_adhoc_build`'s callable runs the ad-hoc build and
    **then** hands the green :class:`AdhocBuildResult` to the pipeline's
    ``process_adhoc_pr`` (which opens the ``issue-<N>`` -> staging PR). The build is faked
    so no container spawns; the recording pipeline proves the result threads through.
    """
    import retinue.pipeline as pipeline_mod
    from retinue.adhoc_build import AdhocBuildResult, AdhocIssue
    from retinue.pipeline import bind_adhoc_build

    green = AdhocBuildResult(branch="issue-31", passed=True)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        pipeline_mod, "build_adhoc_issue", _fake_build_adhoc_issue(captured, green)
    )

    pipeline = _RecordingAdhocPipeline(pr_result="opened-pr")
    settings = _settings(tmp_path, anthropic_credential="k")
    build = bind_adhoc_build(
        settings,  # type: ignore[arg-type]
        _FakeAuth(),  # type: ignore[arg-type]
        pipeline=pipeline,  # type: ignore[arg-type]
        repo_full_name="owner/repo",
        token="ghs_x",
        config=_config(),
        claude_md="## Definition of done\n```\nuv run pytest\n```\n",
    )

    issue = AdhocIssue(repo_full_name="owner/repo", issue_number=31)
    await build(issue, repo_full_name="owner/repo")

    # The build ran (the faked primitive captured the issue) and its green result was then
    # handed to process_adhoc_pr — the PR-opening chain.
    assert captured["issue"] is issue
    assert pipeline.pr_calls == [(issue, green)]


@pytest.mark.asyncio
async def test_bind_adhoc_build_still_chains_process_adhoc_pr_on_a_red_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A red ad-hoc build still calls ``process_adhoc_pr`` (which skips, opening no PR).

    A red build pushed no branch, so ``process_adhoc_pr`` returns ``None`` — but the bound
    build must *still* call it (unconditionally, after every build) rather than branching
    on ``passed`` itself. Dropping the call on a red build would be silent, so this pins it.
    """
    import retinue.pipeline as pipeline_mod
    from retinue.adhoc_build import AdhocBuildResult, AdhocIssue
    from retinue.pipeline import bind_adhoc_build

    red = AdhocBuildResult(branch="issue-31", passed=False)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        pipeline_mod, "build_adhoc_issue", _fake_build_adhoc_issue(captured, red)
    )

    pipeline = _RecordingAdhocPipeline(pr_result=None)  # red -> process_adhoc_pr skips
    settings = _settings(tmp_path, anthropic_credential="k")
    build = bind_adhoc_build(
        settings,  # type: ignore[arg-type]
        _FakeAuth(),  # type: ignore[arg-type]
        pipeline=pipeline,  # type: ignore[arg-type]
        repo_full_name="owner/repo",
        token="ghs_x",
        config=_config(),
        claude_md="## Definition of done\n```\nuv run pytest\n```\n",
    )

    issue = AdhocIssue(repo_full_name="owner/repo", issue_number=31)
    await build(issue, repo_full_name="owner/repo")

    # The red result was still handed to process_adhoc_pr (it skips, opening no PR).
    assert pipeline.pr_calls == [(issue, red)]


@pytest.mark.asyncio
async def test_bind_adhoc_build_resolves_each_role_model_at_the_issue_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each ad-hoc role is constructed *per issue* at the issue's resolved routing level.

    Issue #65: the ad-hoc lane classifies each issue once at build start and constructs its
    planner/implementer/reviewer at that resolved level (not at bind time, and not at the
    table default). The classify hop is faked to resolve ``complex`` — a **non-default**
    level whose ``roles:`` map overrides all three models with ids distinct from the
    ``default`` level's overrides. Since ``resolve_model(..., level=None)`` would resolve
    via the default level, asserting the captured ``model=`` each adapter receives matches
    ``complex`` (not the default) pins the resolved level actually flowing into
    construction — a closure that dropped the level (passing ``level=None``) would build
    the default models and fail. The captured classify-hop kwargs also pin that the shared
    ``pipeline.governor`` meters the classifier charge.
    """
    import retinue.pipeline as pipeline_mod
    from retinue.adhoc_build import AdhocBuildResult, AdhocIssue, ContainerPlanner
    from retinue.orchestrator import ContainerImplementer
    from retinue.pipeline import bind_adhoc_build
    from retinue.reviewer import AgentSdkReviewGenerator
    from retinue.roles import EFFORT_MAX, Role

    captured: dict[str, str] = {}
    captured_effort: dict[str, str] = {}

    def _record(name: str, real: object) -> object:
        def ctor(*args: object, **kwargs: object) -> object:
            if "model" in kwargs:
                captured[name] = kwargs["model"]  # type: ignore[assignment]
            if "effort" in kwargs:
                captured_effort[name] = kwargs["effort"]  # type: ignore[assignment]
            return real(*args, **kwargs)  # type: ignore[operator]

        return ctor

    monkeypatch.setattr(
        pipeline_mod, "ContainerPlanner", _record("planner", ContainerPlanner)
    )
    monkeypatch.setattr(
        pipeline_mod, "ContainerImplementer", _record("implementer", ContainerImplementer)
    )
    monkeypatch.setattr(
        pipeline_mod, "AgentSdkReviewGenerator", _record("reviewer", AgentSdkReviewGenerator)
    )
    # Fake the classify hop (resolves to the non-default ``complex`` level) and the
    # container build, so the closure runs offline — no gh, no classifier HTTP, no Docker.
    # The kwargs the hop receives are captured to pin the governor threaded into it.
    resolved: list[object] = []
    resolve_kwargs: dict[str, object] = {}

    async def _fake_level(issue: object, config: object, **kwargs: object) -> str:
        resolved.append(issue)
        resolve_kwargs.update(kwargs)
        return "complex"

    monkeypatch.setattr(pipeline_mod, "_resolve_adhoc_level", _fake_level)
    monkeypatch.setattr(
        pipeline_mod,
        "build_adhoc_issue",
        _fake_build_adhoc_issue({}, AdhocBuildResult(branch="issue-31", passed=True)),
    )

    # Two levels with *distinct* role models: the ``default`` (``standard``) and the
    # non-default ``complex`` the fake resolves to. If the closure dropped the level and
    # resolved at ``level=None``, every model below would come out ``*-default``, so the
    # ``*-complex`` asserts pin the resolved level flowing through the construction.
    config = RepoConfig(
        staging_branch="staging",
        retry_cap=2,
        routing=RoutingConfig(
            default="standard",
            levels={
                "standard": RoutingLevel(
                    description="Ordinary work.",
                    roles={
                        Role.PLANNER.value: ModelEffort(model="planner-default"),
                        Role.IMPLEMENTER.value: ModelEffort(model="implementer-default"),
                        Role.REVIEWER.value: ModelEffort(model="reviewer-default"),
                    },
                ),
                "complex": RoutingLevel(
                    description="Hard work.",
                    roles={
                        Role.PLANNER.value: ModelEffort(model="planner-complex"),
                        Role.IMPLEMENTER.value: ModelEffort(model="implementer-complex"),
                        Role.REVIEWER.value: ModelEffort(model="reviewer-complex"),
                    },
                ),
            },
        ),
    )
    settings = _settings(tmp_path, anthropic_credential="k")
    # A distinct governor sentinel so the captured classify-hop kwargs prove the pipeline's
    # own governor (not some other object) meters the per-issue classifier charge.
    governor_sentinel = object()
    pipeline = _RecordingAdhocPipeline(governor=governor_sentinel)
    build = bind_adhoc_build(
        settings,  # type: ignore[arg-type]
        _FakeAuth(),  # type: ignore[arg-type]
        pipeline=pipeline,  # type: ignore[arg-type]
        repo_full_name="owner/repo",
        token="ghs_x",
        config=config,
        claude_md="## Definition of done\n```\nuv run pytest\n```\n",
    )

    # No role is constructed at bind time — construction is deferred into the per-issue build.
    assert captured == {}

    issue = AdhocIssue(repo_full_name="owner/repo", issue_number=31)
    await build(issue, repo_full_name="owner/repo")

    # The issue was classified exactly once, metered on the pipeline's own governor.
    assert resolved == [issue]
    assert resolve_kwargs["governor"] is governor_sentinel
    # Each role was built at the resolved (non-default ``complex``) level's model — a
    # closure resolving at ``level=None`` would yield the ``*-default`` ids instead.
    assert captured == {
        "planner": "planner-complex",
        "implementer": "implementer-complex",
        "reviewer": "reviewer-complex",
    }
    # The routing table's roles map named a model but no ``effort:`` for the reviewer, so
    # its effort falls through to the registry default (``max``) — a model-only override
    # does not implicitly change effort.
    assert captured_effort == {"reviewer": EFFORT_MAX}
