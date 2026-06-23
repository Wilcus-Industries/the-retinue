"""Tests for the PRD pipeline orchestration (retinue.pipeline).

The pipeline ties the real adapters together: budget gate -> slice -> build_prd ->
open staging PR -> reconcile on resume, with triage on an implementer failure and the
heimdall loopback / reap on the webhook-driven events. Every collaborator is injected,
so these tests drive the orchestration with fakes — no Docker, gh, Agent SDK, or network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from retinue.budget import AuthMode, BudgetGovernor, BudgetLedger
from retinue.handoff import ChildIssue, MergedPullRequest, ReapOutcome
from retinue.loopback import (
    HeimdallFinding,
    HeimdallReview,
    ReviewState,
    Severity,
    VerdictOutcome,
)
from retinue.orchestrator import PrdBuildResult
from retinue.pipeline import Pipeline
from retinue.pr_opener import (
    OpenPrRequest,
    PullRequest,
)
from retinue.repo_config import RepoConfig
from retinue.slicer import (
    CreatedIssue,
    IssueDraft,
    SlicePlan,
)


class _FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 6, 22, tzinfo=UTC)


@dataclass
class _RecordingNotifier:
    notes: list[object] = field(default_factory=list)

    async def notify(self, notification: object) -> None:
        self.notes.append(notification)


@dataclass
class _FakePrOps:
    heimdall: bool = True
    staging: bool = True
    opened: list[OpenPrRequest] = field(default_factory=list)

    async def heimdall_installed(self, repo_full_name: str) -> bool:
        return self.heimdall

    async def staging_exists(self, *, repo_full_name: str, branch: str) -> bool:
        return self.staging

    async def bring_up_to_date(
        self, *, repo_full_name: str, branch: str, base: str
    ) -> None:
        return None

    async def open_pr(self, request: OpenPrRequest) -> PullRequest:
        self.opened.append(request)
        return PullRequest(number=99, url="https://github.com/owner/repo/pull/99")


@dataclass
class _FakeReapGh:
    children: list[ChildIssue] = field(default_factory=list)
    closed: list[int] = field(default_factory=list)

    async def close_issue(self, *, repo_full_name: str, issue_number: int) -> None:
        self.closed.append(issue_number)

    async def children_of(
        self, *, repo_full_name: str, prd_number: int
    ) -> list[ChildIssue]:
        return self.children


def _config() -> RepoConfig:
    return RepoConfig(staging_branch="staging", retry_cap=2)


def _governor(tmp_path: Path, *, weekly: float = 1_000_000.0) -> BudgetGovernor:
    ledger = BudgetLedger(
        tmp_path / "budget.sqlite3",
        clock=_FixedClock(),
        auth_mode=AuthMode.API_KEY,
        weekly_budget=weekly,
    )
    return BudgetGovernor(ledger)


async def _created(draft: IssueDraft) -> CreatedIssue:
    return CreatedIssue(issue_number=1000)


async def _noop_rebuild(request: object) -> None:
    return None


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


def _settings(tmp_path: Path, **extra: object) -> object:
    from retinue.config import Settings

    base = dict(
        webhook_secret="s",
        dedupe_db_path=str(tmp_path / "dedupe.sqlite3"),
        budget_db_path=str(tmp_path / "budget.sqlite3"),
        weekly_budget=1000.0,
    )
    base.update(extra)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type, call-arg]


@pytest.mark.asyncio
async def test_build_pipeline_factory_wires_a_pipeline(tmp_path: Path) -> None:
    """The production factory mints a token and builds a fully-wired Pipeline.

    The build lane is now bound by default: the factory binds the real
    budget-gated/triaged ``build_prd`` (over the Agent-SDK implementer + container/git/
    secret/report adapters) per repo, so the produced pipeline has a live build seam.
    """
    from retinue.pipeline import build_pipeline_factory

    settings = _settings(tmp_path, ntfy_topic="alerts")
    factory = build_pipeline_factory(settings, _FakeAuth())  # type: ignore[arg-type]
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
        fetch_claude_md=fetch_claude_md,
    )
    pipeline = await factory("owner/repo", _config())

    assert fetched == ["owner/repo"]
    assert "uv run pytest" in pipeline.claude_md


def test_build_push_sink_picks_pushover_when_no_ntfy(tmp_path: Path) -> None:
    """With only Pushover configured, the push sink is the Pushover backend."""
    from retinue.notify import PushoverPushSink
    from retinue.pipeline import _build_push_sink

    settings = _settings(tmp_path, pushover_token="pk", pushover_user="uk")
    sink = _build_push_sink(settings)  # type: ignore[arg-type]
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
    """A ``repo_config.models`` reviewer entry overrides the review generator's model.

    Drives the real production review-generator construction (no ``generate`` override),
    so the override flows end-to-end from the repo config through the role registry into
    the live :class:`~retinue.reviewer.AgentSdkReviewGenerator`.
    """
    from retinue.pipeline import _build_review_reviewer_factory
    from retinue.reviewer import AgentSdkReviewGenerator
    from retinue.roles import Role
    from retinue.wiring import _BoundRoundReviewer

    class _DiffSource:
        async def round_diff(self, *, merged_branches: list[str], base: str) -> str:
            return "diff-body"

    settings = _settings(tmp_path, anthropic_credential="k")
    config = RepoConfig(models={Role.REVIEWER.value: "claude-opus-4-8-custom"})
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


def test_httpx_transport_is_the_default_review_transport() -> None:
    """The factory's default review transport is the real httpx-backed adapter."""
    from retinue.pipeline import HttpxTransport

    assert HttpxTransport().timeout > 0
