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

    async def bring_up_to_date(self, *, branch: str, base: str) -> None:
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
