"""Tests for the lanes' production wiring (retinue.wiring).

``bind_build_prd`` wraps the orchestrator's ``build_prd`` with the budget gate (defer a
run that would start over the cap) and triage (reason about an implementer failure /
notes against the persisted retry cap). ``bind_cron_tick`` and ``bind_adhoc_drain`` are
the per-lane production binds: each constructs its per-repo collaborators itself, so
their leaf module seams (the gh CLIs, the build binders) are monkeypatched and everything
between runs for real — no Docker, gh, or network.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from retinue.adhoc_drain import FlightState
from retinue.budget import (
    ADHOC_DRAIN_ESTIMATED_AMOUNT,
    AuthMode,
    BudgetGovernor,
    BudgetLedger,
)
from retinue.config import Settings
from retinue.container import ContainerRuntime
from retinue.container_build import Implementer, Slice
from retinue.github_app import InstallationAuth
from retinue.notify import (
    CommentRequest,
    LabelRequest,
    Notifier,
    PushRequest,
)
from retinue.orchestrator import GitOps, PrdSlice
from retinue.pipeline import PipelineFactory
from retinue.repo_config import RepoConfig
from retinue.slicer import IssueCreator
from retinue.triage import ImplementerNotes
from retinue.wiring import bind_adhoc_drain, bind_build_prd, bind_cron_tick


class _Clock:
    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 6, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta


@dataclass
class _Implementer:
    """A triage-aware implementer scripting a per-issue outcome (None = clean build)."""

    outcomes: dict[int, object] = field(default_factory=dict)
    calls: list[int] = field(default_factory=list)

    async def implement(
        self, slice_: Slice, *, container: object
    ) -> ImplementerNotes | None:
        self.calls.append(slice_.issue_number)
        outcome = self.outcomes.get(slice_.issue_number)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]

    def auth_env(self) -> dict[str, str]:
        return {}


@dataclass
class _RecordingSinks:
    """Recording push/comment/label sinks; ``comments`` proves an escalation landed."""

    comments: list[CommentRequest] = field(default_factory=list)
    labels: list[LabelRequest] = field(default_factory=list)

    async def push(self, request: PushRequest) -> None:
        return None

    async def comment(self, request: CommentRequest) -> None:
        self.comments.append(request)

    async def label(self, request: LabelRequest) -> None:
        self.labels.append(request)


def _notifier(sinks: _RecordingSinks) -> Notifier:
    return Notifier(push=sinks.push, comment=sinks.comment, label=sinks.label)


async def _created(draft: object) -> object:
    from retinue.slicer import CreatedIssue

    return CreatedIssue(issue_number=999)


def _config() -> RepoConfig:
    return RepoConfig(staging_branch="staging", retry_cap=2, max_parallel=1)


def _governor(tmp_path: Path, clock: _Clock, *, weekly: float = 1000.0) -> BudgetGovernor:
    return BudgetGovernor(
        BudgetLedger(
            tmp_path / "budget.sqlite3",
            clock=clock,
            auth_mode=AuthMode.API_KEY,
            weekly_budget=weekly,
        )
    )


# --- bind_build_prd: budget gate -------------------------------------------------


@pytest.mark.asyncio
async def test_build_prd_defers_when_budget_spent(tmp_path: Path) -> None:
    """An estimate over the rolling-24h cap defers the run; nothing is implemented."""
    clock = _Clock()
    governor = _governor(tmp_path, clock, weekly=1.0)  # cap 0.12 < the 1.0 estimate -> defers
    implementer = _Implementer()
    build_prd = bind_build_prd(
        implementer=cast(Implementer, implementer),
        governor=governor,
        notifier=_notifier(_RecordingSinks()),
        create_issue=cast(IssueCreator, _created),
        retry_store_path=tmp_path / "retries.sqlite3",
        estimated_amount=1.0,
        git=cast(GitOps, _NoGit()),
        auth=cast(InstallationAuth, _NoAuth()),
        runtime=cast(ContainerRuntime, _NoRuntime()),
        resolve_secret=_no_secret,
        report=_no_report,
    )
    result = await build_prd(
        repo_full_name="owner/repo",
        prd_number=7,
        slices=[PrdSlice(repo_full_name="owner/repo", issue_number=100, prd_number=7)],
        config=_config(),
        claude_md="cm",
    )
    assert result.deferred is True
    assert implementer.calls == []


@pytest.mark.asyncio
async def test_build_prd_runs_and_triages_a_failure(tmp_path: Path) -> None:
    """Within budget the run builds; a failing implementer escalates via triage."""
    clock = _Clock()
    governor = _governor(tmp_path, clock)
    sinks = _RecordingSinks()
    implementer = _Implementer(outcomes={100: RuntimeError("boom")})
    build_prd = bind_build_prd(
        implementer=cast(Implementer, implementer),
        governor=governor,
        notifier=_notifier(sinks),
        create_issue=cast(IssueCreator, _created),
        retry_store_path=tmp_path / "retries.sqlite3",
        estimated_amount=1.0,
        git=cast(GitOps, _NoGit()),
        auth=cast(InstallationAuth, _NoAuth()),
        runtime=cast(ContainerRuntime, _NoRuntime()),
        resolve_secret=_no_secret,
        report=_no_report,
    )
    config = RepoConfig(staging_branch="staging", retry_cap=0, max_parallel=1)
    result = await build_prd(
        repo_full_name="owner/repo",
        prd_number=7,
        slices=[PrdSlice(repo_full_name="owner/repo", issue_number=100, prd_number=7)],
        config=config,
        claude_md=_CLAUDE_MD,
    )
    assert result.deferred is False
    # retry_cap 0 -> the failure escalates straight to a human via the notifier, and the
    # red done-check that follows blocks the (uncommitted) slice rather than merging it.
    assert sinks.comments  # the escalation comment landed
    assert result.prd_build is not None
    assert result.prd_build.merged_issues == []


@pytest.mark.asyncio
async def test_build_prd_charges_the_shared_ledger(tmp_path: Path) -> None:
    """An admitted PRD build records its estimate on the shared rolling-24h ledger.

    The PRD lane burns real spend when it builds; the gate must charge the estimate to
    the service-level ledger (not just read it), or the 12%/24h cap never learns about
    PRD-lane spend and the shared budget is decorative.
    """
    clock = _Clock()
    governor = _governor(tmp_path, clock)
    implementer = _Implementer(outcomes={100: RuntimeError("boom")})
    build_prd = bind_build_prd(
        implementer=cast(Implementer, implementer),
        governor=governor,
        notifier=_notifier(_RecordingSinks()),
        create_issue=cast(IssueCreator, _created),
        retry_store_path=tmp_path / "retries.sqlite3",
        estimated_amount=3.0,
        git=cast(GitOps, _NoGit()),
        auth=cast(InstallationAuth, _NoAuth()),
        runtime=cast(ContainerRuntime, _NoRuntime()),
        resolve_secret=_no_secret,
        report=_no_report,
    )
    result = await build_prd(
        repo_full_name="owner/repo",
        prd_number=7,
        slices=[PrdSlice(repo_full_name="owner/repo", issue_number=100, prd_number=7)],
        config=RepoConfig(staging_branch="staging", retry_cap=0, max_parallel=1),
        claude_md=_CLAUDE_MD,
    )
    assert result.deferred is False
    # The admitted run's estimate landed on the shared ledger inside the gate.
    assert await governor._ledger.trailing_24h_spend() == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_deferred_build_prd_charges_nothing(tmp_path: Path) -> None:
    """A deferred PRD run leaves the ledger untouched: no phantom charge without a build."""
    clock = _Clock()
    governor = _governor(tmp_path, clock, weekly=1.0)  # cap 0.12 < the 1.0 estimate -> defers
    build_prd = bind_build_prd(
        implementer=cast(Implementer, _Implementer()),
        governor=governor,
        notifier=_notifier(_RecordingSinks()),
        create_issue=cast(IssueCreator, _created),
        retry_store_path=tmp_path / "retries.sqlite3",
        estimated_amount=1.0,
        git=cast(GitOps, _NoGit()),
        auth=cast(InstallationAuth, _NoAuth()),
        runtime=cast(ContainerRuntime, _NoRuntime()),
        resolve_secret=_no_secret,
        report=_no_report,
    )
    result = await build_prd(
        repo_full_name="owner/repo",
        prd_number=7,
        slices=[PrdSlice(repo_full_name="owner/repo", issue_number=100, prd_number=7)],
        config=_config(),
        claude_md="cm",
    )
    assert result.deferred is True
    assert await governor._ledger.trailing_24h_spend() == pytest.approx(0.0)


# --- bind_round_reviewer: the per-round internal reviewer seam --------------------


@pytest.mark.asyncio
async def test_bind_round_reviewer_reviews_diff_and_returns_fix_slices() -> None:
    """The bound reviewer diffs the round, runs the reviewer, and yields fix slices.

    Proves the production seam: it pulls the round's merged diff over the PRD's
    integration branch, drives ``review_round`` (faked here), files a review-fix issue
    via the issue creator, wires it into the dependent's Blocked by, and returns one
    independently-ready :class:`PrdSlice` per filed issue for a later round to build.
    """
    from retinue.reviewer import (
        EditBlockedByRequest,
        ReviewFinding,
        ReviewInput,
        ReviewPlan,
    )
    from retinue.slicer import CreatedIssue, IssueDraft
    from retinue.wiring import bind_round_reviewer

    diffed: list[tuple[list[str], str]] = []

    class _DiffSource:
        async def round_diff(self, *, merged_branches: list[str], base: str) -> str:
            diffed.append((list(merged_branches), base))
            return "diff --git a/x b/x\n+off-by-one"

    reviewed: list[ReviewInput] = []

    async def generate(review_input: ReviewInput) -> ReviewPlan:
        reviewed.append(review_input)
        return ReviewPlan(
            findings=[
                ReviewFinding(title="fix", body="off-by-one", blocks_issues=[3])
            ]
        )

    created: list[IssueDraft] = []

    async def create_issue(draft: IssueDraft) -> CreatedIssue:
        created.append(draft)
        return CreatedIssue(issue_number=201)

    edits: list[EditBlockedByRequest] = []

    async def edit_blocked_by(request: EditBlockedByRequest) -> None:
        edits.append(request)

    reviewer = bind_round_reviewer(
        diff_source=cast(object, _DiffSource()),  # type: ignore[arg-type]
        generate=generate,
        create_issue=cast(IssueCreator, create_issue),
        edit_blocked_by=edit_blocked_by,
        repo_full_name="owner/repo",
        prd_number=7,
    )

    fixes = await reviewer.review(merged_issues=[2, 3])

    # The round diff was taken over the PRD's integration branch for the merged branches.
    assert diffed == [(["issue-2", "issue-3"], "retinue/prd-7")]
    # The reviewer saw the diff + merged issues; it filed and wired the fix.
    assert reviewed[0].merged_issues == [2, 3]
    assert "off-by-one" in reviewed[0].diff
    assert created and "review-fix" in created[0].labels
    assert edits == [
        EditBlockedByRequest(repo_full_name="owner/repo", issue_number=3, add_blocker=201)
    ]
    # The filed review-fix issue comes back as an independently-ready slice.
    assert [(s.issue_number, s.prd_number) for s in fixes] == [(201, 7)]


# --- bind_cron_tick --------------------------------------------------------------


@pytest.mark.asyncio
async def test_cron_tick_drains_when_in_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound cron tick mints a per-repo token and drains one backlog issue.

    The production bind constructs the backlog gh seam and the cron build itself, so the
    test patches those module seams (a scripted backlog, a recording build) and asserts
    the bound ``(*, repo_full_name, tick_number)`` callable threads them: the listed
    issue is picked and its downstream build runs.
    """
    import retinue.cron as cron_mod
    import retinue.pipeline as pipeline_mod
    from retinue.cron import BacklogIssue

    clock = _Clock()
    governor = _governor(tmp_path, clock)
    built: list[int] = []
    issue = BacklogIssue(
        number=42, labels=["backlog", "priority:low"], created_at=clock.now()
    )

    class _FakeCronGhCli:
        def __init__(self, *, token: str) -> None:
            self.token = token

        async def list_backlog(self, *, repo_full_name: str) -> list[BacklogIssue]:
            return [issue]

    monkeypatch.setattr(cron_mod, "GhCli", _FakeCronGhCli)

    async def _fake_builder(
        settings: object, auth: object, **kwargs: object
    ) -> object:
        async def build(*, repo_full_name: str, issue_number: int) -> None:
            built.append(issue_number)

        return build

    monkeypatch.setattr(pipeline_mod, "build_cron_slice_builder", _fake_builder)

    tick = bind_cron_tick(
        cast(Settings, object()),
        cast(InstallationAuth, _NoAuth()),
        governor=governor,
        fetch_claude_md=_canned_claude_md,
    )
    result = await tick(repo_full_name="owner/repo", tick_number=1)
    assert result.issue_number == 42
    assert built == [42]


# --- bind_adhoc_drain ------------------------------------------------------------


def _bind_test_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    governor: BudgetGovernor,
    listed: list[str],
    built: list[object],
) -> Callable[..., Awaitable[None]]:
    """Bind the production ad-hoc drain over patched leaf seams for the tests below.

    The production bind constructs the gh seam and the build+PR primitive itself, so the
    module seams are patched (one scripted ready issue, a recording build) and a fake
    pipeline factory stands in for the worker's; everything between — token mint, ranking,
    metering, the per-repo lock — runs for real.
    """
    import retinue.adhoc_drain as adhoc_drain_mod
    import retinue.pipeline as pipeline_mod
    from retinue.adhoc_drain import ReadyIssue

    class _FakeAdhocGhCli:
        def __init__(self, *, token: str) -> None:
            self.token = token

        async def list_ready(self, *, repo_full_name: str) -> list[ReadyIssue]:
            listed.append(repo_full_name)
            return [ReadyIssue(number=7, labels=["ready-for-agent"], body="")]

        async def flight_state(
            self, *, repo_full_name: str, issue_number: int
        ) -> FlightState:
            return FlightState.ABSENT

    monkeypatch.setattr(adhoc_drain_mod, "GhCli", _FakeAdhocGhCli)

    def _fake_bind_adhoc_build(
        settings: object, auth: object, **kwargs: object
    ) -> object:
        async def build(issue: object, *, repo_full_name: str) -> None:
            built.append(issue)

        return build

    monkeypatch.setattr(pipeline_mod, "bind_adhoc_build", _fake_bind_adhoc_build)

    async def pipeline_factory(repo_full_name: str, config: RepoConfig) -> object:
        return object()

    return bind_adhoc_drain(
        cast(Settings, object()),
        cast(InstallationAuth, _NoAuth()),
        governor=governor,
        pipeline_factory=cast(PipelineFactory, pipeline_factory),
        fetch_claude_md=_canned_claude_md,
    )


@pytest.mark.asyncio
async def test_adhoc_drain_drives_run_adhoc_drain_with_its_collaborators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound drain drives ``run_adhoc_drain`` over exactly the wired collaborators.

    Fires the returned ``(*, repo_full_name, config)`` callable over the patched leaf
    seams. The scripted issue must reach the recording build (the gh seam was listed,
    the issue ranked + built) and the shared governor must have metered the drain's
    flat per-build charge against the shared cap.
    """
    clock = _Clock()
    ledger = BudgetLedger(
        tmp_path / "budget.sqlite3",
        clock=clock,
        auth_mode=AuthMode.API_KEY,
        weekly_budget=1000.0,
    )
    governor = BudgetGovernor(ledger)
    listed: list[str] = []
    built: list[object] = []
    drain = _bind_test_drain(
        tmp_path, monkeypatch, governor=governor, listed=listed, built=built
    )

    await drain(repo_full_name="owner/repo", config=_config())

    # The gh seam was queried for the repo, the scripted issue reached the build, and the
    # governor metered the flat per-build charge against the shared rolling-24h ledger.
    assert listed == ["owner/repo"]
    assert [issue.issue_number for issue in built] == [7]  # type: ignore[attr-defined]
    assert await ledger.trailing_24h_spend() == ADHOC_DRAIN_ESTIMATED_AMOUNT


@pytest.mark.asyncio
async def test_adhoc_drain_skips_the_build_when_the_shared_budget_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drain whose shared cap is spent meters every issue away — nothing builds.

    Proves the *shared governor* is the one the bound drain meters through: a zero-cap
    ledger declines the metered charge, so the recording build is never called even
    though the gh seam listed a ready issue.
    """
    clock = _Clock()
    # cap 0.12 < the flat 1.0 per-build estimate -> declined
    governor = _governor(tmp_path, clock, weekly=1.0)
    built: list[object] = []
    drain = _bind_test_drain(
        tmp_path, monkeypatch, governor=governor, listed=[], built=built
    )

    await drain(repo_full_name="owner/repo", config=_config())

    assert built == []


# --- inert collaborators ---------------------------------------------------------


class _NoGit:
    async def ensure_integration_branch(self, *, branch: str, base: str) -> None: ...
    async def merge(self, *, source: str, into: str) -> None: ...


class _NoAuth:
    async def installation_token(self, repo_full_name: str) -> object:
        from retinue.github_app import InstallationToken

        return InstallationToken(token="t", clone_url="u")


_CLAUDE_MD = "## Definition of done\n```\nuv run pytest\n```\n"


async def _canned_claude_md(repo_full_name: str) -> str:
    """The lane binds' ``fetch_claude_md`` seam: canned text, no contents-API read."""
    return _CLAUDE_MD


class _RedContainer:
    """A container whose git ops succeed but whose done-check command fails (red check).

    Clone, fetch, checkout, and push all return 0 so the per-slice build container reaches
    the done-check; the done-check command itself fails, so the slice is blocked (and never
    pushed). The implementer is a fake (it does not exec ``claude`` against this container).
    """

    async def run_command(
        self, command: list[str], *, env: Mapping[str, str] | None = None
    ) -> object:
        from retinue.container import RunResult

        if command and command[0] == "git":
            return RunResult(exit_code=0)
        return RunResult(exit_code=1, stderr="check failed")

    async def destroy(self) -> None:
        return None


class _NoRuntime:
    async def start(self, *, image: str, env: dict[str, str]) -> object:
        return _RedContainer()


async def _no_secret(ref: str) -> str | None:
    return None


async def _no_report(report: object) -> None:
    return None
