"""Tests for the orchestrator-build + cron-tick production wiring (retinue.wiring).

``bind_build_prd`` wraps the orchestrator's ``build_prd`` with the budget gate (defer a
run that would start over the cap) and triage (reason about an implementer failure /
notes against the persisted retry cap). ``bind_cron_tick`` drives the cron backlog lane
over its real collaborators. The implementer-spawn seam is the one injected dependency;
everything else is a real adapter, exercised here with fakes — no Docker, gh, or network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from retinue.budget import AuthMode, BudgetGovernor, BudgetLedger
from retinue.container import ContainerRuntime
from retinue.cron import CronGh
from retinue.github_app import InstallationAuth
from retinue.notify import (
    CommentRequest,
    LabelRequest,
    Notifier,
    PushRequest,
)
from retinue.orchestrator import GitOps, Implementer, PrdSlice, Slice
from retinue.repo_config import RepoConfig
from retinue.slicer import IssueCreator
from retinue.triage import ImplementerNotes
from retinue.wiring import bind_build_prd, bind_cron_tick


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
    governor = _governor(tmp_path, clock, weekly=0.0)  # cap is 0 -> any estimate defers
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


# --- bind_cron_tick --------------------------------------------------------------


@pytest.mark.asyncio
async def test_cron_tick_drains_when_in_budget(tmp_path: Path) -> None:
    """A cron tick within budget picks a backlog issue and runs its downstream build."""
    from retinue.cron import BacklogIssue

    clock = _Clock()
    governor = _governor(tmp_path, clock)
    built: list[int] = []

    async def build(*, repo_full_name: str, issue_number: int) -> None:
        built.append(issue_number)

    issue = BacklogIssue(
        number=42, labels=["backlog", "priority:low"], created_at=clock.now()
    )
    tick = bind_cron_tick(
        gh=cast(CronGh, _FakeCronGh([issue])),
        governor=governor,
        clock=clock,
        build=build,
        lock=_Lock(),
    )
    result = await tick(repo_full_name="owner/repo", tick_number=1, estimated_amount=1.0)
    assert result.issue_number == 42
    assert built == [42]


# --- inert collaborators ---------------------------------------------------------


class _NoGit:
    async def ensure_integration_branch(self, *, branch: str, base: str) -> None: ...
    async def merge(self, *, source: str, into: str) -> None: ...


class _NoAuth:
    async def installation_token(self, repo_full_name: str) -> object:
        from retinue.github_app import InstallationToken

        return InstallationToken(token="t", clone_url="u")


_CLAUDE_MD = "## Definition of done\n```\nuv run pytest\n```\n"


class _RedContainer:
    """A container whose git ops succeed but whose done-check command fails (red check).

    Clone, fetch, checkout, and push all return 0 so the per-slice build container reaches
    the done-check; the done-check command itself fails, so the slice is blocked (and never
    pushed). The implementer is a fake (it does not exec ``claude`` against this container).
    """

    async def run_command(self, command: list[str]) -> object:
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


@dataclass
class _FakeCronGh:
    issues: list[object]

    async def list_backlog(self, *, repo_full_name: str) -> list[object]:
        return list(self.issues)


class _Lock:
    async def __aenter__(self) -> _Lock:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None
