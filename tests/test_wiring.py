"""Tests for the lane wiring (the composition root).

``bind_cron_tick`` and ``bind_adhoc_drain`` bind their pure drivers to the real per-repo
collaborators behind a single callable each. The leaf seams (the gh CLIs, the build+PR
primitive, the default-branch lookup) are patched module attributes, so everything between
— token mint, target-branch resolution, ranking, metering, the per-repo lock — runs for
real with no gh, Docker, or network.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
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
from retinue.github_app import InstallationAuth
from retinue.pipeline import PipelineFactory
from retinue.repo_config import RepoConfig
from retinue.wiring import bind_adhoc_drain, bind_cron_tick


class _Clock:
    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 6, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta


class _NoAuth:
    async def installation_token(self, repo_full_name: str) -> object:
        from retinue.github_app import InstallationToken

        return InstallationToken(token="t", clone_url="u")


_CLAUDE_MD = "## Definition of done\n```\nuv run pytest\n```\n"


async def _canned_claude_md(repo_full_name: str) -> str:
    """The lane binds' ``fetch_claude_md`` seam: canned text, no contents-API read."""
    return _CLAUDE_MD


def _config(*, target_branch: str | None = "staging") -> RepoConfig:
    return RepoConfig(target_branch=target_branch, retry_cap=2, max_parallel=1)


def _governor(tmp_path: Path, clock: _Clock, *, weekly: float = 1000.0) -> BudgetGovernor:
    return BudgetGovernor(
        BudgetLedger(
            tmp_path / "budget.sqlite3",
            clock=clock,
            auth_mode=AuthMode.API_KEY,
            weekly_budget=weekly,
        )
    )


# --- bind_cron_tick --------------------------------------------------------------


@pytest.mark.asyncio
async def test_cron_tick_lists_and_locks_when_in_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound cron tick mints a per-repo token and picks one backlog issue.

    The production bind constructs the backlog gh seam itself (the WS1 downstream is a
    no-op trickle placeholder), so the test patches the module seam with a scripted backlog
    and asserts the bound callable threads it: the listed issue is picked.
    """
    import retinue.cron as cron_mod
    from retinue.cron import BacklogIssue

    clock = _Clock()
    governor = _governor(tmp_path, clock)
    issue = BacklogIssue(
        number=42, labels=["backlog", "priority:low"], created_at=clock.now()
    )

    class _FakeCronGhCli:
        def __init__(self, *, token: str) -> None:
            self.token = token

        async def list_backlog(self, *, repo_full_name: str) -> list[BacklogIssue]:
            return [issue]

    monkeypatch.setattr(cron_mod, "GhCli", _FakeCronGhCli)

    tick = bind_cron_tick(
        cast(Settings, object()),
        cast(InstallationAuth, _NoAuth()),
        governor=governor,
        fetch_claude_md=_canned_claude_md,
    )
    result = await tick(repo_full_name="owner/repo", tick_number=1)
    assert result.issue_number == 42


# --- bind_adhoc_drain ------------------------------------------------------------


def _bind_test_drain(
    monkeypatch: pytest.MonkeyPatch,
    *,
    governor: BudgetGovernor,
    listed: list[str],
    built: list[object],
    captured: dict[str, object] | None = None,
) -> Callable[..., Awaitable[None]]:
    """Bind the production ad-hoc drain over patched leaf seams for the tests below.

    The production bind constructs the trigger-label gh seam, the readiness gh seam, and the
    build+PR primitive itself, so those module seams are patched (one scripted ready issue
    with no blockers, a recording build) and a fake pipeline factory stands in for the
    worker's; everything between — token mint, target-branch resolution, ranking, metering,
    the per-repo lock — runs for real.
    """
    import retinue.adhoc_drain as adhoc_drain_mod
    import retinue.pipeline as pipeline_mod
    import retinue.readiness as readiness_mod
    from retinue.adhoc_drain import ReadyIssue

    class _FakeAdhocGhCli:
        def __init__(self, *, token: str) -> None:
            self.token = token

        async def list_ready(
            self, *, repo_full_name: str, label: str
        ) -> list[ReadyIssue]:
            listed.append(repo_full_name)
            return [ReadyIssue(number=7, labels=["ready-for-agent"], body="")]

        async def flight_state(
            self, *, repo_full_name: str, issue_number: int
        ) -> FlightState:
            return FlightState.ABSENT

    class _FakeReadinessGhCli:
        def __init__(self, *, token: str) -> None:
            self.token = token

        async def native_blockers(
            self, *, repo_full_name: str, issue_number: int
        ) -> list[int]:
            return []

        async def is_closed(
            self, *, repo_full_name: str, issue_number: int
        ) -> bool:
            return False

    monkeypatch.setattr(adhoc_drain_mod, "GhCli", _FakeAdhocGhCli)
    monkeypatch.setattr(readiness_mod, "GhCli", _FakeReadinessGhCli)

    def _fake_bind_adhoc_build(
        settings: object, auth: object, **kwargs: object
    ) -> object:
        if captured is not None:
            captured["config"] = kwargs["config"]

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

    The scripted (unblocked) issue must reach the recording build, and the shared governor
    must have metered the drain's flat per-build charge against the shared cap.
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
        monkeypatch, governor=governor, listed=listed, built=built
    )

    await drain(repo_full_name="owner/repo", config=_config())

    assert listed == ["owner/repo"]
    assert [issue.issue_number for issue in built] == [7]  # type: ignore[attr-defined]
    assert await ledger.trailing_24h_spend() == ADHOC_DRAIN_ESTIMATED_AMOUNT


@pytest.mark.asyncio
async def test_adhoc_drain_skips_the_build_when_the_shared_budget_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drain whose shared cap is spent meters every issue away — nothing builds."""
    clock = _Clock()
    # cap 0.12 * 1.0 weekly < the flat 1.0 per-build estimate -> declined
    governor = _governor(tmp_path, clock, weekly=1.0)
    built: list[object] = []
    drain = _bind_test_drain(
        monkeypatch, governor=governor, listed=[], built=built
    )

    await drain(repo_full_name="owner/repo", config=_config())

    assert built == []


@pytest.mark.asyncio
async def test_adhoc_drain_resolves_none_target_branch_to_repo_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``target_branch`` of None is resolved to the repo's default branch before build.

    The wiring boundary owns the ``gh repo view`` lookup; a config that leaves the target
    branch unset must be re-stamped with the resolved name so build-time code never cuts
    ``issue-<N>`` off ``origin/None``.
    """
    import retinue.wiring as wiring_mod

    class _FakeRunner:
        def __init__(self, token: str) -> None:
            self.token = token

        async def __call__(self, argv: list[str]) -> str:
            return "main\n"

    monkeypatch.setattr(wiring_mod, "ReconcileGhRunner", _FakeRunner)

    clock = _Clock()
    governor = _governor(tmp_path, clock)
    captured: dict[str, object] = {}
    drain = _bind_test_drain(
        monkeypatch, governor=governor, listed=[], built=[], captured=captured
    )

    await drain(repo_full_name="owner/repo", config=_config(target_branch=None))

    resolved = cast(RepoConfig, captured["config"])
    assert resolved.target_branch == "main"
