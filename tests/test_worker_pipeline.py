"""Tests for the worker's pipeline wiring: process_prd + the review/reap tasks.

The worker tasks read their collaborators from the Arq ``ctx`` (populated by
``on_startup``). These tests inject fakes into ``ctx`` — a recording pipeline, a config
fetcher, a PRD-body fetcher — so the dispatch and parsing are exercised with no real gh,
Anthropic, Docker, or network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import retinue.github_app as github_app
import retinue.worker as worker
from retinue.dedupe import PrdDedupeStore
from retinue.github_app import InstallationToken
from retinue.handoff import MergedPullRequest, ReapOutcome, ReapResult
from retinue.loopback import (
    HeimdallReview,
    ReviewState,
    Severity,
    VerdictOutcome,
    VerdictResult,
)
from retinue.pipeline import PrdJobResult
from retinue.queue import RUN_ADHOC_DRAIN_TASK
from retinue.repo_config import RepoConfig
from retinue.worker import (
    WorkerSettings,
    on_shutdown,
    on_startup,
    parse_heimdall_review,
    process_prd,
    process_review_job,
    reap_pr_job,
    run_adhoc_drain_job,
)

_CONFIG_YAML = "staging_branch: staging\nretry_cap: 2\n"


@dataclass
class _RecordingPipeline:
    """A fake Pipeline recording every call the worker tasks make against it."""

    prd_calls: list[dict[str, Any]] = field(default_factory=list)
    reviews: list[HeimdallReview] = field(default_factory=list)
    reaps: list[MergedPullRequest] = field(default_factory=list)
    pr_round: tuple[int, list[int]] | None = (7, [100, 101])

    async def process_prd_job(
        self, *, repo_full_name: str, prd_number: int, prd_body: str
    ) -> PrdJobResult:
        self.prd_calls.append(
            {"repo": repo_full_name, "prd": prd_number, "body": prd_body}
        )
        return PrdJobResult(sliced=True, pr_opened=True)

    async def process_review(self, review: HeimdallReview) -> VerdictResult:
        self.reviews.append(review)
        return VerdictResult(outcome=VerdictOutcome.CONVERGED)

    async def reap_pr(self, merged: MergedPullRequest) -> ReapResult:
        self.reaps.append(merged)
        return ReapResult(outcome=ReapOutcome.REAPED, prd_closed=True)

    async def round_for_pr(
        self, *, repo_full_name: str, pr_number: int
    ) -> tuple[int, list[int]] | None:
        return self.pr_round


def _ctx(tmp_path: Path, pipeline: _RecordingPipeline, *, body: str = "") -> dict[str, Any]:
    async def fetch_config(repo_full_name: str) -> str | None:
        return _CONFIG_YAML

    async def fetch_body(repo_full_name: str, issue_number: int) -> str:
        return body

    async def factory(repo_full_name: str, config: RepoConfig) -> _RecordingPipeline:
        return pipeline

    return {
        "fetch_config": fetch_config,
        "fetch_prd_body": fetch_body,
        "pipeline_factory": factory,
        "dedupe": PrdDedupeStore(tmp_path / "dedupe.sqlite3"),
    }


# --- process_prd ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_prd_drives_pipeline_with_fetched_body(tmp_path: Path) -> None:
    """An accepted PRD reaches the pipeline with its fetched issue body."""
    pipeline = _RecordingPipeline()
    body = "Implement the thing with enough body text to slice it responsibly here."
    ctx = _ctx(tmp_path, pipeline, body=body)

    await process_prd(ctx, repo_full_name="owner/repo", issue_number=7, action="opened")

    assert pipeline.prd_calls == [{"repo": "owner/repo", "prd": 7, "body": body}]


@pytest.mark.asyncio
async def test_process_prd_without_pipeline_is_a_noop(tmp_path: Path) -> None:
    """With no pipeline_factory wired the accepted PRD stops after the gate."""
    pipeline = _RecordingPipeline()
    ctx = _ctx(tmp_path, pipeline)
    del ctx["pipeline_factory"]

    await process_prd(ctx, repo_full_name="owner/repo", issue_number=7, action="opened")

    assert pipeline.prd_calls == []


# --- review loopback ------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_review_job_parses_and_drives_loopback(tmp_path: Path) -> None:
    """A review job resolves the PRD, parses findings, and drives the loopback."""
    pipeline = _RecordingPipeline(pr_round=(7, [100]))
    ctx = _ctx(tmp_path, pipeline)

    await process_review_job(
        ctx,
        repo_full_name="owner/repo",
        pr_number=99,
        review_state="changes_requested",
        review_body="high: a blocking problem\nlow: a nit",
    )

    assert len(pipeline.reviews) == 1
    review = pipeline.reviews[0]
    assert review.pr_number == 99
    assert review.prd_number == 7
    assert review.integration_branch == "retinue/prd-7"
    assert review.state is ReviewState.REQUEST_CHANGES
    assert [f.severity for f in review.findings] == [Severity.HIGH, Severity.LOW]


@pytest.mark.asyncio
async def test_process_review_job_skips_unknown_pr(tmp_path: Path) -> None:
    """A review on a PR not in run-state is skipped (not the retinue's PR)."""
    pipeline = _RecordingPipeline(pr_round=None)
    ctx = _ctx(tmp_path, pipeline)

    await process_review_job(
        ctx, repo_full_name="owner/repo", pr_number=5, review_state="approved"
    )

    assert pipeline.reviews == []


# --- reap -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reap_pr_job_resolves_slices_and_reaps(tmp_path: Path) -> None:
    """A merge job resolves the PRD + slice issues from run-state and reaps."""
    pipeline = _RecordingPipeline(pr_round=(7, [100, 101]))
    ctx = _ctx(tmp_path, pipeline)

    await reap_pr_job(ctx, repo_full_name="owner/repo", pr_number=99)

    assert pipeline.reaps == [
        MergedPullRequest(
            repo_full_name="owner/repo",
            pr_number=99,
            prd_number=7,
            slice_issues=[100, 101],
        )
    ]


@pytest.mark.asyncio
async def test_reap_pr_job_skips_unknown_pr(tmp_path: Path) -> None:
    """A merge of a PR the retinue never opened is skipped, not reaped."""
    pipeline = _RecordingPipeline(pr_round=None)
    ctx = _ctx(tmp_path, pipeline)

    await reap_pr_job(ctx, repo_full_name="owner/repo", pr_number=5)

    assert pipeline.reaps == []


# --- ad-hoc drain kick ----------------------------------------------------------


@pytest.mark.asyncio
async def test_run_adhoc_drain_job_drives_the_bound_drain(tmp_path: Path) -> None:
    """A kicked drain job calls the bound drain from ctx with the repo (and its config)."""
    calls: list[dict[str, Any]] = []

    async def drain(*, repo_full_name: str, config: RepoConfig) -> None:
        calls.append({"repo": repo_full_name, "config": config})

    ctx = _ctx(tmp_path, _RecordingPipeline())
    ctx["adhoc_drain"] = drain

    await run_adhoc_drain_job(ctx, repo_full_name="owner/repo")

    assert len(calls) == 1
    assert calls[0]["repo"] == "owner/repo"
    assert calls[0]["config"].staging_branch == "staging"


@pytest.mark.asyncio
async def test_run_adhoc_drain_job_without_drain_is_a_noop(tmp_path: Path) -> None:
    """With no drain wired (bare worker) the kick logs and returns, never crashing."""
    ctx = _ctx(tmp_path, _RecordingPipeline())
    assert "adhoc_drain" not in ctx

    await run_adhoc_drain_job(ctx, repo_full_name="owner/repo")  # must not raise


@pytest.mark.asyncio
async def test_run_adhoc_drain_job_skips_a_deopted_repo(tmp_path: Path) -> None:
    """A repo no longer opted in (no config) is a skip — the drain is never fired."""
    calls: list[str] = []

    async def drain(*, repo_full_name: str, config: RepoConfig) -> None:
        calls.append(repo_full_name)

    ctx = _ctx(tmp_path, _RecordingPipeline())
    ctx["adhoc_drain"] = drain

    async def no_config(repo_full_name: str) -> str | None:
        return None

    ctx["fetch_config"] = no_config

    await run_adhoc_drain_job(ctx, repo_full_name="owner/repo")

    assert calls == []


def test_worker_registers_the_adhoc_drain_task() -> None:
    """WorkerSettings registers a function named ``run_adhoc_drain_job`` (the kick task).

    Mirrors the cron-job registration test: the webhook enqueues
    ``RUN_ADHOC_DRAIN_TASK`` and Arq dequeues by ``__name__``, so a function with that
    exact name must be in ``WorkerSettings.functions`` or the kick is dropped.
    """
    names = {fn.__name__ for fn in WorkerSettings.functions}
    assert RUN_ADHOC_DRAIN_TASK in names
    assert run_adhoc_drain_job.__name__ == RUN_ADHOC_DRAIN_TASK


# --- parse_heimdall_review ------------------------------------------------------


def test_parse_heimdall_review_maps_state_and_findings() -> None:
    """The review parser maps gh state and reads severity:summary finding lines."""
    review = parse_heimdall_review(
        repo_full_name="owner/repo",
        pr_number=99,
        prd_number=7,
        review_state="approved",
        review_body="critical: data loss\nnot a finding line\nmedium: slow path",
    )
    assert review.state is ReviewState.APPROVED
    assert review.integration_branch == "retinue/prd-7"
    assert [(f.severity, f.summary) for f in review.findings] == [
        (Severity.CRITICAL, "data loss"),
        (Severity.MEDIUM, "slow path"),
    ]


def test_parse_heimdall_review_unknown_state_is_commented() -> None:
    """An unrecognised gh review state reads as a plain comment (no verdict)."""
    review = parse_heimdall_review(
        repo_full_name="owner/repo",
        pr_number=1,
        prd_number=2,
        review_state="dismissed",
        review_body="",
    )
    assert review.state is ReviewState.COMMENTED
    assert review.findings == []


# --- on_startup: the production wiring path -------------------------------------


class _FakeAuth:
    """A stand-in :class:`InstallationAuth` that mints a canned token without network.

    Also satisfies :class:`~retinue.github_app.InstalledRepos`: ``installed_repos`` is the
    fixed set the heartbeat enumerator lists (the App's installed repos), so the sweep is
    exercised without a live GitHub listing.
    """

    installed_repos: list[str] = []

    async def installation_token(self, repo_full_name: str) -> InstallationToken:
        return InstallationToken(token="ghs_x", clone_url="https://x/y.git")

    async def installed_repositories(self) -> list[str]:
        return list(self.installed_repos)


async def _fake_claude_md(repo_full_name: str) -> str:
    """Canned CLAUDE.md text standing in for the contents-API fetch (no network)."""
    return "## Definition of done\n```\nuv run pytest\n```\n"


def _worker_settings(tmp_path: Path) -> object:
    from retinue.config import Settings

    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        webhook_secret="s",
        dedupe_db_path=str(tmp_path / "dedupe.sqlite3"),
        budget_db_path=str(tmp_path / "budget.sqlite3"),
        weekly_budget=1000.0,
        ntfy_topic="alerts",
    )


@pytest.mark.asyncio
async def test_on_startup_wires_a_live_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """on_startup takes the auth branch and produces a pipeline with a live build lane.

    With GitHub App auth resolvable, on_startup must install the config fetcher, the
    PRD-body fetcher, and the pipeline_factory — and the factory must yield a Pipeline
    whose build lane is bound (not the dead ``build_prd is None`` of the unwired path).
    No network, Docker, or model: the auth is faked and adapter construction is pure.
    """
    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.setattr(github_app, "build_installation_auth", _FakeAuth)
    # The factory sources CLAUDE.md per repo over the contents API; stub that fetcher so
    # the wiring is exercised without a live GitHub read.
    monkeypatch.setattr(
        worker,
        "_github_claude_md_fetcher",
        lambda auth, client: _fake_claude_md,
    )

    ctx: dict[str, Any] = {}
    try:
        await on_startup(ctx)

        # Took the auth branch: all three downstream seams are installed.
        assert ctx["github_client"] is not None
        assert callable(ctx["fetch_config"])
        assert callable(ctx["fetch_prd_body"])
        assert callable(ctx["pipeline_factory"])
        # The webhook's ad-hoc kick task reads ``adhoc_drain`` from ctx; a deployed
        # worker under live auth must have it bound so the kick actually drains.
        assert callable(ctx["adhoc_drain"])

        # The produced pipeline has a live build lane (the wiring blocker is closed).
        pipeline = await ctx["pipeline_factory"](
            "owner/repo", RepoConfig(staging_branch="staging", retry_cap=2)
        )
        assert pipeline.build_prd is not None
    finally:
        await on_shutdown(ctx)


@pytest.mark.asyncio
async def test_on_startup_adhoc_drain_drives_one_issue_to_the_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound ``ctx['adhoc_drain']`` actually drains: one listed issue reaches the build.

    Extends the live-wiring path by *invoking* the bound drain (not merely asserting it is
    callable): a fake gh seam lists one ready issue, and the ad-hoc build (faked to avoid a
    container) drives the per-repo pipeline's ``process_adhoc_pr``. This proves
    ``_bind_adhoc_drain`` wires the per-repo lock registry + shared governor and threads the
    factory-built pipeline into ``bind_adhoc_build`` — the whole assembly runs end to end
    with no Docker, gh, model, or network.
    """
    import retinue.adhoc_drain as adhoc_drain_mod
    import retinue.pipeline as pipeline_mod
    from retinue.adhoc_build import AdhocBuildResult, AdhocIssue
    from retinue.adhoc_drain import ReadyIssue

    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.setattr(github_app, "build_installation_auth", _FakeAuth)
    monkeypatch.setattr(
        worker, "_github_claude_md_fetcher", lambda auth, client: _fake_claude_md
    )

    # Fake the gh seam the drain constructs: list one ready ad-hoc issue, none in flight.
    class _FakeGhCli:
        def __init__(self, *, token: str) -> None:
            self.token = token

        async def list_ready(self, *, repo_full_name: str) -> list[ReadyIssue]:
            return [ReadyIssue(number=31, labels=["ready-for-agent"], body="")]

        async def in_flight(self, *, repo_full_name: str, issue_number: int) -> bool:
            return False

    monkeypatch.setattr(adhoc_drain_mod, "GhCli", _FakeGhCli)

    # Fake the ad-hoc build so no container spawns, but drive the *real per-repo pipeline*
    # the factory built — proving bind_adhoc_build is handed that pipeline and its
    # process_adhoc_pr runs as the build's PR step.
    pr_calls: list[tuple[int, bool]] = []
    pipelines_seen: list[object] = []

    def _fake_bind_adhoc_build(settings: object, auth: object, **kwargs: object) -> object:
        pipeline = kwargs["pipeline"]
        pipelines_seen.append(pipeline)

        async def build(issue: AdhocIssue, *, repo_full_name: str) -> None:
            # A red result drives the *real* factory-built pipeline's process_adhoc_pr
            # without opening a network PR (a red build skips the PR step, returning None),
            # so the per-repo pipeline is exercised end to end with no gh/network.
            result = AdhocBuildResult(branch=issue.branch, passed=False)
            pr_result = await pipeline.process_adhoc_pr(issue, result)  # type: ignore[attr-defined]
            pr_calls.append((issue.issue_number, pr_result is None))

        return build

    monkeypatch.setattr(pipeline_mod, "bind_adhoc_build", _fake_bind_adhoc_build)

    ctx: dict[str, Any] = {}
    try:
        await on_startup(ctx)
        drain = ctx["adhoc_drain"]
        assert callable(drain)

        config = RepoConfig(staging_branch="staging", retry_cap=2)
        await drain(repo_full_name="owner/repo", config=config)

        # The listed issue (#31) drove the build, which ran the factory-built pipeline's
        # process_adhoc_pr (a red build skips, returning None) — the per-repo pipeline is
        # threaded through, the shared governor metered it (default budget has room), and
        # the per-repo lock serialized the run.
        assert pr_calls == [(31, True)]  # (issue_number, process_adhoc_pr returned None)
        assert pipelines_seen  # bind_adhoc_build was handed the per-repo pipeline
    finally:
        await on_shutdown(ctx)


@pytest.mark.asyncio
async def test_on_startup_without_auth_installs_no_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no GitHub App auth builder, on_startup falls back to the safe not-opted path."""
    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.delattr(github_app, "build_installation_auth", raising=False)

    ctx: dict[str, Any] = {}
    await on_startup(ctx)

    # No auth -> no pipeline; the fetcher is the not-opted-in fallback.
    assert "pipeline_factory" not in ctx
    assert await ctx["fetch_config"]("owner/repo") is None


# --- on_startup: the heartbeat collaborators (issue #43) ------------------------


def _fake_config_fetcher(auth: object, client: object) -> Any:
    """Stand in for the contents-API config fetcher: every repo is opted in (no network)."""

    async def fetch(repo_full_name: str) -> str | None:
        return _CONFIG_YAML

    return fetch


@pytest.mark.asyncio
async def test_on_startup_wires_the_heartbeat_collaborators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under live auth, on_startup installs all four heartbeat collaborators on ctx.

    The registered ``heartbeat_tick`` reads ``heartbeat_enumerate_repos``,
    ``heartbeat_clock``, ``heartbeat_drain``, and ``heartbeat_cron_tick`` from ctx; without
    these it no-ops every tick. The drain must be the *same* object the webhook kick fires
    (``ctx['adhoc_drain']``) so a kick and a sweep are one drain.
    """
    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.setattr(github_app, "build_installation_auth", _FakeAuth)
    monkeypatch.setattr(
        worker, "_github_claude_md_fetcher", lambda auth, client: _fake_claude_md
    )

    ctx: dict[str, Any] = {}
    try:
        await on_startup(ctx)

        assert callable(ctx["heartbeat_enumerate_repos"])
        assert ctx["heartbeat_clock"].now() is not None  # the real wall-clock seam
        assert callable(ctx["heartbeat_cron_tick"])
        # The heartbeat sweep fires the SAME bound drain the webhook kick fires.
        assert ctx["heartbeat_drain"] is ctx["adhoc_drain"]
    finally:
        await on_shutdown(ctx)


@pytest.mark.asyncio
async def test_on_startup_without_auth_leaves_the_heartbeat_unwired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No auth -> no heartbeat collaborators, so the registered tick stays a safe no-op."""
    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.delattr(github_app, "build_installation_auth", raising=False)

    ctx: dict[str, Any] = {}
    await on_startup(ctx)

    assert "heartbeat_enumerate_repos" not in ctx
    assert "heartbeat_clock" not in ctx
    assert "heartbeat_drain" not in ctx
    assert "heartbeat_cron_tick" not in ctx


@pytest.mark.asyncio
async def test_on_startup_heartbeat_enumerate_yields_opted_in_due_repos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound enumerator lists the App's installed, opted-in repos as DueRepos."""
    from retinue.heartbeat import DueRepo

    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.setattr(github_app, "build_installation_auth", _FakeAuth)
    monkeypatch.setattr(
        worker, "_github_claude_md_fetcher", lambda auth, client: _fake_claude_md
    )
    # The App is installed on these repos; the fetcher reports each as opted in.
    monkeypatch.setattr(_FakeAuth, "installed_repos", ["owner/a", "owner/b"])
    monkeypatch.setattr(worker, "github_config_fetcher", _fake_config_fetcher)

    ctx: dict[str, Any] = {}
    try:
        await on_startup(ctx)
        due = await ctx["heartbeat_enumerate_repos"]()
    finally:
        await on_shutdown(ctx)

    assert [r.repo_full_name for r in due] == ["owner/a", "owner/b"]
    assert all(isinstance(r, DueRepo) for r in due)
    assert all(r.config.staging_branch == "staging" for r in due)


@pytest.mark.asyncio
async def test_on_startup_heartbeat_tick_drives_run_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the collaborators wired, ``heartbeat_tick(ctx)`` drives a real sweep, not the skip.

    Proves criterion 2: one opted-in, cron-due repo is enumerated, its safety-net ad-hoc
    drain fires (one ready issue reaches the build), and its backlog cron lane ticks — the
    not-wired ``is None`` guard is never hit. The leaf gh/build seams are faked so no
    container, gh, model, or network runs.
    """
    import retinue.adhoc_drain as adhoc_drain_mod
    import retinue.cron as cron_mod
    import retinue.pipeline as pipeline_mod
    from retinue.adhoc_build import AdhocIssue
    from retinue.adhoc_drain import ReadyIssue
    from retinue.heartbeat import heartbeat_tick

    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.setattr(github_app, "build_installation_auth", _FakeAuth)
    monkeypatch.setattr(
        worker, "_github_claude_md_fetcher", lambda auth, client: _fake_claude_md
    )
    # One installed repo whose cron is due on every tick, opted in via the fake fetcher.
    cron_yaml = "staging_branch: staging\ncron: '* * * * *'\n"

    def _due_config_fetcher(auth: object, client: object) -> Any:
        async def fetch(repo_full_name: str) -> str | None:
            return cron_yaml

        return fetch

    monkeypatch.setattr(_FakeAuth, "installed_repos", ["owner/due"])
    monkeypatch.setattr(worker, "github_config_fetcher", _due_config_fetcher)

    # Fake the ad-hoc drain's gh seam: one ready issue, none in flight.
    class _FakeAdhocGh:
        def __init__(self, *, token: str) -> None:
            self.token = token

        async def list_ready(self, *, repo_full_name: str) -> list[ReadyIssue]:
            return [ReadyIssue(number=31, labels=["ready-for-agent"], body="")]

        async def in_flight(self, *, repo_full_name: str, issue_number: int) -> bool:
            return False

    monkeypatch.setattr(adhoc_drain_mod, "GhCli", _FakeAdhocGh)

    drained: list[int] = []

    def _fake_bind_adhoc_build(settings: object, auth: object, **kwargs: object) -> object:
        async def build(issue: AdhocIssue, *, repo_full_name: str) -> None:
            drained.append(issue.issue_number)

        return build

    monkeypatch.setattr(pipeline_mod, "bind_adhoc_build", _fake_bind_adhoc_build)

    # Fake the cron backlog gh seam (empty backlog) so the cron lane ticks idle, no build.
    class _FakeCronGh:
        def __init__(self, *, token: str) -> None:
            self.token = token

        async def list_backlog(self, *, repo_full_name: str) -> list[object]:
            return []

    monkeypatch.setattr(cron_mod, "GhCli", _FakeCronGh)

    ctx: dict[str, Any] = {}
    try:
        await on_startup(ctx)
        await heartbeat_tick(ctx)
    finally:
        await on_shutdown(ctx)

    # The due repo's safety-net drain fired (issue #31 reached the build) — not the no-op.
    assert drained == [31]
    # The tick counter advanced, proving run_heartbeat ran rather than the not-wired skip.
    assert ctx["heartbeat_tick_number"] == 1
