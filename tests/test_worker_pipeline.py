"""Tests for the worker's pipeline wiring: the reap + ad-hoc drain kick tasks.

The worker tasks read their collaborators from the Arq ``ctx`` (populated by
``on_startup``). These tests inject fakes into ``ctx`` — a recording pipeline and a config
fetcher — so the dispatch is exercised with no real gh, Anthropic, Docker, or network.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import retinue.github_app as github_app
import retinue.worker as worker
from retinue.github_app import InstallationAuthError, InstallationToken
from retinue.handoff import MergedPullRequest, ReapOutcome, ReapResult
from retinue.queue import RUN_ADHOC_DRAIN_TASK
from retinue.repo_config import RepoConfig
from retinue.worker import (
    WorkerSettings,
    on_shutdown,
    on_startup,
    reap_pr_job,
    run_adhoc_drain_job,
)

_CONFIG_YAML = "target_branch: staging\nretry_cap: 2\n"


@dataclass
class _RecordingPipeline:
    """A fake Pipeline recording every call the worker tasks make against it.

    Only the surviving worker-facing surface is modeled: ``round_for_pr`` (the PR ->
    ``(issue, owned_issues)`` resolution the reap reads) and ``reap_pr`` (the merge reap).
    """

    reaps: list[MergedPullRequest] = field(default_factory=list)
    pr_round: tuple[int, list[int]] | None = (7, [100, 101])

    async def reap_pr(self, merged: MergedPullRequest) -> ReapResult:
        self.reaps.append(merged)
        return ReapResult(outcome=ReapOutcome.REAPED, prd_closed=True)

    async def round_for_pr(
        self, *, repo_full_name: str, pr_number: int
    ) -> tuple[int, list[int]] | None:
        return self.pr_round


def _ctx(pipeline: _RecordingPipeline) -> dict[str, Any]:
    async def fetch_config(repo_full_name: str) -> str | None:
        return _CONFIG_YAML

    async def factory(repo_full_name: str, config: RepoConfig) -> _RecordingPipeline:
        return pipeline

    return {
        "fetch_config": fetch_config,
        "pipeline_factory": factory,
    }


CtxFactory = Callable[..., dict[str, Any]]


@pytest.fixture()
def make_ctx() -> CtxFactory:
    """A :func:`_ctx` factory: an Arq ctx wired to the given recording pipeline."""
    return _ctx


# --- reap -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reap_pr_job_resolves_slices_and_reaps(make_ctx: CtxFactory) -> None:
    """A merge job resolves the PRD + slice issues from run-state and reaps."""
    pipeline = _RecordingPipeline(pr_round=(7, [100, 101]))
    ctx = make_ctx(pipeline)

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
async def test_reap_pr_job_skips_unknown_pr(make_ctx: CtxFactory) -> None:
    """A merge of a PR the retinue never opened is skipped, not reaped."""
    pipeline = _RecordingPipeline(pr_round=None)
    ctx = make_ctx(pipeline)

    await reap_pr_job(ctx, repo_full_name="owner/repo", pr_number=5)

    assert pipeline.reaps == []


@pytest.mark.asyncio
async def test_reap_pr_job_without_pipeline_is_a_noop(make_ctx: CtxFactory) -> None:
    """A bare worker (no pipeline_factory wired) drops the merge without reaping."""
    pipeline = _RecordingPipeline()
    ctx = make_ctx(pipeline)
    del ctx["pipeline_factory"]

    await reap_pr_job(ctx, repo_full_name="owner/repo", pr_number=99)

    assert pipeline.reaps == []


@pytest.mark.asyncio
async def test_reap_pr_job_skips_a_deopted_repo(make_ctx: CtxFactory) -> None:
    """A merge for a repo no longer opted in (no config) is a skip, not a reap."""
    pipeline = _RecordingPipeline()
    ctx = make_ctx(pipeline)

    async def no_config(repo_full_name: str) -> str | None:
        return None

    ctx["fetch_config"] = no_config

    await reap_pr_job(ctx, repo_full_name="owner/repo", pr_number=99)

    assert pipeline.reaps == []


# --- ad-hoc drain kick ----------------------------------------------------------


@pytest.mark.asyncio
async def test_run_adhoc_drain_job_drives_the_bound_drain(make_ctx: CtxFactory) -> None:
    """A kicked drain job calls the bound drain from ctx with the repo (and its config)."""
    calls: list[dict[str, Any]] = []

    async def drain(*, repo_full_name: str, config: RepoConfig) -> None:
        calls.append({"repo": repo_full_name, "config": config})

    ctx = make_ctx(_RecordingPipeline())
    ctx["adhoc_drain"] = drain

    await run_adhoc_drain_job(ctx, repo_full_name="owner/repo")

    assert len(calls) == 1
    assert calls[0]["repo"] == "owner/repo"
    assert calls[0]["config"].target_branch == "staging"


@pytest.mark.asyncio
async def test_run_adhoc_drain_job_without_drain_is_a_noop(
    make_ctx: CtxFactory,
) -> None:
    """With no drain wired (bare worker) the kick logs and returns, never crashing."""
    ctx = make_ctx(_RecordingPipeline())
    assert "adhoc_drain" not in ctx

    await run_adhoc_drain_job(ctx, repo_full_name="owner/repo")  # must not raise


@pytest.mark.asyncio
async def test_run_adhoc_drain_job_skips_a_deopted_repo(make_ctx: CtxFactory) -> None:
    """A repo no longer opted in (no config) is a skip — the drain is never fired."""
    calls: list[str] = []

    async def drain(*, repo_full_name: str, config: RepoConfig) -> None:
        calls.append(repo_full_name)

    ctx = make_ctx(_RecordingPipeline())
    ctx["adhoc_drain"] = drain

    async def no_config(repo_full_name: str) -> str | None:
        return None

    ctx["fetch_config"] = no_config

    await run_adhoc_drain_job(ctx, repo_full_name="owner/repo")

    assert calls == []


def _registered_names() -> set[str]:
    """The registered task names; ``arq.worker.func`` wrappers carry ``.name``."""
    return {
        getattr(fn, "name", None) or getattr(fn, "__name__")  # noqa: B009
        for fn in WorkerSettings.functions
    }


def test_worker_registers_the_adhoc_drain_task() -> None:
    """WorkerSettings registers a function named ``run_adhoc_drain_job`` (the kick task).

    The webhook enqueues ``RUN_ADHOC_DRAIN_TASK`` and Arq dequeues by ``__name__``, so a
    function with that exact name must be in ``WorkerSettings.functions`` or the kick is
    dropped.
    """
    assert RUN_ADHOC_DRAIN_TASK in _registered_names()
    assert run_adhoc_drain_job.__name__ == RUN_ADHOC_DRAIN_TASK


def test_re_enqueued_adhoc_drain_keeps_no_result() -> None:
    """The re-kicked drain registers with ``keep_result=0``.

    arq's enqueue dedups on the completed job's *result* key too; a lingering result
    (default 1h) would silently drop the next kick after a drain finishes.
    """
    from arq.worker import Function

    by_name = {
        fn.name: fn for fn in WorkerSettings.functions if isinstance(fn, Function)
    }
    assert RUN_ADHOC_DRAIN_TASK in by_name, "adhoc drain is not registered via arq func()"
    assert by_name[RUN_ADHOC_DRAIN_TASK].keep_result_s == 0


def test_main_drives_job_timeout_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main`` overrides ``WorkerSettings.job_timeout`` from the configured setting.

    Arq reads ``job_timeout`` off the class before ``on_startup`` runs, so — like
    ``redis_settings`` — it is applied at process start. The arq default (300s) cancels a
    real claude build mid-implement; this override is what keeps the build alive.
    """

    class _FakeSettings:
        redis_url = "redis://localhost:6379"
        job_timeout_seconds = 1234

    monkeypatch.setattr(worker, "settings", _FakeSettings())
    monkeypatch.setattr(worker, "run_worker", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_configure_logging", lambda: None)

    worker.main()

    assert WorkerSettings.job_timeout == 1234


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
    """on_startup takes the auth branch and produces a live pipeline + drain.

    With GitHub App auth resolvable, on_startup must install the config fetcher, the
    ``pipeline_factory``, and the bound ad-hoc drain — and the factory must yield a
    Pipeline threading the *shared* service-level governor. No network, Docker, or model:
    the auth is faked and adapter construction is pure.
    """
    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.setattr(github_app, "build_installation_auth", _FakeAuth)
    # The factory sources CLAUDE.md per repo over the contents API; stub that fetcher so
    # the wiring is exercised without a live GitHub read.
    monkeypatch.setattr(
        worker,
        "github_claude_md_fetcher",
        lambda auth, client: _fake_claude_md,
    )

    ctx: dict[str, Any] = {}
    try:
        await on_startup(ctx)

        # Took the auth branch: the downstream seams are installed.
        assert ctx["github_client"] is not None
        assert callable(ctx["fetch_config"])
        assert callable(ctx["pipeline_factory"])
        # The webhook's ad-hoc kick task reads ``adhoc_drain`` from ctx; a deployed
        # worker under live auth must have it bound so the kick actually drains.
        assert callable(ctx["adhoc_drain"])

        # The produced pipeline carries the accepted config and the one shared governor.
        pipeline = await ctx["pipeline_factory"](
            "owner/repo", RepoConfig(target_branch="staging", retry_cap=2)
        )
        assert pipeline.config.target_branch == "staging"
        assert pipeline.governor is ctx["governor"]
    finally:
        await on_shutdown(ctx)


@pytest.mark.asyncio
async def test_on_shutdown_closes_the_github_client() -> None:
    """on_shutdown closes the GitHub HTTP client :func:`on_startup` opened."""

    class _Client:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    client = _Client()
    await on_shutdown({"github_client": client})

    assert client.closed


@pytest.mark.asyncio
async def test_on_shutdown_without_startup_is_a_noop() -> None:
    """on_shutdown on a ctx on_startup never populated must not raise."""
    await on_shutdown({})


@pytest.mark.asyncio
async def test_on_startup_adhoc_drain_drives_one_issue_to_the_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound ``ctx['adhoc_drain']`` actually drains: one listed issue reaches the build.

    Extends the live-wiring path by *invoking* the bound drain (not merely asserting it is
    callable): a fake gh seam lists one ready issue, and the ad-hoc build (faked to avoid a
    container) drives the per-repo pipeline's ``process_adhoc_pr``. This proves
    ``wiring.bind_adhoc_drain`` wires the per-repo lock registry + shared governor and
    threads the factory-built pipeline into ``bind_adhoc_build`` — the whole assembly runs
    end to end with no Docker, gh, model, or network.
    """
    import retinue.adhoc_drain as adhoc_drain_mod
    import retinue.pipeline as pipeline_mod
    from retinue.adhoc_build import AdhocBuildResult, AdhocIssue
    from retinue.adhoc_drain import FlightState, ReadyIssue

    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.setattr(github_app, "build_installation_auth", _FakeAuth)
    monkeypatch.setattr(
        worker, "github_claude_md_fetcher", lambda auth, client: _fake_claude_md
    )

    # Fake the gh seam the drain constructs: list one ready ad-hoc issue, none in flight.
    class _FakeGhCli:
        def __init__(self, *, token: str) -> None:
            self.token = token

        async def list_ready(
            self, *, repo_full_name: str, label: str
        ) -> list[ReadyIssue]:
            return [ReadyIssue(number=31, labels=["ready-for-agent"], body="")]

        async def flight_state(
            self, *, repo_full_name: str, issue_number: int
        ) -> FlightState:
            return FlightState.ABSENT

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

        config = RepoConfig(target_branch="staging", retry_cap=2)
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


@pytest.mark.asyncio
async def test_on_startup_with_unconfigured_auth_falls_back_to_not_opted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present-but-unconfigured auth builder must degrade, not crash the worker.

    Production wires a concrete ``build_installation_auth`` that raises
    ``InstallationAuthError`` when ``github_app_id``/key path are unset — a fresh deploy
    with no GitHub App registered yet. ``on_startup`` must catch that and install the
    safe not-opted-in fetcher so the worker boots and logs SKIPs (the graceful fallback
    DEPLOY.md promises), rather than the builder's exception killing worker startup.
    """
    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))

    def _raise_unconfigured() -> object:
        raise InstallationAuthError(
            "GitHub App auth is unconfigured: set github_app_id and "
            "github_app_private_key_path"
        )

    monkeypatch.setattr(github_app, "build_installation_auth", _raise_unconfigured)

    ctx: dict[str, Any] = {}
    await on_startup(ctx)

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
        worker, "github_claude_md_fetcher", lambda auth, client: _fake_claude_md
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
        worker, "github_claude_md_fetcher", lambda auth, client: _fake_claude_md
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
    assert all(r.config.target_branch == "staging" for r in due)


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
    from retinue.adhoc_drain import FlightState, ReadyIssue
    from retinue.heartbeat import heartbeat_tick

    monkeypatch.setattr(worker, "settings", _worker_settings(tmp_path))
    monkeypatch.setattr(github_app, "build_installation_auth", _FakeAuth)
    monkeypatch.setattr(
        worker, "github_claude_md_fetcher", lambda auth, client: _fake_claude_md
    )
    # One installed repo whose cron is due on every tick, opted in via the fake fetcher.
    cron_yaml = "target_branch: staging\ncron: '* * * * *'\n"

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

        async def list_ready(
            self, *, repo_full_name: str, label: str
        ) -> list[ReadyIssue]:
            return [ReadyIssue(number=31, labels=["ready-for-agent"], body="")]

        async def flight_state(
            self, *, repo_full_name: str, issue_number: int
        ) -> FlightState:
            return FlightState.ABSENT

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
