"""Tests for the ad-hoc pipeline orchestration (retinue.pipeline).

The slimmed pipeline ties the real adapters together for two responsibilities: open the
ad-hoc ``issue-<N>`` -> target-branch PR after a green build (recording the PR<->issue
mapping), and reap that issue on the human's merge. Every collaborator is injected, so
these tests drive the orchestration with fakes — no Docker, gh, Agent SDK, or network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from retinue.handoff import MergedPullRequest, ReapOutcome
from retinue.pipeline import Pipeline
from retinue.repo_config import (
    ModelEffort,
    RepoConfig,
    RoutingConfig,
    RoutingLevel,
)
from retinue.run_ledger import RunLedgerStore, RunState
from tests.fakes import (
    _created,
    _fake_build_adhoc_issue,
    _FakePrOps,
    _FakeReapGh,
    _governor,
    _RecordingAdhocPipeline,
    _RecordingNotifier,
    _run_ledger,
    _settings,
)


def _config() -> RepoConfig:
    return RepoConfig(target_branch="staging", retry_cap=2)


def _pipeline(tmp_path: Path, **overrides: object) -> Pipeline:
    """Build a slimmed Pipeline whose collaborators default to harmless recording fakes."""
    notifier = _RecordingNotifier()
    base: dict[str, object] = dict(
        config=_config(),
        claude_md="## Definition of done\n```\nuv run pytest\n```\n",
        governor=_governor(tmp_path),
        notifier=notifier,
        create_issue=_created,
        pr_ops=_FakePrOps(),
        reap_gh=_FakeReapGh(),
        retry_store_path=tmp_path / "retries.sqlite3",
        run_state_path=tmp_path / "runstate.sqlite3",
        run_ledger=_run_ledger(tmp_path),
    )
    base.update(overrides)
    return Pipeline(**base)  # type: ignore[arg-type]


# --- ad-hoc PR open --------------------------------------------------------------


@pytest.mark.asyncio
async def test_green_adhoc_build_opens_one_pr_into_target_branch(tmp_path: Path) -> None:
    """A green ad-hoc build opens exactly one PR ``issue-<N>`` -> target branch.

    The ad-hoc PR head is the ``issue-<N>`` branch itself — there is no integration
    branch — and it reuses the shared PR-opener prechecks. The PR<->issue mapping is
    recorded so the reap can resolve the PR back to the single ad-hoc issue.
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
    # The pipeline choke point recorded the run-ledger's pr_opened terminal state.
    rows = await pipeline.run_ledger.rows()
    assert len(rows) == 1
    assert rows[0].issue == 31
    assert rows[0].state == RunState.PR_OPENED.value
    assert rows[0].url == "https://github.com/owner/repo/pull/99"


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
    # The pipeline choke point recorded the run-ledger's failed terminal state.
    rows = await pipeline.run_ledger.rows()
    assert len(rows) == 1
    assert rows[0].issue == 31
    assert rows[0].state == RunState.FAILED.value


# --- review gate consumption -----------------------------------------------------


@pytest.mark.asyncio
async def test_gate_blocking_findings_escalate_and_open_no_pr(tmp_path: Path) -> None:
    """A blocking review gate escalates a hitl notification and opens no PR.

    The green branch stays pushed for a human; the single notification fans out to
    push + comment + label, so the blocking findings land as a ``hitl`` escalation.
    """
    from retinue.adhoc_build import AdhocBuildResult, AdhocIssue, ReviewGateOutcome
    from retinue.notify import Notification
    from retinue.reviewer import ReviewFinding
    from retinue.vocab import HITL_LABEL, Severity

    notifier = _RecordingNotifier()
    pr_ops = _FakePrOps()
    pipeline = _pipeline(tmp_path, notifier=notifier, pr_ops=pr_ops)
    issue = AdhocIssue(repo_full_name="owner/repo", issue_number=31)
    gate = ReviewGateOutcome(
        blocking=[ReviewFinding("Still broken", "the bug survives", Severity.HIGH)],
        backlog=[],
    )

    result = await pipeline.process_adhoc_pr(
        issue, AdhocBuildResult(branch="issue-31", passed=True, gate=gate)
    )

    assert result is None  # no PR opened
    assert pr_ops.opened == []
    assert len(notifier.notes) == 1
    note = notifier.notes[0]
    assert isinstance(note, Notification)
    assert note.repo_full_name == "owner/repo"
    assert note.issue_number == 31
    assert note.label == HITL_LABEL
    assert "Still broken" in note.body
    # No mapping recorded — the PR never opened.
    assert await pipeline.round_for_pr(repo_full_name="owner/repo", pr_number=99) is None
    # The pipeline choke point recorded the run-ledger's escalated terminal state, with
    # the GitHub issue URL (the escalations endpoint's read).
    rows = await pipeline.run_ledger.rows()
    assert len(rows) == 1
    assert rows[0].issue == 31
    assert rows[0].state == RunState.ESCALATED.value
    assert rows[0].url == "https://github.com/owner/repo/issues/31"


@pytest.mark.asyncio
async def test_gate_backlog_findings_are_filed_then_the_pr_opens(tmp_path: Path) -> None:
    """Sub-threshold gate findings become ``priority:<severity>`` backlog nits, PR opens."""
    from retinue.adhoc_build import AdhocBuildResult, AdhocIssue, ReviewGateOutcome
    from retinue.issues import CreatedIssue, IssueDraft
    from retinue.reviewer import ReviewFinding
    from retinue.vocab import BACKLOG_LABEL, Severity

    filed: list[IssueDraft] = []

    async def recording_create(draft: IssueDraft) -> CreatedIssue:
        filed.append(draft)
        return CreatedIssue(issue_number=1000 + len(filed))

    pr_ops = _FakePrOps()
    pipeline = _pipeline(tmp_path, pr_ops=pr_ops, create_issue=recording_create)
    issue = AdhocIssue(repo_full_name="owner/repo", issue_number=31)
    gate = ReviewGateOutcome(
        blocking=[],
        backlog=[
            ReviewFinding("Rename var", "cosmetic", Severity.LOW),
            ReviewFinding("Tidy helper", "minor", Severity.MEDIUM),
        ],
    )

    result = await pipeline.process_adhoc_pr(
        issue, AdhocBuildResult(branch="issue-31", passed=True, gate=gate)
    )

    # The PR opened (backlog does not block).
    assert result is not None and result.opened is True
    assert len(pr_ops.opened) == 1
    # Each backlog finding filed a backlog + priority:<severity> nit.
    assert len(filed) == 2
    assert filed[0].labels == [BACKLOG_LABEL, "priority:low"]
    assert filed[1].labels == [BACKLOG_LABEL, "priority:medium"]
    assert {d.title for d in filed} == {"Rename var", "Tidy helper"}


@pytest.mark.asyncio
async def test_clean_gate_opens_the_pr_without_filing_anything(tmp_path: Path) -> None:
    """A clean gate (no findings) opens the PR and files no backlog nits."""
    from retinue.adhoc_build import AdhocBuildResult, AdhocIssue, ReviewGateOutcome
    from retinue.issues import CreatedIssue, IssueDraft

    filed: list[IssueDraft] = []

    async def recording_create(draft: IssueDraft) -> CreatedIssue:
        filed.append(draft)
        return CreatedIssue(issue_number=1)

    pr_ops = _FakePrOps()
    pipeline = _pipeline(tmp_path, pr_ops=pr_ops, create_issue=recording_create)
    issue = AdhocIssue(repo_full_name="owner/repo", issue_number=31)
    gate = ReviewGateOutcome(blocking=[], backlog=[])

    result = await pipeline.process_adhoc_pr(
        issue, AdhocBuildResult(branch="issue-31", passed=True, gate=gate)
    )

    assert result is not None and result.opened is True
    assert filed == []


# --- reap ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reap_pr_closes_the_issue_and_deletes_the_run_state(tmp_path: Path) -> None:
    """The merge reap closes the ad-hoc issue and deletes the round's run-state row.

    The green build records the PR<->issue mapping; the human merge then reaps the single
    ad-hoc issue and deletes the row (the round's terminal event) so ``round_for_pr``
    resolves to ``None`` — no later sweep re-reconciles a finished PR.
    """
    from retinue.adhoc_build import AdhocBuildResult, AdhocIssue

    reap_gh = _FakeReapGh(children=[])  # no Part-of children: ad-hoc has no PRD parent
    pipeline = _pipeline(tmp_path, reap_gh=reap_gh)
    issue = AdhocIssue(repo_full_name="owner/repo", issue_number=31)
    await pipeline.process_adhoc_pr(
        issue, AdhocBuildResult(branch="issue-31", passed=True)
    )

    # The recorded mapping resolves the PR back to the single ad-hoc issue.
    mapping = await pipeline.round_for_pr(repo_full_name="owner/repo", pr_number=99)
    assert mapping == (31, [])
    assert mapping is not None
    prd_number: int = mapping[0]
    slice_numbers: list[int] = mapping[1]

    merged = MergedPullRequest(
        repo_full_name="owner/repo",
        pr_number=99,
        prd_number=prd_number,
        slice_issues=slice_numbers,
    )
    result = await pipeline.reap_pr(merged)

    assert result.outcome is ReapOutcome.REAPED
    assert reap_gh.closed == [31]  # the single ad-hoc issue, closed exactly once
    # The round's row was deleted, so the merged PR no longer resolves.
    assert await pipeline.round_for_pr(repo_full_name="owner/repo", pr_number=99) is None
    # The merge reap recorded the run-ledger's merged terminal state, overwriting the
    # pr_opened state the green build recorded earlier.
    rows = await pipeline.run_ledger.rows()
    assert len(rows) == 1
    assert rows[0].issue == 31
    assert rows[0].state == RunState.MERGED.value


# --- production factory wiring ---------------------------------------------------


class _FakeAuth:
    async def installation_token(self, repo_full_name: str) -> object:
        from retinue.github_app import InstallationToken

        return InstallationToken(token="ghs_x", clone_url="https://x/y.git")


@pytest.mark.asyncio
async def test_build_pipeline_factory_wires_a_pipeline(tmp_path: Path) -> None:
    """The production factory mints a token and builds a wired Pipeline.

    The slimmed pipeline's surviving seams — the accepted repo config (its target branch
    is the PR base), the PR-opener gh ops, and the reap gh seam — are all set on the
    pipeline the factory returns.
    """
    from retinue.pipeline import build_pipeline_factory

    settings = _settings(tmp_path, ntfy_topic="alerts")
    factory = build_pipeline_factory(
        settings,  # type: ignore[arg-type]
        _FakeAuth(),  # type: ignore[arg-type]
        governor=_governor(tmp_path),
    )
    pipeline = await factory("owner/repo", _config())

    assert isinstance(pipeline, Pipeline)
    assert pipeline.config.target_branch == "staging"  # the PR base is wired
    assert pipeline.pr_ops is not None  # the PR-opener gh seam is wired
    assert pipeline.reap_gh is not None  # the reap gh seam is wired
    assert isinstance(pipeline.run_ledger, RunLedgerStore)  # the run-ledger is wired


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


def test_build_push_sink_picks_pushover_when_no_ntfy(tmp_path: Path) -> None:
    """With only Pushover configured, the push sink is the Pushover backend."""
    from retinue.notify import PushoverPushSink, build_push_sink

    settings = _settings(tmp_path, pushover_token="pk", pushover_user="uk")
    sink = build_push_sink(settings)  # type: ignore[arg-type]
    assert isinstance(sink, PushoverPushSink)


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
    ``process_adhoc_pr`` (which opens the ``issue-<N>`` -> target-branch PR). The build is
    faked so no container spawns; the recording pipeline proves the result threads through.
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
    from retinue.container_build import ContainerImplementer
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
        target_branch="staging",
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
