"""Tests for the ad-hoc build primitive (issue #29).

The flow is plan -> materialize -> implement -> done-check -> push, all in one
disposable container. Every collaborator is faked: a fake planner records the issue it
was asked to plan and returns a canned plan string, a fake implementer records the issue
it was asked to build, the done-check runs against the faked container reused from the
done-check tests, and the container records every command so the order (clone -> branch
off staging -> plan -> materialize-plan-file -> implement -> done-check -> push-on-green)
is assertable. No Agent SDK, no Docker, no gh, no network.

A green done-check pushes ``issue-<N>`` (cut off ``config.staging_branch``); a red
done-check pushes nothing.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from retinue.adhoc_build import (
    PLAN_FILE,
    AdhocBuildResult,
    AdhocIssue,
    AdhocReviewer,
    ContainerAdhocReviewer,
    ContainerPlanner,
    PlanError,
    Planner,
    _issue_diff_command,
    _materialize_plan_command,
    _plan_prompt,
    _slice_for_issue,
    build_adhoc_issue,
)
from retinue.container import Container, RunResult
from retinue.done_check import DoneCheckReport
from retinue.impl_retry import ImplRetryStore, impl_retry_key
from retinue.orchestrator import GitOpsError, Implementer, Slice, _implement_prompt
from retinue.repo_config import RepoConfig
from retinue.reviewer import (
    READY_LABEL,
    REVIEW_FIX_LABEL,
    CreatedIssue,
    IssueDraft,
    ReviewFinding,
    ReviewInput,
    ReviewPlan,
)
from retinue.roles import Role, planner_cli_argv, resolve_model
from tests.test_done_check import (
    CLAUDE_MD,
    FakeAuth,
    FakeRuntime,
    _resolver,
    _sink,
)

PLAN_TEXT = "1. write a failing test\n2. make it pass"


class FakePlanner:
    """Records the issue it was asked to plan; returns a canned plan, marks the container."""

    def __init__(self, plan: str = PLAN_TEXT) -> None:
        self.planned: list[AdhocIssue] = []
        self._plan = plan

    async def plan(self, issue: AdhocIssue, *, container: Container) -> str:
        self.planned.append(issue)
        await container.run_command(["plan", issue.branch])
        return self._plan

    def auth_env(self) -> dict[str, str]:
        return {}


class FakeImplementer:
    """Records the slice it was asked to build; marks the container with an implement event."""

    def __init__(self) -> None:
        self.built: list[Slice] = []
        self.plan_paths: list[str | None] = []

    async def implement(
        self, slice_: Slice, *, container: Container, plan_path: str | None = None
    ) -> None:
        self.built.append(slice_)
        self.plan_paths.append(plan_path)
        await container.run_command(["implement", slice_.branch])

    def auth_env(self) -> dict[str, str]:
        return {}


def _issue(issue_number: int = 29) -> AdhocIssue:
    return AdhocIssue(repo_full_name="owner/repo", issue_number=issue_number)


async def _build(
    *,
    runtime: FakeRuntime,
    planner: Planner | None = None,
    implementer: Implementer | None = None,
    config: RepoConfig | None = None,
    captured: list[DoneCheckReport] | None = None,
    issue: AdhocIssue | None = None,
    reviewer: AdhocReviewer | None = None,
) -> AdhocBuildResult:
    return await build_adhoc_issue(
        issue or _issue(),
        config or RepoConfig(),
        CLAUDE_MD,
        planner=planner or FakePlanner(),
        implementer=implementer or FakeImplementer(),
        auth=FakeAuth(),
        runtime=runtime,
        resolve_secret=_resolver({}),
        report=_sink(captured if captured is not None else []),
        reviewer=reviewer,
    )


# --- branch naming ---------------------------------------------------------------


def test_adhoc_issue_branch_name() -> None:
    """The branch an ad-hoc issue is built on is ``issue-<N>``."""
    assert _issue(29).branch == "issue-29"
    assert _issue(7).branch == "issue-7"


# --- order: plan then implement, both in one container ----------------------------


@pytest.mark.asyncio
async def test_runs_planner_then_implementer_in_one_container() -> None:
    """The primitive plans, then implements, both inside a single started container."""
    planner = FakePlanner()
    implementer = FakeImplementer()
    runtime = FakeRuntime()

    await _build(runtime=runtime, planner=planner, implementer=implementer)

    # Exactly one container was started for the whole build.
    assert runtime.log.count("start:" + runtime.log[0].split(":", 1)[1]) >= 1
    assert sum(1 for event in runtime.log if event.startswith("start:")) == 1
    # The planner ran before the implementer.
    plan_idx = runtime.log.index("run:plan issue-29")
    impl_idx = runtime.log.index("run:implement issue-29")
    assert plan_idx < impl_idx
    # Each was asked about exactly this issue.
    assert planner.planned == [_issue()]
    assert implementer.built == [Slice("owner/repo", 29, 29)]


# --- the plan is materialized into the file the implementer reads -----------------


@pytest.mark.asyncio
async def test_plan_is_materialized_into_the_plan_file_before_implementing() -> None:
    """The captured plan is written to PLAN_FILE in-container before the implementer runs."""
    planner = FakePlanner(plan=PLAN_TEXT)
    runtime = FakeRuntime()

    await _build(runtime=runtime, planner=planner)

    expected_blob = base64.b64encode(PLAN_TEXT.encode()).decode()
    materialize = "run:" + " ".join(_materialize_plan_command(PLAN_TEXT))
    assert materialize in runtime.log
    # The blob carries the exact plan bytes.
    assert expected_blob in materialize
    # Materialization happens after the plan is captured and before the implementer runs.
    mat_idx = runtime.log.index(materialize)
    plan_idx = runtime.log.index("run:plan issue-29")
    impl_idx = runtime.log.index("run:implement issue-29")
    assert plan_idx < mat_idx < impl_idx


def test_plan_prompt_names_the_plan_file_and_issue() -> None:
    """The planner prompt points at the issue and the materialized plan file."""
    prompt = _plan_prompt(_issue(29))
    assert "#29" in prompt
    assert PLAN_FILE in prompt


# --- the implementer is pointed at the plan file it must consume ------------------


@pytest.mark.asyncio
async def test_implementer_is_pointed_at_the_plan_file() -> None:
    """The ad-hoc lane tells the implementer to read PLAN_FILE, not just write it.

    Closes #29 AC2: consumption, not mere materialization — the implementer the ad-hoc
    lane runs receives ``PLAN_FILE`` as its plan path, so it is instructed to read the
    plan the planner wrote before building.
    """
    implementer = FakeImplementer()
    runtime = FakeRuntime()

    await _build(runtime=runtime, implementer=implementer)

    assert implementer.plan_paths == [PLAN_FILE]


def test_implement_prompt_with_plan_path_instructs_reading_the_plan() -> None:
    """A plan_path makes the implement prompt point the subagent at PLAN_FILE first."""
    prompt = _implement_prompt(_slice_for_issue(_issue(29)), plan_path=PLAN_FILE)

    assert PLAN_FILE in prompt
    assert "read" in prompt.lower()


# --- branch cut off staging -------------------------------------------------------


@pytest.mark.asyncio
async def test_issue_branch_is_cut_off_the_staging_branch() -> None:
    """The implementer's ``issue-<N>`` branch is created off ``config.staging_branch``."""
    runtime = FakeRuntime()
    config = RepoConfig(staging_branch="trunk")

    await _build(runtime=runtime, config=config)

    # The branch is checked out off origin/<staging_branch>.
    assert "run:git checkout -B issue-29 origin/trunk" in runtime.log
    assert "run:git fetch origin trunk" in runtime.log


# --- green pushes, red pushes nothing ---------------------------------------------


@pytest.mark.asyncio
async def test_green_done_check_pushes_the_issue_branch() -> None:
    """A green done-check pushes ``issue-<N>`` to origin and reports passed."""
    runtime = FakeRuntime()
    captured: list[DoneCheckReport] = []

    result = await _build(runtime=runtime, captured=captured)

    assert result.passed is True
    assert "run:git push origin issue-29" in runtime.log
    assert [r.passed for r in captured] == [True]


@pytest.mark.asyncio
async def test_red_done_check_pushes_nothing() -> None:
    """A red done-check pushes nothing and reports failed."""
    runtime = FakeRuntime(results={"uv": RunResult(exit_code=1, stderr="boom")})
    captured: list[DoneCheckReport] = []

    result = await _build(runtime=runtime, captured=captured)

    assert result.passed is False
    assert not any(event.startswith("run:git push") for event in runtime.log)
    assert [r.passed for r in captured] == [False]


# --- the container is always destroyed --------------------------------------------


@pytest.mark.asyncio
async def test_container_is_destroyed_on_every_path() -> None:
    """The disposable container is torn down even when the done-check is red."""
    runtime = FakeRuntime(results={"uv": RunResult(exit_code=1, stderr="boom")})

    await _build(runtime=runtime)

    assert runtime.container is not None
    assert runtime.container.destroyed is True


@pytest.mark.asyncio
async def test_planner_and_implementer_credentials_are_injected_at_start() -> None:
    """Both the planner's and the implementer's auth env ride the container at start."""

    class CredPlanner(FakePlanner):
        def auth_env(self) -> dict[str, str]:
            return {"PLANNER_TOKEN": "p"}

    class CredImplementer(FakeImplementer):
        def auth_env(self) -> dict[str, str]:
            return {"IMPLEMENTER_TOKEN": "i"}

    runtime = FakeRuntime()
    await _build(runtime=runtime, planner=CredPlanner(), implementer=CredImplementer())

    assert runtime.started_env is not None
    assert runtime.started_env["PLANNER_TOKEN"] == "p"
    assert runtime.started_env["IMPLEMENTER_TOKEN"] == "i"


# --- ContainerPlanner: the real planner adapter -----------------------------------


class ScriptedContainer:
    """A container that returns a scripted RunResult for the planner exec."""

    def __init__(self, result: RunResult) -> None:
        self._result = result
        self.commands: list[list[str]] = []

    async def run_command(self, command: list[str]) -> RunResult:
        self.commands.append(command)
        return self._result

    async def destroy(self) -> None:  # pragma: no cover - unused in these tests
        pass


@pytest.mark.asyncio
async def test_container_planner_captures_the_plan_from_stdout() -> None:
    """The real planner execs the read-only argv and returns the run's captured output."""
    container = ScriptedContainer(RunResult(exit_code=0, stdout="the plan"))
    planner = ContainerPlanner(credential="cred", auth_mode="api_key", model="m")

    plan = await planner.plan(_issue(29), container=container)

    assert plan == "the plan"
    assert container.commands == [
        planner_cli_argv(prompt=_plan_prompt(_issue(29)), model="m")
    ]


@pytest.mark.asyncio
async def test_container_planner_raises_on_a_nonzero_exit() -> None:
    """A planner exec that exits non-zero raises rather than returning a half plan."""
    container = ScriptedContainer(RunResult(exit_code=2, stderr="kaboom"))
    planner = ContainerPlanner(credential="cred")

    with pytest.raises(PlanError):
        await planner.plan(_issue(29), container=container)


def test_container_planner_defaults_to_the_planner_role_model() -> None:
    """The planner's model defaults to the planner role registry entry."""
    planner = ContainerPlanner(credential="cred")
    assert planner.model == resolve_model(Role.PLANNER)


def test_container_planner_routes_subscription_credential() -> None:
    """A subscription auth mode threads the credential as the OAuth env var."""
    planner = ContainerPlanner(credential="tok", auth_mode="subscription")
    assert planner.auth_env() == {"CLAUDE_CODE_OAUTH_TOKEN": "tok"}


# --- the third pass: the internal reviewer reviews the issue-N diff ----------------


class _ReviewRecorder:
    """Records the reviewer's calls and the issues it filed for assertions."""

    def __init__(self) -> None:
        self.reviewed: list[AdhocIssue] = []
        self.created: list[IssueDraft] = []
        self._next_number = 500

    async def review(self, issue: AdhocIssue, *, container: Container) -> None:
        self.reviewed.append(issue)
        await container.run_command(["review", issue.branch])

    def auth_env(self) -> dict[str, str]:
        return {}

    async def create_issue(self, draft: IssueDraft) -> CreatedIssue:
        self._next_number += 1
        self.created.append(draft)
        return CreatedIssue(issue_number=self._next_number)


@pytest.mark.asyncio
async def test_reviewer_runs_after_a_green_build() -> None:
    """AC1: after a green build, the reviewer reviews the issue-N diff in-container."""
    reviewer = _ReviewRecorder()
    runtime = FakeRuntime()

    await _build(runtime=runtime, reviewer=reviewer)

    assert reviewer.reviewed == [_issue()]
    # The review runs after the implementer and after the green push.
    impl_idx = runtime.log.index("run:implement issue-29")
    push_idx = runtime.log.index("run:git push origin issue-29")
    review_idx = runtime.log.index("run:review issue-29")
    assert impl_idx < review_idx
    assert push_idx < review_idx


@pytest.mark.asyncio
async def test_reviewer_does_not_run_on_a_red_build() -> None:
    """A red done-check skips the reviewer — there is no build to review."""
    reviewer = _ReviewRecorder()
    runtime = FakeRuntime(results={"uv": RunResult(exit_code=1, stderr="boom")})

    result = await _build(runtime=runtime, reviewer=reviewer)

    assert result.passed is False
    assert reviewer.reviewed == []


@pytest.mark.asyncio
async def test_review_never_blocks_the_build_or_push() -> None:
    """AC3: a reviewer that raises does not undo the green build or its push."""

    class ExplodingReviewer:
        async def review(self, issue: AdhocIssue, *, container: Container) -> None:
            raise RuntimeError("reviewer blew up")

        def auth_env(self) -> dict[str, str]:
            return {}

    runtime = FakeRuntime()
    captured: list[DoneCheckReport] = []

    result = await _build(
        runtime=runtime, reviewer=ExplodingReviewer(), captured=captured
    )

    # The build still passed and the branch was still pushed — review is advisory.
    assert result.passed is True
    assert "run:git push origin issue-29" in runtime.log
    assert [r.passed for r in captured] == [True]
    # The container is still torn down on the swallowed-error path.
    assert runtime.container is not None
    assert runtime.container.destroyed is True


@pytest.mark.asyncio
async def test_reviewer_credentials_are_injected_at_start() -> None:
    """The reviewer's auth env rides the build container at start, like the others."""

    class CredReviewer(_ReviewRecorder):
        def auth_env(self) -> dict[str, str]:
            return {"REVIEWER_TOKEN": "r"}

    runtime = FakeRuntime()
    await _build(runtime=runtime, reviewer=CredReviewer())

    assert runtime.started_env is not None
    assert runtime.started_env["REVIEWER_TOKEN"] == "r"


@pytest.mark.asyncio
async def test_no_reviewer_runs_no_review_pass() -> None:
    """An absent reviewer seam leaves the two-pass build unchanged (no review pass)."""
    runtime = FakeRuntime()

    result = await _build(runtime=runtime, reviewer=None)

    assert result.passed is True
    assert not any(event.startswith("run:review") for event in runtime.log)


# --- ContainerAdhocReviewer: the real reviewer adapter ----------------------------


class _DiffContainer:
    """A container that scripts the diff stdout and records every command."""

    def __init__(self, diff: str) -> None:
        self._diff = diff
        self.commands: list[list[str]] = []

    async def run_command(self, command: list[str]) -> RunResult:
        self.commands.append(command)
        if command[:2] == ["git", "diff"]:
            return RunResult(exit_code=0, stdout=self._diff)
        return RunResult(exit_code=0)

    async def destroy(self) -> None:  # pragma: no cover - unused here
        pass


class _RefAwareDiffContainer:
    """A container that resolves a diff only when its base ref actually exists.

    Models the refs ``_clone_and_branch`` leaves in the build container: the
    remote-tracking ``origin/<base>`` and the local ``issue-<N>`` branch, but **no** bare
    local ``<base>`` unless it is the clone's default HEAD. A ``git diff`` whose base side
    names an unknown revision exits non-zero (mirroring git's "unknown revision" 404),
    just as the live container would, so the previous bare-``staging`` form is caught.
    """

    def __init__(self, diff: str, *, known_refs: set[str]) -> None:
        self._diff = diff
        self._known_refs = known_refs
        self.commands: list[list[str]] = []

    async def run_command(self, command: list[str]) -> RunResult:
        self.commands.append(command)
        if command[:2] == ["git", "diff"]:
            base = command[2].split("...", 1)[0]
            if base not in self._known_refs:
                return RunResult(
                    exit_code=128,
                    stderr=f"fatal: ambiguous argument '{base}': unknown revision",
                )
            return RunResult(exit_code=0, stdout=self._diff)
        return RunResult(exit_code=0)

    async def destroy(self) -> None:  # pragma: no cover - unused here
        pass


DEFECT_DIFF = """\
diff --git a/retinue/widget.py b/retinue/widget.py
+def total(items):
+    return sum(items) + 1  # planted off-by-one
"""


def _finding_generator(*findings: ReviewFinding):
    captured: list[ReviewInput] = []

    async def generate(review_input: ReviewInput) -> ReviewPlan:
        captured.append(review_input)
        return ReviewPlan(findings=list(findings))

    return generate, captured


def _reviewer(
    *,
    generate,
    create_issue,
    retry_store: ImplRetryStore,
    config: RepoConfig | None = None,
) -> ContainerAdhocReviewer:
    return ContainerAdhocReviewer(
        repo_full_name="owner/repo",
        config=config or RepoConfig(),
        generate=generate,
        create_issue=create_issue,
        retry_store=retry_store,
    )


def test_issue_diff_command_bases_on_the_remote_tracking_staging_ref() -> None:
    """AC1: the diff base is ``origin/<base>``, a ref the build container actually has.

    ``_clone_and_branch`` only ever creates ``origin/<base>`` (the remote-tracking ref)
    and the local ``issue-<N>`` branch; a bare local ``staging`` exists only when it is
    the clone's default HEAD. Basing on ``origin/<base>`` resolves in every case, while
    the local issue tip stays on the right since the review runs in the build container.
    """
    command = _issue_diff_command("issue-29", "staging")
    assert command == ["git", "diff", "origin/staging...issue-29"]


@pytest.mark.asyncio
async def test_reviewer_files_each_finding_as_a_flat_review_fix_issue(
    tmp_path: Path,
) -> None:
    """AC2: each finding is filed as a flat ``review-fix`` + ``ready-for-agent`` issue.

    Flat = no ``Part of #`` footer (ad-hoc work has no parent PRD) and no Blocked-by
    wiring; the fix loops back as ordinary ad-hoc work.
    """
    rec = _ReviewRecorder()
    generate, captured = _finding_generator(
        ReviewFinding(title="Fix off-by-one", body="total() adds a stray +1."),
        ReviewFinding(title="Stale doc", body="README still claims X."),
    )
    store = ImplRetryStore(tmp_path / "retries.sqlite3")
    reviewer = _reviewer(generate=generate, create_issue=rec.create_issue, retry_store=store)
    container = _DiffContainer(DEFECT_DIFF)

    await reviewer.review(_issue(29), container=container)

    # The reviewer reviewed the issue-29 diff over the remote-tracking staging ref.
    assert ["git", "diff", "origin/staging...issue-29"] in container.commands
    assert captured[0].diff == DEFECT_DIFF
    # Two findings -> two flat review-fix issues.
    assert len(rec.created) == 2
    for draft in rec.created:
        assert REVIEW_FIX_LABEL in draft.labels
        assert READY_LABEL in draft.labels
        # Flat: no PRD back-link footer on an ad-hoc review-fix.
        assert "Part of #" not in draft.body


@pytest.mark.asyncio
async def test_clean_review_files_nothing(tmp_path: Path) -> None:
    """A clean review (no findings) files no issue and records no retry attempt."""
    rec = _ReviewRecorder()
    generate, _ = _finding_generator()  # no findings
    store = ImplRetryStore(tmp_path / "retries.sqlite3")
    reviewer = _reviewer(generate=generate, create_issue=rec.create_issue, retry_store=store)

    await reviewer.review(_issue(29), container=_DiffContainer(""))

    assert rec.created == []
    # No findings means no attempt was consumed against the cap.
    assert await store.count(impl_retry_key(_slice_for_issue(_issue(29)))) == 0


@pytest.mark.asyncio
async def test_review_fix_chain_is_bounded_by_the_retry_cap(tmp_path: Path) -> None:
    """AC4: once the issue has spent its retry budget, the review files no more fixes.

    The per-unit retry cap (the same persisted counter triage uses) bounds how many
    review passes one unit may spend filing fixes, so a review-fix issue cannot spawn
    review fixes without limit.
    """
    rec = _ReviewRecorder()
    generate, _ = _finding_generator(
        ReviewFinding(title="Another fix", body="more to fix")
    )
    store = ImplRetryStore(tmp_path / "retries.sqlite3")
    config = RepoConfig(retry_cap=2)
    reviewer = _reviewer(
        generate=generate, create_issue=rec.create_issue, retry_store=store, config=config
    )
    issue = _issue(29)

    # Each review that files fixes consumes one unit of the retry budget.
    await reviewer.review(issue, container=_DiffContainer(DEFECT_DIFF))
    await reviewer.review(issue, container=_DiffContainer(DEFECT_DIFF))
    assert len(rec.created) == 2  # two passes, cap=2 — budget now spent

    # The third pass is over budget: the reviewer files nothing more (chain bounded).
    rec.created.clear()
    await reviewer.review(issue, container=_DiffContainer(DEFECT_DIFF))
    assert rec.created == []


@pytest.mark.asyncio
async def test_reviewer_credential_rides_the_auth_env(tmp_path: Path) -> None:
    """The reviewer's credential is threaded as the container's auth env."""
    rec = _ReviewRecorder()
    generate, _ = _finding_generator()
    store = ImplRetryStore(tmp_path / "retries.sqlite3")
    reviewer = ContainerAdhocReviewer(
        repo_full_name="owner/repo",
        config=RepoConfig(),
        generate=generate,
        create_issue=rec.create_issue,
        retry_store=store,
        credential="tok",
        auth_mode="subscription",
    )
    assert reviewer.auth_env() == {"CLAUDE_CODE_OAUTH_TOKEN": "tok"}


@pytest.mark.asyncio
async def test_review_diffs_a_non_default_staging_branch(tmp_path: Path) -> None:
    """AC2: a non-default staging branch still yields the issue branch's real diff.

    When ``staging_branch`` is not the clone's default HEAD, no bare local ``<base>``
    ref exists — only ``origin/<base>``. The previous ``<base>...issue-<N>`` form would
    name an unknown revision and 404; basing on ``origin/<base>`` resolves, so the
    reviewer feeds the generator the branch's actual diff and files the finding.
    """
    rec = _ReviewRecorder()
    generate, captured = _finding_generator(
        ReviewFinding(title="Fix off-by-one", body="total() adds a stray +1.")
    )
    store = ImplRetryStore(tmp_path / "retries.sqlite3")
    config = RepoConfig(staging_branch="release")
    reviewer = _reviewer(
        generate=generate, create_issue=rec.create_issue, retry_store=store, config=config
    )
    # The build container only has the remote-tracking ref, not a bare local ``release``.
    container = _RefAwareDiffContainer(DEFECT_DIFF, known_refs={"origin/release"})

    await reviewer.review(_issue(29), container=container)

    # The diff resolved against ``origin/release`` and the real diff reached the generator.
    assert ["git", "diff", "origin/release...issue-29"] in container.commands
    assert captured[0].diff == DEFECT_DIFF
    assert len(rec.created) == 1


@pytest.mark.asyncio
async def test_review_raises_on_a_failed_diff_rather_than_treating_it_as_empty(
    tmp_path: Path,
) -> None:
    """AC3: a failed diff command surfaces as an error, not a silent empty review.

    If the diff exits non-zero (e.g. an unresolvable base ref), ``_issue_diff`` must
    raise so the advisory wrapper can log it — feeding the generator an empty diff would
    leave the branch unreviewed with no error surfaced. Here the base ref is unknown, so
    the diff fails and ``review`` propagates the error (the build's wrapper swallows it).
    """
    rec = _ReviewRecorder()
    generate, captured = _finding_generator()
    store = ImplRetryStore(tmp_path / "retries.sqlite3")
    reviewer = _reviewer(generate=generate, create_issue=rec.create_issue, retry_store=store)
    # No known refs: every ``git diff`` 404s, so the diff command fails.
    container = _RefAwareDiffContainer("", known_refs=set())

    with pytest.raises(GitOpsError):
        await reviewer.review(_issue(29), container=container)

    # The generator never ran on a garbage/empty diff, and nothing was filed.
    assert captured == []
    assert rec.created == []
