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
from collections.abc import Mapping

import pytest

from retinue.adhoc_build import (
    PLAN_FILE,
    AdhocBuildResult,
    AdhocIssue,
    ContainerPlanner,
    PlanError,
    Planner,
    ReviewGateOutcome,
    _issue_diff_command,
    _materialize_plan_command,
    _partition_findings,
    _plan_prompt,
    _run_review_gate,
    _slice_for_issue,
    build_adhoc_issue,
    parse_chain_depth,
    render_chain_depth,
)
from retinue.container import Container, RunResult
from retinue.container_build import (
    Implementer,
    ImplementError,
    Slice,
    _implement_prompt,
)
from retinue.done_check import DoneCheckReport
from retinue.repo_config import RepoConfig
from retinue.reviewer import (
    ReviewFinding,
    ReviewInput,
    ReviewPlan,
)
from retinue.roles import Role, planner_cli_argv, resolve_model
from retinue.vocab import Severity
from tests.fakes import (
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
    review_generate=None,
) -> AdhocBuildResult:
    return await build_adhoc_issue(
        issue or _issue(),
        config or RepoConfig(target_branch="staging"),
        CLAUDE_MD,
        planner=planner or FakePlanner(),
        implementer=implementer or FakeImplementer(),
        auth=FakeAuth(),
        runtime=runtime,
        resolve_secret=_resolver({}),
        report=_sink(captured if captured is not None else []),
        review_generate=review_generate,
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
    config = RepoConfig(target_branch="trunk")

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


# --- hollow implement: zero commits fails the ad-hoc build -------------------------


@pytest.mark.asyncio
async def test_implement_landing_no_commits_raises_instead_of_vacuous_green() -> None:
    """An implementer run that lands zero commits fails the build before the done-check.

    The hollow-implement failure the PRD lane already guards against: the agent no-ops,
    exits 0, and the done-check passes vacuously over the untouched tree — pushing an
    empty branch a PR is then opened from. Counting commits since ``origin/<base>``
    right after the implement catches it: a ``0`` count raises ``ImplementError``,
    pushes nothing, and still tears the container down.
    """
    runtime = FakeRuntime(
        results={"git rev-list": RunResult(exit_code=0, stdout="0\n")}
    )

    with pytest.raises(ImplementError, match="landed no commits"):
        await _build(runtime=runtime)

    assert not any("git push" in e for e in runtime.log)
    assert runtime.container is not None and runtime.container.destroyed


@pytest.mark.asyncio
async def test_failed_commit_count_probe_raises_not_passes() -> None:
    """A rev-list probe that itself fails (empty stdout, bad exit) raises, not passes.

    An unreadable count must not be read as "commits exist" — that would re-open the
    vacuous-green hole whenever the probe breaks.
    """
    runtime = FakeRuntime(
        results={"git rev-list": RunResult(exit_code=128, stderr="fatal: bad rev")}
    )

    with pytest.raises(ImplementError):
        await _build(runtime=runtime)


@pytest.mark.asyncio
async def test_commit_count_probe_runs_between_implement_and_done_check() -> None:
    """The guard probes commits after the implement and before the done-check runs."""
    runtime = FakeRuntime()

    await _build(runtime=runtime)

    implement_at = runtime.log.index("run:implement issue-29")
    probe_at = next(i for i, e in enumerate(runtime.log) if "git rev-list" in e)
    check_at = next(i for i, e in enumerate(runtime.log) if "uv run pytest" in e)
    assert implement_at < probe_at < check_at


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

    async def run_command(
        self, command: list[str], *, env: Mapping[str, str] | None = None
    ) -> RunResult:
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


# --- the review gate: diff -> review -> fix pass -> re-review -> partition ----------


DEFECT_DIFF = """\
diff --git a/retinue/widget.py b/retinue/widget.py
+def total(items):
+    return sum(items) + 1  # planted off-by-one
"""


def _finding(title: str, severity: Severity, body: str = "the finding body") -> ReviewFinding:
    return ReviewFinding(title=title, body=body, severity=severity)


def _scripted_generator(*plans: ReviewPlan):
    """A review generator that returns the given plans in order (review1, review2, ...)."""
    captured: list[ReviewInput] = []
    queue = list(plans)

    async def generate(review_input: ReviewInput) -> ReviewPlan:
        captured.append(review_input)
        return queue.pop(0)

    return generate, captured


class _GateContainer:
    """Scripts the gate's git diff / done-check / push commands and records them.

    ``diffs`` is the queue of stdout returned for each ``git diff`` (review1 then review2);
    ``fix_passes`` is whether the fix-pass done-check rerun (every ``uv run`` command)
    goes green. Everything else — the plan-file writes, the implement marker, the push —
    succeeds.
    """

    def __init__(self, *, diffs: list[str], fix_passes: bool = True) -> None:
        self._diffs = list(diffs)
        self._fix_passes = fix_passes
        self.commands: list[list[str]] = []

    async def run_command(
        self, command: list[str], *, env: Mapping[str, str] | None = None
    ) -> RunResult:
        self.commands.append(command)
        if command[:2] == ["git", "diff"]:
            return RunResult(exit_code=0, stdout=self._diffs.pop(0))
        if command[:2] == ["uv", "run"]:
            return RunResult(exit_code=0 if self._fix_passes else 1, stderr="boom")
        return RunResult(exit_code=0)

    async def destroy(self) -> None:  # pragma: no cover - unused here
        pass


async def _gate(
    container: _GateContainer,
    generate,
    *,
    config: RepoConfig | None = None,
    implementer: FakeImplementer | None = None,
) -> ReviewGateOutcome:
    return await _run_review_gate(
        _issue(29),
        config or RepoConfig(target_branch="staging"),
        CLAUDE_MD,
        container=container,
        review_generate=generate,
        implementer=implementer or FakeImplementer(),
    )


def test_issue_diff_command_bases_on_the_remote_tracking_staging_ref() -> None:
    """The gate diff base is ``origin/<base>``, a ref the build container actually has.

    ``clone_and_branch`` only ever creates ``origin/<base>`` (the remote-tracking ref)
    and the local ``issue-<N>`` branch; a bare local ``staging`` exists only when it is
    the clone's default HEAD. Basing on ``origin/<base>`` resolves in every case, while
    the local issue tip stays on the right since the gate runs in the build container.
    """
    assert _issue_diff_command("issue-29", "staging") == [
        "git",
        "diff",
        "origin/staging...issue-29",
    ]


def test_partition_findings_splits_at_high() -> None:
    """Findings at or above HIGH are blocking; below HIGH are backlog."""
    findings = [
        _finding("crit", Severity.CRITICAL),
        _finding("high", Severity.HIGH),
        _finding("med", Severity.MEDIUM),
        _finding("low", Severity.LOW),
    ]
    blocking, backlog = _partition_findings(findings)
    assert [f.title for f in blocking] == ["crit", "high"]
    assert [f.title for f in backlog] == ["med", "low"]


@pytest.mark.asyncio
async def test_clean_review_gate_runs_no_fix_pass() -> None:
    """A clean review₁ short-circuits: no fix pass, no re-push, empty outcome."""
    generate, captured = _scripted_generator(ReviewPlan(findings=[]))
    container = _GateContainer(diffs=[DEFECT_DIFF])
    implementer = FakeImplementer()

    outcome = await _gate(container, generate, implementer=implementer)

    assert outcome == ReviewGateOutcome(blocking=[], backlog=[])
    # Only review₁ ran; the fix implementer never ran and nothing was re-pushed.
    assert len(captured) == 1
    assert implementer.built == []
    assert not any(c[:2] == ["git", "push"] for c in container.commands)


@pytest.mark.asyncio
async def test_fix_pass_repushes_and_partitions_surviving_findings() -> None:
    """Findings trigger a fix pass; a green rerun re-pushes and partitions review₂.

    review₁ finds a high + a low, the implementer fixes them, the done-check reruns
    green, the branch is re-pushed, and review₂'s surviving findings — a lone low — are
    partitioned into backlog (nothing blocking).
    """
    generate, captured = _scripted_generator(
        ReviewPlan(findings=[_finding("bug", Severity.HIGH), _finding("nit", Severity.LOW)]),
        ReviewPlan(findings=[_finding("leftover nit", Severity.LOW)]),
    )
    container = _GateContainer(diffs=[DEFECT_DIFF, "second diff"], fix_passes=True)
    implementer = FakeImplementer()

    outcome = await _gate(container, generate, implementer=implementer)

    assert outcome.regressed is False
    assert outcome.blocking == []
    assert [f.title for f in outcome.backlog] == ["leftover nit"]
    # The fix pass ran the implementer over the plan file, then re-pushed the branch.
    assert implementer.built == [_slice_for_issue(_issue(29))]
    assert ["git", "push", "origin", "issue-29"] in container.commands
    # Both review passes ran, over the two successive diffs.
    assert [rv.diff for rv in captured] == [DEFECT_DIFF, "second diff"]


@pytest.mark.asyncio
async def test_surviving_high_finding_is_blocking() -> None:
    """A finding review₂ still sees at HIGH lands in the blocking bucket."""
    generate, _ = _scripted_generator(
        ReviewPlan(findings=[_finding("bug", Severity.HIGH)]),
        ReviewPlan(findings=[_finding("still broken", Severity.HIGH)]),
    )
    container = _GateContainer(diffs=[DEFECT_DIFF, "second diff"], fix_passes=True)

    outcome = await _gate(container, generate)

    assert outcome.regressed is False
    assert [f.title for f in outcome.blocking] == ["still broken"]
    assert outcome.backlog == []


@pytest.mark.asyncio
async def test_fix_pass_regression_blocks_and_does_not_push() -> None:
    """A fix pass that turns the done-check red is a regression: blocking, no re-push.

    The red fix must not be pushed — the branch stays at its green pre-fix pushed state —
    and the outcome carries a single synthetic blocking finding so a regression escalates
    like any other block. review₂ never runs (there is nothing green to re-review).
    """
    generate, captured = _scripted_generator(
        ReviewPlan(findings=[_finding("bug", Severity.HIGH)]),
    )
    container = _GateContainer(diffs=[DEFECT_DIFF], fix_passes=False)

    outcome = await _gate(container, generate)

    assert outcome.regressed is True
    assert len(outcome.blocking) == 1
    assert outcome.blocking[0].severity >= Severity.HIGH
    assert outcome.backlog == []
    # No re-push of the red fix, and review₂ never ran.
    assert not any(c[:2] == ["git", "push"] for c in container.commands)
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_gate_error_is_fail_closed_blocking() -> None:
    """A gate that raises mid-run blocks the PR instead of propagating.

    The branch is already pushed green when the gate starts, so a raised gate — an LLM
    5xx/429 from the reviewer, a failed fix pass, an unresolvable diff — must not escape
    :func:`_run_review_gate`: escaping would skip ``process_adhoc_pr``, leave no ``hitl``,
    and let the next drain's stranded recovery open the PR gate-bypassed. Instead the gate
    swallows the error into a single synthetic blocking finding so the pipeline escalates.
    """

    async def boom(_: ReviewInput) -> ReviewPlan:
        raise RuntimeError("reviewer 503")

    container = _GateContainer(diffs=[DEFECT_DIFF])

    outcome = await _gate(container, boom)

    assert outcome.regressed is False
    assert len(outcome.blocking) == 1
    assert outcome.blocking[0].severity >= Severity.HIGH
    assert outcome.backlog == []


# --- build_adhoc_issue integration: the gate rides the on_green hook ---------------


@pytest.mark.asyncio
async def test_gate_runs_after_a_green_build_and_is_captured_on_the_result() -> None:
    """A clean gate rides on_green after the green push and is captured on the result."""
    generate, captured = _scripted_generator(ReviewPlan(findings=[]))
    runtime = FakeRuntime()

    result = await _build(runtime=runtime, review_generate=generate)

    assert result.passed is True
    assert result.gate == ReviewGateOutcome(blocking=[], backlog=[])
    # The review ran after the implementer and after the green push.
    impl_idx = runtime.log.index("run:implement issue-29")
    push_idx = runtime.log.index("run:git push origin issue-29")
    diff_idx = next(i for i, e in enumerate(runtime.log) if "git diff origin" in e)
    assert impl_idx < diff_idx
    assert push_idx < diff_idx
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_gate_does_not_run_on_a_red_build() -> None:
    """A red done-check skips the gate — there is no built work to review."""
    generate, captured = _scripted_generator(ReviewPlan(findings=[]))
    runtime = FakeRuntime(results={"uv": RunResult(exit_code=1, stderr="boom")})

    result = await _build(runtime=runtime, review_generate=generate)

    assert result.passed is False
    assert result.gate is None
    assert captured == []


@pytest.mark.asyncio
async def test_gate_error_blocks_the_pr_without_stranding_the_branch() -> None:
    """A gate error on a green build yields a blocking result, never a raised build.

    ``build_adhoc_issue`` pushes the branch before the gate runs, so a gate that raised
    out of ``on_green`` would leave the branch pushed with no PR and — because the raise
    skips ``process_adhoc_pr`` — no ``hitl`` escalation, which the next drain would
    "recover" into a gate-bypassed PR. The build must instead return ``passed=True`` with a
    blocking gate so the pipeline escalates the issue to a human.
    """

    async def boom(_: ReviewInput) -> ReviewPlan:
        raise RuntimeError("reviewer 503")

    runtime = FakeRuntime()

    result = await _build(runtime=runtime, review_generate=boom)

    assert result.passed is True
    assert result.gate is not None
    assert len(result.gate.blocking) == 1
    assert result.gate.blocking[0].severity >= Severity.HIGH


@pytest.mark.asyncio
async def test_no_review_generate_leaves_the_gate_unset() -> None:
    """An absent reviewer seam leaves the two-pass build unchanged (gate is None)."""
    runtime = FakeRuntime()

    result = await _build(runtime=runtime, review_generate=None)

    assert result.passed is True
    assert result.gate is None
    assert not any("git diff origin" in e for e in runtime.log)


@pytest.mark.asyncio
async def test_gate_findings_trigger_a_second_push_via_the_fix_pass() -> None:
    """Gate findings drive a fix pass whose green rerun re-pushes the branch.

    Through the real ``build_adhoc_issue`` container lifecycle, a non-clean review₁ makes
    the gate run the implementer a second time and re-push, so the branch is pushed twice
    (the initial green push plus the post-fix re-push).
    """
    generate, _ = _scripted_generator(
        ReviewPlan(findings=[_finding("nit", Severity.LOW)]),
        ReviewPlan(findings=[]),
    )
    runtime = FakeRuntime()
    implementer = FakeImplementer()

    result = await _build(
        runtime=runtime, review_generate=generate, implementer=implementer
    )

    assert result.passed is True
    assert result.gate == ReviewGateOutcome(blocking=[], backlog=[])
    # Two pushes: the initial green push and the gate's post-fix re-push.
    assert sum(1 for e in runtime.log if e == "run:git push origin issue-29") == 2
    # The implementer ran twice: the build, then the fix pass.
    assert implementer.built == [_slice_for_issue(_issue(29))] * 2


# --- chain-depth marker: render / parse round-trip (kept; drain reads it back) -----


def test_chain_depth_marker_round_trips() -> None:
    """The chain-depth lineage marker renders into a body and parses back out."""
    body = f"some finding text.\n\n{render_chain_depth(2)}"
    assert parse_chain_depth(body) == 2


def test_a_body_without_a_marker_is_chain_origin_depth_zero() -> None:
    """A hand-filed issue carries no marker, so it starts the chain at depth 0."""
    assert parse_chain_depth("just a plain issue body, no lineage marker") == 0


def test_from_fetched_issue_reads_chain_depth_from_the_body() -> None:
    """A fetched body carrying ``Chain-depth: <n>`` yields ``chain_depth == n``.

    The seam the ad-hoc drain calls instead of building ``AdhocIssue`` by hand: it parses
    the lineage marker so the drain reconstructs the depth off the fetched body.
    """
    body = f"a review-fix to apply.\n\n{render_chain_depth(2)}"
    issue = AdhocIssue.from_fetched_issue("owner/repo", 503, body)

    assert issue == AdhocIssue(
        repo_full_name="owner/repo", issue_number=503, chain_depth=2
    )


def test_from_fetched_issue_defaults_a_marker_less_body_to_depth_zero() -> None:
    """A marker-less fetched body is a chain origin, so ``chain_depth == 0``."""
    issue = AdhocIssue.from_fetched_issue("owner/repo", 29, "a hand-filed nit, no marker")

    assert issue.chain_depth == 0
    assert issue == AdhocIssue(repo_full_name="owner/repo", issue_number=29)


