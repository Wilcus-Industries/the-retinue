"""Ad-hoc build primitive: plan -> materialize -> implement -> review, gated on done-check.

The scheduler drain routes a ``ready-for-agent`` issue here. There is no integration branch
and no merge: the issue is built directly on an ``issue-<N>`` branch cut off the resolved
target branch (``config.require_target_branch()``), and a green build pushes that branch for
the PR-open step. The whole build runs in **one disposable container** that is destroyed on
every path (:func:`build_adhoc_issue`):

1. **clone + branch** — the container clones the repo over the installation token and
   checks out a fresh ``issue-<N>`` branch off the resolved target branch,
2. **plan** — the read-only planner (the :data:`~retinue.roles.Role.PLANNER` registry
   entry, Opus on the in-container CLI) maps the code with an Explore subagent and emits a
   plan, captured from its output (it writes nothing to the workspace),
3. **materialize** — the captured plan is written byte-exact into :data:`PLAN_FILE`, the
   one file the implementer reads, so the plan crosses from the read-only planner to the
   write-capable implementer through the workspace rather than a second model call,
4. **implement** — the same implementer the PRD lane uses (Sonnet/high on the in-container
   CLI) is pointed at :data:`PLAN_FILE` via its ``plan_path`` and told to read the plan
   before building, then implements TDD-first and commits to the ``issue-<N>`` branch; an
   implement run that lands **zero commits** on the branch fails the build (the shared
   hollow-implement guard) instead of pushing an empty branch,
5. **done-check** — the repo's done-check runs in the *same* container over the real
   changes, and the outcome is posted to the report sink,
6. **push** — only on a green done-check is ``issue-<N>`` pushed to origin; a red check
   pushes nothing,
7. **review gate** — after a green push, the in-session gate (:func:`_run_review_gate`)
   runs the internal reviewer (the :data:`~retinue.roles.Role.REVIEWER` Opus role) over
   the ``issue-<N>`` diff. On a clean review the gate is a no-op. On findings it runs one
   critique-and-fix pass: the same implementer fixes the flagged findings in the *same*
   container, the done-check re-runs, and — only if it stays green — the branch is
   re-pushed and the reviewer runs again over the fixed diff. The surviving findings are
   partitioned by severity into a :class:`ReviewGateOutcome`: those at or above
   :attr:`~retinue.vocab.Severity.HIGH` are *blocking* (the pipeline escalates the issue
   and opens no PR), the rest are *backlog* nits the pipeline files as
   ``priority:<severity>`` follow-ups before opening the PR. A fix pass that turns the
   done-check red is a regression: the gate flags it blocking and does **not** push the
   red fix, so the branch stays at its green pre-fix pushed state.

Every side-effecting collaborator — the planner spawn, the implementer spawn, the
reviewer, the container runtime, the auth, the secret resolver, and the report sink — is
injected, so the whole flow is exercised in tests with no Agent SDK, no Docker, no gh, and
no network. The whole container lifecycle — start, clone+branch, implement, the
hollow-implement guard, done-check, push-on-green, report, teardown — is the shared
:func:`retinue.container_build.build_issue_in_container` (the same lifecycle the PRD lane
runs), and the gate reuses the internal reviewer's
:class:`~retinue.reviewer.ReviewGenerator` seam and the same implementer; this module
only adds the planner seam, the plan materialization, threading :data:`PLAN_FILE` into
the implementer as its ``plan_path``, the no-merge plan->execute ordering, and the
in-session review gate.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from retinue.container import Container, ContainerRuntime
from retinue.container_build import (
    GitOpsError,
    Implementer,
    Slice,
    build_issue_in_container,
    implement_env,
    push_branch,
    write_file_command,
)
from retinue.done_check import (
    DEFAULT_IMAGE,
    ReportSink,
    SecretResolver,
    parse_done_check,
    run_done_check_commands,
)
from retinue.github_app import InstallationAuth
from retinue.repo_config import RepoConfig
from retinue.reviewer import (
    ReviewFinding,
    ReviewGenerator,
    ReviewInput,
)
from retinue.roles import Role, planner_cli_argv, resolve_model
from retinue.vocab import Severity, issue_branch

logger = logging.getLogger(__name__)

# The single file the captured plan is materialized into and the implementer reads. It
# lives under a dot-dir so it is unobtrusive in the worktree; the path is the contract
# between the read-only planner and the write-capable implementer, named in the planner's
# prompt so the plan it emits is framed as "what will land in this file".
PLAN_FILE = ".retinue/plan.md"

# The lineage marker stamped into a filed review-fix issue's body and read back when that
# issue loops in as ad-hoc work. It carries the chain's *depth* — how many review-fix hops
# separate this issue from the chain origin — so each hop inherits one decreasing budget
# and the chain terminates at ``config.retry_cap`` regardless of the fresh GitHub issue
# number each hop is filed under. Rendered like the slicer's ``Part of #<prd>`` footer and
# parsed back with :data:`_CHAIN_DEPTH_RE`. An issue with no marker is depth 0 (a chain
# origin: a hand-filed nit or the first ad-hoc build).
_CHAIN_DEPTH_PREFIX = "Chain-depth:"
_CHAIN_DEPTH_RE = re.compile(rf"^{re.escape(_CHAIN_DEPTH_PREFIX)}\s*(\d+)\s*$", re.MULTILINE)


def render_chain_depth(depth: int) -> str:
    """Render the ``Chain-depth: <depth>`` lineage marker for a filed review-fix body."""
    return f"{_CHAIN_DEPTH_PREFIX} {depth}"


def parse_chain_depth(body: str) -> int:
    """Read the chain depth from an issue ``body``; ``0`` when no marker is present.

    An issue carrying no :data:`_CHAIN_DEPTH_PREFIX` marker is a chain origin (a
    hand-filed nit or the first ad-hoc build), so it starts the chain at depth 0.
    """
    match = _CHAIN_DEPTH_RE.search(body)
    return int(match.group(1)) if match else 0


@dataclass(frozen=True)
class AdhocIssue:
    """One standalone issue the ad-hoc lane builds directly off staging.

    Attributes:
        repo_full_name: The target repo, e.g. "owner/repo".
        issue_number: The issue's GitHub number; the build commits to the derived
            ``issue-<N>`` branch.
        chain_depth: How many review-fix hops separate this issue from its chain origin.
            A hand-filed or first-built issue is depth 0; a review-fix this build files
            is stamped (and loops back as) the next depth. The review pass uses it — not
            the per-issue GitHub number — to bound the ``#29 -> #501 -> #503 -> ...``
            chain, since each hop is filed under a fresh number with no shared state.
    """

    repo_full_name: str
    issue_number: int
    chain_depth: int = 0

    @classmethod
    def from_fetched_issue(
        cls, repo_full_name: str, issue_number: int, body: str
    ) -> AdhocIssue:
        """Build the issue from a fetched GitHub issue, reading its lineage depth.

        The canonical constructor the ad-hoc drain (:mod:`retinue.lane`'s ranked drain,
        #32) must call instead of ``AdhocIssue(repo_full_name=..., issue_number=...)``.
        It parses the ``Chain-depth: <n>`` marker the prior review-fix hop stamped into
        ``body`` (:func:`parse_chain_depth`) and threads it into :attr:`chain_depth`, so
        the #39 review-fix chain bound — which lives entirely in the issue body — is read
        back and stops being inert. Building the issue by hand instead would default every
        hop to depth 0, leaving the ``#29 -> #501 -> #503 -> ...`` chain unbounded.

        Args:
            repo_full_name: The target repo, e.g. ``"owner/repo"``.
            issue_number: The fetched issue's GitHub number.
            body: The fetched issue body; its ``Chain-depth:`` marker carries the depth
                (a marker-less body is a chain origin, depth 0).
        """
        return cls(
            repo_full_name=repo_full_name,
            issue_number=issue_number,
            chain_depth=parse_chain_depth(body),
        )

    @property
    def branch(self) -> str:
        """The branch the build commits to: ``issue-<N>``."""
        return issue_branch(self.issue_number)


class Planner(Protocol):
    """Spawns one read-only planner that produces a plan for an issue. The planner seam.

    A production implementation execs the read-only headless ``claude`` CLI *inside the
    disposable build container* the orchestration passes in (the same container the
    implementer later builds in), maps the code with an Explore subagent, and returns the
    plan captured from the run's output. Tests inject a fake that records the request and
    returns a canned plan without any real spawn. The plan text is the contract — the
    orchestration materializes it into :data:`PLAN_FILE` for the implementer to read.
    """

    async def plan(self, issue: AdhocIssue, *, container: Container) -> str:
        """Plan ``issue`` in ``container`` (read-only) and return the captured plan."""
        ...

    def auth_env(self) -> dict[str, str]:
        """The env the planner authenticates with, merged into the container at start."""
        ...


def _plan_prompt(issue: AdhocIssue) -> str:
    """Assemble the planner's per-issue prompt: which issue to plan, where the plan lands.

    Names the issue to plan and tells the planner its emitted plan will be materialized
    into :data:`PLAN_FILE` for the implementer to read, so the planner frames its output
    as the implementation plan that file will carry rather than incidental prose.
    """
    return (
        f"Produce an implementation plan for issue #{issue.issue_number} of "
        f"{issue.repo_full_name}. Your plan will be saved to '{PLAN_FILE}' and read by "
        "the implementer, so write the plan itself as your response."
    )


def _materialize_plan_command(plan: str) -> list[str]:
    """Argv that writes the captured ``plan`` into :data:`PLAN_FILE`, byte-exact.

    Reuses the orchestrator's base64 in-container file writer so the plan's markdown —
    backticks, quotes, newlines — survives untouched and nothing in it is interpreted as
    shell syntax. The parent dot-dir is created first so the write into it can't fail.
    """
    return write_file_command(PLAN_FILE, plan)


_ENSURE_PLAN_DIR_COMMAND = ["mkdir", "-p", ".retinue"]


class PlanError(RuntimeError):
    """The read-only planner run ended in an error rather than producing a plan.

    Raised when the in-container ``claude`` plan exec exits non-zero, so the build surfaces
    the failure rather than materializing an empty or half plan and implementing against it.
    """


@dataclass(frozen=True)
class ContainerPlanner:
    """Real :class:`Planner`: produce a plan by exec-ing the read-only ``claude`` CLI.

    Satisfies the planner protocol ``plan(issue, *, container) -> str`` so it drops in
    where the fake planner sits in tests and at the wiring site. It execs the read-only
    headless ``claude`` argv (:func:`retinue.roles.planner_cli_argv`) inside the already
    cloned, branch-checked-out container and returns the plan captured from the run's
    stdout. A non-zero exit raises :class:`PlanError`. The plan is captured from output,
    so unlike the implementer there is no ``--output-format json`` result contract.

    Attributes:
        credential: The Anthropic credential (API key or subscription OAuth token).
        auth_mode: ``"api_key"`` (credential rides ``ANTHROPIC_API_KEY``) or
            ``"subscription"`` (credential rides ``CLAUDE_CODE_OAUTH_TOKEN``).
        model: The planning model id; defaults to the
            :data:`~retinue.roles.Role.PLANNER` registry entry (Opus 4.8), which a repo's
            routing level can replace at the wiring site.
    """

    credential: str
    auth_mode: str = "api_key"
    model: str = field(default_factory=lambda: resolve_model(Role.PLANNER))

    async def plan(self, issue: AdhocIssue, *, container: Container) -> str:
        """Exec the read-only ``claude`` planner in ``container``; return its plan."""
        argv = planner_cli_argv(prompt=_plan_prompt(issue), model=self.model)
        result = await container.run_command(argv)
        if not result.ok:
            raise PlanError(
                f"planner for {issue.branch} exited {result.exit_code}: {result.stderr}"
            )
        logger.info("Planner for %s produced a plan in-container", issue.branch)
        return result.stdout

    def auth_env(self) -> dict[str, str]:
        """The credential env the orchestration merges into the build container at start."""
        return implement_env(self.credential, self.auth_mode)


def _issue_diff_command(branch: str, base: str) -> list[str]:
    """Argv for the diff a freshly-built ``branch`` contributed over the staging ``base``.

    Uses the three-dot form (``origin/<base>...branch``) so the diff is the issue branch's
    own contribution since it was cut off the staging base — the work the build produced —
    rather than also folding in whatever else advanced staging in parallel. Both sides must
    resolve against the refs the build container actually has:
    :func:`retinue.container_build.clone_and_branch` runs ``git fetch origin <base>`` then
    ``git checkout -B issue-<N> origin/<base>``, so it creates the remote-tracking
    ``origin/<base>`` ref and the local ``issue-<N>`` branch but **no** bare local
    ``<base>`` (that exists only when the target branch happens to be the clone's default
    HEAD). The base is therefore ``origin/<base>`` — mirroring the orchestrator's
    the round-diff base, whose base side is
    also a resolvable ref — while the branch stays the *local* ``issue-<N>`` ref the build
    just committed to (no ``origin/`` prefix): the review runs in the same container that
    built it, so the local tip is the surface to review.
    """
    return ["git", "diff", f"origin/{base}...{branch}"]


# The blocking threshold: a finding at or above this severity blocks the PR; the rest are
# filed as backlog nits. The PRD pins it to HIGH — a correctness bug or shipped-broken
# behaviour stops the PR, a cosmetic or minor concern does not.
_BLOCKING_THRESHOLD = Severity.HIGH

# The framing line that leads the fix-pass plan materialized into :data:`PLAN_FILE` so the
# implementer reads the review's findings as the work to do, not incidental prose.
_FIX_PLAN_HEADER = "Address these review findings before the PR opens:"


@dataclass(frozen=True)
class ReviewGateOutcome:
    """How the in-session review gate came out for one built issue.

    Attributes:
        blocking: Findings at or above :data:`_BLOCKING_THRESHOLD` that the surviving
            (post-fix) review still sees. A non-empty list stops the PR — the pipeline
            escalates the issue and opens no PR, leaving the green branch pushed.
        backlog: Sub-threshold findings the pipeline files as ``priority:<severity>``
            backlog nits before opening the PR.
        regressed: True when the fix pass turned the done-check red. The red fix is not
            pushed (the branch stays at its green pre-fix state) and the outcome carries a
            single synthetic blocking finding, so a regression escalates like any block.
    """

    blocking: list[ReviewFinding]
    backlog: list[ReviewFinding]
    regressed: bool = False


def _partition_findings(
    findings: list[ReviewFinding],
) -> tuple[list[ReviewFinding], list[ReviewFinding]]:
    """Split ``findings`` into ``(blocking, backlog)`` at :data:`_BLOCKING_THRESHOLD`."""
    blocking = [f for f in findings if f.severity >= _BLOCKING_THRESHOLD]
    backlog = [f for f in findings if f.severity < _BLOCKING_THRESHOLD]
    return blocking, backlog


def _regression_finding() -> ReviewFinding:
    """The synthetic blocking finding for a fix pass that turned the done-check red."""
    return ReviewFinding(
        title="Review fix pass regressed the done-check",
        body=(
            "The in-session review fix pass turned the repo's done-check red. The fix was "
            "not pushed, so the branch stays at its green pre-fix state, but the issue "
            "needs a human before it can proceed to a PR."
        ),
        severity=Severity.HIGH,
    )


def _gate_error_finding(exc: Exception) -> ReviewFinding:
    """The synthetic blocking finding for a gate that raised before it could clear.

    The green branch is pushed *before* the gate runs, so a gate that escaped would leave
    the branch pushed with no PR and no ``hitl`` — a state the next drain classifies as
    stranded and "recovers" into a gate-bypassed PR. Turning the error into a blocking
    finding keeps the gate fail-closed: the pipeline escalates the issue to a human instead
    of ever shipping unreviewed work.
    """
    return ReviewFinding(
        title="Review gate errored before it could clear the issue",
        body=(
            "The in-session review gate raised before it could finish reviewing the built "
            f"issue ({exc!r}). The build is green and pushed, but it was never fully "
            "reviewed, so the PR is held and the issue escalated to a human."
        ),
        severity=Severity.HIGH,
    )


def _render_fix_plan(findings: list[ReviewFinding]) -> str:
    """Render the review findings into the plan the fix-pass implementer reads.

    Each finding becomes a numbered item carrying its title, severity, and body, led by
    :data:`_FIX_PLAN_HEADER` so the implementer frames the file as the fixes to make.
    """
    lines = [_FIX_PLAN_HEADER, ""]
    for index, finding in enumerate(findings, start=1):
        lines.append(f"{index}. **{finding.title}** ({finding.severity.name.lower()})")
        lines.append("")
        lines.append(finding.body.rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


async def _issue_diff(
    issue: AdhocIssue, config: RepoConfig, container: Container
) -> str:
    """Return the issue branch's contribution over the target base, from the build.

    A non-zero diff exit (e.g. an unresolvable base ref) raises rather than returning the
    empty stdout: a failed diff must not be silently treated as an empty review surface —
    that would leave the very branch the gate is meant to review unreviewed with no error
    surfaced. The gate runs in the same container that built the branch, so the local tip
    is the surface to review against the remote-tracking ``origin/<base>`` ref.
    """
    command = _issue_diff_command(issue.branch, config.require_target_branch())
    result = await container.run_command(command)
    if not result.ok:
        raise GitOpsError(
            f"review gate diff for {issue.branch} exited "
            f"{result.exit_code}: {result.stderr}"
        )
    return result.stdout


async def _run_review_gate(
    issue: AdhocIssue,
    config: RepoConfig,
    claude_md: str,
    *,
    container: Container,
    review_generate: ReviewGenerator,
    implementer: Implementer,
) -> ReviewGateOutcome:
    """Run the in-session pre-PR review gate, fail-closed on any error.

    Delegates to :func:`_review_gate_pass` for the actual critique-and-fix run and, on any
    exception it raises, returns a blocking outcome (:func:`_gate_error_finding`) instead of
    propagating. This is the gate's safety contract: the green branch is pushed *before* the
    gate runs, so a gate that escaped would strand the branch pushed-but-unreviewed with no
    ``hitl`` — a state the next drain "recovers" into a gate-bypassed PR. Swallowing the
    error into a block routes the issue through the pipeline's human escalation instead.
    """
    try:
        return await _review_gate_pass(
            issue,
            config,
            claude_md,
            container=container,
            review_generate=review_generate,
            implementer=implementer,
        )
    except Exception as exc:
        logger.warning(
            "Review gate for %s raised (%r); blocking the PR and escalating",
            issue.branch,
            exc,
        )
        return ReviewGateOutcome(blocking=[_gate_error_finding(exc)], backlog=[])


async def _review_gate_pass(
    issue: AdhocIssue,
    config: RepoConfig,
    claude_md: str,
    *,
    container: Container,
    review_generate: ReviewGenerator,
    implementer: Implementer,
) -> ReviewGateOutcome:
    """Run the in-session pre-PR review over a freshly-built, green-pushed issue.

    The gate runs in the same container that built the branch (already pushed at its green
    pre-fix state), one critique-and-fix pass:

    1. diff the issue branch over the target base and run the reviewer (review₁);
    2. a clean review is a clean outcome — nothing blocking, nothing backlog;
    3. a review with only sub-threshold findings needs no fix pass: the build is already
       green and shippable, so its nits are filed as backlog and the PR opens from the
       green pre-fix branch. Running a fix pass over cosmetic nits would risk regressing
       the done-check and false-escalating a shippable build to a human;
    4. otherwise (there is a blocking finding) materialize review₁'s findings into
       :data:`PLAN_FILE` and let the same implementer fix them in-place, then re-run the
       done-check;
    5. a red re-run is a regression: return a blocking outcome and do **not** push the red
       fix (the green pre-fix branch stays pushed);
    6. a green re-run re-pushes the branch, then the reviewer runs again (review₂) over the
       fixed diff and the *surviving* findings are partitioned by severity into blocking
       (>= :data:`_BLOCKING_THRESHOLD`) and backlog (below it).

    Raises whatever the reviewer, the fix-pass implementer, or the diff raises; the
    :func:`_run_review_gate` wrapper turns any such error into a blocking outcome.
    """
    diff1 = await _issue_diff(issue, config, container)
    plan1 = await review_generate(
        ReviewInput(
            repo_full_name=issue.repo_full_name,
            issue_number=issue.issue_number,
            diff=diff1,
        )
    )
    if not plan1.findings:
        logger.info("Review gate for %s found nothing to fix", issue.branch)
        return ReviewGateOutcome(blocking=[], backlog=[])

    blocking1, backlog1 = _partition_findings(plan1.findings)
    if not blocking1:
        # Nit-only: the build is already green and pushed, so there is nothing to gate on.
        # File the nits as backlog and open the PR from the green pre-fix branch rather than
        # run a fix pass whose regression would false-escalate a shippable build to a human.
        logger.info(
            "Review gate for %s found %d sub-threshold nit(s) and nothing blocking; "
            "filing them as backlog, no fix pass",
            issue.branch,
            len(backlog1),
        )
        return ReviewGateOutcome(blocking=[], backlog=backlog1)

    logger.info(
        "Review gate for %s found %d finding(s) incl. %d blocking; running one fix pass",
        issue.branch,
        len(plan1.findings),
        len(blocking1),
    )
    await _materialize_plan(container, _render_fix_plan(plan1.findings))
    await implementer.implement(
        _slice_for_issue(issue), container=container, plan_path=PLAN_FILE
    )
    commands = parse_done_check(claude_md)
    passed, _ = await run_done_check_commands(container, commands, secret_values=[])
    if not passed:
        logger.warning(
            "Review gate fix pass for %s regressed the done-check; not pushing the fix",
            issue.branch,
        )
        return ReviewGateOutcome(
            blocking=[_regression_finding()], backlog=[], regressed=True
        )

    await push_branch(container, issue.branch)
    diff2 = await _issue_diff(issue, config, container)
    plan2 = await review_generate(
        ReviewInput(
            repo_full_name=issue.repo_full_name,
            issue_number=issue.issue_number,
            diff=diff2,
        )
    )
    blocking, backlog = _partition_findings(plan2.findings)
    logger.info(
        "Review gate for %s: %d blocking, %d backlog after the fix pass",
        issue.branch,
        len(blocking),
        len(backlog),
    )
    return ReviewGateOutcome(blocking=blocking, backlog=backlog)


@dataclass(frozen=True)
class AdhocBuildResult:
    """Result of building one ad-hoc issue.

    Attributes:
        branch: The ``issue-<N>`` branch the build targeted (pushed only when ``passed``).
        passed: True when the done-check was green (and the branch was pushed); False on
            a red check, where nothing was pushed.
        gate: The in-session review gate's outcome, captured on a green build when a
            ``review_generate`` was injected; ``None`` on a red build (the gate never ran)
            or when no reviewer was wired. The pipeline consumes it to escalate blocking
            findings or file backlog nits before opening the PR.
    """

    branch: str
    passed: bool
    gate: ReviewGateOutcome | None = None


async def build_adhoc_issue(
    issue: AdhocIssue,
    config: RepoConfig,
    claude_md: str,
    *,
    planner: Planner,
    implementer: Implementer,
    auth: InstallationAuth,
    runtime: ContainerRuntime,
    resolve_secret: SecretResolver,
    report: ReportSink,
    review_generate: ReviewGenerator | None = None,
    image: str = DEFAULT_IMAGE,
) -> AdhocBuildResult:
    """Build one ad-hoc issue in one container: plan -> implement -> push -> review gate.

    Runs the shared per-issue lifecycle
    (:func:`retinue.container_build.build_issue_in_container`) in a single disposable
    container, destroyed on every path — parse the done-check, resolve the secrets, start
    the container, clone and cut ``issue-<N>`` off the resolved target branch, implement,
    guard against a hollow implement (zero commits fails the build), done-check, push on
    green, report — with the ad-hoc lane's hooks threaded in:

    1. **pre-implement**: run the read-only planner to produce a plan, captured from its
       output, and materialize it into :data:`PLAN_FILE` for the implementer to read,
    2. **plan_path**: the implementer is pointed at :data:`PLAN_FILE` so it reads the plan
       first,
    3. **credentials**: the planner's and the implementer's credential env ride the
       container at ``start`` — the two roles that exec in-container; the reviewer runs
       over HTTP, so its credential is not injected,
    4. **on green**: run the in-session review gate (:func:`_run_review_gate`, when a
       ``review_generate`` is injected) over the ``issue-<N>`` diff — reviewer, one
       critique-and-fix pass by the same implementer, re-push on a green re-run, and a
       severity partition of the surviving findings. The gate's
       :class:`ReviewGateOutcome` is captured onto the result for the pipeline to act on.
       A red build never runs the gate (there is no built work to review).

    Args:
        issue: The ad-hoc issue to build (repo, issue number).
        config: The accepted repo config; its target branch is the issue-branch base
            and its ``secrets`` are injected into the container.
        claude_md: The repo's ``CLAUDE.md`` text, carrying the done-check command (also
            re-run by the gate's fix pass).
        planner: Execs the read-only planner in the container (the planner seam).
        implementer: Execs the implementer subagent in the container (the Agent SDK seam);
            the gate reuses it for the fix pass.
        auth: Mints the installation token used to clone (the auth seam).
        runtime: Spawns the disposable build container (the Docker seam).
        resolve_secret: Resolves the config's declared secret names/refs to values.
        report: Sink the done-check outcome is posted to (commit status / comment).
        review_generate: The headless reviewer the gate runs (the reviewer seam); absent
            means no review gate (and ``gate=None`` on the result).
        image: Container image the build runs in.

    Returns:
        An :class:`AdhocBuildResult`: ``passed=True`` when the green branch was pushed
        (with ``gate`` set when a reviewer ran), ``passed=False`` (and ``gate=None``) when
        a red done-check pushed nothing.

    Raises:
        Propagates whatever the build container raises (e.g. a missing secret, a clone
        failure, a :class:`PlanError` from the planner exec, or an
        :class:`~retinue.container_build.ImplementError` from a hollow implement).
    """

    async def plan_and_materialize(container: Container) -> None:
        plan = await planner.plan(issue, container=container)
        await _materialize_plan(container, plan)

    # The gate's outcome is produced inside the on_green hook (which returns nothing), so a
    # one-slot holder carries it back out to the result.
    gate_holder: list[ReviewGateOutcome] = []
    on_green: Callable[[Container], Awaitable[None]] | None = None
    if review_generate is not None:
        present_generate = review_generate

        async def gate_on_green(container: Container) -> None:
            outcome = await _run_review_gate(
                issue,
                config,
                claude_md,
                container=container,
                review_generate=present_generate,
                implementer=implementer,
            )
            gate_holder.append(outcome)

        on_green = gate_on_green

    passed = await build_issue_in_container(
        _slice_for_issue(issue),
        config,
        claude_md,
        base=config.require_target_branch(),
        implementer=implementer,
        auth=auth,
        runtime=runtime,
        resolve_secret=resolve_secret,
        report=report,
        image=image,
        lane_label="Ad-hoc issue",
        extra_auth_envs=[planner.auth_env()],
        pre_implement=plan_and_materialize,
        plan_path=PLAN_FILE,
        on_green=on_green,
    )
    gate = gate_holder[0] if gate_holder else None
    return AdhocBuildResult(branch=issue.branch, passed=passed, gate=gate)


async def _materialize_plan(container: Container, plan: str) -> None:
    """Write the captured plan into :data:`PLAN_FILE` inside ``container``, byte-exact."""
    await container.run_command(_ENSURE_PLAN_DIR_COMMAND)
    await container.run_command(_materialize_plan_command(plan))


def _slice_for_issue(issue: AdhocIssue) -> Slice:
    """Adapt an :class:`AdhocIssue` to the :class:`~retinue.container_build.Slice` seam.

    The implementer seam is shared with the PRD lane, whose contract is a ``Slice``. An
    ad-hoc issue has no parent PRD, so it stands on its own integration target: the
    per-issue PRD number is the issue number itself (the same convention the cron lane uses
    for a standalone backlog nit), which only feeds the ``issue-<N>`` branch the implementer
    already commits to here — the ad-hoc lane never merges onto that target.
    """
    return Slice(
        repo_full_name=issue.repo_full_name,
        issue_number=issue.issue_number,
        prd_number=issue.issue_number,
    )
