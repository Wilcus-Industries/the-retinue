"""The PRD pipeline: tie the real adapters together behind one orchestration object.

Once :func:`retinue.worker.gate_prd` ACCEPTS a PRD, the worker drives the real
pipeline through a :class:`Pipeline`. The pipeline is the single seam the worker
injects (mirroring how ``fetch_config`` / ``dedupe`` are injected onto the Arq
context): production builds one from :class:`retinue.config.Settings` via
:func:`build_pipeline_factory`, wiring each step to its real adapter; tests inject a fake
``Pipeline`` (or a real one over recording-fake collaborators).

The PRD path, in order, mirrors the build pipeline the rest of the modules document:

1. **slice** (:func:`retinue.slicer.slice_prd`) — turn the PRD body into labeled,
   dependency-ordered slices, or escalate a thin/malformed PRD,
2. **build** (the injected ``build_prd`` seam over :func:`retinue.orchestrator.build_prd`)
   — fan the slices out to implementers and merge the green ones onto ``retinue/prd-<n>``,
3. **open the staging PR** (:func:`retinue.pr_opener.open_staging_pr`) — behind the
   heimdall precheck, record the PR<->PRD mapping for a later resume.

The webhook-driven events route to their own entry points: a ``pull_request_review``
to :meth:`Pipeline.process_review` (:func:`retinue.loopback.process_review`) and a
merged ``pull_request`` to :meth:`Pipeline.reap_pr` (:func:`retinue.handoff.reap_merged_pr`).
A worker restart resumes through :meth:`Pipeline.reconcile` (:func:`reconcile_run`).

The orchestrator ``build_prd`` call is an injected seam (so a fake drops in for tests),
but production now binds it to the real, budget-gated, triaged build over the
:class:`~retinue.orchestrator.ContainerImplementer` per repo inside
:func:`build_pipeline_factory` — every side-effecting collaborator, the build lane
included, is a real adapter wired there.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import retinue.loopback as _loopback_gh
import retinue.pr_opener as _pr_gh
import retinue.reviewer as _reviewer_gh
import retinue.slicer as _slicer_gh
from retinue.budget import (
    AuthMode,
    BudgetGovernor,
    BudgetLedger,
    SystemClock,
)
from retinue.container import Container, ContainerRuntime, DockerRuntime
from retinue.done_check import (
    DEFAULT_IMAGE,
    DoneCheckError,
    EnvSecretResolver,
    GhReportSink,
    ReportSink,
    SecretResolver,
    parse_done_check,
)
from retinue.handoff import (
    Handoff,
    HandoffGh,
    MergedPullRequest,
    ReapResult,
    announce_handoff,
    reap_merged_pr,
)
from retinue.loopback import (
    GhCliRebuilder,
    HeimdallReview,
    HeimdallRoundStore,
    Rebuilder,
    VerdictResult,
    process_review,
)
from retinue.notify import (
    GhCommentSink,
    GhLabelSink,
    Notification,
    Notifier,
    NtfyPushSink,
    PushoverPushSink,
    PushRequest,
    PushSink,
)
from retinue.orchestrator import (
    ContainerGitOps,
    ContainerImplementer,
    PrdBuildResult,
    PrdSlice,
    RoundReviewer,
)
from retinue.pr_opener import GhCliPrOps, PrOpenResult, PrOps, open_staging_pr
from retinue.reconcile import (
    GhCliReconcile,
    ReconcileGh,
    ReconcileResult,
    RunStateStore,
    reconcile_run,
)
from retinue.repo_config import RepoConfig
from retinue.reviewer import (
    AgentSdkReviewGenerator,
    GhCliBlockedByEditor,
    ReviewGenerator,
)
from retinue.roles import Role, resolve_model
from retinue.slicer import (
    HITL_LABEL,
    ClaudeSliceGenerator,
    GhCliIssueCreator,
    IssueCreator,
    SliceGenerator,
    SliceOutcome,
    slice_prd,
)
from retinue.wiring import (
    BoundBuildResult,
    ReviewerFactory,
    RoundDiffSource,
    bind_build_prd,
    bind_round_reviewer,
)

if TYPE_CHECKING:
    from retinue.config import Settings
    from retinue.github_app import InstallationAuth

logger = logging.getLogger(__name__)

# The orchestrator build seam. Bound to the real budget-gated build over
# :func:`retinue.orchestrator.build_prd` in production (per repo in the factory); injected
# as a fake in tests. Returns the per-slice build outcome the PR-opener gates on.
BuildPrd = Callable[..., Awaitable[PrdBuildResult]]

# The handoff seam invoked when heimdall converges (the loopback's Handoff shape). Bound
# to :func:`retinue.handoff.announce_handoff` in production.
HandoffSeam = Callable[..., Awaitable[None]]


@dataclass(frozen=True)
class PrdJobResult:
    """Outcome of driving one accepted PRD through the pipeline.

    Attributes:
        sliced: True when the PRD was sliced into issues (False when it escalated thin).
        pr_opened: True when the staging PR opened after the build.
        prd_build: The full-PRD build result, or ``None`` when the PRD never built.
        pr_open: The PR-open result, or ``None`` when the PR step was not reached.
        done_check_missing: True when the repo had no parseable done-check gate, so the
            build was escalated and skipped (a clean terminal skip, not a crash).
    """

    sliced: bool
    pr_opened: bool
    prd_build: PrdBuildResult | None = None
    pr_open: PrOpenResult | None = None
    done_check_missing: bool = False


@dataclass
class Pipeline:
    """Ties the real adapters into the PRD pipeline and the webhook event handlers.

    The collaborators are the already-built real adapters (or recording fakes in a
    test); the SQLite-backed stores are constructed lazily from their paths so one
    pipeline owns one durable file per concern. The orchestrator build and the handoff
    are injected seams (``build_prd`` / ``handoff``) so a fake drops in for tests; the
    factory binds ``build_prd`` to the real budget-gated build in production.

    Attributes:
        config: The accepted repo config gating every step (staging branch, retry cap).
        claude_md: The repo's ``CLAUDE.md`` text carrying the done-check command.
        governor: The shared service-level budget governor.
        notifier: The shared escalation fan-out (push + comment + label).
        create_issue: The gh issue creator (slicer's seam) reused across slice/loopback.
        slice_generate: The headless Agent-SDK slicer producing a SlicePlan.
        pr_ops: The PR-opener gh seam (heimdall precheck, staging check, sync, open).
        reap_gh: The reap gh seam (issue close + PRD child enumeration).
        round_store_path: SQLite file backing the per-PR heimdall round counter.
        retry_store_path: SQLite file backing the per-slice implementer-retry counter.
        run_state_path: SQLite file backing the per-PRD run-state (slices + PR mapping).
        build_prd: The orchestrator build seam (injected; bound to the real build_prd).
        handoff: The convergence handoff seam (bound to announce_handoff).
        rebuild: The heimdall rebuild seam (re-file fix-issues + re-trigger review).
        reconcile_gh: The reconcile gh seam GitHub truth is read through on resume.
    """

    config: RepoConfig
    claude_md: str
    governor: BudgetGovernor
    notifier: Notifier
    create_issue: IssueCreator
    slice_generate: SliceGenerator
    pr_ops: PrOps
    reap_gh: Handoff
    round_store_path: Path
    retry_store_path: Path
    run_state_path: Path
    build_prd: BuildPrd | None = None
    handoff: HandoffSeam | None = None
    rebuild: Rebuilder | None = None
    reconcile_gh: ReconcileGh | None = None

    _round_store: HeimdallRoundStore = field(init=False, repr=False)
    _run_state: RunStateStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._round_store = HeimdallRoundStore(self.round_store_path)
        self._run_state = RunStateStore(self.run_state_path)

    async def process_prd_job(
        self, *, repo_full_name: str, prd_number: int, prd_body: str
    ) -> PrdJobResult:
        """Drive an accepted PRD: slice -> build -> open the staging PR.

        Slices the PRD (escalating a thin one and stopping); on a real slice it records
        the owned slice set, runs the full-PRD build, then opens the staging PR behind
        the heimdall precheck and records the PR<->PRD mapping for a later resume.

        Two guards keep a misconfigured or no-op build from crashing the job. A repo with
        no parseable done-check gate is escalated and skipped (a clean terminal result, so
        the Arq job succeeds instead of crash-looping on the build's ``DoneCheckError``).
        A build that merged no slices (budget-deferred or all-blocked) never pushed an
        integration branch, so the staging-PR step is skipped — opening a PR for a head
        that doesn't exist would 404.

        Args:
            repo_full_name: The target repo, e.g. "owner/repo".
            prd_number: The PRD's tracking issue number.
            prd_body: The PRD issue body to slice.

        Returns:
            A :class:`PrdJobResult` recording whether the PRD sliced, built, and opened.
        """
        slice_result = await slice_prd(
            repo_full_name=repo_full_name,
            prd_number=prd_number,
            prd_body=prd_body,
            generate=self.slice_generate,
            create_issue=self.create_issue,
            notifier=self.notifier,
        )
        if slice_result.outcome is not SliceOutcome.SLICED:
            return PrdJobResult(sliced=False, pr_opened=False)

        slices = _slices_from_numbers(
            repo_full_name, prd_number, slice_result.created_numbers
        )
        await self._run_state.record_slices(
            repo_full_name=repo_full_name,
            prd_number=prd_number,
            issue_numbers=slice_result.created_numbers,
        )
        if not await self._has_done_check_gate(repo_full_name, prd_number):
            return PrdJobResult(sliced=True, pr_opened=False, done_check_missing=True)

        build = await self._build(repo_full_name, prd_number, slices)
        if not build.merged_issues:
            # Budget-deferred or all-blocked: no integration branch was pushed, so there
            # is no head to open a PR from. Surface the build (its deferral is visible in
            # the empty merged_issues) without attempting the doomed PR open.
            return PrdJobResult(sliced=True, pr_opened=False, prd_build=build)

        pr_open = await self._open_pr(repo_full_name, prd_number)
        return PrdJobResult(
            sliced=True,
            pr_opened=pr_open.opened,
            prd_build=build,
            pr_open=pr_open,
        )

    async def _has_done_check_gate(
        self, repo_full_name: str, prd_number: int
    ) -> bool:
        """True when the repo's CLAUDE.md carries a parseable done-check gate.

        An opted-in repo whose CLAUDE.md has no "Definition of done" block would make the
        build's :func:`retinue.done_check.parse_done_check` raise, failing the Arq task
        into an infinite retry. Detecting it here lets the pipeline escalate the
        misconfiguration through the notifier (push + comment + label, matching the slicer
        and PR-opener escalations) and skip the build cleanly — never running a phantom
        gate, never crash-looping. Returns False after escalating; True when the gate
        parses.
        """
        try:
            parse_done_check(self.claude_md)
        except DoneCheckError as exc:
            await self.notifier.notify(
                Notification(
                    repo_full_name=repo_full_name,
                    issue_number=prd_number,
                    title=f"Retinue can't build PRD #{prd_number}: no done-check gate",
                    body=(
                        "skipped: no done-check gate. The repo's CLAUDE.md has no "
                        f"parseable 'Definition of done' block ({exc}). Add one so the "
                        "build has a gate to run, then re-trigger the PRD."
                    ),
                    label=HITL_LABEL,
                )
            )
            logger.warning(
                "Skipping build for %s PRD #%d: no done-check gate (%s)",
                repo_full_name,
                prd_number,
                exc,
            )
            return False
        return True

    async def process_review(self, review: HeimdallReview) -> VerdictResult:
        """Run the heimdall loopback for one review: rebuild / converge / escalate.

        Drives :func:`retinue.loopback.process_review` with the pipeline's persisted
        round store, issue creator, rebuild seam, handoff, and notifier — the converge
        path hands off through :func:`retinue.handoff.announce_handoff`.

        Args:
            review: The parsed heimdall bot review.

        Returns:
            The :class:`retinue.loopback.VerdictResult` for the review.
        """
        return await process_review(
            review,
            self.config,
            round_store=self._round_store,
            create_issue=self.create_issue,
            rebuild=self._require_rebuild(),
            handoff=self._handoff_seam(),
            notifier=self.notifier,
        )

    async def reap_pr(self, merged: MergedPullRequest) -> ReapResult:
        """React to a human-merged PR: close its slice issues, then reap the PRD."""
        return await reap_merged_pr(merged, gh=self.reap_gh)

    async def round_for_pr(
        self, *, repo_full_name: str, pr_number: int
    ) -> tuple[int, list[int]] | None:
        """Resolve a PR to its ``(prd_number, slice_numbers)`` from the run-state store.

        The webhook routes review/merge events by PR number, but the loopback and reap
        need the parent PRD and its owned slice set; the PR<->PRD mapping recorded when
        the staging PR opened (:meth:`_open_pr`) is the source. Returns ``None`` for a PR
        the retinue never opened (so the worker can skip a foreign PR's event).
        """
        return await self._run_state.round_for_pr(
            repo_full_name=repo_full_name, pr_number=pr_number
        )

    async def reconcile(
        self, *, repo_full_name: str, prd_number: int, slices: list[PrdSlice]
    ) -> ReconcileResult:
        """Reconcile an in-flight PRD round against GitHub truth on worker restart."""
        return await reconcile_run(
            repo_full_name=repo_full_name,
            prd_number=prd_number,
            slices=slices,
            gh=self._require_reconcile_gh(),
        )

    async def _build(
        self, repo_full_name: str, prd_number: int, slices: list[PrdSlice]
    ) -> PrdBuildResult:
        """Run the full-PRD build through the injected orchestrator seam."""
        return await self._require_build_prd()(
            repo_full_name=repo_full_name,
            prd_number=prd_number,
            slices=slices,
            config=self.config,
            claude_md=self.claude_md,
        )

    async def _open_pr(self, repo_full_name: str, prd_number: int) -> PrOpenResult:
        """Open the staging PR for a built PRD and record the PR<->PRD mapping."""
        result = await open_staging_pr(
            repo_full_name=repo_full_name,
            prd_number=prd_number,
            prd_issue_number=prd_number,
            config=self.config,
            ops=self.pr_ops,
            notifier=self.notifier,
        )
        if result.opened and result.pull_request is not None:
            await self._run_state.record_pr(
                repo_full_name=repo_full_name,
                prd_number=prd_number,
                pr_number=result.pull_request.number,
            )
        return result

    def _handoff_seam(self) -> HandoffSeam:
        """The handoff invoked on convergence: the injected one, else announce_handoff."""
        if self.handoff is not None:
            return self.handoff

        async def _announce(*, repo_full_name: str, pr_number: int) -> None:
            await announce_handoff(
                repo_full_name=repo_full_name,
                pr_number=pr_number,
                notifier=self.notifier,
            )

        return _announce

    def _require_build_prd(self) -> BuildPrd:
        if self.build_prd is None:
            raise PipelineNotWiredError("build_prd")
        return self.build_prd

    def _require_rebuild(self) -> Rebuilder:
        if self.rebuild is None:
            raise PipelineNotWiredError("rebuild")
        return self.rebuild

    def _require_reconcile_gh(self) -> ReconcileGh:
        if self.reconcile_gh is None:
            raise PipelineNotWiredError("reconcile_gh")
        return self.reconcile_gh


class PipelineNotWiredError(RuntimeError):
    """A pipeline step was reached without its collaborator wired in.

    Raised rather than silently no-oping so a pipeline reached through a step whose
    optional seam was never injected (e.g. a fake pipeline with no ``rebuild`` or
    ``reconcile_gh``) fails loudly at first use instead of misbehaving silently.
    """

    def __init__(self, seam: str) -> None:
        super().__init__(f"pipeline collaborator not wired: {seam}")
        self.seam = seam


def _slices_from_numbers(
    repo_full_name: str, prd_number: int, issue_numbers: list[int]
) -> list[PrdSlice]:
    """Build :class:`PrdSlice` objects for freshly-sliced issue numbers (no edges yet).

    The slicer resolves intra-PRD ``blocked_by`` into the rendered issue bodies and gh's
    native links; the build's dependency order is re-derived there, so the in-process
    slice objects carry no edges — every slice is independently ready for the first round.
    """
    return [
        PrdSlice(
            repo_full_name=repo_full_name,
            issue_number=number,
            prd_number=prd_number,
        )
        for number in issue_numbers
    ]


# --- production wiring: build the real pipeline from Settings ----------------------


class SubprocessGhRunner[R]:
    """Real ``GhRunner`` for the ``run(args, *, env) -> GhResult`` seam (slicer/pr/loopback).

    Spawns ``gh`` as a child (argv list, no shell, so nothing is interpolated into a
    command line) with ``env`` merged over the ambient environment, then builds the
    *target module's own* ``GhResult`` from the captured ``(exit_code, stdout, stderr)``
    via the injected ``result`` factory. The slicer, PR-opener, and loopback each define a
    structurally-identical ``GhResult``; constructing the real one per call keeps each
    module's runner Protocol satisfied with a single subprocess implementation.

    Args:
        result: The module's ``GhResult`` constructor (``(exit_code, stdout, stderr)``).
    """

    def __init__(
        self, result: Callable[..., R]
    ) -> None:
        self._result = result

    async def run(self, args: list[str], *, env: dict[str, str]) -> R:
        """Run ``gh <args>`` with ``env`` in the environment and capture the result."""
        merged_env = {**os.environ, **env}
        process = await asyncio.create_subprocess_exec(
            "gh",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )
        stdout, stderr = await process.communicate()
        return self._result(
            exit_code=process.returncode or 0,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )


class _ReconcileGhRunner:
    """Real reconcile gh runner (``__call__(argv) -> str``) authenticated by token.

    Satisfies :class:`retinue.reconcile.GhRunner`: runs one ``gh`` argv with the
    installation token in ``GH_TOKEN`` and returns stdout, raising on a non-zero exit so
    a failed reconcile query surfaces rather than reading as empty truth.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    async def __call__(self, argv: list[str]) -> str:
        from retinue.reconcile import gh_env

        process = await asyncio.create_subprocess_exec(
            "gh",
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=gh_env(self._token, dict(os.environ)),
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"gh {' '.join(argv)} exited {process.returncode}: "
                f"{stderr.decode(errors='replace').strip()}"
            )
        return stdout.decode()


def _build_push_sink(settings: Settings) -> PushSink:
    """Pick the push sink from settings: ntfy (topic) or Pushover (token+user).

    Exactly one backend is expected; with neither configured the push is a logged no-op
    so the comment + label (the durable record) still land. ntfy wins when both are set.
    """
    if settings.ntfy_topic:
        return NtfyPushSink(
            topic=settings.ntfy_topic, token=settings.ntfy_token or None
        )
    if settings.pushover_token and settings.pushover_user:
        return PushoverPushSink(token=settings.pushover_token, user=settings.pushover_user)

    async def _noop(request: PushRequest) -> None:
        logger.warning("No push channel configured; skipping push %r", request.title)

    return _noop


# The orchestrator build's estimated charge, gated against the rolling-24h budget cap.
# The build's true cost is only known after the implementer/done-check runs, so the gate
# uses a conservative fixed estimate; the meter (the governor's mid-run pause/resume)
# tracks the real spend once the run is underway. Kept here (not a Settings field) so the
# public config schema is unchanged.
_BUILD_ESTIMATED_AMOUNT = 1.0


class _MergeContainerGitOps:
    """A :class:`GitOps` that lazily starts its own merge container, then delegates.

    The orchestrator's merge phase runs ``git`` inside a container holding a clone of the
    repo, but ``build_prd`` starts no such container — the done-check's per-slice
    containers are created and destroyed inside the build round, before any merge. This
    adapter closes that gap: on the first branch/merge call it starts a fresh container
    via the injected :class:`ContainerRuntime`, clones the repo over the installation
    token, and wraps it in :class:`ContainerGitOps`; every later call within the same
    build reuses that one container (so the integration branch persists across the
    round's merges). :meth:`aclose` destroys it, and the build seam wrapper calls it in a
    ``finally`` so the container is never leaked.

    Args:
        repo_full_name: The repo to clone for the merges, e.g. "owner/repo".
        auth: Mints the installation token whose URL the clone authenticates with.
        runtime: Spawns the disposable merge container (the Docker seam).
        image: The container image the merges run in; defaults to the done-check image.
    """

    def __init__(
        self,
        *,
        repo_full_name: str,
        auth: InstallationAuth,
        runtime: ContainerRuntime,
        image: str = DEFAULT_IMAGE,
    ) -> None:
        self._repo_full_name = repo_full_name
        self._auth = auth
        self._runtime = runtime
        self._image = image
        self._container: Container | None = None
        self._delegate: ContainerGitOps | None = None

    async def ensure_integration_branch(self, *, branch: str, base: str) -> None:
        """Ensure ``branch`` exists in the (lazily started) merge container."""
        delegate = await self._ensure_delegate()
        await delegate.ensure_integration_branch(branch=branch, base=base)

    async def merge(self, *, source: str, into: str) -> None:
        """Merge ``source`` into ``into`` in the (lazily started) merge container."""
        delegate = await self._ensure_delegate()
        await delegate.merge(source=source, into=into)

    async def round_diff(self, *, merged_branches: list[str], base: str) -> str:
        """Diff a round's ``merged_branches`` over ``base`` in the merge container.

        The merge container is already started (the round merged into ``base`` through it),
        so the internal reviewer reads the round's merged surface from the same clone the
        merges advanced — no second container or clone.
        """
        delegate = await self._ensure_delegate()
        return await delegate.round_diff(merged_branches=merged_branches, base=base)

    async def _ensure_delegate(self) -> ContainerGitOps:
        """Start + clone the merge container on first use; reuse it thereafter."""
        if self._delegate is not None:
            return self._delegate
        token = await self._auth.installation_token(self._repo_full_name)
        container = await self._runtime.start(image=self._image, env={})
        result = await container.run_command(["git", "clone", token.clone_url, "."])
        if not result.ok:
            await container.destroy()
            raise GitOpsCloneError(
                f"clone of {self._repo_full_name} for merge failed "
                f"(exit {result.exit_code}): {result.stderr}"
            )
        self._container = container
        self._delegate = ContainerGitOps(container)
        return self._delegate

    async def aclose(self) -> None:
        """Destroy the merge container if one was started. Idempotent."""
        container, self._container, self._delegate = self._container, None, None
        if container is not None:
            await container.destroy()


class GitOpsCloneError(RuntimeError):
    """The merge container could not clone the repo, so no merge can run.

    Raised rather than returning a sentinel so a doomed merge round fails loudly instead
    of silently reporting an empty integration branch.
    """


def build_pipeline_factory(
    settings: Settings,
    auth: InstallationAuth,
    *,
    build_prd: BuildPrd | None = None,
    fetch_claude_md: ClaudeMdFetcher | None = None,
) -> Callable[[str, RepoConfig], Awaitable[Pipeline]]:
    """Build the production pipeline factory over the real adapters.

    Returns an async ``(repo_full_name, config) -> Pipeline`` that mints a per-repo
    installation token, then constructs every gh/Anthropic/push/build adapter against it:
    the slicer's gh issue creator and Agent-SDK generator, the PR-opener gh ops, the reap
    and reconcile gh seams, the heimdall rebuilder, the shared notifier (push + comment +
    label), the shared budget governor, and the orchestrator build lane.

    The orchestrator ``build_prd`` seam defaults to the real budget-gated, triaged build
    bound per repo via :func:`retinue.wiring.bind_build_prd` over the real
    :class:`~retinue.orchestrator.ContainerImplementer`, container/git/secret/report
    adapters — so a constructed :class:`Pipeline` has a live build lane. A caller may pass
    a ``build_prd`` to override it (a fake in tests); passing one skips the per-repo bind.

    Args:
        settings: The runtime settings carrying budget, Anthropic, and push config.
        auth: The GitHub App installation auth used to mint per-repo tokens.
        build_prd: An explicit build seam overriding the real per-repo bind (e.g. a fake).
        fetch_claude_md: Reads the target repo's ``CLAUDE.md`` text (the done-check
            command source); ``None`` falls back to empty text, which the done-check
            reads as no recognisable command. Production injects the contents-API fetcher.

    Returns:
        An async pipeline factory keyed by repo and config.
    """
    governor = BudgetGovernor(
        BudgetLedger(
            settings.budget_db_path,
            clock=SystemClock(),
            auth_mode=AuthMode.from_config(settings.auth_mode),
            weekly_budget=settings.weekly_budget,
            daily_cap_fraction=settings.budget_daily_cap_fraction,
        )
    )
    push = _build_push_sink(settings)
    state_dir = _state_dir(settings)
    retry_store_path = state_dir / "impl-retries.sqlite3"
    # One subprocess runner per module's own GhResult (structurally identical), so each
    # adapter's runner Protocol is satisfied by the same real ``gh`` spawn.
    slicer_runner = SubprocessGhRunner(_slicer_gh.GhResult)
    pr_runner = SubprocessGhRunner(_pr_gh.GhResult)
    loopback_runner = SubprocessGhRunner(_loopback_gh.GhResult)

    async def factory(repo_full_name: str, config: RepoConfig) -> Pipeline:
        token = (await auth.installation_token(repo_full_name)).token
        notifier = Notifier(
            push=push,
            comment=GhCommentSink(token=token),
            label=GhLabelSink(token=token),
        )
        create_issue = GhCliIssueCreator(
            slicer_runner, token=token, repo_full_name=repo_full_name
        )
        slice_generate = ClaudeSliceGenerator(
            token=settings.anthropic_credential,
            auth_mode=settings.auth_mode,
            model=resolve_model(Role.SLICER, config),
        ).generate
        pr_ops = GhCliPrOps(pr_runner, token=token)
        reap_gh = HandoffGh(token=token)
        rebuild = GhCliRebuilder(
            loopback_runner,
            create_issue=create_issue,
            token=token,
            reviewer_login=settings.heimdall_bot_login,
        )
        reconcile_gh = GhCliReconcile(
            _ReconcileGhRunner(token), merge_base=config.staging_branch
        )
        bound_build_prd = build_prd or _bind_build_prd_for_repo(
            settings,
            auth,
            repo_full_name=repo_full_name,
            token=token,
            governor=governor,
            notifier=notifier,
            create_issue=create_issue,
            retry_store_path=retry_store_path,
            config=config,
        )
        return Pipeline(
            config=config,
            claude_md=await _fetch_claude_md(fetch_claude_md, repo_full_name),
            governor=governor,
            notifier=notifier,
            create_issue=create_issue,
            slice_generate=slice_generate,
            pr_ops=pr_ops,
            reap_gh=reap_gh,
            round_store_path=state_dir / "heimdall-rounds.sqlite3",
            retry_store_path=retry_store_path,
            run_state_path=state_dir / "run-state.sqlite3",
            build_prd=bound_build_prd,
            rebuild=rebuild,
            reconcile_gh=reconcile_gh,
        )

    return factory


# Reads the target repo's ``CLAUDE.md`` text given its full name. Injected (over the
# GitHub contents API in production) so the build's done-check command is parsed from the
# real repo text rather than an empty string; absent, the factory falls back to "".
ClaudeMdFetcher = Callable[[str], Awaitable[str]]


async def _fetch_claude_md(
    fetch_claude_md: ClaudeMdFetcher | None, repo_full_name: str
) -> str:
    """Read the target repo's ``CLAUDE.md`` text, or "" when no fetcher is wired."""
    if fetch_claude_md is None:
        return ""
    return await fetch_claude_md(repo_full_name)


# The internal reviewer's single Anthropic Messages API call runs at the "max" effort
# tier on Opus 4.8; a high-effort Opus turn can take minutes, so the transport's timeout
# matches the SDK's 10-minute default rather than a short connect-style cap.
_REVIEW_HTTP_TIMEOUT_SECONDS = 600.0


@dataclass(frozen=True)
class HttpxTransport:
    """Production :class:`~retinue.reviewer.HttpTransport`: POST one request via httpx.

    The reviewer assembles the full request body and headers (model, effort tier, the
    json-schema response format, and the credential's auth header); this transport only
    POSTs them and reads the status code + JSON body back into the reviewer's
    :class:`~retinue.reviewer.HttpResponse`. The single POST is the only network edge, so
    it sits behind the reviewer's injected seam and the rest of the review flow is
    exercised in tests with a fake transport — no httpx, no network.
    """

    timeout: float = _REVIEW_HTTP_TIMEOUT_SECONDS

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, object]
    ) -> _reviewer_gh.HttpResponse:
        """POST ``json`` to ``url`` with ``headers``; return status + parsed JSON body."""
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=json)
        return _reviewer_gh.HttpResponse(
            status_code=response.status_code, body=response.json()
        )


def _bind_build_prd_for_repo(
    settings: Settings,
    auth: InstallationAuth,
    *,
    repo_full_name: str,
    token: str,
    governor: BudgetGovernor,
    notifier: Notifier,
    create_issue: IssueCreator,
    retry_store_path: Path,
    config: RepoConfig,
) -> BuildPrd:
    """Bind the real budget-gated, triaged orchestrator build for one repo.

    Constructs the build lane's real adapters — the container-exec implementer, the Docker
    runtime, the lazy merge-container git ops, the env secret resolver, and the gh report
    sink — then binds them through :func:`retinue.wiring.bind_build_prd`. The merge
    container the lazy git ops starts is destroyed after each build in a ``finally``, so a
    long-lived worker never leaks a container across PRD runs.

    Args:
        settings: Carries the Anthropic credential/auth mode and budget config.
        auth: Mints the installation token the clone authenticates with (the done-check
            clones over its URL); also passed through to the orchestrator build.
        repo_full_name: The target repo the build runs against.
        token: The minted installation token the gh report sink authenticates with.
        governor: The shared service-level budget governor (gate + meter).
        notifier: The escalation fan-out used by triage's escalate path.
        create_issue: The gh issue creator used by triage's reslice path.
        retry_store_path: SQLite file backing the persisted per-slice retry counter.
        config: The repo's validated config; its ``models`` block overrides the
            implementer's and reviewer's model per the role registry.

    Returns:
        The bound ``build_prd`` seam the pipeline drives.
    """
    runtime = DockerRuntime()
    git = _MergeContainerGitOps(
        repo_full_name=repo_full_name, auth=auth, runtime=runtime
    )
    implementer = ContainerImplementer(
        credential=settings.anthropic_credential,
        auth_mode=settings.auth_mode,
        model=resolve_model(Role.IMPLEMENTER, config),
    )
    resolve_secret: SecretResolver = EnvSecretResolver()
    report: ReportSink = GhReportSink(token=token)
    review_reviewer = _build_review_reviewer_factory(
        settings,
        repo_full_name=repo_full_name,
        token=token,
        create_issue=create_issue,
        diff_source=git,
        config=config,
    )
    bound = bind_build_prd(
        implementer=implementer,
        governor=governor,
        notifier=notifier,
        create_issue=create_issue,
        retry_store_path=retry_store_path,
        estimated_amount=_BUILD_ESTIMATED_AMOUNT,
        git=git,
        auth=auth,
        runtime=runtime,
        resolve_secret=resolve_secret,
        report=report,
        review_reviewer=review_reviewer,
    )

    async def run(**kwargs: object) -> PrdBuildResult:
        try:
            result = await bound(**kwargs)
        finally:
            await git.aclose()
        return _prd_build_from_bound(result, prd_number=kwargs.get("prd_number"))

    return run


def _build_review_reviewer_factory(
    settings: Settings,
    *,
    repo_full_name: str,
    token: str,
    create_issue: IssueCreator,
    diff_source: RoundDiffSource,
    config: RepoConfig,
    generate: ReviewGenerator | None = None,
) -> ReviewerFactory:
    """Build the per-PRD internal-reviewer factory for one repo's live build.

    The reviewer is per-PRD (its ``Part of #<prd>`` footer and round diff depend on the
    PRD), but the build lane is bound per repo, so this returns a
    ``(repo_full_name, prd_number) -> RoundReviewer`` factory the build calls at run time.
    Each reviewer wires the real Agent-SDK review generator (over the httpx transport,
    with its model resolved from the role registry against the repo's ``models``
    override), the gh issue creator reused from the slicer, and the gh Blocked-by editor —
    so a live build reviews every merged round and the review-fix follow-ups it files build
    later. ``generate`` defaults to the real Agent-SDK reviewer; tests inject a fake to
    keep the review flow off the network.
    """
    review_generate = generate or AgentSdkReviewGenerator(
        credential=settings.anthropic_credential,
        transport=HttpxTransport(),
        model=resolve_model(Role.REVIEWER, config),
    )
    edit_blocked_by = GhCliBlockedByEditor(
        runner=SubprocessGhRunner(_reviewer_gh.GhResult), token=token
    )

    def factory(factory_repo: str, prd_number: int) -> RoundReviewer:
        return bind_round_reviewer(
            diff_source=diff_source,
            generate=review_generate,
            create_issue=create_issue,
            edit_blocked_by=edit_blocked_by,
            repo_full_name=factory_repo,
            prd_number=prd_number,
        )

    return factory


def _prd_build_from_bound(
    result: BoundBuildResult, *, prd_number: object
) -> PrdBuildResult:
    """Adapt a :class:`BoundBuildResult` to the pipeline's :class:`PrdBuildResult` seam.

    ``bind_build_prd`` returns the budget-gate-aware :class:`BoundBuildResult`, but the
    pipeline's ``build_prd`` seam is a :class:`PrdBuildResult` (nothing in the pipeline
    consumes the deferral flag). A run that built yields its inner ``prd_build``; a run the
    budget gate *deferred* yields an empty build on the integration branch — honest: the
    deferred PRD merged nothing — so the staging-PR step sees no merged slices.
    """
    from retinue.orchestrator import integration_branch

    if result.prd_build is not None:
        return result.prd_build
    number = prd_number if isinstance(prd_number, int) else 0
    return PrdBuildResult(
        integration_branch=integration_branch(number),
        merged_issues=[],
        blocked_issues=[],
        escalated_issues=[],
        skipped_issues=[],
    )


def _state_dir(settings: Settings) -> Path:
    """The directory the pipeline's durable SQLite stores live in.

    Co-locates the run-state/round/retry stores next to the dedupe DB so a single mounted
    volume holds all of the worker's durable state.
    """
    return Path(settings.dedupe_db_path).resolve().parent
