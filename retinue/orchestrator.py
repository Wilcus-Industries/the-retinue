"""Orchestrator: build one ready slice, or build a full PRD in dependency order.

The single-slice primitive (issue #6) is :func:`build_slice`. For one ready slice the
orchestrator:

1. **spawn** — runs one implementer subagent (the Agent SDK seam) in an isolated git
   worktree inside the disposable container; it implements TDD-first and commits to
   an ``issue-<N>`` branch,
2. **done-check** — runs the repo's done-check via :func:`retinue.done_check.run_done_check`
   (auth -> clone -> inject -> run -> report -> teardown), which yields a pass/fail,
3. **merge** — only on a green done-check, ensures the integration branch
   ``retinue/prd-<n>`` exists (created off the config's ``staging_branch`` when absent)
   and merges ``issue-<N>`` into it. A red done-check **blocks** the merge: no red
   slice is ever merged.

The full-PRD driver (issue #7) is :func:`build_prd`. It wraps the single-slice
primitive: pick the ready set (every ``blocked_by`` ref merged/closed), fan out
implementers in parallel bounded by ``config.max_parallel``, merge the green branches
in topological order under the done-check (resolving a conflict or escalating), and
loop rounds until the ready set drains — all under a single-run lock so at most one
orchestrator run executes at a time.

Every side-effecting collaborator is injected — the implementer spawn, the container
runtime, the auth, the secret resolver, the report sink, the git operations, the
conflict resolver, and the single-run lock — so the whole flow is exercised in tests
with no Agent SDK, no Docker, no gh, no network, and no concurrency races.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Protocol

from retinue.container import Container, ContainerRuntime, RunResult
from retinue.done_check import (
    DEFAULT_IMAGE,
    ReportSink,
    SecretResolver,
    run_done_check,
)
from retinue.github_app import InstallationAuth
from retinue.repo_config import RepoConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Slice:
    """One ready slice: a single issue an implementer builds on its own branch.

    Attributes:
        repo_full_name: The target repo, e.g. "owner/repo".
        issue_number: The slice's GitHub issue number; the implementer commits to
            the ``issue-<N>`` branch derived from it.
        prd_number: The parent PRD number; the integration branch is
            ``retinue/prd-<prd_number>``.
    """

    repo_full_name: str
    issue_number: int
    prd_number: int

    @property
    def branch(self) -> str:
        """The branch the implementer commits the slice to: ``issue-<N>``."""
        return f"issue-{self.issue_number}"


def integration_branch(prd_number: int) -> str:
    """The integration branch a PRD's slices are merged onto: ``retinue/prd-<n>``."""
    return f"retinue/prd-{prd_number}"


class Implementer(Protocol):
    """Spawns one implementer subagent that builds a slice. The Agent SDK seam.

    A production implementation spawns a Claude Agent-SDK subagent in an isolated git
    worktree inside the disposable container; the subagent implements TDD-first and
    commits to the slice's ``issue-<N>`` branch. Tests inject a fake that records the
    request without any real spawn. The contract is the commit on ``slice.branch``;
    the orchestrator does not read a return value, it gates on the done-check that
    follows.
    """

    async def implement(self, slice_: Slice) -> None:
        """Build ``slice_``, committing the work to its ``issue-<N>`` branch."""
        ...


class MergeConflict(Exception):
    """A merge could not complete because of a conflict.

    The git seam raises this (or a subclass) instead of returning a sentinel, so a
    half-merged slice is never reported as merged. The full-PRD driver catches it to
    hand the conflict to the resolver; the single-slice primitive lets it propagate.
    Carries the source/target branches for the resolver and the report.
    """

    def __init__(self, source: str, into: str) -> None:
        super().__init__(f"merge conflict merging {source} into {into}")
        self.source = source
        self.into = into


class GitOps(Protocol):
    """Git operations on the integration branch. The merge seam.

    A production implementation runs ``git`` inside the disposable container against
    the cloned repo; tests inject a fake that records branch creation and merges. A
    merge that cannot complete (a conflict) raises :class:`MergeConflict` rather than
    returning a sentinel, so a half-merged slice is never reported as merged.
    """

    async def ensure_integration_branch(self, *, branch: str, base: str) -> None:
        """Ensure ``branch`` exists, creating it off ``base`` when it is absent."""
        ...

    async def merge(self, *, source: str, into: str) -> None:
        """Merge ``source`` into ``into``; raise :class:`MergeConflict` on a conflict."""
        ...


# --- real container-backed GitOps adapter ----------------------------------------
#
# The production seam runs ``git`` inside the disposable container that already holds
# the cloned repo (the same workspace ``build_slice``/``build_prd`` clone into). No git
# SDK and no network of its own: ``external_dep none`` — the only collaborator is the
# already-injected :class:`~retinue.container.Container`. The bug-prone, pure parts —
# argv assembly and classifying a failed merge as a conflict vs. a hard git error — are
# factored into the free functions below so they are tested without a live container.

# Identity used for the merge commit ``git`` records. Merges are non-interactive, so a
# committer identity must be configured or ``git commit`` refuses to run; it is set
# per-command via ``-c`` rather than mutating global config in the shared workspace.
_GIT_AUTHOR_NAME = "the-retinue"
_GIT_AUTHOR_EMAIL = "retinue@users.noreply.github.com"
_GIT_IDENTITY_FLAGS = [
    "-c",
    f"user.name={_GIT_AUTHOR_NAME}",
    "-c",
    f"user.email={_GIT_AUTHOR_EMAIL}",
]

# Substrings git prints to stdout/stderr when a merge stops on a content conflict, as
# opposed to a hard error (unknown ref, not a repo, …). Matched case-insensitively.
_CONFLICT_MARKERS = (
    "conflict",
    "automatic merge failed",
    "fix conflicts and then commit",
)


def _branch_exists_command(branch: str) -> list[str]:
    """Argv that exits 0 iff local ``branch`` already exists in the workspace."""
    return ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"]


def _create_branch_commands(branch: str, base: str) -> list[list[str]]:
    """Argv list that creates ``branch`` off ``base`` and checks it out.

    ``base`` is referenced via ``origin/<base>`` so the branch is rooted on the freshly
    cloned remote tip rather than whatever happens to be checked out, then a local
    ``branch`` is created at that point and made current.
    """
    return [
        ["git", "fetch", "origin", base],
        ["git", "checkout", "-B", branch, f"origin/{base}"],
    ]


def _merge_commands(source: str, into: str) -> list[list[str]]:
    """Argv list that checks out ``into`` and merges ``source`` into it.

    The merge is a non-fast-forward, no-edit commit (``--no-ff --no-edit``) under a
    fixed committer identity, so every integration is an explicit, attributable merge
    commit. ``source`` is taken from the remote (``origin/<source>``) to merge the tip
    the implementer pushed.
    """
    return [
        ["git", "checkout", into],
        ["git", "fetch", "origin", source],
        [
            "git",
            *_GIT_IDENTITY_FLAGS,
            "merge",
            "--no-ff",
            "--no-edit",
            f"origin/{source}",
        ],
    ]


_ABORT_MERGE_COMMAND = ["git", "merge", "--abort"]


def _is_merge_conflict(result: RunResult) -> bool:
    """Whether a failed ``git merge`` stopped on a content conflict (vs. a hard error).

    A conflict is recoverable by the resolver; a hard error (bad ref, not a repo) is
    not. Git signals a conflict with exit code 1 *and* a conflict marker in its output,
    so both are required — a different non-zero exit, or a code-1 failure without a
    marker, is treated as a hard error and surfaced as such rather than as a conflict.
    """
    if result.exit_code != 1:
        return False
    blob = f"{result.stdout}\n{result.stderr}".lower()
    return any(marker in blob for marker in _CONFLICT_MARKERS)


class ContainerGitOps:
    """Production :class:`GitOps` that runs ``git`` in the cloned-repo container.

    The integration-branch merges happen inside the same disposable container the
    done-check cloned the repo into, so this seam has no external dependency of its own
    beyond the injected :class:`~retinue.container.Container`. A merge that stops on a
    content conflict is aborted (to leave the workspace clean for the resolver) and
    surfaced as :class:`MergeConflict`; any other ``git`` failure is a hard
    :class:`GitOpsError`, never silently swallowed.
    """

    def __init__(self, container: Container) -> None:
        self._container = container

    async def ensure_integration_branch(self, *, branch: str, base: str) -> None:
        """Ensure ``branch`` exists locally, creating it off ``origin/<base>`` if absent."""
        exists = await self._container.run_command(_branch_exists_command(branch))
        if exists.ok:
            logger.info("Integration branch %s already exists", branch)
            return
        for command in _create_branch_commands(branch, base):
            await self._run_checked(command, action=f"create {branch} off {base}")
        logger.info("Created integration branch %s off %s", branch, base)

    async def merge(self, *, source: str, into: str) -> None:
        """Merge ``source`` into ``into``; raise :class:`MergeConflict` on a conflict.

        Runs checkout + fetch + merge. A merge that stops on a content conflict is
        aborted to leave the workspace clean, then raised as :class:`MergeConflict`; any
        other non-zero ``git`` exit is a :class:`GitOpsError`.
        """
        commands = _merge_commands(source, into)
        for command in commands[:-1]:
            await self._run_checked(command, action=f"prepare merge of {source}")
        result = await self._container.run_command(commands[-1])
        if result.ok:
            logger.info("Merged %s into %s", source, into)
            return
        if _is_merge_conflict(result):
            await self._container.run_command(_ABORT_MERGE_COMMAND)
            raise MergeConflict(source=source, into=into)
        raise GitOpsError(
            f"git merge of {source} into {into} failed "
            f"(exit {result.exit_code}): {result.stderr}"
        )

    async def _run_checked(self, command: list[str], *, action: str) -> RunResult:
        """Run ``command`` in the container; raise :class:`GitOpsError` on failure."""
        result = await self._container.run_command(command)
        if not result.ok:
            raise GitOpsError(
                f"git failed to {action} (exit {result.exit_code}): {result.stderr}"
            )
        return result


class GitOpsError(RuntimeError):
    """A ``git`` command failed for a reason other than a recoverable merge conflict.

    Distinct from :class:`MergeConflict`: a conflict is handed to the resolver, but a
    hard error (unknown ref, not a repository, checkout failure) means the integration
    branch could not be advanced at all, so it propagates rather than masquerading as a
    conflict the resolver could fix.
    """


class ConflictResolution(enum.Enum):
    """Whether a conflict resolver believes it fixed the merge."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class ConflictResolver(Protocol):
    """Attempts to resolve a merge conflict. The conflict-resolving merger seam.

    A production implementation re-runs the merge with a conflict-resolving agent
    inside the container; tests inject a fake that scripts the outcome. Returning
    ``RESOLVED`` means the resolver staged a resolution and the merge should be
    retried; ``UNRESOLVED`` means it gave up and the slice must escalate. A claimed
    resolution is still verified — the retried merge is the real gate, so a resolver
    that lies (leaves the conflict in place) escalates the slice rather than merging.
    """

    async def __call__(self, *, source: str, into: str) -> ConflictResolution:
        """Attempt to resolve the conflict merging ``source`` into ``into``."""
        ...


class BuildOutcome(enum.Enum):
    """Why the orchestrator merged a slice or blocked it."""

    MERGED = "merged"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class BuildResult:
    """Result of building one ready slice.

    Attributes:
        outcome: ``MERGED`` when the green slice was merged into the integration
            branch, ``BLOCKED`` when a red done-check stopped the merge.
        integration_branch: The integration branch the slice targets,
            ``retinue/prd-<n>`` — merged into on MERGED, left untouched on BLOCKED.
    """

    outcome: BuildOutcome
    integration_branch: str

    @property
    def merged(self) -> bool:
        """True only when the slice was actually merged into the integration branch."""
        return self.outcome is BuildOutcome.MERGED


async def build_slice(
    slice_: Slice,
    config: RepoConfig,
    claude_md: str,
    *,
    implementer: Implementer,
    git: GitOps,
    auth: InstallationAuth,
    runtime: ContainerRuntime,
    resolve_secret: SecretResolver,
    report: ReportSink,
    image: str = DEFAULT_IMAGE,
) -> BuildResult:
    """Build one ready slice: spawn the implementer, gate on the done-check, merge.

    The implementer builds and commits the slice to its ``issue-<N>`` branch, then the
    repo's done-check runs in a fresh disposable container. The done-check result gates
    the merge: a green check merges ``issue-<N>`` into the integration branch
    ``retinue/prd-<n>`` (created off ``config.staging_branch`` if absent), while a red
    check blocks the merge so no failing slice is ever integrated.

    Args:
        slice_: The ready slice to build (repo, issue number, PRD number).
        config: The accepted repo config; its ``staging_branch`` is the base for a
            new integration branch and its ``secrets`` are injected into the check.
        claude_md: The repo's ``CLAUDE.md`` text, carrying the done-check command.
        implementer: Spawns the implementer subagent (the Agent SDK seam).
        git: Integration-branch git operations (the merge seam).
        auth: Mints the installation token used to clone (the auth seam).
        runtime: Spawns the disposable container the done-check runs in (Docker seam).
        resolve_secret: Resolves the config's declared secret names/refs to values.
        report: Sink the done-check outcome is posted to (commit status / comment).
        image: Container image the done-check runs in.

    Returns:
        A :class:`BuildResult`: ``MERGED`` when the green slice was merged, or
        ``BLOCKED`` when a red done-check stopped it.

    Raises:
        Propagates whatever ``run_done_check`` raises (e.g. a missing secret), and any
        merge error the git seam raises on a conflict.
    """
    branch = integration_branch(slice_.prd_number)

    await implementer.implement(slice_)
    passed = await _run_slice_done_check(
        slice_,
        config,
        claude_md,
        auth=auth,
        runtime=runtime,
        resolve_secret=resolve_secret,
        report=report,
        image=image,
    )

    if not passed:
        # A red slice is never merged: leave the integration branch untouched.
        logger.info(
            "Blocking merge of %s into %s: done-check failed",
            slice_.branch,
            branch,
        )
        return BuildResult(outcome=BuildOutcome.BLOCKED, integration_branch=branch)

    await _merge_green_slice(slice_, branch, config=config, git=git)
    return BuildResult(outcome=BuildOutcome.MERGED, integration_branch=branch)


async def _run_slice_done_check(
    slice_: Slice,
    config: RepoConfig,
    claude_md: str,
    *,
    auth: InstallationAuth,
    runtime: ContainerRuntime,
    resolve_secret: SecretResolver,
    report: ReportSink,
    image: str,
) -> bool:
    """Run the repo's done-check for ``slice_``; return True only when it is green."""
    check = await run_done_check(
        slice_.repo_full_name,
        config,
        claude_md,
        auth=auth,
        runtime=runtime,
        resolve_secret=resolve_secret,
        report=report,
        image=image,
    )
    return check.passed


async def _merge_green_slice(
    slice_: Slice, branch: str, *, config: RepoConfig, git: GitOps
) -> None:
    """Ensure the integration branch and merge a green slice onto it."""
    await git.ensure_integration_branch(branch=branch, base=config.staging_branch)
    await git.merge(source=slice_.branch, into=branch)
    logger.info("Merged %s into %s after green done-check", slice_.branch, branch)


# --- full-PRD driver (issue #7) --------------------------------------------------


class OrchestratorBusyError(Exception):
    """A second orchestrator run was attempted while one is already in flight.

    The single-run guarantee: :func:`build_prd` runs inside an injected lock that
    rejects a concurrent holder. Catching this (rather than blocking) makes the
    "at most one run at a time" contract observable to the caller.
    """

    def __init__(self) -> None:
        super().__init__("an orchestrator run is already in flight")


@dataclass(frozen=True)
class PrdSlice(Slice):
    """A PRD slice: a :class:`Slice` plus the issue numbers it is blocked by.

    Attributes:
        blocked_by: Issue numbers this slice depends on. A slice is *ready* only once
            every blocker is merged in this run (or is absent from the PRD's slice
            set, meaning it was already merged/closed before the run began).
    """

    blocked_by: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class PrdBuildResult:
    """Outcome of a full-PRD build, partitioned by what happened to each slice.

    Every input slice lands in exactly one bucket — none is dropped from all of them.

    Attributes:
        integration_branch: The integration branch every slice targeted.
        merged_issues: Issue numbers merged into the integration branch, in merge
            (topological) order.
        blocked_issues: Issue numbers whose done-check was red, so they were not
            merged.
        escalated_issues: Issue numbers whose merge hit a conflict that could not be
            resolved under the done-check.
        skipped_issues: Issue numbers that never became ready because an upstream
            slice they transitively depend on was blocked or escalated, pruning the
            subtree. Reported here rather than silently dropped.
    """

    integration_branch: str
    merged_issues: list[int]
    blocked_issues: list[int]
    escalated_issues: list[int]
    skipped_issues: list[int]


async def build_prd(
    slices: list[PrdSlice],
    config: RepoConfig,
    claude_md: str,
    *,
    implementer: Implementer,
    git: GitOps,
    auth: InstallationAuth,
    runtime: ContainerRuntime,
    resolve_secret: SecretResolver,
    report: ReportSink,
    lock: AbstractAsyncContextManager[object],
    resolve_conflict: ConflictResolver | None = None,
    image: str = DEFAULT_IMAGE,
) -> PrdBuildResult:
    """Build a full PRD: ready set -> parallel fan-out -> topological merge -> loop.

    Runs under ``lock`` so at most one orchestrator run executes at a time. Each round
    picks the ready set (every ``blocked_by`` merged this run or already merged/closed
    before it), fans the slices out to implementers bounded by ``config.max_parallel``,
    then merges the green branches in dependency order under the done-check — resolving
    a conflict or escalating. Rounds repeat until no ready slice remains.

    Args:
        slices: The PRD's slices with their ``blocked_by`` graph.
        config: The accepted repo config; ``max_parallel`` bounds the fan-out and
            ``staging_branch`` is the base for a new integration branch.
        claude_md: The repo's ``CLAUDE.md`` text, carrying the done-check command.
        implementer: Spawns the implementer subagent (the Agent SDK seam).
        git: Integration-branch git operations (the merge seam).
        auth: Mints the installation token used to clone (the auth seam).
        runtime: Spawns the disposable container the done-check runs in (Docker seam).
        resolve_secret: Resolves the config's declared secret names/refs to values.
        report: Sink the done-check outcome is posted to.
        lock: The single-run lock; entering it raises :class:`OrchestratorBusyError`
            when a run is already in flight.
        resolve_conflict: Attempts to resolve a merge conflict; absent means any
            conflict escalates.
        image: Container image the done-check runs in.

    Returns:
        A :class:`PrdBuildResult` partitioning the slices into
        merged/blocked/escalated/skipped — a subtree pruned by a failed upstream
        slice lands in ``skipped``, so every slice is accounted for.

    Raises:
        OrchestratorBusyError: A run is already in flight (from the injected lock).
    """
    branch = integration_branch(slices[0].prd_number) if slices else integration_branch(0)
    async with lock:
        state = _PrdState(slices)
        while True:
            ready = state.ready_set()
            if not ready:
                break
            built = await _build_round(
                ready,
                config,
                claude_md,
                implementer=implementer,
                auth=auth,
                runtime=runtime,
                resolve_secret=resolve_secret,
                report=report,
                image=image,
            )
            await _merge_round(
                built,
                branch,
                config=config,
                git=git,
                resolve_conflict=resolve_conflict,
                state=state,
            )
        # The ready set has drained; any still-pending slice was pruned by a failed
        # upstream slice. Surface it as skipped rather than dropping it silently.
        state.drain_skipped()

    return PrdBuildResult(
        integration_branch=branch,
        merged_issues=state.merged,
        blocked_issues=state.blocked,
        escalated_issues=state.escalated,
        skipped_issues=state.skipped,
    )


class _PrdState:
    """Tracks ready-set selection and per-slice outcomes across rounds.

    A slice is *ready* when it is still pending and every blocker is either merged
    this run or absent from the PRD's slice set (already merged/closed beforehand). A
    blocked or escalated slice is terminal: its dependents never become ready, so a
    failure naturally prunes the subtree below it. That pruning is made observable —
    once the ready set drains, every still-pending slice is recorded as *skipped* via
    :meth:`drain_skipped`, so the abandoned subtree is reported, not silently dropped.
    """

    def __init__(self, slices: list[PrdSlice]) -> None:
        self._pending = list(slices)
        self._slice_numbers = {s.issue_number for s in slices}
        self.merged: list[int] = []
        self.blocked: list[int] = []
        self.escalated: list[int] = []
        self.skipped: list[int] = []

    def ready_set(self) -> list[PrdSlice]:
        """Slices whose every blocker is merged (or was already merged/closed)."""
        return [s for s in self._pending if self._is_ready(s)]

    def _is_ready(self, slice_: PrdSlice) -> bool:
        return all(
            blocker in self.merged or blocker not in self._slice_numbers
            for blocker in slice_.blocked_by
        )

    def record(self, slice_: PrdSlice, bucket: list[int]) -> None:
        """Move ``slice_`` out of pending into a terminal bucket."""
        self._pending.remove(slice_)
        bucket.append(slice_.issue_number)

    def drain_skipped(self) -> None:
        """Record every still-pending slice as skipped once no slice can become ready.

        Called after the ready set drains: any slice still pending was pruned by a
        blocked or escalated upstream slice and can never become ready. Recording it
        here surfaces the abandoned subtree instead of letting it vanish from the
        result. Issue order is preserved so the bucket is deterministic.
        """
        for slice_ in list(self._pending):
            self.record(slice_, self.skipped)


async def _build_round(
    ready: list[PrdSlice],
    config: RepoConfig,
    claude_md: str,
    *,
    implementer: Implementer,
    auth: InstallationAuth,
    runtime: ContainerRuntime,
    resolve_secret: SecretResolver,
    report: ReportSink,
    image: str,
) -> list[tuple[PrdSlice, bool]]:
    """Fan out the ready slices, bounded by ``max_parallel``; return (slice, passed).

    Each slice's implementer runs then its done-check runs, concurrently across the
    round but capped at ``config.max_parallel`` live builds. Only the done-check phase
    runs here; merges are serialized afterwards in dependency order.
    """
    semaphore = asyncio.Semaphore(config.max_parallel or len(ready) or 1)

    async def build_one(slice_: PrdSlice) -> tuple[PrdSlice, bool]:
        async with semaphore:
            await implementer.implement(slice_)
            passed = await _run_slice_done_check(
                slice_,
                config,
                claude_md,
                auth=auth,
                runtime=runtime,
                resolve_secret=resolve_secret,
                report=report,
                image=image,
            )
            return slice_, passed

    return await asyncio.gather(*(build_one(s) for s in ready))


async def _merge_round(
    built: list[tuple[PrdSlice, bool]],
    branch: str,
    *,
    config: RepoConfig,
    git: GitOps,
    resolve_conflict: ConflictResolver | None,
    state: _PrdState,
) -> None:
    """Merge a round's green slices in dependency order, recording each outcome.

    Merges are serialized (a shared integration branch) and ordered by issue number,
    which respects the ``blocked_by`` graph since a slice is only ready once its
    blockers merged. A red slice is recorded blocked; a conflict is resolved under the
    done-check or escalated.
    """
    for slice_, passed in sorted(built, key=lambda item: item[0].issue_number):
        if not passed:
            state.record(slice_, state.blocked)
            continue
        merged = await _merge_or_escalate(
            slice_, branch, config=config, git=git, resolve_conflict=resolve_conflict
        )
        state.record(slice_, state.merged if merged else state.escalated)


async def _merge_or_escalate(
    slice_: PrdSlice,
    branch: str,
    *,
    config: RepoConfig,
    git: GitOps,
    resolve_conflict: ConflictResolver | None,
) -> bool:
    """Merge a green slice; on a conflict, resolve-and-retry once or escalate.

    Returns True when the slice merged, False when it escalated (an unresolvable
    conflict, no resolver, or a retry that still conflicts).
    """
    try:
        await _merge_green_slice(slice_, branch, config=config, git=git)
        return True
    except MergeConflict as conflict:
        if resolve_conflict is None:
            logger.warning("Escalating %s: merge conflict, no resolver", slice_.branch)
            return False
        resolution = await resolve_conflict(source=conflict.source, into=conflict.into)
        if resolution is ConflictResolution.UNRESOLVED:
            logger.warning("Escalating %s: conflict unresolved", slice_.branch)
            return False
        return await _retry_merge_after_resolution(slice_, branch, config=config, git=git)


async def _retry_merge_after_resolution(
    slice_: PrdSlice, branch: str, *, config: RepoConfig, git: GitOps
) -> bool:
    """Retry a merge after a claimed resolution; escalate if it still conflicts."""
    try:
        await _merge_green_slice(slice_, branch, config=config, git=git)
        return True
    except MergeConflict:
        logger.warning("Escalating %s: still conflicts after resolution", slice_.branch)
        return False
