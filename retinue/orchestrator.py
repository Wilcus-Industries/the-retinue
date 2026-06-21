"""Single-slice orchestrator: spawn an implementer, gate on the done-check, merge.

This is the single-slice form of the merger primitive (issue #6). For one ready
slice the orchestrator:

1. **spawn** — runs one implementer subagent (the Agent SDK seam) in an isolated git
   worktree inside the disposable container; it implements TDD-first and commits to
   an ``issue-<N>`` branch,
2. **done-check** — runs the repo's done-check via :func:`retinue.done_check.run_done_check`
   (auth -> clone -> inject -> run -> report -> teardown), which yields a pass/fail,
3. **merge** — only on a green done-check, ensures the integration branch
   ``retinue/prd-<n>`` exists (created off the config's ``staging_branch`` when absent)
   and merges ``issue-<N>`` into it. A red done-check **blocks** the merge: no red
   slice is ever merged.

Every side-effecting collaborator is injected — the implementer spawn, the container
runtime, the auth, the secret resolver, the report sink, and the git operations — so
the whole flow is exercised in tests with no Agent SDK, no Docker, and no network.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Protocol

from retinue.container import ContainerRuntime
from retinue.done_check import DEFAULT_IMAGE, ReportSink, SecretResolver, run_done_check
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


class GitOps(Protocol):
    """Git operations on the integration branch. The merge seam.

    A production implementation runs ``git`` inside the disposable container against
    the cloned repo; tests inject a fake that records branch creation and merges. A
    merge that cannot complete (a conflict) raises rather than returning a sentinel,
    so a half-merged slice is never reported as merged.
    """

    async def ensure_integration_branch(self, *, branch: str, base: str) -> None:
        """Ensure ``branch`` exists, creating it off ``base`` when it is absent."""
        ...

    async def merge(self, *, source: str, into: str) -> None:
        """Merge ``source`` into ``into``; raise on a conflict that cannot resolve."""
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

    if not check.passed:
        # A red slice is never merged: leave the integration branch untouched.
        logger.info(
            "Blocking merge of %s into %s: done-check failed",
            slice_.branch,
            branch,
        )
        return BuildResult(outcome=BuildOutcome.BLOCKED, integration_branch=branch)

    await git.ensure_integration_branch(branch=branch, base=config.staging_branch)
    await git.merge(source=slice_.branch, into=branch)
    logger.info("Merged %s into %s after green done-check", slice_.branch, branch)
    return BuildResult(outcome=BuildOutcome.MERGED, integration_branch=branch)
