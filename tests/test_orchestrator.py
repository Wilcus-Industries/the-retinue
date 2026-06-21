"""Tests for the single-slice orchestrator (issue #6).

The flow is spawn-implementer -> done-check -> merge-or-block. Every collaborator is
faked: a fake implementer records that it was asked to build the slice, the done-check
runs against the faked container/auth seams reused from the done-check tests, and a fake
git-ops records branch creation and merges. No Agent SDK, no Docker, no gh, no network.

A green done-check merges ``issue-<N>`` into the integration branch ``retinue/prd-<n>``
(created off ``staging`` if absent); a red done-check blocks the merge.
"""

from __future__ import annotations

import pytest

from retinue.container import Container, RunResult
from retinue.done_check import DoneCheckReport
from retinue.orchestrator import (
    BuildOutcome,
    BuildResult,
    Implementer,
    MergeConflict,
    Slice,
    build_slice,
    integration_branch,
)
from retinue.repo_config import RepoConfig
from tests.test_done_check import (
    CLAUDE_MD,
    FakeAuth,
    FakeRuntime,
    _resolver,
    _sink,
)


class FakeImplementer:
    """Records the slice it was asked to build; no real Agent SDK spawn."""

    def __init__(self) -> None:
        self.built: list[Slice] = []

    async def implement(self, slice_: Slice) -> None:
        self.built.append(slice_)


class FakeGitOps:
    """In-memory git: records branch creation and merges, scripts existence/conflicts.

    ``existing`` is the set of branches that already exist on the remote. ``conflicts``
    is the set of source branches whose merge should raise (a conflict the orchestrator
    surfaces). ``log`` records each ensure/merge event in order.
    """

    def __init__(
        self,
        existing: set[str] | None = None,
        conflicts: set[str] | None = None,
    ) -> None:
        self.existing = set(existing or set())
        self._conflicts = set(conflicts or set())
        self.log: list[str] = []
        self.merges: list[tuple[str, str]] = []

    async def ensure_integration_branch(self, *, branch: str, base: str) -> None:
        if branch in self.existing:
            self.log.append(f"exists:{branch}")
            return
        self.log.append(f"create:{branch}<-{base}")
        self.existing.add(branch)

    async def merge(self, *, source: str, into: str) -> None:
        self.log.append(f"merge:{source}->{into}")
        if source in self._conflicts:
            raise MergeConflictError(source, into)
        self.merges.append((source, into))


class MergeConflictError(MergeConflict):
    """A merge could not complete because of a conflict (a typed ``MergeConflict``)."""


def _slice(issue_number: int = 7, prd_number: int = 1) -> Slice:
    return Slice(
        repo_full_name="owner/repo",
        issue_number=issue_number,
        prd_number=prd_number,
    )


async def _build(
    *,
    runtime: FakeRuntime,
    git: FakeGitOps,
    implementer: Implementer | None = None,
    config: RepoConfig | None = None,
    captured: list[DoneCheckReport] | None = None,
    slice_: Slice | None = None,
) -> BuildResult:
    return await build_slice(
        slice_ or _slice(),
        config or RepoConfig(),
        CLAUDE_MD,
        implementer=implementer or FakeImplementer(),
        git=git,
        auth=FakeAuth(),
        runtime=runtime,
        resolve_secret=_resolver({}),
        report=_sink(captured if captured is not None else []),
    )


# --- branch naming ---------------------------------------------------------------


def test_integration_branch_name() -> None:
    """The integration branch for PRD <n> is ``retinue/prd-<n>``."""
    assert integration_branch(1) == "retinue/prd-1"
    assert integration_branch(42) == "retinue/prd-42"


# --- happy path: green done-check merges -----------------------------------------


@pytest.mark.asyncio
async def test_green_done_check_merges_into_integration_branch() -> None:
    """A ready slice with a green done-check is merged into retinue/prd-<n>."""
    implementer = FakeImplementer()
    runtime = FakeRuntime()
    git = FakeGitOps()

    result = await _build(runtime=runtime, git=git, implementer=implementer)

    assert result.outcome is BuildOutcome.MERGED
    assert result.merged is True
    # The implementer was asked to build exactly this slice.
    assert implementer.built == [_slice()]
    # The merge happened from issue-7 into the integration branch.
    assert git.merges == [("issue-7", "retinue/prd-1")]


@pytest.mark.asyncio
async def test_integration_branch_created_off_staging_when_absent() -> None:
    """When retinue/prd-<n> is absent it is created off the config's staging branch."""
    git = FakeGitOps()
    config = RepoConfig(staging_branch="staging")

    await _build(runtime=FakeRuntime(), git=git, config=config)

    assert "create:retinue/prd-1<-staging" in git.log
    # Create precedes the merge.
    create_index = git.log.index("create:retinue/prd-1<-staging")
    merge_index = git.log.index("merge:issue-7->retinue/prd-1")
    assert create_index < merge_index


@pytest.mark.asyncio
async def test_integration_branch_reused_when_present() -> None:
    """An existing integration branch is reused, not recreated off staging."""
    git = FakeGitOps(existing={"retinue/prd-1"})

    await _build(runtime=FakeRuntime(), git=git)

    assert "exists:retinue/prd-1" in git.log
    assert not any(event.startswith("create:") for event in git.log)


@pytest.mark.asyncio
async def test_custom_staging_branch_is_the_merge_base() -> None:
    """A repo config's non-default staging branch is the base for the new branch."""
    git = FakeGitOps()
    config = RepoConfig(staging_branch="integration")

    await _build(runtime=FakeRuntime(), git=git, config=config)

    assert "create:retinue/prd-1<-integration" in git.log


# --- red done-check blocks the merge ---------------------------------------------


@pytest.mark.asyncio
async def test_red_done_check_blocks_the_merge() -> None:
    """A failing done-check blocks the merge: no merge runs, outcome is BLOCKED."""
    runtime = FakeRuntime(results={"uv": RunResult(exit_code=1, stderr="boom")})
    git = FakeGitOps()

    result = await _build(runtime=runtime, git=git)

    assert result.outcome is BuildOutcome.BLOCKED
    assert result.merged is False
    # No red slice is merged, and no integration branch work happened.
    assert git.merges == []
    assert git.log == []


@pytest.mark.asyncio
async def test_red_done_check_still_built_and_reported() -> None:
    """A red slice is still built and the failing done-check is still reported."""
    runtime = FakeRuntime(results={"uv": RunResult(exit_code=1, stderr="boom")})
    implementer = FakeImplementer()
    captured: list[DoneCheckReport] = []

    await _build(
        runtime=runtime, git=FakeGitOps(), implementer=implementer, captured=captured
    )

    assert implementer.built == [_slice()]
    assert len(captured) == 1
    assert captured[0].passed is False


# --- ordering: implement precedes done-check -------------------------------------


@pytest.mark.asyncio
async def test_implementer_runs_before_done_check() -> None:
    """The implementer commits the slice before the done-check clones and runs it."""
    events: list[str] = []

    class RecordingImplementer:
        async def implement(self, slice_: Slice) -> None:
            events.append("implement")

    runtime = FakeRuntime()
    # Record the first container command (the clone) to prove ordering.
    original_start = runtime.start

    async def start(*, image: str, env: dict[str, str]) -> Container:
        events.append("done-check")
        return await original_start(image=image, env=env)

    runtime.start = start  # type: ignore[method-assign]

    await _build(runtime=runtime, git=FakeGitOps(), implementer=RecordingImplementer())

    assert events == ["implement", "done-check"]


# --- merge conflict surfaces -----------------------------------------------------


@pytest.mark.asyncio
async def test_merge_conflict_propagates() -> None:
    """A merge conflict on a green slice propagates rather than silently passing."""
    git = FakeGitOps(conflicts={"issue-7"})

    with pytest.raises(MergeConflictError):
        await _build(runtime=FakeRuntime(), git=git)
