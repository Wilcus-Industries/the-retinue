"""Tests for the convergence handoff and the merge reap (issue #12).

Two flows, both with every gh/push/network touch faked or injected:

1. **convergence handoff** — when heimdall converges on the staging PR, the retinue
   fires a "test & merge" notification (a push + a PR comment) and NEVER merges. There
   is no merge seam to call; the test asserts the announcement landed and that the
   handoff signature is the loopback ``Handoff`` shape so it wires in directly.
2. **merge reap** — on a ``pull_request`` closed+merged signal (the human merged the
   PR), the retinue closes the PR's slice issues, then reaps the PRD: it closes the PRD
   IFF every non-``hitl`` child issue is closed. An open ``hitl`` child does not block
   the reap; an open non-``hitl`` child does.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable

import pytest

from retinue.handoff import (
    ChildIssue,
    Handoff,
    MergedPullRequest,
    ReapOutcome,
    announce_handoff,
    reap_merged_pr,
)
from retinue.loopback import Handoff as LoopbackHandoff
from retinue.notify import CommentRequest, LabelRequest, Notifier, PushRequest


class _RecordingSinks:
    """Captures notifier sink calls so a test can assert the announcement fired."""

    def __init__(self) -> None:
        self.pushes: list[PushRequest] = []
        self.comments: list[CommentRequest] = []
        self.labels: list[LabelRequest] = []

    async def push(self, request: PushRequest) -> None:
        self.pushes.append(request)

    async def comment(self, request: CommentRequest) -> None:
        self.comments.append(request)

    async def label(self, request: LabelRequest) -> None:
        self.labels.append(request)


def _notifier(sinks: _RecordingSinks) -> Notifier:
    return Notifier(push=sinks.push, comment=sinks.comment, label=sinks.label)


class _RecordingGh:
    """A fake gh seam: records issue closes and scripts a PRD's children."""

    def __init__(self, children: dict[int, list[ChildIssue]] | None = None) -> None:
        self.closed: list[tuple[str, int]] = []
        self._children = children or {}

    async def close_issue(self, *, repo_full_name: str, issue_number: int) -> None:
        self.closed.append((repo_full_name, issue_number))

    async def children_of(self, *, repo_full_name: str, prd_number: int) -> list[ChildIssue]:
        return self._children.get(prd_number, [])


# --- convergence handoff: fire "test & merge" (push + PR comment), never merge ----


@pytest.mark.asyncio
async def test_convergence_fires_test_and_merge_push_and_comment() -> None:
    """The handoff fires a push + a PR comment telling a human to test & merge."""
    sinks = _RecordingSinks()

    await announce_handoff(
        repo_full_name="owner/repo",
        pr_number=42,
        notifier=_notifier(sinks),
    )

    # A push went out (the out-of-band heads-up).
    assert len(sinks.pushes) == 1
    # A PR comment landed on the PR (the durable in-repo record).
    assert len(sinks.comments) == 1
    assert sinks.comments[0].issue_number == 42
    assert sinks.comments[0].repo_full_name == "owner/repo"
    # The announcement is a "test & merge" handoff, naming the PR.
    body = sinks.comments[0].body.lower()
    assert "test" in body and "merge" in body
    assert "#42" in sinks.comments[0].body


@pytest.mark.asyncio
async def test_convergence_never_merges() -> None:
    """The handoff has no merge seam: the retinue never merges on convergence."""
    sinks = _RecordingSinks()

    await announce_handoff(
        repo_full_name="owner/repo",
        pr_number=42,
        notifier=_notifier(sinks),
    )

    # There is no merge collaborator at all on the handoff signature.
    params = inspect.signature(announce_handoff).parameters
    assert not any("merge" in name.lower() for name in params)


def test_announce_handoff_matches_loopback_handoff_seam() -> None:
    """``announce_handoff`` is callable as the loopback ``Handoff`` seam."""
    # The loopback seam is ``Callable[..., Awaitable[None]]``; the converge path calls
    # it with ``repo_full_name=`` and ``pr_number=``. announce_handoff must accept both
    # so it can be partially applied (notifier bound) into the loopback seam.
    def handoff(*, repo_full_name: str, pr_number: int) -> Awaitable[None]:
        return announce_handoff(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            notifier=_notifier(_RecordingSinks()),
        )

    seam: LoopbackHandoff = handoff
    params = inspect.signature(announce_handoff).parameters
    assert "repo_full_name" in params
    assert "pr_number" in params
    assert callable(seam)


# --- merge reap: a merged PR closes slice issues, then reaps the PRD --------------


@pytest.mark.asyncio
async def test_merged_pr_closes_slice_issues_then_reaps_prd_when_all_closed() -> None:
    """A merged PR closes its slices, then closes the PRD when all non-hitl kids closed."""
    # Both children are closed non-hitl issues -> PRD is reaped.
    gh = _RecordingGh(
        children={
            1: [
                ChildIssue(number=10, closed=True, hitl=False),
                ChildIssue(number=11, closed=True, hitl=False),
            ]
        }
    )
    merged = MergedPullRequest(
        repo_full_name="owner/repo",
        pr_number=42,
        prd_number=1,
        slice_issues=[10, 11],
    )

    result = await reap_merged_pr(merged, gh=gh)

    # The PR's slice issues were closed first.
    assert ("owner/repo", 10) in gh.closed
    assert ("owner/repo", 11) in gh.closed
    # Then the PRD itself was reaped (closed).
    assert ("owner/repo", 1) in gh.closed
    assert result.outcome is ReapOutcome.REAPED
    assert result.prd_closed is True


@pytest.mark.asyncio
async def test_open_non_hitl_child_blocks_the_reap() -> None:
    """An open non-hitl child leaves the PRD open: the reap does not fire."""
    gh = _RecordingGh(
        children={
            1: [
                ChildIssue(number=10, closed=True, hitl=False),
                ChildIssue(number=99, closed=False, hitl=False),
            ]
        }
    )
    merged = MergedPullRequest(
        repo_full_name="owner/repo",
        pr_number=42,
        prd_number=1,
        slice_issues=[10],
    )

    result = await reap_merged_pr(merged, gh=gh)

    # The slice issue was closed, but the PRD was NOT closed.
    assert ("owner/repo", 10) in gh.closed
    assert ("owner/repo", 1) not in gh.closed
    assert result.outcome is ReapOutcome.KEPT_OPEN
    assert result.prd_closed is False


@pytest.mark.asyncio
async def test_open_hitl_child_does_not_block_the_reap() -> None:
    """An open hitl child is excluded from the reap gate: the PRD still closes."""
    gh = _RecordingGh(
        children={
            1: [
                ChildIssue(number=10, closed=True, hitl=False),
                ChildIssue(number=50, closed=False, hitl=True),
            ]
        }
    )
    merged = MergedPullRequest(
        repo_full_name="owner/repo",
        pr_number=42,
        prd_number=1,
        slice_issues=[10],
    )

    result = await reap_merged_pr(merged, gh=gh)

    # The open child is hitl, so it does not block: the PRD is reaped.
    assert ("owner/repo", 1) in gh.closed
    assert result.outcome is ReapOutcome.REAPED
    assert result.prd_closed is True


@pytest.mark.asyncio
async def test_reap_closes_slice_issues_before_checking_children() -> None:
    """Slice issues close first; the just-closed slice can complete the reap gate."""
    # The only non-hitl child is the slice issue itself, still reported open by the
    # children query. Closing it first must let the reap fire.
    gh = _RecordingGh(
        children={
            1: [ChildIssue(number=10, closed=False, hitl=False)]
        }
    )
    merged = MergedPullRequest(
        repo_full_name="owner/repo",
        pr_number=42,
        prd_number=1,
        slice_issues=[10],
    )

    result = await reap_merged_pr(merged, gh=gh)

    assert ("owner/repo", 10) in gh.closed
    assert ("owner/repo", 1) in gh.closed
    assert result.outcome is ReapOutcome.REAPED


@pytest.mark.asyncio
async def test_reap_with_no_children_closes_prd() -> None:
    """A PRD whose children are all merged-and-closed reaps with an empty gate."""
    gh = _RecordingGh(children={1: []})
    merged = MergedPullRequest(
        repo_full_name="owner/repo",
        pr_number=42,
        prd_number=1,
        slice_issues=[],
    )

    result = await reap_merged_pr(merged, gh=gh)

    assert ("owner/repo", 1) in gh.closed
    assert result.outcome is ReapOutcome.REAPED


def test_handoff_module_exposes_no_merge_seam() -> None:
    """The reap gh seam protocol has no merge method: the retinue never merges."""
    methods = {name for name in dir(Handoff) if not name.startswith("_")}
    assert not any("merge" in name.lower() for name in methods)
    assert "close_issue" in methods
    assert "children_of" in methods
