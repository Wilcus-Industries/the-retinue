"""Convergence handoff + merge reap (issue #12): notify, then reap on the human merge.

This module owns the two ends of the retinue's "the retinue never merges" contract:

1. **convergence handoff** (:func:`announce_handoff`) — when heimdall converges on the
   staging PR (:mod:`retinue.loopback` reaches ``CONVERGED`` and calls its ``Handoff``
   seam), the retinue fires a single "test & merge" notification: a push (the
   out-of-band heads-up) plus a PR comment (the durable, in-repo record) telling a human
   to test and merge. There is **no** merge collaborator — the retinue hands off and a
   human merges. The signature is the loopback ``Handoff`` shape (``repo_full_name=`` +
   ``pr_number=``) so it wires straight in.
2. **merge reap** (:func:`reap_merged_pr`) — subscribing to a ``pull_request``
   closed+merged signal, on the human's merge the retinue closes the PR's slice issues
   and then *reaps* the PRD: it closes the PRD IFF every non-``hitl`` child issue is
   closed. An open ``hitl`` child (a deliberately human-only slice) does not block the
   reap; an open non-``hitl`` child does.

The push + PR comment reuse the shared :class:`retinue.notify.Notifier` fan-out. The
gh issue-close and PRD child-enumeration are an injected :class:`Handoff` gh seam,
mirroring the gh-seam style of :mod:`retinue.pr_opener` / :mod:`retinue.orchestrator`,
so both flows run with no real ``gh``, push service, or network in a unit test.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Protocol

from retinue.notify import Notification, Notifier

logger = logging.getLogger(__name__)

# The "test & merge" handoff is a heads-up for a human, not an escalation. It carries a
# findable label so the converged-but-unmerged PRs are routable, but it never merges.
TEST_AND_MERGE_LABEL = "test-and-merge"


async def announce_handoff(
    *,
    repo_full_name: str,
    pr_number: int,
    notifier: Notifier,
) -> None:
    """Announce a converged PR for a human to test & merge — the retinue never merges.

    Fires one notification through the shared :class:`~retinue.notify.Notifier`: a push
    heads-up plus a PR comment (and a findable label) telling a human the PR is clean and
    ready to test and merge. No merge happens here and no merge seam is accepted — the
    human merges, and :func:`reap_merged_pr` then reacts to that merge.

    The keyword signature matches the loopback ``Handoff`` seam
    (:data:`retinue.loopback.Handoff`), so the converge path wires this in directly.

    Args:
        repo_full_name: e.g. "owner/repo"; targets the PR comment and label.
        pr_number: The converged staging PR to hand off; the comment lands on it.
        notifier: The shared push + comment + label fan-out.

    Raises:
        Whatever ``notifier`` raises when the durable comment/label record cannot be
        written (a push-sink failure is logged and swallowed by the notifier).
    """
    logger.info(
        "Handing off converged PR #%d (%s) for a human to test & merge",
        pr_number,
        repo_full_name,
    )
    await notifier.notify(
        Notification(
            repo_full_name=repo_full_name,
            issue_number=pr_number,
            title=f"Retinue: test & merge PR #{pr_number}",
            body=(
                f"Heimdall converged on PR #{pr_number}: it is clean and ready. "
                "Please test and merge it — the retinue does not merge. Once you merge, "
                "the retinue will close the slice issues and reap the PRD."
            ),
            label=TEST_AND_MERGE_LABEL,
        )
    )


@dataclass(frozen=True)
class MergedPullRequest:
    """A ``pull_request`` closed+merged signal: the human merged the staging PR.

    Parsed from the ``pull_request`` webhook payload (action ``closed`` with
    ``merged == true``) before the reap reasons about it.

    Attributes:
        repo_full_name: e.g. "owner/repo"; targets the issue closes and the child query.
        pr_number: The merged PR number, for logging and the reap record.
        prd_number: The parent PRD whose children gate the reap; children reference it
            via ``Part of #<prd>`` (the slicer convention).
        slice_issues: The PR's slice issue numbers to close on merge.
    """

    repo_full_name: str
    pr_number: int
    prd_number: int
    slice_issues: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ChildIssue:
    """One child issue of a PRD, as reported by the child-enumeration gh seam.

    A child is any issue carrying ``Part of #<prd>`` (the slicer convention). The reap
    gate looks at the non-``hitl`` children only.

    Attributes:
        number: The child issue number.
        closed: True when the issue is already closed.
        hitl: True when the issue carries the ``hitl`` label — a deliberately human-only
            slice that does NOT block the reap.
    """

    number: int
    closed: bool
    hitl: bool


class Handoff(Protocol):
    """The gh operations behind the merge reap. The reap gh seam.

    A production implementation runs ``gh`` against the target repo (``gh issue close``
    and a child-enumeration query that finds issues carrying ``Part of #<prd>`` with
    their closed-state and ``hitl`` label); tests inject a fake that records the closes
    and scripts the children. There is deliberately **no** merge method — the retinue
    never merges; the human does, and this seam only reacts to that merge.
    """

    async def close_issue(self, *, repo_full_name: str, issue_number: int) -> None:
        """Close ``issue_number`` on ``repo_full_name`` (a ``gh issue close``)."""
        ...

    async def children_of(
        self, *, repo_full_name: str, prd_number: int
    ) -> list[ChildIssue]:
        """Return the PRD's children (issues carrying ``Part of #<prd>``)."""
        ...


class ReapOutcome(enum.Enum):
    """Whether the merge reap closed the PRD or left it open."""

    REAPED = "reaped"
    KEPT_OPEN = "kept_open"


@dataclass(frozen=True)
class ReapResult:
    """Outcome of reacting to a merged PR.

    Attributes:
        outcome: ``REAPED`` when the PRD was closed (all non-``hitl`` children closed),
            ``KEPT_OPEN`` when an open non-``hitl`` child still blocks it.
        closed_slice_issues: The slice issue numbers closed on this merge, in order.
        prd_closed: True only when the PRD itself was closed.
    """

    outcome: ReapOutcome
    closed_slice_issues: list[int] = field(default_factory=list)
    prd_closed: bool = False


async def reap_merged_pr(merged: MergedPullRequest, *, gh: Handoff) -> ReapResult:
    """React to a human-merged PR: close its slice issues, then reap the PRD.

    Closes the PR's slice issues first, then reaps the PRD — closing it IFF every
    non-``hitl`` child is closed. Slice issues are closed *before* the child gate is
    evaluated so a slice that the children query still reports open is already accounted
    for. An open ``hitl`` child is excluded from the gate (a deliberately human-only
    slice must not hold the PRD open); an open non-``hitl`` child keeps the PRD open.

    Args:
        merged: The parsed ``pull_request`` closed+merged signal.
        gh: The injected reap gh seam (issue close + child enumeration).

    Returns:
        A :class:`ReapResult`: ``REAPED`` with ``prd_closed`` true, or ``KEPT_OPEN``.

    Raises:
        Whatever ``gh`` raises on a real gh failure.
    """
    for issue_number in merged.slice_issues:
        await gh.close_issue(
            repo_full_name=merged.repo_full_name, issue_number=issue_number
        )
    logger.info(
        "Merged PR #%d (%s): closed %d slice issue(s)",
        merged.pr_number,
        merged.repo_full_name,
        len(merged.slice_issues),
    )

    if not await _all_non_hitl_children_closed(merged, gh):
        logger.info(
            "PRD #%d (%s) kept open: a non-hitl child is still open",
            merged.prd_number,
            merged.repo_full_name,
        )
        return ReapResult(
            outcome=ReapOutcome.KEPT_OPEN,
            closed_slice_issues=list(merged.slice_issues),
        )

    await gh.close_issue(
        repo_full_name=merged.repo_full_name, issue_number=merged.prd_number
    )
    logger.info(
        "Reaped PRD #%d (%s): every non-hitl child is closed",
        merged.prd_number,
        merged.repo_full_name,
    )
    return ReapResult(
        outcome=ReapOutcome.REAPED,
        closed_slice_issues=list(merged.slice_issues),
        prd_closed=True,
    )


async def _all_non_hitl_children_closed(
    merged: MergedPullRequest, gh: Handoff
) -> bool:
    """Whether every non-``hitl`` child of the PRD is closed (the reap gate).

    A ``hitl`` child is excluded from the gate — it is a deliberately human-only slice
    that must not hold the PRD open. A child the merge just closed is treated as closed
    even if the children query still reports it open, since the close already ran.
    """
    just_closed = set(merged.slice_issues)
    children = await gh.children_of(
        repo_full_name=merged.repo_full_name, prd_number=merged.prd_number
    )
    return all(
        child.closed or child.number in just_closed
        for child in children
        if not child.hitl
    )
