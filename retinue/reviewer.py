"""Internal reviewer: file review-fix follow-ups after a round's merge.

:func:`review_round` is the entry point. After a PRD round merges (see
:func:`retinue.orchestrator.build_prd`), the reviewer takes that round's merged diff
and merged issue numbers, runs the headless Agent-SDK review seam (``generate``,
injected) over them, and for each genuine finding:

1. files a follow-up issue via the slicer's ``create_issue`` seam, reusing the
   ``ready-for-agent`` + ``Part of #<prd>`` shape and adding a ``review-fix`` label
   so the agent loop routes it as a correctness/stale-doc fix, and
2. wires that new issue into the ``## Blocked by`` of each dependent open issue it
   flags (the ``edit_blocked_by`` seam, a ``gh issue edit``), so the fix builds in a
   later round *before* the work layered on top of the defect.

The reviewer **never edits code** — it only files and wires issues. All three
side-effecting seams (the Agent-SDK reviewer, the gh issue creator reused from the
slicer, and the gh issue-body editor) are injected, so the flow is unit-testable
without the Agent SDK, gh, or network.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from retinue.slicer import READY_LABEL, CreatedIssue, IssueCreator, IssueDraft

logger = logging.getLogger(__name__)

REVIEW_FIX_LABEL = "review-fix"


@dataclass(frozen=True)
class ReviewInput:
    """What the reviewer reviews: one merged round's diff and its merged issues.

    Attributes:
        repo_full_name: e.g. "owner/repo"; targets the issue creation and edits.
        prd_number: The parent PRD; review-fix issues link back via ``Part of #``.
        merged_issues: Issue numbers merged in the round, in merge order. These are
            the issues whose work the review-fix may need to block.
        diff: The round's merged diff — the review surface for correctness and stale
            docs.
    """

    repo_full_name: str
    prd_number: int
    merged_issues: list[int]
    diff: str


@dataclass
class ReviewFinding:
    """One thing the reviewer flagged in the round's diff.

    Attributes:
        title: Follow-up issue title.
        body: The finding's what/why — the review-fix issue body, enriched in place
            with the ``Part of`` footer before creation.
        blocks_issues: Issue numbers (from ``merged_issues``) whose work is layered on
            this defect; the filed review-fix issue is wired into each one's
            ``## Blocked by`` so the fix lands first. Empty for a standalone fix (e.g.
            a stale doc) that nothing depends on.
    """

    title: str
    body: str
    blocks_issues: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ReviewPlan:
    """The headless reviewer's output: the findings to file as review-fix issues."""

    findings: list[ReviewFinding]


@dataclass(frozen=True)
class EditBlockedByRequest:
    """Payload handed to the issue-body editor: add one Blocked-by reference.

    A ``gh issue edit`` that appends ``add_blocker`` to ``issue_number``'s
    ``## Blocked by`` block, so the dependent builds only after the fix merges.
    """

    repo_full_name: str
    issue_number: int
    add_blocker: int


@dataclass(frozen=True)
class ReviewResult:
    """Result of reviewing one merged round.

    Attributes:
        filed_issues: Issue numbers of the review-fix follow-ups filed, in finding
            order (empty when the review was clean).
    """

    filed_issues: list[int] = field(default_factory=list)


# Injected seams. ``generate`` runs the headless Agent-SDK reviewer over the round's
# diff; ``create_issue`` (reused from the slicer) files one issue via gh;
# ``edit_blocked_by`` wires the new issue into a dependent's ``## Blocked by``. All are
# async and faked in tests — no Agent SDK, gh, or network.
ReviewGenerator = Callable[[ReviewInput], Awaitable[ReviewPlan]]
BlockedByEditor = Callable[[EditBlockedByRequest], Awaitable[None]]


async def review_round(
    review_input: ReviewInput,
    *,
    generate: ReviewGenerator,
    create_issue: IssueCreator,
    edit_blocked_by: BlockedByEditor,
) -> ReviewResult:
    """Review a merged round; file and wire a review-fix issue per finding.

    Runs ``generate`` over the round's merged diff + issue numbers, then for every
    finding files a ``review-fix`` + ``ready-for-agent`` + ``Part of #<prd>`` issue and
    wires it into the ``## Blocked by`` of each dependent open issue it flags. A clean
    review (no findings) files nothing. The reviewer never edits code.

    Args:
        review_input: The round's diff, merged issue numbers, repo, and PRD number.
        generate: Async headless reviewer (Agent SDK seam) producing a ReviewPlan.
        create_issue: Async issue creator (gh seam) filing one review-fix issue;
            reused from the slicer so the labeling/Part-of shape is shared.
        edit_blocked_by: Async issue-body editor (gh seam) appending a Blocked-by ref
            to a dependent issue.

    Returns:
        A :class:`ReviewResult` with the filed review-fix issue numbers in finding
        order — empty when the review was clean.
    """
    plan = await generate(review_input)
    if not plan.findings:
        logger.info(
            "Review of round (PRD #%d, %s) found nothing to fix",
            review_input.prd_number,
            review_input.repo_full_name,
        )
        return ReviewResult()

    filed: list[int] = []
    for finding in plan.findings:
        created = await _file_review_fix(finding, review_input, create_issue)
        await _wire_blocked_by(created.issue_number, finding, review_input, edit_blocked_by)
        filed.append(created.issue_number)
    return ReviewResult(filed_issues=filed)


async def _file_review_fix(
    finding: ReviewFinding,
    review_input: ReviewInput,
    create_issue: IssueCreator,
) -> CreatedIssue:
    """File one finding as a labeled, PRD-linked review-fix issue via the gh seam."""
    draft = IssueDraft(
        title=finding.title,
        body=f"{finding.body.rstrip()}\n\nPart of #{review_input.prd_number}",
        labels=[READY_LABEL, REVIEW_FIX_LABEL],
    )
    return await create_issue(draft)


async def _wire_blocked_by(
    fix_number: int,
    finding: ReviewFinding,
    review_input: ReviewInput,
    edit_blocked_by: BlockedByEditor,
) -> None:
    """Add ``fix_number`` to each flagged dependent's ``## Blocked by`` block.

    Only issues that were merged in this round can be wired — a finding that names an
    issue outside the round is dropped with a warning rather than editing an unrelated
    issue, since the reviewer must not touch work it did not just review.
    """
    for dependent in finding.blocks_issues:
        if dependent not in review_input.merged_issues:
            logger.warning(
                "Dropping review-fix wiring: #%d is not in the reviewed round %s",
                dependent,
                review_input.merged_issues,
            )
            continue
        await edit_blocked_by(
            EditBlockedByRequest(
                repo_full_name=review_input.repo_full_name,
                issue_number=dependent,
                add_blocker=fix_number,
            )
        )
