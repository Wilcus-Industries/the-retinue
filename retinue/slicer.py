"""Headless PRD slicer: turn a PRD body into tracer-bullet vertical slices.

:func:`slice_prd` is the entry point. It checks the PRD is substantive, runs the
headless slice generator (the Agent SDK, injected as the ``generate`` seam) over
the body, then creates one GitHub issue per slice via the ``create_issue`` seam
(``gh``, injected). Every slice issue is labeled ``ready-for-agent`` and carries
``Part of #<prd>``; a genuinely human-only slice also gets ``hitl``. Intra-PRD
``blocked_by`` references are resolved to the real created issue numbers, so the
``## Blocked by`` graph is resolvable in dependency order.

A thin or malformed PRD — too little to slice, or a generator that yields no
slices — is **not** invented around: it escalates through the shared
:class:`retinue.notify.Notifier` (push + comment + label) and creates no issues.

Both side-effecting seams (the Agent-SDK generator and the gh issue creator) are
injected so the slicer is unit-testable without network, Agent SDK, or gh.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from retinue.notify import Notification, Notifier

logger = logging.getLogger(__name__)

READY_LABEL = "ready-for-agent"
HITL_LABEL = "hitl"

# A PRD body shorter than this (after stripping) is too thin to slice responsibly.
_MIN_PRD_BODY_CHARS = 40


@dataclass
class IssueDraft:
    """One vertical slice to file as a GitHub issue.

    The generator produces drafts with ``blocked_by`` holding **1-based indices
    into the same plan**; :func:`slice_prd` rewrites those to real issue numbers
    and renders them into the body before creation, and appends the labels +
    ``Part of`` line. ``labels`` is filled by the slicer, not the generator.

    Attributes:
        title: Issue title.
        body: Issue body (the slice's what/why/acceptance), enriched in place.
        blocked_by: 1-based indices of sibling slices this one depends on.
        hitl: True only for a genuinely human-only slice (secret / external
            account / design call) that the agent loop must not attempt.
        labels: Labels applied to the issue; populated by the slicer.
    """

    title: str
    body: str
    blocked_by: list[int] = field(default_factory=list)
    hitl: bool = False
    labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SlicePlan:
    """The headless generator's output: an ordered list of slice drafts.

    Order is dependency order — a slice may only depend on earlier ones — so the
    created issue numbers are known by the time a later slice references them.
    """

    slices: list[IssueDraft]


@dataclass(frozen=True)
class CreatedIssue:
    """The result of filing one slice issue."""

    issue_number: int


class SliceOutcome(enum.Enum):
    """Why the slicer sliced the PRD or escalated instead."""

    SLICED = "sliced"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class SliceResult:
    """Result of slicing one PRD.

    Attributes:
        outcome: Whether the PRD was sliced or escalated.
        created_numbers: Issue numbers created, in plan order (empty on escalate).
    """

    outcome: SliceOutcome
    created_numbers: list[int] = field(default_factory=list)


# Injected seams. ``generate`` runs the headless Agent-SDK slicer over the PRD
# body; ``create_issue`` files one issue via gh. Both are async and faked in tests.
SliceGenerator = Callable[[str], Awaitable[SlicePlan]]
IssueCreator = Callable[[IssueDraft], Awaitable[CreatedIssue]]


async def slice_prd(
    *,
    repo_full_name: str,
    prd_number: int,
    prd_body: str,
    generate: SliceGenerator,
    create_issue: IssueCreator,
    notifier: Notifier,
) -> SliceResult:
    """Slice a PRD into labeled, dependency-ordered issues, or escalate.

    A substantive PRD is run through ``generate`` and each resulting slice is
    filed with ``ready-for-agent`` + ``Part of #<prd>`` (and ``hitl`` when the
    slice is human-only), with its ``## Blocked by`` graph resolved to real issue
    numbers. A thin PRD, or a generator that yields no slices, escalates through
    ``notifier`` and creates nothing.

    Args:
        repo_full_name: e.g. "owner/repo".
        prd_number: The PRD issue number; slices link back via ``Part of #``.
        prd_body: The PRD issue body to slice.
        generate: Async headless slicer (Agent SDK seam) producing a SlicePlan.
        create_issue: Async issue creator (gh seam) filing one slice issue.
        notifier: Shared notify primitive used to escalate a thin/malformed PRD.

    Returns:
        A :class:`SliceResult`: ``SLICED`` with the created issue numbers, or
        ``ESCALATED`` with an empty list.
    """
    if not _is_substantive(prd_body):
        logger.warning("PRD #%d in %s is too thin to slice; escalating", prd_number, repo_full_name)
        return await _escalate(
            repo_full_name,
            prd_number,
            "PRD is too thin or malformed to slice. A human needs to flesh it out.",
            notifier,
        )

    plan = await generate(prd_body)
    if not plan.slices:
        logger.warning("Slicer produced no slices for PRD #%d; escalating", prd_number)
        return await _escalate(
            repo_full_name,
            prd_number,
            "The slicer produced no vertical slices for this PRD. A human should review it.",
            notifier,
        )

    return await _create_slices(repo_full_name, prd_number, plan, create_issue)


def _is_substantive(prd_body: str) -> bool:
    """A PRD with real content to slice: non-trivial length and not a bare stub."""
    stripped = prd_body.strip()
    if len(stripped) < _MIN_PRD_BODY_CHARS:
        return False
    return not stripped.lower().startswith("todo")


async def _escalate(
    repo_full_name: str,
    prd_number: int,
    reason: str,
    notifier: Notifier,
) -> SliceResult:
    """Route a thin/malformed PRD through the notifier and create no slices."""
    await notifier.notify(
        Notification(
            repo_full_name=repo_full_name,
            issue_number=prd_number,
            title=f"Retinue can't slice PRD #{prd_number}",
            body=reason,
            label=HITL_LABEL,
        )
    )
    return SliceResult(outcome=SliceOutcome.ESCALATED)


async def _create_slices(
    repo_full_name: str,
    prd_number: int,
    plan: SlicePlan,
    create_issue: IssueCreator,
) -> SliceResult:
    """File each slice in order, resolving blocked-by to real issue numbers.

    Slices are created in plan order so a later slice's dependency on an earlier
    one resolves to an already-known issue number. Index ``i`` (1-based) in any
    ``blocked_by`` list maps to ``created_numbers[i - 1]``.
    """
    created_numbers: list[int] = []
    for draft in plan.slices:
        _finalize_draft(draft, prd_number, created_numbers)
        result = await create_issue(draft)
        created_numbers.append(result.issue_number)
    return SliceResult(outcome=SliceOutcome.SLICED, created_numbers=created_numbers)


def _finalize_draft(
    draft: IssueDraft,
    prd_number: int,
    created_numbers: list[int],
) -> None:
    """Apply labels and render the Part-of + Blocked-by footer onto ``draft``.

    Mutates ``draft`` in place: sets its labels (``ready-for-agent`` always,
    ``hitl`` for a human-only slice) and rewrites the body to carry the
    ``Part of #<prd>`` line plus a resolved ``## Blocked by`` block.
    """
    draft.labels = [READY_LABEL]
    if draft.hitl:
        draft.labels.append(HITL_LABEL)

    footer = [f"Part of #{prd_number}"]
    blocked_refs = _resolve_blocked_by(draft.blocked_by, created_numbers)
    if blocked_refs:
        footer.append("## Blocked by\n" + "\n".join(f"#{n}" for n in blocked_refs))
    draft.body = f"{draft.body.rstrip()}\n\n" + "\n\n".join(footer)


def _resolve_blocked_by(blocked_by: list[int], created_numbers: list[int]) -> list[int]:
    """Map 1-based plan indices to real created issue numbers.

    An index referencing a slice that has not been created yet (a forward or
    out-of-range reference) is dropped with a warning rather than rendered as a
    dangling ``#`` reference — the generator is expected to emit dependency order.
    """
    resolved: list[int] = []
    for index in blocked_by:
        if 1 <= index <= len(created_numbers):
            resolved.append(created_numbers[index - 1])
        else:
            logger.warning(
                "Dropping unresolvable blocked-by index %d (only %d slices created so far)",
                index,
                len(created_numbers),
            )
    return resolved
