"""Tests for the headless PRD slicer.

The slicer turns a PRD body into tracer-bullet vertical-slice issues. The two
external seams are faked: the Agent-SDK slice generator (``generate``) and the gh
issue creator (``create_issue``). A well-formed PRD yields labeled, dependency-
ordered slices; a thin or malformed PRD escalates through the notifier and creates
no slices. No real network, Agent SDK, or gh is touched.
"""

from __future__ import annotations

import pytest

from retinue.notify import (
    CommentRequest,
    LabelRequest,
    Notifier,
    PushRequest,
)
from retinue.slicer import (
    CreatedIssue,
    IssueDraft,
    SliceOutcome,
    SlicePlan,
    slice_prd,
)

PRD_NUMBER = 1
REPO = "owner/repo"

# A well-formed PRD: two ordinary slices plus one genuinely human-only slice, with
# a Blocked-by graph the slicer must resolve to created issue numbers.
WELL_FORMED_PRD = """\
## Testing Decisions

Use pytest with mocked sinks; no real network in tests.

## Slices

1. Transport spine — webhook + queue.
2. Worker gate — depends on the transport spine.
3. Provision the production Pushover account and API token.
"""

THIN_PRD = "TODO: write this later."


class _Recorder:
    """Captures notify sink calls and gh issue creations for assertions."""

    def __init__(self) -> None:
        self.pushes: list[PushRequest] = []
        self.comments: list[CommentRequest] = []
        self.labels: list[LabelRequest] = []
        self.created: list[IssueDraft] = []
        self._next_number = 100

    async def push(self, request: PushRequest) -> None:
        self.pushes.append(request)

    async def comment(self, request: CommentRequest) -> None:
        self.comments.append(request)

    async def label(self, request: LabelRequest) -> None:
        self.labels.append(request)

    async def create_issue(self, draft: IssueDraft) -> CreatedIssue:
        self._next_number += 1
        self.created.append(draft)
        return CreatedIssue(issue_number=self._next_number)

    def notifier(self) -> Notifier:
        return Notifier(push=self.push, comment=self.comment, label=self.label)


def _well_formed_plan() -> SlicePlan:
    """The fake Agent-SDK output for the well-formed PRD.

    The slicer's own logic — labels, Part-of, blocked-by resolution, escalation —
    is what is under test; the generator that produces this plan is a faked seam.
    """
    return SlicePlan(
        slices=[
            IssueDraft(title="Transport spine", body="webhook + queue", blocked_by=[]),
            IssueDraft(title="Worker gate", body="gate the PRD", blocked_by=[1]),
            IssueDraft(
                title="Provision Pushover account",
                body="create the prod account + token",
                blocked_by=[],
                hitl=True,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_well_formed_prd_creates_labeled_slices() -> None:
    """Each slice issue carries ready-for-agent + Part of #<prd>; gh is mocked."""
    rec = _Recorder()

    async def generate(prd_body: str) -> SlicePlan:
        return _well_formed_plan()

    result = await slice_prd(
        repo_full_name=REPO,
        prd_number=PRD_NUMBER,
        prd_body=WELL_FORMED_PRD,
        generate=generate,
        create_issue=rec.create_issue,
        notifier=rec.notifier(),
    )

    assert result.outcome is SliceOutcome.SLICED
    assert len(rec.created) == 3
    for draft in rec.created:
        assert "ready-for-agent" in draft.labels
        assert f"Part of #{PRD_NUMBER}" in draft.body


@pytest.mark.asyncio
async def test_blocked_by_graph_resolves_to_created_issue_numbers() -> None:
    """A slice's intra-PRD dependency is rewritten to the real created issue number."""
    rec = _Recorder()

    async def generate(prd_body: str) -> SlicePlan:
        return _well_formed_plan()

    result = await slice_prd(
        repo_full_name=REPO,
        prd_number=PRD_NUMBER,
        prd_body=WELL_FORMED_PRD,
        generate=generate,
        create_issue=rec.create_issue,
        notifier=rec.notifier(),
    )

    assert result.outcome is SliceOutcome.SLICED
    # First created slice gets number 101; the second slice was "blocked_by=[1]".
    spine_number = result.created_numbers[0]
    gate_draft = rec.created[1]
    assert f"## Blocked by\n#{spine_number}" in gate_draft.body


@pytest.mark.asyncio
async def test_human_only_slice_is_hitl_others_are_not() -> None:
    """Only the genuinely human-only slice carries the hitl label."""
    rec = _Recorder()

    async def generate(prd_body: str) -> SlicePlan:
        return _well_formed_plan()

    await slice_prd(
        repo_full_name=REPO,
        prd_number=PRD_NUMBER,
        prd_body=WELL_FORMED_PRD,
        generate=generate,
        create_issue=rec.create_issue,
        notifier=rec.notifier(),
    )

    by_title = {draft.title: draft for draft in rec.created}
    assert "hitl" in by_title["Provision Pushover account"].labels
    assert "hitl" not in by_title["Transport spine"].labels
    assert "hitl" not in by_title["Worker gate"].labels


@pytest.mark.asyncio
async def test_thin_prd_escalates_and_creates_no_slices() -> None:
    """A thin/malformed PRD escalates (push + comment + label) and slices nothing."""
    rec = _Recorder()

    async def generate(prd_body: str) -> SlicePlan:
        raise AssertionError("generate must not be called for a thin PRD")

    result = await slice_prd(
        repo_full_name=REPO,
        prd_number=PRD_NUMBER,
        prd_body=THIN_PRD,
        generate=generate,
        create_issue=rec.create_issue,
        notifier=rec.notifier(),
    )

    assert result.outcome is SliceOutcome.ESCALATED
    assert rec.created == []
    assert len(rec.pushes) == 1
    assert len(rec.comments) == 1
    assert len(rec.labels) == 1
    assert rec.labels[0].label == "hitl"


@pytest.mark.asyncio
async def test_empty_generated_plan_escalates() -> None:
    """A well-formed PRD whose generator yields zero slices escalates, not silently."""
    rec = _Recorder()

    async def generate(prd_body: str) -> SlicePlan:
        return SlicePlan(slices=[])

    result = await slice_prd(
        repo_full_name=REPO,
        prd_number=PRD_NUMBER,
        prd_body=WELL_FORMED_PRD,
        generate=generate,
        create_issue=rec.create_issue,
        notifier=rec.notifier(),
    )

    assert result.outcome is SliceOutcome.ESCALATED
    assert rec.created == []
    assert len(rec.comments) == 1
