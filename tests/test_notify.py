"""Tests for the reusable notify primitive.

The notifier fans one escalation out to three sinks — a push channel (ntfy /
Pushover), an issue comment, and a label — through injected callables so no real
network, gh, or push service is touched. Every escalation in the retinue routes
through here, so the contract is: all three sinks fire for one ``notify`` call.
"""

from __future__ import annotations

import logging

import pytest

from retinue.notify import (
    CommentRequest,
    LabelRequest,
    Notification,
    Notifier,
    PushRequest,
)


class _RecordingSinks:
    """Captures each sink call so a test can assert all three fired."""

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


def _notification() -> Notification:
    return Notification(
        repo_full_name="owner/repo",
        issue_number=42,
        title="Retinue needs a human",
        body="PRD #42 is too thin to slice.",
        label="hitl",
    )


@pytest.mark.asyncio
async def test_notify_fires_all_three_sinks() -> None:
    """One notify call pushes, comments, and labels — exactly once each."""
    sinks = _RecordingSinks()
    notifier = Notifier(push=sinks.push, comment=sinks.comment, label=sinks.label)

    await notifier.notify(_notification())

    assert len(sinks.pushes) == 1
    assert len(sinks.comments) == 1
    assert len(sinks.labels) == 1


@pytest.mark.asyncio
async def test_notify_routes_fields_to_each_sink() -> None:
    """Each sink receives the targeting + payload fields it needs."""
    sinks = _RecordingSinks()
    notifier = Notifier(push=sinks.push, comment=sinks.comment, label=sinks.label)

    await notifier.notify(_notification())

    push = sinks.pushes[0]
    assert push.title == "Retinue needs a human"
    assert "too thin" in push.body

    comment = sinks.comments[0]
    assert comment.repo_full_name == "owner/repo"
    assert comment.issue_number == 42
    assert "too thin" in comment.body

    label = sinks.labels[0]
    assert label.repo_full_name == "owner/repo"
    assert label.issue_number == 42
    assert label.label == "hitl"


@pytest.mark.asyncio
async def test_notify_continues_when_push_sink_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing push sink must not stop the comment and label from being applied.

    The comment + label are the durable, in-repo record of an escalation; a flaky
    push channel must never swallow them. The push failure is logged, not raised.
    """
    sinks = _RecordingSinks()

    async def failing_push(request: PushRequest) -> None:
        raise RuntimeError("ntfy unreachable")

    notifier = Notifier(push=failing_push, comment=sinks.comment, label=sinks.label)

    with caplog.at_level(logging.WARNING, logger="retinue.notify"):
        await notifier.notify(_notification())

    assert len(sinks.comments) == 1
    assert len(sinks.labels) == 1
    assert "push" in caplog.text.lower()


@pytest.mark.asyncio
async def test_notify_raises_when_comment_sink_fails() -> None:
    """A failing comment sink must propagate — unlike push, it is not swallowed.

    The comment is the durable, in-repo record of an escalation; losing it is a
    real failure the caller must see. The label sink is irrelevant here.
    """
    sinks = _RecordingSinks()

    async def failing_comment(request: CommentRequest) -> None:
        raise RuntimeError("gh issue comment failed")

    notifier = Notifier(push=sinks.push, comment=failing_comment, label=sinks.label)

    with pytest.raises(RuntimeError, match="gh issue comment failed"):
        await notifier.notify(_notification())


@pytest.mark.asyncio
async def test_notify_raises_when_label_sink_fails() -> None:
    """A failing label sink must propagate — unlike push, it is not swallowed.

    The label makes the escalated issue findable and routes the agent loop;
    losing it is a real failure the caller must see.
    """
    sinks = _RecordingSinks()

    async def failing_label(request: LabelRequest) -> None:
        raise RuntimeError("gh issue edit failed")

    notifier = Notifier(push=sinks.push, comment=sinks.comment, label=failing_label)

    with pytest.raises(RuntimeError, match="gh issue edit failed"):
        await notifier.notify(_notification())
