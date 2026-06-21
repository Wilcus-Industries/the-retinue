"""The reusable notification primitive: one escalation, three sinks.

Every escalation in the retinue — a thin PRD, a stuck agent, a human-only
decision — routes through :class:`Notifier`. A single :meth:`Notifier.notify`
call fans a :class:`Notification` out to three injected sinks:

* a **push** channel (ntfy or Pushover) for the out-of-band heads-up,
* an issue **comment** for the durable, in-repo record of why, and
* a **label** so the issue is findable and the agent loop can route on it.

The three sinks are injected as async callables so the real ntfy/Pushover HTTP
call and the real ``gh`` invocations live behind seams the tests fake out — no
network, no ``gh``, no push service in a unit test. Other modules import this
module and call :meth:`Notifier.notify`; they never re-implement the fan-out.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Notification:
    """One escalation to deliver to all three sinks.

    Attributes:
        repo_full_name: e.g. "owner/repo"; targets the comment and label sinks.
        issue_number: The issue to comment on and label.
        title: Short push title / subject line.
        body: Human-readable explanation, reused for the push and the comment.
        label: The label to apply (e.g. ``hitl``).
    """

    repo_full_name: str
    issue_number: int
    title: str
    body: str
    label: str


@dataclass(frozen=True)
class PushRequest:
    """Payload handed to the push sink (ntfy / Pushover)."""

    title: str
    body: str


@dataclass(frozen=True)
class CommentRequest:
    """Payload handed to the comment sink (a ``gh issue comment``)."""

    repo_full_name: str
    issue_number: int
    body: str


@dataclass(frozen=True)
class LabelRequest:
    """Payload handed to the label sink (a ``gh issue edit --add-label``)."""

    repo_full_name: str
    issue_number: int
    label: str


# Async sink seams. The real implementations (an ntfy/Pushover POST and two gh
# invocations) are injected so the fan-out is testable without side effects.
PushSink = Callable[[PushRequest], Awaitable[None]]
CommentSink = Callable[[CommentRequest], Awaitable[None]]
LabelSink = Callable[[LabelRequest], Awaitable[None]]


class Notifier:
    """Fans one :class:`Notification` out to a push, a comment, and a label sink.

    The comment and the label are the durable, in-repo record of an escalation;
    the push is a best-effort out-of-band ping. A failing push sink is therefore
    logged and swallowed so it can never block the comment + label from landing.
    A failing comment or label sink *does* raise — losing the in-repo record is a
    real failure the caller must see.
    """

    def __init__(self, *, push: PushSink, comment: CommentSink, label: LabelSink) -> None:
        self._push = push
        self._comment = comment
        self._label = label

    async def notify(self, notification: Notification) -> None:
        """Deliver ``notification`` to all three sinks.

        Args:
            notification: The escalation to deliver.

        Raises:
            Exception: Whatever the comment or label sink raises — those are the
                durable record and a failure there must surface. A push-sink
                failure is logged and not raised.
        """
        await self._try_push(notification)
        await self._comment(
            CommentRequest(
                repo_full_name=notification.repo_full_name,
                issue_number=notification.issue_number,
                body=notification.body,
            )
        )
        await self._label(
            LabelRequest(
                repo_full_name=notification.repo_full_name,
                issue_number=notification.issue_number,
                label=notification.label,
            )
        )

    async def _try_push(self, notification: Notification) -> None:
        """Send the best-effort push; log and swallow any failure.

        A flaky ntfy/Pushover endpoint must not cost us the comment + label, so
        every exception here is caught. ``asyncio.CancelledError`` is re-raised so
        task cancellation still propagates.
        """
        try:
            await self._push(
                PushRequest(title=notification.title, body=notification.body)
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Push sink failed for %s#%d; comment+label still applied",
                notification.repo_full_name,
                notification.issue_number,
                exc_info=True,
            )
