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
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlencode

from retinue.gh import GhCliError, auth_env, run_gh_subprocess

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


# --- Real push sink: ntfy / Pushover over HTTP -----------------------------
#
# This is the production adapter behind the ``PushSink`` seam. It is an async
# callable matching the ``PushSink`` protocol — ``await sink(PushRequest(...))``
# — so it drops straight into ``Notifier(push=...)`` where the tests inject a
# fake. The HTTP POST is the only impure step; it runs in a worker thread via
# ``asyncio.to_thread`` so the blocking stdlib client never stalls the loop, and
# the request-building / response-parsing are pure methods the unit tests drive
# without touching the network.

_DEFAULT_NTFY_URL = "https://ntfy.sh"
_PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


@dataclass(frozen=True)
class _HttpPost:
    """A fully-built HTTP POST — the pure description of one push call.

    Building this is side-effect-free, so the URL, headers, and body can be
    asserted in a unit test without sending anything.
    """

    url: str
    data: bytes
    headers: dict[str, str]


class NtfyPushSink:
    """Real :data:`PushSink` backed by an `ntfy <https://ntfy.sh>`_ topic.

    ntfy publishes a notification by POSTing the message body to
    ``{base_url}/{topic}`` with the title carried in an ``X-Title`` header.
    A token, when supplied, is sent as a ``Bearer`` ``Authorization`` header.

    Args:
        topic: The ntfy topic to publish to.
        base_url: ntfy server base (defaults to the public ``https://ntfy.sh``).
        token: Optional access token for a protected topic.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        *,
        topic: str,
        base_url: str = _DEFAULT_NTFY_URL,
        token: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        if not topic:
            raise ValueError("ntfy topic must be non-empty")
        self._topic = topic
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    def build_request(self, request: PushRequest) -> _HttpPost:
        """Assemble the ntfy POST for ``request`` without sending it."""
        headers = {
            "X-Title": request.title,
            "Content-Type": "text/plain; charset=utf-8",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return _HttpPost(
            url=f"{self._base_url}/{self._topic}",
            data=request.body.encode("utf-8"),
            headers=headers,
        )

    async def __call__(self, request: PushRequest) -> None:
        """Publish ``request`` to ntfy. Raises on any HTTP/transport error."""
        await _send(self.build_request(request), self._timeout)


class PushoverPushSink:
    """Real :data:`PushSink` backed by `Pushover <https://pushover.net>`_.

    Pushover takes a form-encoded POST carrying the application ``token`` and the
    target ``user`` key alongside the ``title`` and ``message``. Both credentials
    are required; the response is JSON whose ``status`` is ``1`` on success.

    Args:
        token: Pushover application API token.
        user: Pushover user/group key to deliver to.
        timeout: Per-request timeout in seconds.
    """

    def __init__(self, *, token: str, user: str, timeout: float = 10.0) -> None:
        if not token or not user:
            raise ValueError("Pushover token and user must both be non-empty")
        self._token = token
        self._user = user
        self._timeout = timeout

    def build_request(self, request: PushRequest) -> _HttpPost:
        """Assemble the Pushover form POST for ``request`` without sending it."""
        form = urlencode(
            {
                "token": self._token,
                "user": self._user,
                "title": request.title,
                "message": request.body,
            }
        )
        return _HttpPost(
            url=_PUSHOVER_URL,
            data=form.encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    @staticmethod
    def parse_response(payload: bytes) -> None:
        """Validate a Pushover JSON response; raise on a non-success status.

        Pushover returns ``{"status": 1, ...}`` on success and ``status`` ``0``
        with an ``errors`` list otherwise, so a 200 with ``status`` ``0`` is
        still a failure that must surface.
        """
        try:
            body = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise PushDeliveryError(f"unparseable Pushover response: {exc}") from exc
        if body.get("status") != 1:
            errors = body.get("errors") or ["unknown error"]
            raise PushDeliveryError(f"Pushover rejected the push: {errors}")

    async def __call__(self, request: PushRequest) -> None:
        """Send ``request`` to Pushover. Raises on transport or status failure."""
        payload = await _send(self.build_request(request), self._timeout)
        self.parse_response(payload)


class PushDeliveryError(RuntimeError):
    """A push was attempted but the service reported it as not delivered."""


async def _send(post: _HttpPost, timeout: float) -> bytes:
    """Run the blocking POST off the event loop and return the response body.

    The stdlib HTTP client is synchronous, so it runs in a worker thread to keep
    the async ``PushSink`` contract. Transport and non-2xx errors propagate as
    :class:`PushDeliveryError`; the ``Notifier`` catches them so a flaky push
    never blocks the comment + label.
    """
    return await asyncio.to_thread(_post_sync, post, timeout)


def _post_sync(post: _HttpPost, timeout: float) -> bytes:
    req = urllib.request.Request(  # noqa: S310 - URLs are our own constants, not user input
        post.url, data=post.data, headers=post.headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body: bytes = resp.read()
            return body
    except urllib.error.HTTPError as exc:
        raise PushDeliveryError(f"push HTTP {exc.code} from {post.url}") from exc
    except urllib.error.URLError as exc:
        raise PushDeliveryError(f"push transport error to {post.url}: {exc}") from exc


# --- Real comment sink: gh issue comment ------------------------------------
#
# This is the production adapter behind the ``CommentSink`` seam. It posts the
# escalation's durable in-repo record by shelling out to ``gh issue comment``,
# mirroring the rest of the retinue's gh-cli adapters (cron.GhCli,
# handoff.HandoffGh): the command-assembly and the auth-env build are pure and
# unit-tested directly, and the one impure edge — the subprocess spawn — sits
# behind an injected ``runner`` so the seam is exercisable without a real ``gh``,
# Docker, or network. A failing comment is a lost in-repo record, so a non-zero
# ``gh`` exit raises :class:`CommentDeliveryError` rather than being swallowed.

# An async runner for a ``gh`` argv: given the argv (no leading "gh") and the auth
# env, it runs the command and returns nothing, raising on a non-zero exit. The
# default spawns a real ``gh`` child; tests inject a fake to drive the pure
# command-assembly + auth-env without a process.
GhCommentRunner = Callable[[Sequence[str], Mapping[str, str]], Awaitable[None]]


class CommentDeliveryError(RuntimeError):
    """A ``gh issue comment`` invocation failed (non-zero exit).

    Carries the argv and captured stderr so a lost-comment failure is debuggable
    rather than a bare ``CalledProcessError``.
    """

    def __init__(self, argv: Sequence[str], *, returncode: int, stderr: str) -> None:
        self.argv = list(argv)
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"gh exited {returncode} for {' '.join(argv)}: {stderr.strip()}"
        )


class GhCommentSink:
    """Real :data:`CommentSink`: posts an issue comment via ``gh issue comment``.

    Runs ``gh issue comment <number> --repo <owner/repo> --body <body>`` to write
    the durable, in-repo record of an escalation. Authenticates by injecting the
    GitHub token into the child env as ``GH_TOKEN`` (the variable ``gh`` reads), so
    the token is never placed on the command line where a process listing or log
    could leak it.

    The command assembly (:meth:`build_argv`) and the auth env build
    (:meth:`build_auth_env`) are pure and unit-tested directly. The subprocess
    spawn is the one impure edge, factored behind the injected ``runner`` (default:
    :func:`_run_gh_comment_subprocess`) so the seam is exercisable without a real
    ``gh``, Docker, or network.

    Args:
        token: The GitHub token ``gh`` authenticates with, placed in the child env
            as ``GH_TOKEN``. ``None`` runs ``gh`` with the ambient auth (e.g. a
            logged-in CLI), useful for local runs.
        runner: The injected argv runner; defaults to the real subprocess spawn.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        runner: GhCommentRunner | None = None,
    ) -> None:
        self._token = token
        self._runner = runner or _run_gh_comment_subprocess

    @staticmethod
    def build_argv(request: CommentRequest) -> list[str]:
        """Assemble the ``gh issue comment`` argv for ``request`` (no leading ``gh``).

        The body rides as a ``--body`` value rather than being interpolated into a
        shell, so a comment containing shell metacharacters is posted verbatim.
        """
        return [
            "issue",
            "comment",
            str(request.issue_number),
            "--repo",
            request.repo_full_name,
            "--body",
            request.body,
        ]

    def build_auth_env(self) -> dict[str, str]:
        """The child env carrying the token as ``GH_TOKEN`` (just it when supplied).

        The token goes in the env, never on the argv, so it never lands in a
        process listing or a log of the command. With no token, an empty mapping
        lets the runner fall back to the host's own ``gh`` auth.
        """
        return auth_env(self._token)

    async def __call__(self, request: CommentRequest) -> None:
        """Post ``request`` as an issue comment. Raises on a non-zero ``gh`` exit."""
        await self._runner(self.build_argv(request), self.build_auth_env())


async def _run_gh_comment_subprocess(
    argv: Sequence[str], env: Mapping[str, str]
) -> None:
    """Spawn ``gh`` with ``env`` layered over the ambient env; raise on failure.

    The default :data:`GhCommentRunner`: delegates the spawn to
    :func:`retinue.gh.run_gh_subprocess` and surfaces a non-zero exit as
    :class:`CommentDeliveryError` so a lost comment fails loudly — the comment is the
    durable record the caller must not silently lose.
    """
    try:
        await run_gh_subprocess(["gh", *argv], env)
    except GhCliError as exc:
        raise CommentDeliveryError(
            argv, returncode=exc.returncode, stderr=exc.stderr
        ) from exc


# --- Real label sink: gh issue edit --add-label ------------------------------
#
# This is the production adapter behind the ``LabelSink`` seam. It applies the
# escalation's routing label by shelling out to ``gh issue edit --add-label``,
# mirroring the comment sink above (and the rest of the retinue's gh-cli
# adapters): the command-assembly and the auth-env build are pure and unit-tested
# directly, and the one impure edge — the subprocess spawn — sits behind an
# injected ``runner`` so the seam is exercisable without a real ``gh``, Docker, or
# network. The label makes the escalated issue findable and routes the agent
# loop, so a non-zero ``gh`` exit raises :class:`LabelDeliveryError` rather than
# being swallowed.

# An async runner for a ``gh`` argv, identical in shape to GhCommentRunner: given
# the argv (no leading "gh") and the auth env, it runs the command and returns
# nothing, raising on a non-zero exit. The default spawns a real ``gh`` child;
# tests inject a fake to drive the pure command-assembly + auth-env.
GhLabelRunner = Callable[[Sequence[str], Mapping[str, str]], Awaitable[None]]


class LabelDeliveryError(RuntimeError):
    """A ``gh issue edit --add-label`` invocation failed (non-zero exit).

    Carries the argv and captured stderr so a lost-label failure is debuggable
    rather than a bare ``CalledProcessError``.
    """

    def __init__(self, argv: Sequence[str], *, returncode: int, stderr: str) -> None:
        self.argv = list(argv)
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"gh exited {returncode} for {' '.join(argv)}: {stderr.strip()}"
        )


class GhLabelSink:
    """Real :data:`LabelSink`: applies a label via ``gh issue edit --add-label``.

    Runs ``gh issue edit <number> --repo <owner/repo> --add-label <label>`` to make
    the escalated issue findable and to route the agent loop. Authenticates by
    injecting the GitHub token into the child env as ``GH_TOKEN`` (the variable
    ``gh`` reads), so the token is never placed on the command line where a process
    listing or log could leak it.

    The command assembly (:meth:`build_argv`) and the auth env build
    (:meth:`build_auth_env`) are pure and unit-tested directly. The subprocess
    spawn is the one impure edge, factored behind the injected ``runner`` (default:
    :func:`_run_gh_label_subprocess`) so the seam is exercisable without a real
    ``gh``, Docker, or network.

    Args:
        token: The GitHub token ``gh`` authenticates with, placed in the child env
            as ``GH_TOKEN``. ``None`` runs ``gh`` with the ambient auth (e.g. a
            logged-in CLI), useful for local runs.
        runner: The injected argv runner; defaults to the real subprocess spawn.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        runner: GhLabelRunner | None = None,
    ) -> None:
        self._token = token
        self._runner = runner or _run_gh_label_subprocess

    @staticmethod
    def build_argv(request: LabelRequest) -> list[str]:
        """Assemble the ``gh issue edit`` argv for ``request`` (no leading ``gh``).

        The label rides as an ``--add-label`` value rather than being interpolated
        into a shell, so a label containing shell metacharacters is applied verbatim.
        """
        return [
            "issue",
            "edit",
            str(request.issue_number),
            "--repo",
            request.repo_full_name,
            "--add-label",
            request.label,
        ]

    def build_auth_env(self) -> dict[str, str]:
        """The child env carrying the token as ``GH_TOKEN`` (just it when supplied).

        The token goes in the env, never on the argv, so it never lands in a
        process listing or a log of the command. With no token, an empty mapping
        lets the runner fall back to the host's own ``gh`` auth.
        """
        return auth_env(self._token)

    async def __call__(self, request: LabelRequest) -> None:
        """Apply ``request``'s label. Raises on a non-zero ``gh`` exit."""
        await self._runner(self.build_argv(request), self.build_auth_env())


async def _run_gh_label_subprocess(
    argv: Sequence[str], env: Mapping[str, str]
) -> None:
    """Spawn ``gh`` with ``env`` layered over the ambient env; raise on failure.

    The default :data:`GhLabelRunner`: delegates the spawn to
    :func:`retinue.gh.run_gh_subprocess` and surfaces a non-zero exit as
    :class:`LabelDeliveryError` so a lost label fails loudly — the label routes the
    agent loop and the caller must not silently lose it.
    """
    try:
        await run_gh_subprocess(["gh", *argv], env)
    except GhCliError as exc:
        raise LabelDeliveryError(
            argv, returncode=exc.returncode, stderr=exc.stderr
        ) from exc
