"""Tests for the reusable notify primitive.

The notifier fans one escalation out to three sinks — a push channel (ntfy /
Pushover), an issue comment, and a label — through injected callables so no real
network, gh, or push service is touched. Every escalation in the retinue routes
through here, so the contract is: all three sinks fire for one ``notify`` call.
"""

from __future__ import annotations

import logging
import urllib.parse

import pytest

from retinue.notify import (
    CommentRequest,
    LabelRequest,
    Notification,
    Notifier,
    NtfyPushSink,
    PushDeliveryError,
    PushoverPushSink,
    PushRequest,
    build_basic_auth_header,
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


# --- Real push sink: pure request-building / parsing (no network) -----------


def _push() -> PushRequest:
    return PushRequest(title="Retinue needs a human", body="PRD #42 is too thin.")


def test_build_basic_auth_header_encodes_credentials() -> None:
    """The Basic auth header base64-encodes ``user:password`` per RFC 7617."""
    # "user:pass" -> base64 -> dXNlcjpwYXNz
    assert build_basic_auth_header("user", "pass") == "Basic dXNlcjpwYXNz"


def test_ntfy_request_assembly_carries_title_body_and_topic() -> None:
    """ntfy posts the body to ``{base}/{topic}`` with the title in ``X-Title``."""
    sink = NtfyPushSink(topic="retinue", base_url="https://ntfy.example.com/")

    post = sink.build_request(_push())

    assert post.url == "https://ntfy.example.com/retinue"
    assert post.data == b"PRD #42 is too thin."
    assert post.headers["X-Title"] == "Retinue needs a human"
    # No token configured -> no Authorization header.
    assert "Authorization" not in post.headers


def test_ntfy_request_adds_bearer_token_when_configured() -> None:
    """A configured ntfy token rides as a ``Bearer`` Authorization header."""
    sink = NtfyPushSink(topic="retinue", token="tk_secret")

    post = sink.build_request(_push())

    assert post.url == "https://ntfy.sh/retinue"
    assert post.headers["Authorization"] == "Bearer tk_secret"


def test_ntfy_rejects_empty_topic() -> None:
    """An empty topic is a misconfiguration, caught at construction."""
    with pytest.raises(ValueError, match="topic"):
        NtfyPushSink(topic="")


def test_pushover_request_assembly_form_encodes_credentials_and_message() -> None:
    """Pushover form-encodes token, user, title, and message into the POST body."""
    sink = PushoverPushSink(token="app_tok", user="usr_key")

    post = sink.build_request(_push())

    assert post.url.endswith("/messages.json")
    assert post.headers["Content-Type"] == "application/x-www-form-urlencoded"
    fields = dict(urllib.parse.parse_qsl(post.data.decode("utf-8")))
    assert fields == {
        "token": "app_tok",
        "user": "usr_key",
        "title": "Retinue needs a human",
        "message": "PRD #42 is too thin.",
    }


def test_pushover_requires_token_and_user() -> None:
    """Both Pushover credentials are mandatory at construction."""
    with pytest.raises(ValueError, match="token and user"):
        PushoverPushSink(token="", user="usr")


def test_pushover_parse_response_accepts_success_status() -> None:
    """A ``status: 1`` response parses cleanly (no raise)."""
    PushoverPushSink.parse_response(b'{"status": 1, "request": "abc"}')


def test_pushover_parse_response_raises_on_rejected_status() -> None:
    """A 200 with ``status: 0`` is still a delivery failure that must surface."""
    payload = b'{"status": 0, "errors": ["user key is invalid"]}'
    with pytest.raises(PushDeliveryError, match="user key is invalid"):
        PushoverPushSink.parse_response(payload)


def test_pushover_parse_response_raises_on_unparseable_body() -> None:
    """A non-JSON body cannot be confirmed as delivered, so it raises."""
    with pytest.raises(PushDeliveryError, match="unparseable"):
        PushoverPushSink.parse_response(b"<html>502 Bad Gateway</html>")
