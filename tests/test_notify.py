"""Tests for the reusable notify primitive.

The notifier fans one escalation out to three sinks — a push channel (ntfy /
Pushover), an issue comment, and a label — through injected callables so no real
network, gh, or push service is touched. Every escalation in the retinue routes
through here, so the contract is: all three sinks fire for one ``notify`` call.
"""

from __future__ import annotations

import logging
import urllib.parse
from collections.abc import Mapping, Sequence

import pytest

from retinue.notify import (
    CommentDeliveryError,
    CommentRequest,
    GhCommentSink,
    GhLabelSink,
    LabelDeliveryError,
    LabelRequest,
    Notification,
    Notifier,
    NtfyPushSink,
    PushDeliveryError,
    PushoverPushSink,
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


# --- Real push sink: pure request-building / parsing (no network) -----------


def _push() -> PushRequest:
    return PushRequest(title="Retinue needs a human", body="PRD #42 is too thin.")


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


# --- Real comment sink: gh issue comment (no subprocess) --------------------


class _RecordingGhRunner:
    """Captures the argv + env handed to the gh runner; never spawns a process."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    async def __call__(
        self, argv: Sequence[str], env: Mapping[str, str]
    ) -> None:
        self.calls.append((list(argv), dict(env)))


def _comment() -> CommentRequest:
    return CommentRequest(
        repo_full_name="owner/repo",
        issue_number=42,
        body="PRD #42 is too thin to slice.",
    )


def test_gh_comment_argv_assembles_issue_comment_command() -> None:
    """The argv is ``issue comment <n> --repo <repo> --body <body>`` (no shell)."""
    argv = GhCommentSink.build_argv(_comment())

    assert argv == [
        "issue",
        "comment",
        "42",
        "--repo",
        "owner/repo",
        "--body",
        "PRD #42 is too thin to slice.",
    ]


def test_gh_comment_argv_carries_body_verbatim_without_shell_interpolation() -> None:
    """A body with shell metacharacters rides as one --body arg, never a shell line."""
    request = CommentRequest(
        repo_full_name="owner/repo",
        issue_number=7,
        body="watch out: `rm -rf /` && $(whoami)",
    )

    argv = GhCommentSink.build_argv(request)

    assert argv[-2:] == ["--body", "watch out: `rm -rf /` && $(whoami)"]


def test_gh_comment_auth_env_injects_token_as_gh_token() -> None:
    """A configured token rides in the child env as ``GH_TOKEN``, not on the argv."""
    sink = GhCommentSink(token="tk_secret")

    env = sink.build_auth_env()

    assert env == {"GH_TOKEN": "tk_secret"}
    assert "tk_secret" not in GhCommentSink.build_argv(_comment())


def test_gh_comment_auth_env_empty_without_token() -> None:
    """With no token the auth env is empty, falling back to the host's gh auth."""
    assert GhCommentSink().build_auth_env() == {}


@pytest.mark.asyncio
async def test_gh_comment_sink_dispatches_argv_and_env_to_runner() -> None:
    """Calling the sink hands the assembled argv + auth env to the injected runner."""
    runner = _RecordingGhRunner()
    sink = GhCommentSink(token="tk_secret", runner=runner)

    await sink(_comment())

    assert len(runner.calls) == 1
    argv, env = runner.calls[0]
    assert argv == GhCommentSink.build_argv(_comment())
    assert env == {"GH_TOKEN": "tk_secret"}


@pytest.mark.asyncio
async def test_gh_comment_sink_propagates_runner_failure() -> None:
    """A failing gh runner propagates — a lost comment is not swallowed."""

    async def failing_runner(
        argv: Sequence[str], env: Mapping[str, str]
    ) -> None:
        raise CommentDeliveryError(argv, returncode=1, stderr="not found")

    sink = GhCommentSink(runner=failing_runner)

    with pytest.raises(CommentDeliveryError, match="not found"):
        await sink(_comment())


def test_comment_delivery_error_reports_argv_and_stderr() -> None:
    """The error surfaces the argv and stderr so a lost comment is debuggable."""
    err = CommentDeliveryError(
        ["issue", "comment", "42"], returncode=1, stderr="gh: not found\n"
    )

    assert err.returncode == 1
    assert err.argv == ["issue", "comment", "42"]
    assert "gh: not found" in str(err)


# --- Real label sink: gh issue edit --add-label (no subprocess) -------------


def _label() -> LabelRequest:
    return LabelRequest(
        repo_full_name="owner/repo",
        issue_number=42,
        label="hitl",
    )


def test_gh_label_argv_assembles_issue_edit_command() -> None:
    """The argv is ``issue edit <n> --repo <repo> --add-label <label>`` (no shell)."""
    argv = GhLabelSink.build_argv(_label())

    assert argv == [
        "issue",
        "edit",
        "42",
        "--repo",
        "owner/repo",
        "--add-label",
        "hitl",
    ]


def test_gh_label_argv_carries_label_verbatim_without_shell_interpolation() -> None:
    """A label with shell metacharacters rides as one --add-label arg, never a line."""
    request = LabelRequest(
        repo_full_name="owner/repo",
        issue_number=7,
        label="needs $(whoami) && `id`",
    )

    argv = GhLabelSink.build_argv(request)

    assert argv[-2:] == ["--add-label", "needs $(whoami) && `id`"]


def test_gh_label_auth_env_injects_token_as_gh_token() -> None:
    """A configured token rides in the child env as ``GH_TOKEN``, not on the argv."""
    sink = GhLabelSink(token="tk_secret")

    env = sink.build_auth_env()

    assert env == {"GH_TOKEN": "tk_secret"}
    assert "tk_secret" not in GhLabelSink.build_argv(_label())


def test_gh_label_auth_env_empty_without_token() -> None:
    """With no token the auth env is empty, falling back to the host's gh auth."""
    assert GhLabelSink().build_auth_env() == {}


@pytest.mark.asyncio
async def test_gh_label_sink_dispatches_argv_and_env_to_runner() -> None:
    """Calling the sink hands the assembled argv + auth env to the injected runner."""
    runner = _RecordingGhRunner()
    sink = GhLabelSink(token="tk_secret", runner=runner)

    await sink(_label())

    assert len(runner.calls) == 1
    argv, env = runner.calls[0]
    assert argv == GhLabelSink.build_argv(_label())
    assert env == {"GH_TOKEN": "tk_secret"}


@pytest.mark.asyncio
async def test_gh_label_sink_propagates_runner_failure() -> None:
    """A failing gh runner propagates — a lost label is not swallowed."""

    async def failing_runner(
        argv: Sequence[str], env: Mapping[str, str]
    ) -> None:
        raise LabelDeliveryError(argv, returncode=1, stderr="label not found")

    sink = GhLabelSink(runner=failing_runner)

    with pytest.raises(LabelDeliveryError, match="label not found"):
        await sink(_label())


def test_label_delivery_error_reports_argv_and_stderr() -> None:
    """The error surfaces the argv and stderr so a lost label is debuggable."""
    err = LabelDeliveryError(
        ["issue", "edit", "42"], returncode=1, stderr="gh: label not found\n"
    )

    assert err.returncode == 1
    assert err.argv == ["issue", "edit", "42"]
    assert "gh: label not found" in str(err)
