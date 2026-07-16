"""Shared Anthropic Messages-API wire helpers: constants, transport seam, parsing.

Every role that drives the Messages API over raw HTTP (the classifier and internal
reviewer today) shares these instead of keeping private copies: the protocol version and
OAuth beta constants, the credential-routed request headers, the narrow HTTP-POST seam
(:class:`HttpTransport` / :class:`HttpResponse`), and the structured-output response
parse (:func:`extract_json_object`). The Agent-SDK-driven slicer shares the constants
(its client kwargs carry the same beta header) but owns its own SDK response parsing.

Credential routing follows the claude-api conventions: a subscription OAuth token
(``sk-ant-oat...``) rides ``Authorization: Bearer`` with the ``oauth-2025-04-20`` beta
header; any other credential is a raw API key on ``x-api-key``. ``anthropic-version``
is always sent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from retinue.roles import is_oauth_credential

ANTHROPIC_VERSION = "2023-06-01"
OAUTH_BETA = "oauth-2025-04-20"
MESSAGES_URL = "https://api.anthropic.com/v1/messages"


@dataclass(frozen=True)
class HttpResponse:
    """The slice of an HTTP response a Messages-API role reads: status and JSON body."""

    status_code: int
    body: dict[str, Any]


class HttpTransport(Protocol):
    """Async HTTP POST seam (httpx-style). The network edge of a Messages-API role.

    A production implementation wraps an httpx client; tests inject a fake that
    returns a canned :class:`HttpResponse`. Kept narrow — one POST — so the real
    flow is exercisable without network.
    """

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any]
    ) -> HttpResponse:
        """POST ``json`` to ``url`` with ``headers`` and return the response."""
        ...


def request_headers(credential: str) -> dict[str, str]:
    """Build the Messages API request headers, routing ``credential`` to its scheme.

    An OAuth subscription token is sent as ``Authorization: Bearer`` with the OAuth beta
    header; any other value is treated as a raw API key on ``x-api-key``.
    """
    headers = {
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    if is_oauth_credential(credential):
        headers["authorization"] = f"Bearer {credential}"
        headers["anthropic-beta"] = OAUTH_BETA
    else:
        headers["x-api-key"] = credential
    return headers


def extract_json_object(
    body: dict[str, Any], *, who: str, error: type[Exception]
) -> dict[str, Any]:
    """Parse a Messages API response body's text content into the schema JSON object.

    Concatenates the ``text`` content blocks and loads them as JSON. Empty text,
    malformed JSON, or a non-object payload raises ``error`` (the calling role's typed
    failure, e.g. ``ClassificationError``) with ``who`` naming the role in the message —
    so no shape of a 200 response can raise anything but the role's error, and the caller
    validates only its own schema fields on the returned object.
    """
    content = body.get("content")
    blocks = content if isinstance(content, list) else []
    text = "".join(
        block["text"]
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )
    if not text.strip():
        raise error("Messages API response carried no text content")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise error(f"{who} emitted invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise error(f"{who} JSON is not an object")
    return parsed
