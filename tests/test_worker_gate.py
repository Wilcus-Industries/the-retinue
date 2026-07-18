"""Tests for the GitHub-backed opt-in config fetcher.

The worker resolves a repo's opt-in by fetching ``.github/retinue.yml`` through
:func:`retinue.github_app.github_config_fetcher`. These tests cover its pure parts (the
contents URL, the auth headers, the base64 decode) and its offline transport behavior
(200 -> text, 404 -> None, 5xx -> raise).
"""

from __future__ import annotations

import base64

import httpx
import pytest

from retinue.github_app import (
    GITHUB_API_BASE_URL,
    RETINUE_CONFIG_PATH,
    InstallationToken,
    _auth_headers,
    _decode_contents_payload,
    _repo_contents_url,
    github_config_fetcher,
)

VALID_CONFIG = "target_branch: release\nretry_cap: 2\n"


class _FakeAuth:
    """An InstallationAuth stub returning a canned token, recording the repo asked."""

    def __init__(self, token: str = "ghs_canned") -> None:
        self.token = token
        self.asked_for: str | None = None

    async def installation_token(self, repo_full_name: str) -> InstallationToken:
        self.asked_for = repo_full_name
        return InstallationToken(token=self.token, clone_url="https://x/y.git")


def _contents_payload(text: str) -> dict[str, object]:
    """A GitHub contents-API response body for a base64-encoded file."""
    return {
        "encoding": "base64",
        "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
    }


def test_contents_url_targets_the_opt_in_file() -> None:
    url = _repo_contents_url("owner/repo", RETINUE_CONFIG_PATH)
    assert url == f"{GITHUB_API_BASE_URL}/repos/owner/repo/contents/{RETINUE_CONFIG_PATH}"


def test_auth_headers_use_bearer_and_pin_the_api() -> None:
    headers = _auth_headers("ghs_tok")
    assert headers["Authorization"] == "Bearer ghs_tok"
    assert headers["Accept"] == "application/vnd.github+json"
    assert headers["X-GitHub-Api-Version"] == "2022-11-28"


def test_decode_contents_payload_yields_raw_text() -> None:
    assert _decode_contents_payload(_contents_payload(VALID_CONFIG)) == VALID_CONFIG


def test_decode_contents_payload_rejects_unexpected_encoding() -> None:
    with pytest.raises(ValueError, match="encoding"):
        _decode_contents_payload({"encoding": "none", "content": "x"})


@pytest.mark.asyncio
async def test_github_fetcher_returns_raw_config_text() -> None:
    """The real fetcher mints a token, reads the file, and returns the raw text."""
    auth = _FakeAuth(token="ghs_live")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=_contents_payload(VALID_CONFIG))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetch = github_config_fetcher(auth, client)
        raw = await fetch("owner/repo")

    assert raw == VALID_CONFIG
    assert auth.asked_for == "owner/repo"
    assert captured["auth"] == "Bearer ghs_live"
    assert captured["url"] == _repo_contents_url("owner/repo", RETINUE_CONFIG_PATH)


@pytest.mark.asyncio
async def test_github_fetcher_maps_404_to_not_opted_in() -> None:
    """A missing config file (404) reads as None, matching the not-opted-in fake."""
    auth = _FakeAuth()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        raw = await github_config_fetcher(auth, client)("owner/repo")

    assert raw is None


@pytest.mark.asyncio
async def test_github_fetcher_raises_on_transient_error() -> None:
    """A 5xx is raised, not swallowed: the job must retry, not read as opted out."""
    auth = _FakeAuth()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"message": "unavailable"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await github_config_fetcher(auth, client)("owner/repo")
