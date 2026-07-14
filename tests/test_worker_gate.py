"""Tests for the PRD opt-in gate in the worker.

The gate decides, per dequeued PRD: is the repo opted in (a valid
``.github/retinue.yml`` present), and is this PRD new (not already deduped)? It
fetches the config text through an injected async callable so the GitHub fetch is
mocked out; the dedupe store is a real SQLite file in a tmp dir.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from retinue.dedupe import PrdDedupeStore
from retinue.github_app import InstallationToken
from retinue.queue import PrdJob
from retinue.worker import (
    GITHUB_API_BASE_URL,
    RETINUE_CONFIG_PATH,
    GateOutcome,
    _auth_headers,
    _contents_url,
    _decode_contents_payload,
    gate_prd,
    github_config_fetcher,
)

VALID_CONFIG = "staging_branch: release\nretry_cap: 2\n"


@pytest_asyncio.fixture()
async def store(tmp_path: Path) -> AsyncIterator[PrdDedupeStore]:
    """A dedupe store closed at teardown so no worker thread outlives the test."""
    dedupe = PrdDedupeStore(tmp_path / "dedupe.sqlite3")
    yield dedupe
    await dedupe.close()


@pytest.fixture()
def job() -> PrdJob:
    return PrdJob(repo_full_name="owner/repo", issue_number=7, action="opened")


async def _fetch_valid(repo_full_name: str) -> str | None:
    return VALID_CONFIG


async def _fetch_missing(repo_full_name: str) -> str | None:
    return None


async def _fetch_malformed(repo_full_name: str) -> str | None:
    return "staging_branch: [unclosed\n"


@pytest.mark.asyncio
async def test_valid_config_is_accepted_and_parsed(
    job: PrdJob, store: PrdDedupeStore
) -> None:
    """A repo with a valid config is accepted and the parsed config is returned."""
    result = await gate_prd(job, fetch_config=_fetch_valid, dedupe=store)
    assert result.outcome is GateOutcome.ACCEPTED
    assert result.config is not None
    assert result.config.staging_branch == "release"
    assert result.config.retry_cap == 2


@pytest.mark.asyncio
async def test_missing_config_is_skipped(
    job: PrdJob, store: PrdDedupeStore, caplog: pytest.LogCaptureFixture
) -> None:
    """A repo with no config file is an observable skip with no parsed config."""
    with caplog.at_level(logging.INFO, logger="retinue.worker"):
        result = await gate_prd(job, fetch_config=_fetch_missing, dedupe=store)
    assert result.outcome is GateOutcome.NOT_OPTED_IN
    assert result.config is None
    assert "not opted in" in caplog.text.lower()


@pytest.mark.asyncio
async def test_malformed_config_is_skipped_not_crashing(
    job: PrdJob, store: PrdDedupeStore, caplog: pytest.LogCaptureFixture
) -> None:
    """A malformed config is skipped and logged, never raising out of the gate."""
    with caplog.at_level(logging.WARNING, logger="retinue.worker"):
        result = await gate_prd(job, fetch_config=_fetch_malformed, dedupe=store)
    assert result.outcome is GateOutcome.MALFORMED_CONFIG
    assert result.config is None
    assert "malformed" in caplog.text.lower()


@pytest.mark.asyncio
async def test_duplicate_prd_is_skipped(
    job: PrdJob, store: PrdDedupeStore, caplog: pytest.LogCaptureFixture
) -> None:
    """A second gate of the same PRD is a dedupe skip; the first is accepted."""
    first = await gate_prd(job, fetch_config=_fetch_valid, dedupe=store)
    assert first.outcome is GateOutcome.ACCEPTED

    redelivery = PrdJob(repo_full_name="owner/repo", issue_number=7, action="labeled")
    with caplog.at_level(logging.INFO, logger="retinue.worker"):
        second = await gate_prd(redelivery, fetch_config=_fetch_valid, dedupe=store)
    assert second.outcome is GateOutcome.DUPLICATE
    assert second.config is None
    assert "duplicate" in caplog.text.lower()


@pytest.mark.asyncio
async def test_malformed_config_does_not_consume_dedupe_claim(
    job: PrdJob, store: PrdDedupeStore
) -> None:
    """A malformed config must not burn the dedupe slot, so a later fix can run.

    If the repo fixes its config and the PRD is redelivered, the gate should be
    able to accept it — the earlier malformed attempt must not have claimed the key.
    """
    first = await gate_prd(job, fetch_config=_fetch_malformed, dedupe=store)
    assert first.outcome is GateOutcome.MALFORMED_CONFIG

    retry = await gate_prd(job, fetch_config=_fetch_valid, dedupe=store)
    assert retry.outcome is GateOutcome.ACCEPTED


# --- The real GitHub-backed fetcher: pure parts + offline transport. ---------


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
    url = _contents_url("owner/repo")
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
    """The real fetcher mints a token, reads the file, and returns the same shape.

    The decoded text must match what the injected fake hands the gate, so the live
    response parses through ``load_repo_config`` identically.
    """
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
    assert captured["url"] == _contents_url("owner/repo")


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
