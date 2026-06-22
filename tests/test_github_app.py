"""Tests for the real GitHub App installation-auth adapter (the InstallationAuth seam).

These cover the adapter's pure, parseable parts — JWT claims, auth-header assembly,
URL/command construction, and payload parsing — plus the caching/refresh logic, all
driven through injected fakes. Nothing here touches the network, ``cryptography``, or a
live GitHub; the RSA signer and the two HTTP edges are injected. The fake-backed
end-to-end orchestration test lives in ``tests/test_done_check.py`` and is the contract
this adapter must satisfy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from retinue.github_app import (
    GitHubAppCredentials,
    GitHubInstallationAuth,
    InstallationAuthError,
    InstallationToken,
    _access_tokens_url,
    _app_jwt_claims,
    _bearer_header,
    _clone_url,
    _installation_url,
    _parse_access_token_response,
    _parse_installation_id,
    _token_header,
)

# --- pure helpers: claims, headers, URLs --------------------------------------------


def test_app_jwt_claims_backdates_iat_and_caps_lifetime_under_ten_minutes() -> None:
    """iat is backdated 60s for clock skew; the lifetime stays under GitHub's 10m cap."""
    claims = _app_jwt_claims("12345", now=1_000_000.0)
    assert claims["iss"] == "12345"
    assert claims["iat"] == 1_000_000 - 60
    lifetime = claims["exp"] - claims["iat"]
    assert 0 < lifetime < 600


def test_bearer_header_uses_app_jwt() -> None:
    header = _bearer_header("the.jwt.token")
    assert header["Authorization"] == "Bearer the.jwt.token"
    assert header["Accept"] == "application/vnd.github+json"
    assert header["X-GitHub-Api-Version"] == "2022-11-28"


def test_token_header_uses_installation_token() -> None:
    assert _token_header("ghs_abc")["Authorization"] == "token ghs_abc"


def test_installation_and_access_token_urls() -> None:
    base = "https://api.github.com"
    assert _installation_url(base, "owner/repo") == f"{base}/repos/owner/repo/installation"
    assert _access_tokens_url(base, 42) == f"{base}/app/installations/42/access_tokens"


def test_clone_url_embeds_token_as_x_access_token() -> None:
    """The clone URL matches the contract the fake/done-check depend on."""
    assert (
        _clone_url("owner/repo", "ghs_tok")
        == "https://x-access-token:ghs_tok@github.com/owner/repo.git"
    )


# --- payload parsing ----------------------------------------------------------------


def test_parse_installation_id_reads_integer_id() -> None:
    assert _parse_installation_id({"id": 99, "app_id": 1}) == 99


def test_parse_installation_id_raises_when_app_not_installed() -> None:
    """A GitHub error body (no integer id) is an auth error, not a silent 0."""
    with pytest.raises(InstallationAuthError):
        _parse_installation_id({"message": "Not Found"})


def test_parse_access_token_response_returns_token_and_epoch_expiry() -> None:
    token, expires_at = _parse_access_token_response(
        {"token": "ghs_live", "expires_at": "2026-06-22T12:00:00Z"}
    )
    assert token == "ghs_live"
    # 2026-06-22T12:00:00Z as a UTC epoch second.
    assert expires_at == pytest.approx(1_782_129_600.0)


@pytest.mark.parametrize(
    "payload",
    [
        {"expires_at": "2026-06-22T12:00:00Z"},  # no token
        {"token": "", "expires_at": "2026-06-22T12:00:00Z"},  # empty token
        {"token": "ghs_x"},  # no expires_at
        {"token": "ghs_x", "expires_at": "not-a-date"},  # malformed expiry
    ],
)
def test_parse_access_token_response_raises_on_bad_payload(
    payload: Mapping[str, Any],
) -> None:
    with pytest.raises(InstallationAuthError):
        _parse_access_token_response(payload)


# --- the adapter: full mint, header threading, caching, refresh, errors -------------


class _FakeHttp:
    """Records requests and replays scripted ``(status, body)`` responses in order."""

    def __init__(self, responses: list[tuple[int, Mapping[str, Any]]]) -> None:
        self._responses = responses
        self.gets: list[tuple[str, Mapping[str, str]]] = []
        self.posts: list[tuple[str, Mapping[str, str]]] = []

    async def get(
        self, url: str, headers: Mapping[str, str]
    ) -> tuple[int, Mapping[str, Any]]:
        self.gets.append((url, headers))
        return self._responses.pop(0)

    async def post(
        self, url: str, headers: Mapping[str, str]
    ) -> tuple[int, Mapping[str, Any]]:
        self.posts.append((url, headers))
        return self._responses.pop(0)


def _creds() -> GitHubAppCredentials:
    return GitHubAppCredentials(app_id="555", private_key="PEM", api_base_url="https://api")


def _fake_signer(claims: Mapping[str, Any], private_key: str) -> str:
    """A signer that records its inputs in the returned token (no real RSA)."""
    return f"jwt:{claims['iss']}:{private_key}"


def _auth(
    http: _FakeHttp, *, clock_value: float = 1_000.0
) -> GitHubInstallationAuth:
    return GitHubInstallationAuth(
        _creds(),
        http_get=http.get,
        http_post=http.post,
        sign_jwt=_fake_signer,
        clock=lambda: clock_value,
    )


@pytest.mark.asyncio
async def test_mint_signs_resolves_installation_and_threads_bearer() -> None:
    """A cold mint signs a JWT, GETs the installation, POSTs to mint, bearer-authed."""
    http = _FakeHttp(
        [
            (200, {"id": 7}),
            (201, {"token": "ghs_minted", "expires_at": "2030-01-01T00:00:00Z"}),
        ]
    )
    token = await _auth(http).installation_token("owner/repo")

    assert token == InstallationToken(
        token="ghs_minted",
        clone_url="https://x-access-token:ghs_minted@github.com/owner/repo.git",
    )
    # GET resolves the installation, POST mints — both with the app-JWT bearer.
    assert http.gets[0][0] == "https://api/repos/owner/repo/installation"
    assert http.posts[0][0] == "https://api/app/installations/7/access_tokens"
    assert http.gets[0][1]["Authorization"] == "Bearer jwt:555:PEM"
    assert http.posts[0][1]["Authorization"] == "Bearer jwt:555:PEM"


@pytest.mark.asyncio
async def test_token_is_cached_until_near_expiry() -> None:
    """A second call inside the validity window reuses the mint — no new HTTP."""
    http = _FakeHttp(
        [
            (200, {"id": 7}),
            (201, {"token": "ghs_a", "expires_at": "2030-01-01T00:00:00Z"}),
        ]
    )
    auth = _auth(http, clock_value=1_000.0)

    first = await auth.installation_token("owner/repo")
    second = await auth.installation_token("owner/repo")

    assert first == second
    assert len(http.posts) == 1  # not re-minted


@pytest.mark.asyncio
async def test_token_refreshes_once_inside_the_expiry_skew() -> None:
    """Within the skew window of expiry the cache refreshes with a fresh mint."""
    # First token expires at epoch 2000; second mint is far in the future.
    http = _FakeHttp(
        [
            (200, {"id": 7}),
            (201, {"token": "ghs_old", "expires_at": "1970-01-01T00:33:20Z"}),  # =2000s
            (200, {"id": 7}),
            (201, {"token": "ghs_new", "expires_at": "2030-01-01T00:00:00Z"}),
        ]
    )
    auth = GitHubInstallationAuth(
        _creds(),
        http_get=http.get,
        http_post=http.post,
        sign_jwt=_fake_signer,
        # 1990 is within 60s of the 2000s expiry, so the cached token is "expiring".
        clock=lambda: 1_990.0,
    )

    first = await auth.installation_token("owner/repo")
    second = await auth.installation_token("owner/repo")

    assert first.token == "ghs_old"
    assert second.token == "ghs_new"
    assert len(http.posts) == 2  # re-minted


@pytest.mark.asyncio
async def test_non_2xx_on_installation_lookup_raises() -> None:
    http = _FakeHttp([(404, {"message": "Not Found"})])
    with pytest.raises(InstallationAuthError, match="404"):
        await _auth(http).installation_token("owner/repo")


@pytest.mark.asyncio
async def test_non_2xx_on_mint_raises() -> None:
    http = _FakeHttp([(200, {"id": 7}), (403, {"message": "Forbidden"})])
    with pytest.raises(InstallationAuthError, match="403"):
        await _auth(http).installation_token("owner/repo")


@pytest.mark.asyncio
async def test_signer_failure_propagates_as_auth_error() -> None:
    """A signer that cannot sign (e.g. missing cryptography) surfaces as auth failure."""

    def broken_signer(claims: Mapping[str, Any], private_key: str) -> str:
        raise InstallationAuthError("no cryptography backend")

    auth = GitHubInstallationAuth(
        _creds(),
        http_get=_FakeHttp([]).get,
        http_post=_FakeHttp([]).post,
        sign_jwt=broken_signer,
        clock=lambda: 1_000.0,
    )
    with pytest.raises(InstallationAuthError, match="cryptography"):
        await auth.installation_token("owner/repo")
