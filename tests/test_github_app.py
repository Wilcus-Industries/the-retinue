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
from pathlib import Path
from typing import Any

import pytest

from retinue.github_app import (
    _LIST_PER_PAGE,
    GitHubAppCredentials,
    GitHubInstallationAuth,
    InstallationAuthError,
    InstallationToken,
    _access_tokens_url,
    _app_jwt_claims,
    _bearer_header,
    _clone_url,
    _installation_repositories_url,
    _installation_url,
    _installations_url,
    _parse_access_token_response,
    _parse_installation_id,
    _parse_installation_ids,
    _parse_installation_repositories,
    _token_header,
    build_installation_auth,
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


# --- installed-repository enumeration (the heartbeat's repo seam) --------------------


def test_installations_and_repositories_urls_carry_paging() -> None:
    base = "https://api.github.com"
    assert (
        _installations_url(base, page=2, per_page=100)
        == f"{base}/app/installations?per_page=100&page=2"
    )
    assert (
        _installation_repositories_url(base, page=1, per_page=100)
        == f"{base}/installation/repositories?per_page=100&page=1"
    )


def test_parse_installation_ids_keeps_integer_ids_and_skips_odd_entries() -> None:
    """A page yields its integer ids and the raw entry count; a bad entry is skipped.

    The raw count (here 4) — not the filtered list length — is what the paging loop
    compares against the page size, so a full page with a skipped entry still advances.
    """
    ids, raw_count = _parse_installation_ids([{"id": 7}, {"no": "id"}, {"id": 9}, "junk"])
    assert ids == [7, 9]
    assert raw_count == 4


def test_parse_installation_ids_raises_when_not_an_array() -> None:
    with pytest.raises(InstallationAuthError):
        _parse_installation_ids({"message": "Bad credentials"})


def test_parse_installation_repositories_reads_full_names() -> None:
    """The wrapped page yields each full_name and the raw count; a nameless entry is skipped.

    The raw count (here 3) drives page termination, so a full page that drops a malformed
    entry still pages on instead of stopping a page early.
    """
    names, raw_count = _parse_installation_repositories(
        {
            "total_count": 2,
            "repositories": [
                {"full_name": "owner/a"},
                {"no": "name"},
                {"full_name": "owner/b"},
            ],
        }
    )
    assert names == ["owner/a", "owner/b"]
    assert raw_count == 3


def test_parse_installation_repositories_raises_without_array() -> None:
    with pytest.raises(InstallationAuthError):
        _parse_installation_repositories({"message": "Not Found"})


# --- the adapter: full mint, header threading, caching, refresh, errors -------------


class _FakeHttp:
    """Records requests and replays scripted ``(status, body)`` responses in order.

    The body is the parsed JSON — a mapping for the token/installation reads, a JSON array
    for ``GET /app/installations`` — so it is typed ``object`` to match the HTTP seam.
    """

    def __init__(self, responses: list[tuple[int, object]]) -> None:
        self._responses = responses
        self.gets: list[tuple[str, Mapping[str, str]]] = []
        self.posts: list[tuple[str, Mapping[str, str]]] = []

    async def get(
        self, url: str, headers: Mapping[str, str]
    ) -> tuple[int, object]:
        self.gets.append((url, headers))
        return self._responses.pop(0)

    async def post(
        self, url: str, headers: Mapping[str, str]
    ) -> tuple[int, object]:
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
async def test_installed_repositories_lists_repos_across_installations() -> None:
    """The App-wide enumeration mints a token per installation and pages its repos.

    Flow per installation: list installations (app-JWT GET), mint an installation token
    (POST), then GET that installation's repositories — flattened across installations in
    listing order. The repos seed the heartbeat sweep.
    """
    http = _FakeHttp(
        [
            (200, [{"id": 7}, {"id": 8}]),  # list installations (short page)
            (201, {"token": "ghs_7", "expires_at": "2030-01-01T00:00:00Z"}),  # mint 7
            (200, {"repositories": [{"full_name": "owner/a"}]}),  # repos of 7
            (201, {"token": "ghs_8", "expires_at": "2030-01-01T00:00:00Z"}),  # mint 8
            (200, {"repositories": [{"full_name": "owner/b"}]}),  # repos of 8
        ]
    )

    repos = await _auth(http).installed_repositories()

    assert repos == ["owner/a", "owner/b"]
    # The installations listing is app-JWT authed; the repos listing rides the minted token.
    assert http.gets[0][0] == "https://api/app/installations?per_page=100&page=1"
    assert http.gets[0][1]["Authorization"] == "Bearer jwt:555:PEM"
    assert http.posts[0][0] == "https://api/app/installations/7/access_tokens"
    assert http.gets[1][0] == "https://api/installation/repositories?per_page=100&page=1"
    assert http.gets[1][1]["Authorization"] == "token ghs_7"


@pytest.mark.asyncio
async def test_installed_repositories_is_empty_with_no_installations() -> None:
    """No installations means an empty sweep — no token mint, no repos listing."""
    http = _FakeHttp([(200, [])])
    assert await _auth(http).installed_repositories() == []
    assert http.posts == []


@pytest.mark.asyncio
async def test_installed_repositories_raises_on_non_2xx_installations() -> None:
    http = _FakeHttp([(401, {"message": "Bad credentials"})])
    with pytest.raises(InstallationAuthError, match="401"):
        await _auth(http).installed_repositories()


def _full_installations_entries(start: int) -> list[dict[str, object]]:
    """A genuinely full installations page (``_LIST_PER_PAGE`` raw entries)."""
    return [{"id": start + offset} for offset in range(_LIST_PER_PAGE)]


def _full_repository_entries(start: int) -> list[dict[str, object]]:
    """A genuinely full repositories page's entries (``_LIST_PER_PAGE`` raw entries)."""
    return [
        {"full_name": f"owner/repo{start + offset}"}
        for offset in range(_LIST_PER_PAGE)
    ]


@pytest.mark.asyncio
async def test_list_installation_ids_full_page_with_malformed_entry_advances() -> None:
    """A full first page (100 raw entries) with one malformed entry still pages on.

    The malformed entry parses to 99 ids, but termination is decided from the raw entry
    count (100), so the loop fetches page 2 instead of stopping a page early.
    """
    first_page = _full_installations_entries(0)
    first_page[50] = {"no": "id"}  # malformed: a full raw page that filters to 99
    http = _FakeHttp([(200, first_page), (200, [{"id": 999}])])

    ids = await _auth(http)._list_installation_ids(_bearer_header("jwt"))

    expected = [entry["id"] for entry in first_page if "id" in entry] + [999]
    assert ids == expected
    assert len(http.gets) == 2  # advanced past the full page
    assert http.gets[1][0] == "https://api/app/installations?per_page=100&page=2"


@pytest.mark.asyncio
async def test_list_installation_ids_pages_until_short_page() -> None:
    """Two full pages then a short page returns every id across all pages, in order."""
    page1 = _full_installations_entries(0)
    page2 = _full_installations_entries(_LIST_PER_PAGE)
    page3 = [{"id": 9001}, {"id": 9002}]  # short page ends enumeration
    http = _FakeHttp([(200, page1), (200, page2), (200, page3)])

    ids = await _auth(http)._list_installation_ids(_bearer_header("jwt"))

    expected = [entry["id"] for entry in page1 + page2 + page3]
    assert ids == expected
    assert len(http.gets) == 3


@pytest.mark.asyncio
async def test_list_installation_repositories_full_page_with_malformed_entry_advances() -> None:
    """A full first repos page (100 raw entries) with one malformed entry still pages on."""
    first_entries = _full_repository_entries(0)
    first_entries[50] = {"no": "name"}  # malformed: a full raw page that filters to 99
    http = _FakeHttp(
        [
            (200, {"repositories": first_entries}),
            (200, {"repositories": [{"full_name": "owner/last"}]}),
        ]
    )

    names = await _auth(http)._list_installation_repositories("ghs_tok")

    expected = [
        entry["full_name"] for entry in first_entries if "full_name" in entry
    ] + ["owner/last"]
    assert names == expected
    assert len(http.gets) == 2
    assert (
        http.gets[1][0]
        == "https://api/installation/repositories?per_page=100&page=2"
    )


@pytest.mark.asyncio
async def test_list_installation_repositories_pages_until_short_page() -> None:
    """Two full repos pages then a short page returns every repo across all pages, in order."""
    page1 = _full_repository_entries(0)
    page2 = _full_repository_entries(_LIST_PER_PAGE)
    page3 = [{"full_name": "owner/tail"}]  # short page ends it
    http = _FakeHttp(
        [
            (200, {"repositories": page1}),
            (200, {"repositories": page2}),
            (200, {"repositories": page3}),
        ]
    )

    names = await _auth(http)._list_installation_repositories("ghs_tok")

    expected = [entry["full_name"] for entry in page1 + page2 + page3]
    assert names == expected
    assert len(http.gets) == 3


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


# --- the production factory: reads config + PEM, returns a wired adapter -------------


def test_build_installation_auth_reads_pem_and_carries_app_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The factory loads the PEM from the configured path and wires the app id in.

    Settings is fed from env (a tmp PEM file, no network); the returned adapter must be
    a real ``GitHubInstallationAuth`` whose credentials carry the configured app_id and
    the PEM text read off disk. ``httpx_edges`` is built but fires no request here.
    """
    pem = tmp_path / "app.pem"
    pem.write_text("-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----")
    monkeypatch.setenv("WEBHOOK_SECRET", "irrelevant")
    monkeypatch.setenv("GITHUB_APP_ID", "246810")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", str(pem))

    auth = build_installation_auth()

    assert isinstance(auth, GitHubInstallationAuth)
    assert auth._credentials.app_id == "246810"
    assert auth._credentials.private_key == pem.read_text()
