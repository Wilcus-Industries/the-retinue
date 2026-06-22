"""GitHub App installation auth: the token seam the done-check worker clones with.

The worker authenticates as the GitHub App *installation* (not a user) to mint a
short-lived token, then clones the target repo over HTTPS with that token. The real
JWT-signing + ``POST /app/installations/{id}/access_tokens`` exchange talks to GitHub,
so it lives behind the :class:`InstallationAuth` protocol: production wires a concrete
client, tests inject a fake. The orchestrator depends only on the protocol, which keeps
the auth->clone step exercisable without network.

The production adapter is :class:`GitHubInstallationAuth`. It signs a short-lived app
JWT with the app's RSA private key, resolves the repo's installation id, exchanges the
JWT for an installation access token, and caches that token until shortly before it
expires so repeated checks against the same repo reuse one mint. The two impure edges —
RSA signing and the HTTP calls — are themselves injected (``sign_jwt`` and
``http_post``/``http_get``) so the caching, header assembly, URL building, and payload
parsing are all exercised without network or ``cryptography``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, TypedDict


@dataclass(frozen=True)
class InstallationToken:
    """A minted installation access token and the clone URL it authorises.

    Attributes:
        token: The short-lived installation access token (an opaque secret).
        clone_url: The HTTPS clone URL with the token embedded, ready for ``git clone``.
    """

    token: str
    clone_url: str


class InstallationAuth(Protocol):
    """Mints an installation token for a repo. The auth->clone seam.

    A production implementation signs a GitHub App JWT and exchanges it for an
    installation access token scoped to ``repo_full_name``; tests inject a fake that
    returns a canned token. Implementations raise on auth failure rather than
    returning a sentinel, so a doomed clone never starts.
    """

    async def installation_token(self, repo_full_name: str) -> InstallationToken:
        """Return a fresh installation token authorised to clone ``repo_full_name``."""
        ...


# --- credentials --------------------------------------------------------------------


@dataclass(frozen=True)
class GitHubAppCredentials:
    """The static identity of the GitHub App used to mint installation tokens.

    Attributes:
        app_id: The GitHub App's numeric id, the JWT ``iss`` claim.
        private_key: The app's RSA private key in PEM form, used to sign the app JWT.
        api_base_url: The REST API root, e.g. ``https://api.github.com`` (override for
            GitHub Enterprise). No trailing slash.
    """

    app_id: str
    private_key: str
    api_base_url: str = "https://api.github.com"


# --- pure helpers: claims, headers, URLs, payload parsing ---------------------------

# Refresh a cached token this many seconds before its stated expiry, so a token is
# never handed out moments from expiring mid-clone.
_EXPIRY_SKEW_SECONDS = 60
# GitHub rejects an app JWT whose lifetime exceeds 10 minutes; stay safely under it.
_JWT_LIFETIME_SECONDS = 9 * 60


class AppJwtClaims(TypedDict):
    """The registered claims of a GitHub App JWT: issued-at, expiry, and issuer."""

    iat: int
    exp: int
    iss: str


def _app_jwt_claims(app_id: str, *, now: float) -> AppJwtClaims:
    """Build the registered claims for a GitHub App JWT.

    ``iat`` is backdated 60s to tolerate minor clock skew between us and GitHub, and
    ``exp`` stays under GitHub's 10-minute ceiling. ``iss`` is the app id.
    """
    issued_at = int(now) - 60
    return {
        "iat": issued_at,
        "exp": issued_at + _JWT_LIFETIME_SECONDS,
        "iss": app_id,
    }


def _bearer_header(app_jwt: str) -> dict[str, str]:
    """Headers for app-authed requests (resolve installation, mint token)."""
    return {
        "Authorization": f"Bearer {app_jwt}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _token_header(token: str) -> dict[str, str]:
    """Headers for installation-authed requests, made with a minted access token."""
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _installation_url(api_base_url: str, repo_full_name: str) -> str:
    """REST URL resolving the installation that has access to ``owner/repo``."""
    return f"{api_base_url}/repos/{repo_full_name}/installation"


def _access_tokens_url(api_base_url: str, installation_id: int) -> str:
    """REST URL that mints an access token for a given installation id."""
    return f"{api_base_url}/app/installations/{installation_id}/access_tokens"


def _clone_url(repo_full_name: str, token: str) -> str:
    """Embed an installation token into the repo's HTTPS clone URL.

    GitHub accepts the literal username ``x-access-token`` with the installation
    token as the password for token-authenticated HTTPS clones.
    """
    return f"https://x-access-token:{token}@github.com/{repo_full_name}.git"


def _parse_installation_id(payload: Mapping[str, Any]) -> int:
    """Pull the installation id out of a ``GET .../installation`` response body.

    Raises:
        InstallationAuthError: The body has no integer ``id`` (e.g. the app is not
            installed on the repo, so GitHub returned an error shape).
    """
    installation_id = payload.get("id")
    if not isinstance(installation_id, int):
        raise InstallationAuthError(
            f"no installation id in response: {dict(payload)!r}"
        )
    return installation_id


def _parse_access_token_response(payload: Mapping[str, Any]) -> tuple[str, float]:
    """Parse a ``POST .../access_tokens`` body into ``(token, expires_at_epoch)``.

    GitHub returns ``token`` and an ISO-8601 ``expires_at`` (UTC, e.g.
    ``2026-06-22T12:34:56Z``). The expiry is converted to an epoch second so the
    cache can compare it against a monotone-ish wall clock.

    Raises:
        InstallationAuthError: ``token`` or ``expires_at`` is missing or malformed.
    """
    token = payload.get("token")
    expires_at = payload.get("expires_at")
    if not isinstance(token, str) or not token:
        raise InstallationAuthError(f"no token in response: {dict(payload)!r}")
    if not isinstance(expires_at, str):
        raise InstallationAuthError(
            f"no expires_at in response: {dict(payload)!r}"
        )
    try:
        parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InstallationAuthError(
            f"malformed expires_at {expires_at!r}: {exc}"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return token, parsed.timestamp()


# --- errors -------------------------------------------------------------------------


class InstallationAuthError(RuntimeError):
    """Auth could not mint a token: bad credentials, no installation, or a bad reply.

    Raised rather than returning a sentinel so a doomed clone never starts (the
    contract :class:`InstallationAuth` documents).
    """


# --- injected impure edges ----------------------------------------------------------

# Signs JWT claims into a compact RS256 token using the app's PEM private key. The
# real implementation (PyJWT + cryptography) is injected so the adapter's caching and
# parsing stay testable without an RSA backend.
JwtSigner = Callable[[Mapping[str, Any], str], str]

# Performs one HTTP request and returns ``(status_code, parsed_json_body)``. Both the
# GET (resolve installation) and POST (mint token) go through this seam so the adapter
# is exercised with a recording fake instead of a live GitHub.
HttpGet = Callable[[str, Mapping[str, str]], Awaitable[tuple[int, Mapping[str, Any]]]]
HttpPost = Callable[[str, Mapping[str, str]], Awaitable[tuple[int, Mapping[str, Any]]]]


def sign_app_jwt(claims: Mapping[str, Any], private_key: str) -> str:
    """Sign ``claims`` as an RS256 JWT with the app's PEM ``private_key``.

    The default :class:`GitHubInstallationAuth` signer. PyJWT's RS256 backend needs
    ``cryptography`` at call time; the import is local so neither importing this module
    nor the unit tests (which inject a fake signer) require it.

    Raises:
        InstallationAuthError: The key cannot sign (e.g. ``cryptography`` is missing or
            the PEM is invalid).
    """
    import jwt  # local: only the production signing path needs PyJWT + cryptography

    try:
        return jwt.encode(dict(claims), private_key, algorithm="RS256")
    except Exception as exc:  # PyJWT raises a family of errors; all mean "cannot sign"
        raise InstallationAuthError(f"failed to sign app JWT: {exc}") from exc


# --- the production adapter ---------------------------------------------------------


@dataclass
class _CachedToken:
    token: str
    expires_at: float
    clone_url: str


class GitHubInstallationAuth:
    """Mints and caches GitHub App installation tokens. The real :class:`InstallationAuth`.

    For a repo it signs an app JWT, resolves the repo's installation id, mints an
    installation access token, and caches it per repo until shortly before expiry so a
    burst of done-checks against the same repo reuses one mint. RSA signing, the HTTP
    GET, and the HTTP POST are all injected so caching, header assembly, URL building,
    and payload parsing are tested without network or ``cryptography``; the
    module-level :func:`sign_app_jwt` and the httpx-backed helpers wire the real edges
    in production.
    """

    def __init__(
        self,
        credentials: GitHubAppCredentials,
        *,
        http_get: HttpGet,
        http_post: HttpPost,
        sign_jwt: JwtSigner = sign_app_jwt,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._credentials = credentials
        self._http_get = http_get
        self._http_post = http_post
        self._sign_jwt = sign_jwt
        self._clock = clock
        self._cache: dict[str, _CachedToken] = {}

    async def installation_token(self, repo_full_name: str) -> InstallationToken:
        """Return a token for ``repo_full_name``, reusing the cache when still fresh.

        Raises:
            InstallationAuthError: Signing failed, the app is not installed on the
                repo, GitHub returned a non-2xx status, or the reply was malformed.
        """
        cached = self._cache.get(repo_full_name)
        if cached is not None and not self._is_expiring(cached.expires_at):
            return InstallationToken(token=cached.token, clone_url=cached.clone_url)

        minted = await self._mint(repo_full_name)
        self._cache[repo_full_name] = minted
        return InstallationToken(token=minted.token, clone_url=minted.clone_url)

    def _is_expiring(self, expires_at: float) -> bool:
        return self._clock() >= expires_at - _EXPIRY_SKEW_SECONDS

    async def _mint(self, repo_full_name: str) -> _CachedToken:
        app_jwt = self._sign_jwt(
            _app_jwt_claims(self._credentials.app_id, now=self._clock()),
            self._credentials.private_key,
        )
        bearer = _bearer_header(app_jwt)
        base = self._credentials.api_base_url

        status, body = await self._http_get(
            _installation_url(base, repo_full_name), bearer
        )
        _raise_for_status(status, body, "resolve installation")
        installation_id = _parse_installation_id(body)

        status, body = await self._http_post(
            _access_tokens_url(base, installation_id), bearer
        )
        _raise_for_status(status, body, "mint access token")
        token, expires_at = _parse_access_token_response(body)

        return _CachedToken(
            token=token,
            expires_at=expires_at,
            clone_url=_clone_url(repo_full_name, token),
        )


def _raise_for_status(
    status: int, body: Mapping[str, Any], operation: str
) -> None:
    """Turn a non-2xx GitHub response into :class:`InstallationAuthError`."""
    if 200 <= status < 300:
        return
    message = body.get("message") if isinstance(body, Mapping) else None
    raise InstallationAuthError(
        f"GitHub returned {status} on {operation}: {message or dict(body)!r}"
    )


def httpx_edges(
    timeout_seconds: float = 10.0,
) -> tuple[HttpGet, HttpPost]:
    """Build the production httpx-backed ``(http_get, http_post)`` for the adapter.

    Each call opens a short-lived async client, issues one request, and returns the
    status code plus the parsed JSON body. A non-JSON body is reported as an empty
    mapping so the caller's status check still surfaces the failure.
    """
    import httpx  # local: only production wiring needs the HTTP client

    def _parse_body(response: httpx.Response) -> Mapping[str, Any]:
        try:
            parsed = response.json()
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, Mapping) else {}

    async def http_get(
        url: str, headers: Mapping[str, str]
    ) -> tuple[int, Mapping[str, Any]]:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(url, headers=dict(headers))
        return response.status_code, _parse_body(response)

    async def http_post(
        url: str, headers: Mapping[str, str]
    ) -> tuple[int, Mapping[str, Any]]:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(url, headers=dict(headers))
        return response.status_code, _parse_body(response)

    return http_get, http_post
