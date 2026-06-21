"""GitHub App installation auth: the token seam the done-check worker clones with.

The worker authenticates as the GitHub App *installation* (not a user) to mint a
short-lived token, then clones the target repo over HTTPS with that token. The real
JWT-signing + ``POST /app/installations/{id}/access_tokens`` exchange talks to GitHub,
so it lives behind the :class:`InstallationAuth` protocol: production wires a concrete
client, tests inject a fake. The orchestrator depends only on the protocol, which keeps
the auth->clone step exercisable without network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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
