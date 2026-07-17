"""Blocked-by readiness: an issue is schedulable only when every blocker is closed.

The severity-pivot scheduler (:mod:`retinue.scheduler`) never dispatches a blocked issue.
An issue's blockers are the **union** of two sources (PRD #80):

* the ``## Blocked by`` block in its body — the ``#N`` references the slicer/humans write
  (:func:`parse_body_blockers`), and
* GitHub's native "blocked by" issue relations (the :meth:`ReadinessGh.native_blockers`
  seam).

A blocker is *satisfied* only when it is **closed**; an open — or unresolvable —
blocker keeps the issue out of scheduling until it closes. Only same-repo blockers are
considered (cross-repo blockers are out of scope for WS1). The readiness computation is
pure over an injected :class:`ReadinessGh` seam, so it is exercised with a fake — no live
``gh``, no network.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from retinue.gh import GhBytesRunner, GhCliError, auth_env, parse_json_array, run_gh_subprocess

logger = logging.getLogger(__name__)

# The ``## Blocked by`` block: the heading (case-insensitively) through to the next
# ``##`` heading or end of body. Every ``#N`` inside it — on the heading line or the
# lines below — is a blocker reference.
_BLOCK_RE = re.compile(
    r"^##\s+Blocked by\b.*?(?=^##\s|\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
_REF_RE = re.compile(r"#(\d+)")


def parse_body_blockers(body: str) -> list[int]:
    """Return the blocker issue numbers referenced in ``body``'s ``## Blocked by`` block.

    Matches the block the slicer and humans write: a ``## Blocked by`` heading followed
    by ``#N`` references (on the heading line or the lines under it, until the next
    ``##`` heading). Deduplicated in first-seen order; a body with no block yields an
    empty list.
    """
    match = _BLOCK_RE.search(body)
    if not match:
        return []
    refs = (int(n) for n in _REF_RE.findall(match.group(0)))
    return list(dict.fromkeys(refs))


@dataclass(frozen=True)
class BlockableIssue:
    """One candidate the scheduler wants to know the readiness of.

    Attributes:
        number: The issue number (self-references among its blockers are ignored).
        body: The issue body, scanned for the ``## Blocked by`` block.
    """

    number: int
    body: str


@runtime_checkable
class ReadinessGh(Protocol):
    """The gh queries readiness needs: native blockers and issue closed-state.

    A production implementation (:class:`GhCli`) reads GitHub's native issue-dependency
    relations and each blocker's state via ``gh``; tests inject a fake scripting both.
    """

    async def native_blockers(
        self, *, repo_full_name: str, issue_number: int
    ) -> list[int]:
        """Return the same-repo issue numbers GitHub records as blocking ``issue_number``."""
        ...

    async def is_closed(self, *, repo_full_name: str, issue_number: int) -> bool:
        """Whether ``issue_number`` is closed; a missing/unreadable issue reads as open."""
        ...


async def resolve_ready(
    candidates: Iterable[BlockableIssue],
    *,
    repo_full_name: str,
    gh: ReadinessGh,
) -> set[int]:
    """Return the numbers of the candidates whose every blocker is closed.

    For each candidate, the blocker set is the union of its body ``## Blocked by``
    references and GitHub's native relations (:meth:`ReadinessGh.native_blockers`), with
    self-references dropped. A candidate is ready when that set is empty or every blocker
    in it is closed. Each distinct blocker's closed-state is fetched at most once and
    cached across the whole candidate set, so a shared blocker is not re-queried.

    Args:
        candidates: The issues to test for readiness.
        repo_full_name: The target repo (same-repo blockers only).
        gh: The injected readiness gh seam.

    Returns:
        The subset of candidate numbers that are ready (unblocked).
    """
    candidates = list(candidates)
    blockers_by_issue: dict[int, set[int]] = {}
    for candidate in candidates:
        native = await gh.native_blockers(
            repo_full_name=repo_full_name, issue_number=candidate.number
        )
        union = set(parse_body_blockers(candidate.body)) | set(native)
        union.discard(candidate.number)
        blockers_by_issue[candidate.number] = union

    closed_cache: dict[int, bool] = {}
    for blockers in blockers_by_issue.values():
        for blocker in blockers:
            if blocker not in closed_cache:
                closed_cache[blocker] = await gh.is_closed(
                    repo_full_name=repo_full_name, issue_number=blocker
                )

    ready: set[int] = set()
    for number, blockers in blockers_by_issue.items():
        if all(closed_cache[b] for b in blockers):
            ready.add(number)
        else:
            open_blockers = sorted(b for b in blockers if not closed_cache[b])
            logger.info(
                "Issue #%d (%s) is blocked by open issue(s) %s; not schedulable",
                number,
                repo_full_name,
                open_blockers,
            )
    return ready


class GhCli:
    """Production :class:`ReadinessGh`: native relations + issue state via the ``gh`` CLI.

    ``native_blockers`` reads GitHub's issue-dependency relations through
    ``gh api repos/<repo>/issues/<n>/dependencies/blocked_by`` — an endpoint that returns
    an empty array when the issue has no native blockers, and whose absence (older GitHub,
    a 404) is read as "no native blockers" so the body ``## Blocked by`` union still
    governs. ``is_closed`` reads the blocker's state via ``gh issue view --json state``; a
    missing issue (404) reads as open, so an unresolvable blocker keeps the dependent out
    of scheduling. The subprocess spawn is the one impure edge, behind the injected
    ``runner`` seam.

    Args:
        token: The GitHub token placed in the child env as ``GH_TOKEN`` (``None`` uses
            ambient auth).
        runner: The injected argv runner; defaults to the real subprocess spawn.
    """

    def __init__(
        self, *, token: str | None = None, runner: GhBytesRunner | None = None
    ) -> None:
        self._token = token
        self._runner = runner or run_gh_subprocess

    async def native_blockers(
        self, *, repo_full_name: str, issue_number: int
    ) -> list[int]:
        """Return the same-repo issue numbers GitHub records as blocking ``issue_number``."""
        argv = [
            "gh",
            "api",
            f"repos/{repo_full_name}/issues/{issue_number}/dependencies/blocked_by",
        ]
        try:
            stdout = await self._runner(argv, auth_env(self._token))
        except GhCliError:
            # The dependencies endpoint is unavailable (older GitHub / not enabled): the
            # body ``## Blocked by`` union still governs, so treat as no native blockers.
            return []
        numbers: list[int] = []
        for entry in parse_json_array(stdout):
            if isinstance(entry, dict) and "number" in entry:
                numbers.append(int(entry["number"]))
        return numbers

    async def is_closed(self, *, repo_full_name: str, issue_number: int) -> bool:
        """Whether ``issue_number`` is closed; a missing/unreadable issue reads as open."""
        argv = [
            "gh",
            "issue",
            "view",
            str(issue_number),
            "--repo",
            repo_full_name,
            "--json",
            "state",
        ]
        try:
            stdout = await self._runner(argv, auth_env(self._token))
        except GhCliError:
            return False
        payload = _parse_state(stdout)
        return payload == "closed"


def _parse_state(stdout: bytes) -> str:
    """Parse ``gh issue view --json state`` into the lowercased state string."""
    try:
        payload = json.loads(stdout)
        return str(payload["state"]).lower()
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"gh issue view state payload is malformed: {stdout!r}") from exc
