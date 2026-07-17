"""Durable run-state: the PR<->issue mapping, plus the gh seam it reads truth through.

The scheduler drain is stateless per pass, but the ad-hoc PR flow still needs a durable
ledger: when a build opens an ``issue-<N>`` -> target-branch PR, the PR number is recorded
here keyed by the issue, so a later merge webhook can resolve the PR back to the issue it
closes (:meth:`RunStateStore.round_for_pr`). The store mirrors the durable-SQLite style of
:class:`retinue.impl_retry.ImplRetryStore`.

The gh seam (:class:`GhRunner` / :class:`ReconcileGhRunner`) runs one ``gh`` argv and
returns its stdout; it is the generic subprocess seam the issue-facts fetch and the
PR-state query both spawn through, kept behind one callable so command assembly and
payload parsing are unit-testable without a real ``gh`` or network.
"""

from __future__ import annotations

import enum
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import aiosqlite

from retinue.gh import run_gh

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_state (
    prd_key   TEXT PRIMARY KEY,
    slices    TEXT NOT NULL DEFAULT '',
    pr_number INTEGER
)
"""


def run_state_key(repo_full_name: str, prd_number: int) -> str:
    """Return the run-state identity of a build: its repo and (issue) number.

    Args:
        repo_full_name: e.g. "owner/repo".
        prd_number: The tracking issue number keyed on (the ad-hoc issue itself).

    Returns:
        A stable ``"owner/repo#<n>"`` key.
    """
    return f"{repo_full_name}#{prd_number}"


class RunStateStore:
    """Durable run-state: the owned issue set and the PR<->issue mapping.

    One row per build, keyed by repo + number, holding the owned issue numbers and the
    PR number (recorded once a PR opens). Mirrors the durable-SQLite style of
    :class:`retinue.impl_retry.ImplRetryStore`.

    Args:
        db_path: Path to the SQLite database file. Created on first use; parent
            directories are created if missing.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)

    async def record_slices(
        self, *, repo_full_name: str, prd_number: int, issue_numbers: list[int]
    ) -> None:
        """Record the issue numbers a build owns (idempotent on re-run).

        The upsert overwrites any prior set for the key, so re-recording is a no-op
        rather than a duplicate. The PR mapping (if any) is preserved.
        """
        key = run_state_key(repo_full_name, prd_number)
        encoded = _encode_slices(issue_numbers)
        async with self._connect() as db:
            await db.execute(_SCHEMA)
            await db.execute(
                """
                INSERT INTO run_state (prd_key, slices) VALUES (?, ?)
                ON CONFLICT(prd_key) DO UPDATE SET slices = excluded.slices
                """,
                (key, encoded),
            )
            await db.commit()

    async def slices_of(self, *, repo_full_name: str, prd_number: int) -> list[int]:
        """Return the recorded issue numbers for a key (empty if unseen)."""
        key = run_state_key(repo_full_name, prd_number)
        async with self._connect() as db:
            await db.execute(_SCHEMA)
            async with db.execute(
                "SELECT slices FROM run_state WHERE prd_key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
        return _decode_slices(row[0]) if row is not None else []

    async def record_pr(
        self, *, repo_full_name: str, prd_number: int, pr_number: int
    ) -> None:
        """Record the PR number opened for a build (the PR<->issue mapping).

        The upsert preserves any recorded issue set, so recording the PR after the
        issues does not lose them.
        """
        key = run_state_key(repo_full_name, prd_number)
        async with self._connect() as db:
            await db.execute(_SCHEMA)
            await db.execute(
                """
                INSERT INTO run_state (prd_key, pr_number) VALUES (?, ?)
                ON CONFLICT(prd_key) DO UPDATE SET pr_number = excluded.pr_number
                """,
                (key, pr_number),
            )
            await db.commit()

    async def pr_of(self, *, repo_full_name: str, prd_number: int) -> int | None:
        """Return the recorded PR number for a key (None if none recorded)."""
        key = run_state_key(repo_full_name, prd_number)
        async with self._connect() as db:
            await db.execute(_SCHEMA)
            async with db.execute(
                "SELECT pr_number FROM run_state WHERE prd_key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    async def round_for_pr(
        self, *, repo_full_name: str, pr_number: int
    ) -> tuple[int, list[int]] | None:
        """Return the ``(issue_number, owned_issues)`` a PR maps to, or None if unknown.

        The reverse of :meth:`record_pr`: a merged-PR event arrives keyed by PR number,
        but the reap needs the parent issue and its owned set. Scoped to the repo so a PR
        number is never confused across repos.
        """
        prefix = f"{repo_full_name}#"
        async with self._connect() as db:
            await db.execute(_SCHEMA)
            async with db.execute(
                "SELECT prd_key, slices FROM run_state "
                "WHERE pr_number = ? AND prd_key LIKE ?",
                (pr_number, f"{prefix}%"),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        prd_number = int(str(row[0]).rsplit("#", 1)[-1])
        return prd_number, _decode_slices(row[1])

    async def delete_round(self, *, repo_full_name: str, prd_number: int) -> None:
        """Delete a build's row — its terminal event, so no stale mapping lingers.

        Deleting a row never recorded is a no-op, so the cleanup is safe to repeat.
        """
        key = run_state_key(repo_full_name, prd_number)
        async with self._connect() as db:
            await db.execute(_SCHEMA)
            await db.execute("DELETE FROM run_state WHERE prd_key = ?", (key,))
            await db.commit()

    def _connect(self) -> aiosqlite.Connection:
        """Open a fresh DB connection, ensuring the parent dir exists first."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        return aiosqlite.connect(self._db_path)


def _encode_slices(issue_numbers: list[int]) -> str:
    """Encode issue numbers as a comma-separated string for one TEXT column."""
    return ",".join(str(number) for number in issue_numbers)


def _decode_slices(encoded: str) -> list[int]:
    """Decode a comma-separated issue string back into issue numbers."""
    return [int(part) for part in encoded.split(",") if part]


class PrState(enum.Enum):
    """The lifecycle state GitHub reports for a pull request."""

    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"


class GhRunner(Protocol):
    """Runs one ``gh`` invocation and returns its stdout. The gh-subprocess seam.

    The actual subprocess spawn (and its installation-token env) lives behind this one
    callable so command-assembly and payload-parsing are unit-testable without spawning
    a process or touching the network. A production runner shells out to ``gh`` with
    :func:`gh_env` in the child environment; tests inject a fake that returns canned JSON
    and records the argv it was handed.
    """

    async def __call__(self, argv: list[str]) -> str:
        """Run ``gh`` with ``argv`` and return its captured stdout (raises on failure)."""
        ...


def gh_env(token: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build the child-process environment that authenticates ``gh`` as the installation.

    ``gh`` reads its credential from ``GH_TOKEN`` (preferred over ``GITHUB_TOKEN``); the
    installation access token minted by :class:`retinue.github_app.InstallationAuth` goes
    there. ``GH_PROMPT_DISABLED`` keeps a non-interactive worker from ever blocking on a
    prompt.

    Args:
        token: The installation access token to authenticate ``gh`` with.
        base_env: The environment to extend (e.g. ``os.environ``); defaults to empty so
            the build is pure and testable. A copy is returned; the input is untouched.

    Returns:
        A new env dict carrying ``GH_TOKEN`` and the non-interactive flags.
    """
    env = dict(base_env or {})
    env["GH_TOKEN"] = token
    env["GH_PROMPT_DISABLED"] = "1"
    return env


class ReconcileGhRunner:
    """Production :class:`GhRunner`: one ``gh`` argv, token-authenticated, stdout back.

    Delegates the spawn to :func:`retinue.gh.run_gh` with :func:`gh_env` layered over the
    ambient environment, so a failed query raises (:class:`retinue.gh.GhCommandError`)
    rather than reading as empty truth.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    async def __call__(self, argv: list[str]) -> str:
        """Run ``gh`` with ``argv`` and return its captured stdout (raises on failure)."""
        return await run_gh(argv, gh_env(self._token, dict(os.environ)))


def _pr_state_argv(repo_full_name: str, pr_number: int) -> list[str]:
    """Assemble the ``gh`` argv reading one PR's lifecycle state as JSON."""
    return [
        "pr",
        "view",
        str(pr_number),
        "--repo",
        repo_full_name,
        "--json",
        "state",
    ]


def _parse_pr_state(stdout: str) -> PrState:
    """Parse a ``pr view --json state`` payload into a :class:`PrState`."""
    payload = json.loads(stdout)
    return PrState(str(payload["state"]).lower())


@dataclass(frozen=True)
class GhCliReconcile:
    """Reads a PR's lifecycle state by shelling out to ``gh``. The reap PR-state query.

    Assembles the ``gh pr view`` argv, runs it through the injected :class:`GhRunner`, and
    parses the JSON stdout into a :class:`PrState`. The subprocess spawn and its
    installation-token env live in the runner (see :func:`gh_env`), so this class is pure
    command-assembly plus payload-parsing.

    Args:
        runner: The injected gh-subprocess seam that runs an argv and returns stdout.
    """

    runner: GhRunner

    async def pr_state(self, *, repo_full_name: str, pr_number: int) -> PrState:
        """Return the lifecycle state GitHub reports for ``pr_number``."""
        stdout = await self.runner(_pr_state_argv(repo_full_name, pr_number))
        return _parse_pr_state(stdout)
