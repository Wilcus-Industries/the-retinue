"""Resume-from-GitHub reconciliation on worker restart (issue #13).

The retinue can die at any phase of an in-flight PRD round — mid-build, after the
staging PR opened, in the loopback. On restart it must continue only the *unfinished*
work and never duplicate an issue, branch, or PR. This module computes which phase to
resume at and hands the caller a typed :class:`ReconcileResult` to route on.

GitHub is the source of truth. The reconciler reads it through the injected
:class:`ReconcileGh` seam — which slice issues are closed, which ``issue-<N>`` branches
are merged, and whether the ``retinue/prd-<n>`` -> staging PR exists. The SQLite
:class:`RunStateStore` is only a secondary ledger: it remembers which slices a PRD round
owns and the PR<->PRD mapping once a PR opens, so a restart knows what to reconcile.
It mirrors the durable-SQLite style of :class:`retinue.dedupe.PrdDedupeStore` /
:class:`retinue.impl_retry.ImplRetryStore`.

The resume decision, in order (GitHub-truth first, so a lagged ledger never re-does
landed work):

1. **PR exists** -> resume at :attr:`ResumePhase.LOOPBACK`. The PR's existence proves
   the build round finished; we re-enter the heimdall loopback rather than rebuild.
2. **every slice finished, no PR** -> resume at :attr:`ResumePhase.PR_OPEN`. The build
   is done but the PR never opened, so we open it (the PR-opener is idempotent behind
   its own prechecks).
3. **some slice unfinished** -> resume at :attr:`ResumePhase.BUILD`, handing the build
   only the unfinished slices (issue still open AND branch not merged), with their
   ``blocked_by`` graph intact so :func:`retinue.orchestrator.build_prd` keeps order.
4. **no slices and no PR** -> :attr:`ResumePhase.DONE`: nothing to resume.

A slice counts as *finished* when GitHub shows EITHER its issue closed OR its
``issue-<N>`` branch merged — either side proves the work landed, so a crash between the
merge and the issue-close still resumes correctly (no duplicate branch, no duplicate
issue). Every gh query is injected and faked in tests; the run-state lives in a temp
SQLite file — no real ``gh``, no network.
"""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import aiosqlite

from retinue.orchestrator import PrdSlice, integration_branch

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_state (
    prd_key   TEXT PRIMARY KEY,
    slices    TEXT NOT NULL DEFAULT '',
    pr_number INTEGER
)
"""


def run_state_key(repo_full_name: str, prd_number: int) -> str:
    """Return the run-state identity of a PRD round: its repo and PRD number.

    Args:
        repo_full_name: e.g. "owner/repo".
        prd_number: The PRD's tracking issue number.

    Returns:
        A stable ``"owner/repo#<prd>"`` key.
    """
    return f"{repo_full_name}#{prd_number}"


class RunStateStore:
    """Durable per-PRD run-state: the owned slice set and the PR<->PRD mapping.

    GitHub is the source of truth for *what happened*; this store only remembers *what
    the round owns* so a restart knows which slices to reconcile and which PR maps to
    the PRD. One row per PRD round, keyed by repo + PRD number, holding the slice issue
    numbers (recorded when the round begins) and the staging PR number (recorded once a
    PR opens). Mirrors the durable-SQLite style of :class:`retinue.dedupe.PrdDedupeStore`.

    Args:
        db_path: Path to the SQLite database file. Created on first use; parent
            directories are created if missing.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)

    async def record_slices(
        self, *, repo_full_name: str, prd_number: int, issue_numbers: list[int]
    ) -> None:
        """Record the slice issue numbers a PRD round owns (idempotent on re-run).

        The upsert overwrites any prior slice set for the PRD, so re-recording the same
        round is a no-op rather than a duplicate. The PR mapping (if any) is preserved.

        Args:
            repo_full_name: e.g. "owner/repo".
            prd_number: The PRD's tracking issue number.
            issue_numbers: The slice issue numbers the round owns.
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
        """Return the recorded slice issue numbers for a PRD (empty if unseen).

        Args:
            repo_full_name: e.g. "owner/repo".
            prd_number: The PRD's tracking issue number.

        Returns:
            The recorded slice issue numbers, or ``[]`` for a PRD never recorded.
        """
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
        """Record the staging PR number opened for a PRD (the PR<->PRD mapping).

        The upsert preserves any recorded slice set, so recording the PR after the
        slices does not lose them.

        Args:
            repo_full_name: e.g. "owner/repo".
            prd_number: The PRD's tracking issue number.
            pr_number: The opened staging PR number.
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
        """Return the recorded staging PR number for a PRD (None if none recorded).

        Args:
            repo_full_name: e.g. "owner/repo".
            prd_number: The PRD's tracking issue number.

        Returns:
            The recorded PR number, or ``None`` when no PR has been recorded.
        """
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
        """Return the ``(prd_number, slice_numbers)`` a PR maps to, or None if unknown.

        The reverse of :meth:`record_pr`: a merged-PR or review event arrives keyed by PR
        number, but the loopback and reap need the parent PRD and its owned slice set.
        Scoped to the repo so a PR number is never confused across repos.

        Args:
            repo_full_name: e.g. "owner/repo".
            pr_number: The staging PR number recorded for some PRD round.

        Returns:
            ``(prd_number, slice_numbers)`` when a row maps the PR, else ``None``.
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

    def _connect(self) -> aiosqlite.Connection:
        """Open a fresh DB connection, ensuring the parent dir exists first."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        return aiosqlite.connect(self._db_path)


def _encode_slices(issue_numbers: list[int]) -> str:
    """Encode slice issue numbers as a comma-separated string for one TEXT column."""
    return ",".join(str(number) for number in issue_numbers)


def _decode_slices(encoded: str) -> list[int]:
    """Decode a comma-separated slice string back into issue numbers."""
    return [int(part) for part in encoded.split(",") if part]


class ReconcileGh(Protocol):
    """The gh queries reconciliation reads GitHub truth through. The reconcile gh seam.

    A production implementation runs ``gh`` against the target repo (an issue-state
    lookup, a "is this branch merged into staging" query, and a PR-existence query for
    ``retinue/prd-<n>`` -> staging); tests inject a fake that scripts the truth. Modeled
    as one protocol so the whole reconciliation injects through a single collaborator,
    mirroring the gh-seam style of :mod:`retinue.pr_opener` / :mod:`retinue.handoff`.
    """

    async def issue_closed(self, *, repo_full_name: str, issue_number: int) -> bool:
        """Return True when ``issue_number`` is closed on the repo."""
        ...

    async def branch_merged(self, *, repo_full_name: str, branch: str) -> bool:
        """Return True when ``branch`` (an ``issue-<N>`` branch) is merged."""
        ...

    async def staging_pr(self, *, repo_full_name: str, prd_number: int) -> int | None:
        """Return the open ``retinue/prd-<n>`` -> staging PR number, or None."""
        ...


class GhRunner(Protocol):
    """Runs one ``gh`` invocation and returns its stdout. The gh-subprocess seam.

    The production :class:`GhCliReconcile` assembles ``gh`` argv and parses the JSON
    stdout; the actual subprocess spawn (and its installation-token env) lives behind
    this one callable so the command-assembly and payload-parsing are unit-testable
    without spawning a process or touching the network. A production runner shells out
    to ``gh`` with :func:`gh_env` in the child environment; tests inject a fake that
    returns canned JSON and records the argv it was handed.
    """

    async def __call__(self, argv: list[str]) -> str:
        """Run ``gh`` with ``argv`` and return its captured stdout (raises on failure)."""
        ...


def gh_env(token: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build the child-process environment that authenticates ``gh`` as the installation.

    ``gh`` reads its credential from ``GH_TOKEN`` (preferred over ``GITHUB_TOKEN``); the
    installation access token minted by :class:`retinue.github_app.InstallationAuth` goes
    there. ``GH_PROMPT_DISABLED`` keeps a non-interactive worker from ever blocking on a
    prompt. The token is the same ``Authorization: Bearer`` credential ``gh`` sends.

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


def _issue_state_argv(repo_full_name: str, issue_number: int) -> list[str]:
    """Assemble the ``gh`` argv that reads one issue's open/closed state as JSON."""
    return [
        "api",
        f"repos/{repo_full_name}/issues/{issue_number}",
        "--jq",
        "{state: .state}",
    ]


def _compare_argv(repo_full_name: str, base: str, head: str) -> list[str]:
    """Assemble the ``gh`` argv comparing ``base...head`` (does base contain head?)."""
    return [
        "api",
        f"repos/{repo_full_name}/compare/{base}...{head}",
        "--jq",
        "{ahead_by: .ahead_by, status: .status}",
    ]


def _staging_pr_argv(repo_full_name: str, head: str, base: str) -> list[str]:
    """Assemble the ``gh`` argv listing the open ``head`` -> ``base`` PRs as JSON."""
    return [
        "pr",
        "list",
        "--repo",
        repo_full_name,
        "--head",
        head,
        "--base",
        base,
        "--state",
        "open",
        "--json",
        "number",
    ]


def _parse_issue_closed(stdout: str) -> bool:
    """Parse an issue-state payload: True when GitHub reports the issue ``closed``."""
    payload = json.loads(stdout)
    return bool(payload.get("state") == "closed")


def _parse_branch_merged(stdout: str) -> bool:
    """Parse a compare payload: True when the base already contains the head branch.

    ``base...head`` reports how far ``head`` is *ahead of* ``base``; ``ahead_by == 0``
    means every commit on the issue branch already landed on the integration branch, so
    the slice's work is merged (GitHub's ``status`` is then ``identical`` or ``behind``).
    """
    payload = json.loads(stdout)
    return int(payload.get("ahead_by", 0)) == 0


def _parse_staging_pr(stdout: str) -> int | None:
    """Parse a ``pr list --json number`` payload: the first PR number, or None if empty."""
    payload = json.loads(stdout)
    if not payload:
        return None
    return int(payload[0]["number"])


class GhCliReconcile:
    """Production :class:`ReconcileGh`: reads GitHub truth by shelling out to ``gh``.

    Each query assembles a ``gh`` argv, runs it through the injected :class:`GhRunner`,
    and parses the JSON stdout into the protocol's return type. ``issue_closed`` reads the
    issue state; ``branch_merged`` asks whether the integration branch already contains the
    ``issue-<N>`` branch (a ``base...head`` compare with ``ahead_by == 0``); ``staging_pr``
    lists the open ``retinue/prd-<n>`` -> staging PR. The subprocess spawn and its
    installation-token env live in the runner (see :func:`gh_env`), so this class is pure
    command-assembly plus payload-parsing.

    The ``branch_merged`` query asks whether a *merge target* already contains the
    ``issue-<N>`` branch. The orchestrator merges slice branches into the integration
    branch, but the protocol signature carries only the branch name (no PRD number), so
    the comparison base is fixed at construction: ``merge_base`` is the branch a landed
    slice's commits are guaranteed to reach (the staging branch the integration flow
    lands into). A slice whose commits are all already on that base reports ``ahead_by 0``.

    Args:
        runner: The injected gh-subprocess seam that runs an argv and returns stdout.
        merge_base: The branch ``branch_merged`` compares a slice branch against (the
            staging branch landed work reaches). Also the base a staging PR opens into.
            Defaults to ``"staging"``.
    """

    def __init__(self, runner: GhRunner, *, merge_base: str = "staging") -> None:
        self._runner = runner
        self._merge_base = merge_base

    async def issue_closed(self, *, repo_full_name: str, issue_number: int) -> bool:
        """Return True when GitHub reports ``issue_number`` closed on the repo."""
        stdout = await self._runner(_issue_state_argv(repo_full_name, issue_number))
        return _parse_issue_closed(stdout)

    async def branch_merged(self, *, repo_full_name: str, branch: str) -> bool:
        """Return True when the merge base already contains ``branch`` (ahead_by 0)."""
        stdout = await self._runner(
            _compare_argv(repo_full_name, self._merge_base, branch)
        )
        return _parse_branch_merged(stdout)

    async def staging_pr(self, *, repo_full_name: str, prd_number: int) -> int | None:
        """Return the open ``retinue/prd-<n>`` -> staging PR number, or None."""
        head = integration_branch(prd_number)
        stdout = await self._runner(
            _staging_pr_argv(repo_full_name, head, self._merge_base)
        )
        return _parse_staging_pr(stdout)


class ResumePhase(enum.Enum):
    """The phase a reconciled PRD round resumes at after a restart.

    The phases mirror the build pipeline: BUILD -> PR_OPEN -> LOOPBACK, plus DONE for a
    round with nothing left to do. The caller routes into the matching entry point —
    :func:`retinue.orchestrator.build_prd`, :func:`retinue.pr_opener.open_staging_pr`,
    or :func:`retinue.loopback.process_review`.
    """

    BUILD = "build"
    PR_OPEN = "pr_open"
    LOOPBACK = "loopback"
    DONE = "done"


@dataclass(frozen=True)
class ReconcileResult:
    """The reconciled resume plan for one PRD round — what the caller routes on.

    Attributes:
        phase: The phase to resume at (see :class:`ResumePhase`).
        unfinished_slices: On ``BUILD``, the slices to build — only those whose issue is
            still open and branch unmerged, with their ``blocked_by`` graph intact so
            :func:`retinue.orchestrator.build_prd` preserves dependency order. Empty on
            every other phase.
        finished_issues: Slice issue numbers GitHub shows already landed (issue closed or
            branch merged), in input order — reported, not silently dropped.
        pr_number: On ``LOOPBACK``, the open staging PR to resume the loopback on;
            ``None`` on every other phase.
        integration_branch: The PRD's integration branch, ``retinue/prd-<n>``.
    """

    phase: ResumePhase
    integration_branch: str
    unfinished_slices: list[PrdSlice] = field(default_factory=list)
    finished_issues: list[int] = field(default_factory=list)
    pr_number: int | None = None


async def reconcile_run(
    *,
    repo_full_name: str,
    prd_number: int,
    slices: list[PrdSlice],
    gh: ReconcileGh,
) -> ReconcileResult:
    """Reconcile an in-flight PRD round against GitHub truth and pick the resume phase.

    GitHub is the source of truth, queried first: an existing PR proves the build round
    finished, so the round resumes at the loopback (never rebuilding). Otherwise each
    slice is classed finished (issue closed OR branch merged) or unfinished, and the
    round resumes at PR-open (all finished) or build (some unfinished, handing the
    builder only the unfinished slices). A round with no slices and no PR is DONE. The
    either-side finished rule means a crash between a slice's branch-merge and its
    issue-close still resumes without a duplicate branch or issue.

    Args:
        repo_full_name: The target repo, e.g. "owner/repo".
        prd_number: The PRD's tracking issue number; the integration branch is
            ``retinue/prd-<prd_number>``.
        slices: The PRD round's slices with their ``blocked_by`` graph.
        gh: The injected gh seam reconciliation reads GitHub truth through.

    Returns:
        A :class:`ReconcileResult` the caller routes on (phase + the slices/PR to resume
        with). Every input slice is accounted for as finished or unfinished.
    """
    branch = integration_branch(prd_number)

    pr_number = await gh.staging_pr(
        repo_full_name=repo_full_name, prd_number=prd_number
    )
    if pr_number is not None:
        logger.info(
            "Resuming PRD #%d (%s) at loopback: PR #%d already open",
            prd_number,
            repo_full_name,
            pr_number,
        )
        return ReconcileResult(
            phase=ResumePhase.LOOPBACK,
            integration_branch=branch,
            pr_number=pr_number,
        )

    finished, unfinished = await _partition_slices(repo_full_name, slices, gh)
    phase = _phase_without_pr(slices, unfinished)
    logger.info(
        "Resuming PRD #%d (%s) at %s: %d finished, %d unfinished",
        prd_number,
        repo_full_name,
        phase.value,
        len(finished),
        len(unfinished),
    )
    return ReconcileResult(
        phase=phase,
        integration_branch=branch,
        unfinished_slices=unfinished,
        finished_issues=finished,
    )


def _phase_without_pr(
    slices: list[PrdSlice], unfinished: list[PrdSlice]
) -> ResumePhase:
    """Pick the resume phase when no PR exists, from the slice/unfinished split."""
    if not slices:
        return ResumePhase.DONE
    if not unfinished:
        # Every slice landed but the PR never opened: resume at the PR-open phase.
        return ResumePhase.PR_OPEN
    return ResumePhase.BUILD


async def _partition_slices(
    repo_full_name: str, slices: list[PrdSlice], gh: ReconcileGh
) -> tuple[list[int], list[PrdSlice]]:
    """Split slices into (finished issue numbers, unfinished slices) by GitHub truth.

    A slice is finished when GitHub shows EITHER its issue closed OR its branch merged;
    either proves the work landed, so a crash between the two still resumes correctly.
    Input order is preserved in both buckets so the result is deterministic.
    """
    finished: list[int] = []
    unfinished: list[PrdSlice] = []
    for slice_ in slices:
        if await _slice_finished(repo_full_name, slice_, gh):
            finished.append(slice_.issue_number)
        else:
            unfinished.append(slice_)
    return finished, unfinished


async def _slice_finished(
    repo_full_name: str, slice_: PrdSlice, gh: ReconcileGh
) -> bool:
    """Whether GitHub shows a slice's work landed (issue closed or branch merged)."""
    if await gh.issue_closed(
        repo_full_name=repo_full_name, issue_number=slice_.issue_number
    ):
        return True
    return await gh.branch_merged(repo_full_name=repo_full_name, branch=slice_.branch)
