"""Ad-hoc drain: ready-for-agent non-PRD issues, deduped, locked, budget-gated (#32, #33).

One ad-hoc drain runs per repo. It lists every open ``ready-for-agent`` issue via the gh
seam, keeps only the ones the **ad-hoc** lane decision claims (dropping any
``prd``-labeled issue and any issue carrying a ``Part of #<prd>`` link — those route to the
orchestrator lane), ranks the survivors by ``priority:<severity>`` (no-priority lowest),
and drives the ad-hoc build+PR primitive for each up to the concurrency cap
(``config.max_parallel``). The lane filter **mirrors** :func:`retinue.lane.classify`'s
ad-hoc decision (reusing :class:`~retinue.lane.IssueFacts`) but deliberately does **not**
call ``classify``: routing standalone ``priority:critical``/``high`` issues through
classify would preempt them onto the orchestrator lane and exclude them from the drain.

Each surviving issue is materialized into an :class:`~retinue.adhoc_build.AdhocIssue`
through :meth:`~retinue.adhoc_build.AdhocIssue.from_fetched_issue` — fed the issue body the
gh seam surfaces — never the bare constructor. ``from_fetched_issue`` parses the
``Chain-depth:`` lineage marker out of the body into
:attr:`~retinue.adhoc_build.AdhocIssue.chain_depth`; building the issue by hand would
default every hop to depth 0 and silently make the #39/#40 review-fix chain bound inert.
The gh list seam therefore surfaces each issue's ``body`` alongside its labels.

The drain is hardened for production (#33) with four guards:

* **dedup + stranded recovery via GitHub truth** — the drain classifies each issue against
  the reconcile-style source of truth (:mod:`retinue.reconcile`): an issue with a branch
  *and* an open PR is in flight (skip it, no duplicate branch/PR); an issue with a pushed
  ``issue-<N>`` branch but *no* open PR is **stranded** — a prior green build
  (push-only-on-green) whose PR never opened — so the drain opens its PR without rebuilding;
  every other issue is built fresh. Truth is read once per drain through
  :meth:`SupportsFlightSnapshot.flight_snapshot` (two whole-repo ``gh`` queries), then each
  candidate is classified in memory (:meth:`FlightSnapshot.state_for`); a seam offering only
  the per-issue :meth:`AdhocGh.flight_state` is classified through that fallback;
* **admit-then-release lock + build guard** — an injected lock serializes only the short
  list/classify/**reserve** critical section (a second entry *during that section* raises
  :class:`AdhocDrainBusyError`); the long builds run **outside** the lock, so a critical
  issue kicked mid-drain enters the freed lock and builds concurrently (#83). An in-process
  :class:`AdhocBuildGuard` — reserved under the lock, released when each build finishes —
  keeps two overlapping drains from double-building the same issue (a just-started build has
  not opened its PR yet, so flight-state alone can't dedup it). The lock is *separate* from
  the orchestrator's, so the drain still runs alongside a PRD build;
* **shared budget governor** — every build meters against the one service-level
  :class:`~retinue.budget.BudgetGovernor` the PRD lane uses, so a build that would cross
  the rolling-24h cap is skipped and the shared budget is never overshot;
* **PRD-first ordering with preemption** — when a PRD build is in flight, only a
  ``priority:critical``/``high`` issue (the same rule :func:`retinue.lane.classify`
  preempts on) builds; ordinary ad-hoc work waits for the PRD to finish.

Every leaf I/O — the gh queries, the budget store, the downstream build — is injected and
faked, so the whole drain runs with no real ``gh``, no Docker, and no network — mirroring
the injected-seam style of :mod:`retinue.cron`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from retinue.adhoc_build import AdhocIssue
from retinue.budget import BudgetGovernor
from retinue.cron import GhCliError, GhRunner, _parse_json_array, _run_gh_subprocess
from retinue.lane import IssueFacts, preempts_prd_first
from retinue.loopback import Severity
from retinue.repo_config import RepoConfig
from retinue.slicer import READY_LABEL
from retinue.webhook import PRD_LABEL

logger = logging.getLogger(__name__)


class AdhocDrainBusyError(Exception):
    """A second ad-hoc drain hit the list/classify/reserve critical section while it was held.

    :func:`run_adhoc_drain` holds an injected lock only around its short
    list/classify/**reserve** critical section — not the long builds, which run outside it
    (#83) — and rejects a concurrent holder rather than blocking, so contention *on that
    section* is observable to the caller. A drain kicked while another is merely *building*
    is **not** rejected: the lock is already free, and the :class:`AdhocBuildGuard` stops the
    two from double-building the same issue. This lock is *separate* from the orchestrator's
    :class:`retinue.orchestrator.OrchestratorBusyError` lock, so an ad-hoc drain still runs
    concurrently with a PRD build. Mirrors :class:`retinue.cron.CronBusyError`.
    """

    def __init__(self) -> None:
        super().__init__("an ad-hoc drain is already in flight")


class AdhocDrainLock:
    """The production critical-section lock: a non-blocking in-process guard for one repo.

    Satisfies the ``AbstractAsyncContextManager`` :func:`run_adhoc_drain` enters around its
    list/classify/**reserve** critical section (not the builds, which run outside it — #83):
    the first holder enters, and a *second* concurrent ``__aenter__`` raises
    :class:`AdhocDrainBusyError` rather than blocking — so contention on that short section is
    observable to the caller (mirroring the test fake's reject-don't-block behavior). Because
    the lock is released before the long builds, a drain kicked while another is *building*
    finds it free; overlap-safety for the builds themselves is the :class:`AdhocBuildGuard`'s
    job. One lock instance guards one repo; the worker keeps a per-repo registry so two repos
    drain concurrently while a repo's own kicked and swept drains serialize through it.

    The guard is a plain in-process flag (no real wall-clock, Redis, or file lock), which is
    correct because the whole drain runs inside a single worker process; a cross-process lock
    is out of scope for the single-worker deployment.
    """

    def __init__(self) -> None:
        self._held = False

    async def __aenter__(self) -> AdhocDrainLock:
        if self._held:
            raise AdhocDrainBusyError
        self._held = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        self._held = False


class AdhocBuildGuard:
    """In-process set of issue numbers a repo's drains are actively building.

    The double-build guard that replaces the old lock-through-builds invariant (#83): now
    that builds run *outside* the :class:`AdhocDrainLock`, a second drain can enter its
    critical section while a first drain's builds are still in flight. A just-started build
    has not opened its PR yet, so flight-state classification alone would re-pick it as
    ``ABSENT`` and build it twice. The buildable set is :meth:`reserve`\\ d under the lock and
    :meth:`release`\\ d when the builds finish, so an overlapping drain skips any number
    already in flight.

    asyncio is single-threaded and reservation happens synchronously (no ``await`` between
    :meth:`reserve` and releasing the lock), so reserve-under-lock is atomic — no extra
    locking needed. One guard instance per repo, held in the worker's registry alongside the
    lock so a repo's kicked and swept drains share it.
    """

    def __init__(self) -> None:
        self._building: set[int] = set()

    def reserve(self, numbers: Iterable[int]) -> None:
        """Mark ``numbers`` as actively building (called under the drain lock)."""
        self._building.update(numbers)

    def release(self, numbers: Iterable[int]) -> None:
        """Drop ``numbers`` from the in-flight set once their builds finish."""
        self._building.difference_update(numbers)

    def __contains__(self, number: int) -> bool:
        return number in self._building


# How many ready-for-agent issues to pull per drain. The cap on concurrent *builds* is
# ``config.max_parallel``; this generous-but-bounded page just keeps the visible set from
# an unbounded fetch, mirroring the cron lane's list limit.
_DEFAULT_LIST_LIMIT = 200


@dataclass(frozen=True)
class ReadyIssue:
    """One open ``ready-for-agent`` issue, as reported by the ad-hoc gh seam.

    The body is surfaced (unlike the cron lane's backlog seam) because the drain feeds it
    to :meth:`~retinue.adhoc_build.AdhocIssue.from_fetched_issue`, which reads the
    ``Chain-depth:`` lineage marker out of it — and because :meth:`is_adhoc` scans it for
    the ``Part of #<prd>`` link (mirroring :func:`retinue.lane.classify`'s decision, not
    calling it) to split orchestrator-lane slices from ad-hoc work.

    Attributes:
        number: The issue number; the build commits to the derived ``issue-<N>`` branch.
        labels: The issue's label names (carries ``ready-for-agent`` and, optionally, a
            ``priority:<severity>`` or the ``prd`` label).
        body: The issue body, scanned for the ``Part of #<prd>`` link and the
            ``Chain-depth:`` marker.
    """

    number: int
    labels: list[str]
    body: str = ""

    def _facts(self) -> IssueFacts:
        """The routing-relevant facts of this issue, for the lane decision and ranking."""
        return IssueFacts(labels=list(self.labels), body=self.body)

    def is_adhoc(self) -> bool:
        """Whether this issue is ad-hoc work the drain builds directly.

        Mirrors :func:`retinue.lane.classify`'s ad-hoc decision (reusing
        :class:`~retinue.lane.IssueFacts` for the ``Part of #<prd>`` scan): a
        ``ready-for-agent`` issue is ad-hoc unless it carries the ``prd`` label or a
        ``Part of #<prd>`` link — both of which route to the orchestrator lane and are
        excluded here. Unlike a raw ``classify`` call this does *not* fold in classify's
        priority **preemption**, which reorders a standalone ``priority:critical``/``high``
        onto the orchestrator lane: that is an ordering optimization, not a statement that
        the work is not ad-hoc, and the drain ranks those at the top itself.
        """
        facts = self._facts()
        if not facts.has_label(READY_LABEL):
            return False
        if facts.has_label(PRD_LABEL):
            return False
        return facts.prd_link() is None

    def severity(self) -> Severity | None:
        """The issue's ``priority:<severity>`` as a :class:`Severity`, or ``None``.

        Reuses :meth:`retinue.lane.IssueFacts.priority`, so an unknown ``priority:*`` value
        is treated as no priority (ranked lowest) rather than raising.
        """
        return self._facts().priority()

    def preempts(self) -> bool:
        """Whether this issue's priority jumps PRD-first ordering (``critical``/``high``).

        Reuses :func:`retinue.lane.preempts_prd_first` — the single source of truth for the
        preemption rule :func:`retinue.lane.classify` applies — so the drain and the
        classifier agree on what preempts.
        """
        return preempts_prd_first(self.severity())


class FlightState(Enum):
    """GitHub's verdict on an issue's ``issue-<N>`` branch and open PR — the drain's truth.

    The drain reads this to decide what to do with each ready issue:

    * :attr:`ABSENT` — no ``issue-<N>`` branch and no open PR: nothing was built, so build
      it fresh.
    * :attr:`STRANDED` — the ``issue-<N>`` branch exists but no open PR: a prior build
      pushed the branch (push-only-on-green, so the branch is provably green) yet never
      opened its PR — e.g. the PR-open precheck failed. The drain opens the PR **without
      rebuilding** (the rebuild would waste budget re-deriving a known-green branch).
    * :attr:`IN_FLIGHT` — an open PR exists: a build is under way or already landed, so the
      drain skips the issue (the original branch-or-PR dedup, now narrowed to "PR exists").
    """

    ABSENT = "absent"
    STRANDED = "stranded"
    IN_FLIGHT = "in_flight"


@dataclass(frozen=True)
class FlightSnapshot:
    """Whole-repo GitHub truth for flight-state classification, fetched once per drain.

    Carries the two whole-repo sets the drain classifies every candidate against in memory,
    replacing the per-issue branch-ref + open-PR spawns (:meth:`AdhocGh.flight_state`) with a
    single prefetch: the head branch names of the repo's open PRs, and the existing
    ``issue-<N>`` branch names. :meth:`state_for` applies the exact :class:`FlightState`
    rule, preserving the stranded-branch recovery (commit cea5d14).

    Attributes:
        open_pr_heads: The head branch names of the repo's open PRs (an open PR keeps its
            head alive, so these are a subset of ``issue_branches``).
        issue_branches: The existing ``issue-<N>`` branch names on the repo.
    """

    open_pr_heads: frozenset[str]
    issue_branches: frozenset[str]

    def state_for(self, issue_number: int) -> FlightState:
        """Classify one issue from the snapshot, mirroring per-issue ``flight_state``.

        The same order the per-issue query used: a missing ``issue-<N>`` branch is
        :attr:`FlightState.ABSENT`; a branch with an open PR is :attr:`FlightState.IN_FLIGHT`;
        a branch with no open PR is :attr:`FlightState.STRANDED` (pushed green, PR never
        opened).
        """
        branch = _issue_branch(issue_number)
        if branch not in self.issue_branches:
            return FlightState.ABSENT
        if branch in self.open_pr_heads:
            return FlightState.IN_FLIGHT
        return FlightState.STRANDED


class AdhocGh(Protocol):
    """The gh queries behind the ad-hoc drain. The ad-hoc lane's gh seam.

    A production implementation runs ``gh issue list --label ready-for-agent`` (with each
    issue's labels and body) for :meth:`list_ready`, and the branch-existence + open-PR
    lookups for :meth:`flight_state`; tests inject a fake that scripts both. Modeled as one
    protocol so the whole drain injects through a single collaborator, mirroring the
    gh-seam style of :mod:`retinue.cron` / :mod:`retinue.reconcile`. A production seam should
    also implement :class:`SupportsFlightSnapshot` so the drain classifies flight state with
    one whole-repo prefetch instead of a per-issue :meth:`flight_state` spawn.
    """

    async def list_ready(self, *, repo_full_name: str) -> list[ReadyIssue]:
        """Return the repo's open ``ready-for-agent`` issues with their labels and body."""
        ...

    async def flight_state(
        self, *, repo_full_name: str, issue_number: int
    ) -> FlightState:
        """Classify the issue against GitHub truth: absent, stranded, or in flight.

        The dedup + stranded-recovery source of truth, mirroring the reconcile-style
        GitHub-truth approach (:mod:`retinue.reconcile`): reads whether the issue's
        ``issue-<N>`` branch exists and whether an open PR for it exists, and returns the
        matching :class:`FlightState`. The drain builds an :attr:`~FlightState.ABSENT`
        issue, opens the PR for a :attr:`~FlightState.STRANDED` one, and skips an
        :attr:`~FlightState.IN_FLIGHT` one.

        This is the *per-issue* fallback; a seam that also implements
        :class:`SupportsFlightSnapshot` is classified with one whole-repo query instead.
        """
        ...


@runtime_checkable
class SupportsFlightSnapshot(Protocol):
    """An optional whole-repo flight-state prefetch the drain prefers over per-issue calls.

    A gh seam that can answer the flight-state question for the *whole* repo in one shot
    (rather than a branch-ref + open-PR spawn per issue) implements this. The production
    :class:`GhCli` does — collapsing the old N-issue x 2-spawn classification into two
    whole-repo ``gh`` queries — while a seam offering only per-issue :meth:`AdhocGh.flight_state`
    (e.g. a minimal test fake) is classified through that fallback instead. Runtime-checkable
    so :func:`run_adhoc_drain` can pick the fast path with ``isinstance``.
    """

    async def flight_snapshot(self, *, repo_full_name: str) -> FlightSnapshot:
        """Fetch the repo's whole flight-state truth in one prefetch (open PRs + branches)."""
        ...


class GhCli:
    """The production :class:`AdhocGh`: lists ``ready-for-agent`` issues via the ``gh`` CLI.

    Runs ``gh issue list --repo <repo> --label ready-for-agent --state open --json
    number,labels,body`` and parses the JSON into :class:`ReadyIssue` objects. The ``body``
    field is requested — unlike the cron lane's backlog query — because the drain feeds it
    to :meth:`~retinue.adhoc_build.AdhocIssue.from_fetched_issue` (lineage marker) and
    scans it in :meth:`ReadyIssue.is_adhoc` for the ``Part of #<prd>`` link (mirroring
    :func:`retinue.lane.classify`'s decision, not calling it). Authenticates by injecting
    the GitHub token into the child env as ``GH_TOKEN``, so no token is ever placed on the
    command line.

    The subprocess spawn is the one impure edge, factored behind the injected ``runner``
    (the cron lane's :data:`~retinue.cron.GhRunner` and :func:`~retinue.cron._run_gh_subprocess`,
    reused) so command assembly, the auth env, and payload parsing are unit-testable
    without a real ``gh``, Docker, or network.

    Args:
        token: The GitHub token ``gh`` authenticates with, placed in the child env as
            ``GH_TOKEN``. ``None`` runs ``gh`` with the ambient auth (e.g. a logged-in CLI).
        runner: The injected argv runner; defaults to the real subprocess spawn.
        list_limit: The max number of ready-for-agent issues to pull per drain.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        runner: GhRunner | None = None,
        list_limit: int = _DEFAULT_LIST_LIMIT,
    ) -> None:
        self._token = token
        self._runner = runner or _run_gh_subprocess
        self._list_limit = list_limit

    async def list_ready(self, *, repo_full_name: str) -> list[ReadyIssue]:
        """Return the repo's open ``ready-for-agent`` issues with their labels and body.

        Raises:
            GhCliError: ``gh`` exited non-zero (propagated from the runner).
            ValueError: ``gh`` returned a payload that did not parse as the expected
                issue listing.
        """
        argv = _list_ready_argv(repo_full_name, limit=self._list_limit)
        stdout = await self._runner(argv, self._auth_env())
        return [_parse_ready_issue(entry) for entry in _parse_json_array(stdout)]

    async def flight_state(
        self, *, repo_full_name: str, issue_number: int
    ) -> FlightState:
        """Classify the issue from GitHub truth: absent, stranded, or in flight.

        Queries GitHub truth in two legs. First the branch-ref lookup (a missing ref 404s,
        surfaced by the runner as :class:`GhCliError`, read here as "no branch"): no branch
        short-circuits to :attr:`FlightState.ABSENT` — an open PR keeps its head branch
        alive, so a missing branch proves no open PR can exist, and the second leg is
        skipped. When the branch exists, the open-PR list for the ``issue-<N>`` head
        decides: an open PR means :attr:`FlightState.IN_FLIGHT`, none means the branch is
        :attr:`FlightState.STRANDED` (pushed green, PR never opened). Mirrors the
        reconcile-style source-of-truth dedup.

        Raises:
            ValueError: the open-PR query returned a payload that did not parse as a JSON
                array.
        """
        branch = _issue_branch(issue_number)
        if not await self._branch_exists(repo_full_name, branch):
            return FlightState.ABSENT
        if await self._open_pr_exists(repo_full_name, branch):
            return FlightState.IN_FLIGHT
        return FlightState.STRANDED

    async def flight_snapshot(self, *, repo_full_name: str) -> FlightSnapshot:
        """Fetch whole-repo flight-state truth in two queries, for in-memory classification.

        Replaces the per-issue branch-ref + open-PR spawns with two repo-wide ``gh`` calls:
        ``gh pr list --state open --json headRefName`` for every open PR's head branch, and
        ``gh api .../git/matching-refs/heads/issue-`` enumerating every existing ``issue-*``
        branch ref. The drain then classifies each candidate in memory
        (:meth:`FlightSnapshot.state_for`), preserving the exact :class:`FlightState`
        semantics — including the stranded-branch recovery.

        Raises:
            ValueError: a ``gh`` payload did not parse as the expected JSON array (or an
                entry was missing its head/ref field).
        """
        open_pr_heads = await self._open_pr_heads(repo_full_name)
        issue_branches = await self._issue_branches(repo_full_name)
        return FlightSnapshot(
            open_pr_heads=open_pr_heads, issue_branches=issue_branches
        )

    async def _open_pr_heads(self, repo_full_name: str) -> frozenset[str]:
        """The head branch names of every open PR on the repo (the whole in-flight set)."""
        argv = _open_pr_heads_argv(repo_full_name, limit=self._list_limit)
        stdout = await self._runner(argv, self._auth_env())
        return frozenset(_parse_pr_heads(stdout))

    async def _issue_branches(self, repo_full_name: str) -> frozenset[str]:
        """Every existing ``issue-<N>`` branch name on the repo (the whole branch set)."""
        argv = _issue_refs_argv(repo_full_name)
        stdout = await self._runner(argv, self._auth_env())
        return frozenset(_parse_issue_branches(stdout))

    async def _branch_exists(self, repo_full_name: str, branch: str) -> bool:
        """Whether ``branch`` resolves to a ref on the repo (a 404 means it does not)."""
        argv = _branch_ref_argv(repo_full_name, branch)
        try:
            await self._runner(argv, self._auth_env())
        except GhCliError:
            # A missing ref is a 404 / non-zero exit — not an error to propagate, just a
            # False answer to the existence question (mirrors pr_opener's staging_exists).
            return False
        return True

    async def _open_pr_exists(self, repo_full_name: str, branch: str) -> bool:
        """Whether an open PR with ``branch`` as its head exists on the repo."""
        argv = _open_pr_argv(repo_full_name, branch)
        stdout = await self._runner(argv, self._auth_env())
        return len(_parse_json_array(stdout)) > 0

    def _auth_env(self) -> Mapping[str, str]:
        """The child-process env carrying the token as ``GH_TOKEN`` (empty when none).

        The token goes in the env, never on the argv, so it never lands in a process
        listing or a log of the command.
        """
        return {"GH_TOKEN": self._token} if self._token else {}


def _list_ready_argv(repo_full_name: str, *, limit: int) -> list[str]:
    """Assemble the ``gh issue list`` argv for the open ``ready-for-agent`` issues.

    Pulls ``number``, ``labels``, and ``body`` as JSON — exactly the fields the drain needs
    to classify (the ``Part of #<prd>`` link), rank (the ``priority:*`` label), and rebuild
    each issue through :meth:`~retinue.adhoc_build.AdhocIssue.from_fetched_issue` (the
    ``Chain-depth:`` marker).
    """
    return [
        "gh",
        "issue",
        "list",
        "--repo",
        repo_full_name,
        "--label",
        READY_LABEL,
        "--state",
        "open",
        "--json",
        "number,labels,body",
        "--limit",
        str(limit),
    ]


def _issue_branch(issue_number: int) -> str:
    """The ``issue-<N>`` branch an ad-hoc build commits to (the dedup branch name)."""
    return f"issue-{issue_number}"


def _branch_ref_argv(repo_full_name: str, branch: str) -> list[str]:
    """Assemble the ``gh api`` argv reading one branch ref (404s when it does not exist).

    Hits ``repos/<repo>/git/ref/heads/<branch>`` rather than ``gh pr``/``gh branch`` so a
    missing branch is a clean 404 the adapter reads as "no branch", mirroring
    :meth:`retinue.pr_opener.GhCliPrOps.staging_exists`.
    """
    return [
        "gh",
        "api",
        f"repos/{repo_full_name}/git/ref/heads/{branch}",
    ]


def _open_pr_argv(repo_full_name: str, head: str) -> list[str]:
    """Assemble the ``gh pr list`` argv for the open PRs with ``head`` as their branch.

    Mirrors :func:`retinue.reconcile._staging_pr_argv` (``pr list --head ... --state open
    --json number``); a non-empty array means an open PR already exists for the issue.
    """
    return [
        "gh",
        "pr",
        "list",
        "--repo",
        repo_full_name,
        "--head",
        head,
        "--state",
        "open",
        "--json",
        "number",
    ]


# The ``git`` ref prefix a branch name sits behind in the matching-refs payload, and the
# ``issue-`` prefix the whole-repo branch enumeration matches on.
_REF_HEADS_PREFIX = "refs/heads/"
_ISSUE_BRANCH_PREFIX = "issue-"


def _open_pr_heads_argv(repo_full_name: str, *, limit: int) -> list[str]:
    """Assemble the ``gh pr list`` argv for every open PR's head branch (whole-repo).

    One repo-wide query returning just ``headRefName`` for the open PRs, so the drain can
    classify every candidate's in-flight state in memory rather than a ``gh pr list --head``
    per issue.
    """
    return [
        "gh",
        "pr",
        "list",
        "--repo",
        repo_full_name,
        "--state",
        "open",
        "--json",
        "headRefName",
        "--limit",
        str(limit),
    ]


def _issue_refs_argv(repo_full_name: str) -> list[str]:
    """Assemble the ``gh api`` argv enumerating every existing ``issue-*`` branch ref.

    Hits ``repos/<repo>/git/matching-refs/heads/issue-``, which returns *all* refs under
    ``heads/issue-`` (an empty array when none, never a 404), so one paginated query lists
    every ``issue-<N>`` branch instead of a branch-ref lookup per issue.
    """
    return [
        "gh",
        "api",
        "--paginate",
        f"repos/{repo_full_name}/git/matching-refs/heads/{_ISSUE_BRANCH_PREFIX}",
    ]


def _parse_pr_heads(stdout: bytes) -> set[str]:
    """Parse ``gh pr list --json headRefName`` into the set of open-PR head branch names.

    A malformed entry (not an object, or missing ``headRefName``) raises :class:`ValueError`
    rather than silently dropping an in-flight PR — which would risk a duplicate build.
    """
    heads: set[str] = set()
    for entry in _parse_json_array(stdout):
        if not isinstance(entry, dict) or "headRefName" not in entry:
            raise ValueError(f"gh pr list entry is malformed: {entry!r}")
        heads.add(str(entry["headRefName"]))
    return heads


def _parse_issue_branches(stdout: bytes) -> set[str]:
    """Parse a matching-refs payload into the set of existing ``issue-<N>`` branch names.

    Each entry's ``ref`` (``refs/heads/issue-<N>``) is stripped of the ``refs/heads/``
    prefix. A malformed entry (not an object, or missing ``ref``) raises :class:`ValueError`
    rather than silently dropping a branch — which would risk a duplicate build.
    """
    branches: set[str] = set()
    for entry in _parse_json_array(stdout):
        if not isinstance(entry, dict) or "ref" not in entry:
            raise ValueError(f"gh matching-refs entry is malformed: {entry!r}")
        ref = str(entry["ref"])
        if ref.startswith(_REF_HEADS_PREFIX):
            branches.add(ref[len(_REF_HEADS_PREFIX) :])
    return branches


def _parse_ready_issue(entry: object) -> ReadyIssue:
    """Parse one ``gh`` issue object into a :class:`ReadyIssue`.

    A malformed entry raises :class:`ValueError` rather than silently dropping the issue.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"gh issue entry is not an object: {entry!r}")
    try:
        number = int(entry["number"])
        labels = [str(label["name"]) for label in entry["labels"]]
        body = str(entry.get("body", "") or "")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"gh issue entry is malformed: {entry!r} ({exc})") from exc
    return ReadyIssue(number=number, labels=labels, body=body)


# The downstream the drain drives for each ranked ad-hoc issue: the ad-hoc build primitive
# (:func:`retinue.adhoc_build.build_adhoc_issue`) followed by the PR step
# (:meth:`retinue.pipeline.Pipeline.process_adhoc_pr`), behind one injected callable so the
# drain is exercised without Docker, gh, the Agent SDK, or network. The drain hands it the
# materialized :class:`~retinue.adhoc_build.AdhocIssue` (built through
# ``from_fetched_issue``, so its ``chain_depth`` is live) and the repo name.
AdhocBuild = Callable[..., Awaitable[None]]

# The PR-open-only recovery the drain drives for a :attr:`FlightState.STRANDED` issue: open
# the PR for an already-pushed (green) ``issue-<N>`` branch without rebuilding it. Same
# ``(issue, *, repo_full_name) -> None`` shape as :data:`AdhocBuild`, injected and faked so
# the drain runs with no gh or network; bound in production to the repo pipeline's
# :meth:`retinue.pipeline.Pipeline.process_adhoc_pr` over a synthesized green result.
AdhocPrOpen = Callable[..., Awaitable[None]]


async def run_adhoc_drain(
    *,
    repo_full_name: str,
    gh: AdhocGh,
    build: AdhocBuild,
    open_pr: AdhocPrOpen,
    config: RepoConfig,
    governor: BudgetGovernor,
    estimated_amount: float,
    lock: AbstractAsyncContextManager[object],
    guard: AdhocBuildGuard,
    prd_in_flight: bool = False,
) -> list[AdhocIssue]:
    """Drain the repo's ad-hoc work, hardened for production: list, filter, classify, act.

    The drain **admits under the lock, then builds outside it** (#83): ``lock`` is held only
    around the list/classify/**reserve** critical section (a second entry *during that
    section* raises :class:`AdhocDrainBusyError`), then released before the long builds run —
    so a critical issue kicked mid-drain enters the freed lock and builds concurrently rather
    than being rejected. The lock is *separate* from the orchestrator's, so the drain still
    runs concurrently with a PRD build. Inside the lock:

    1. **list** the repo's open ``ready-for-agent`` issues (number, labels, body),
    2. **filter** to the ad-hoc lane via :meth:`ReadyIssue.is_adhoc` — mirroring
       :func:`retinue.lane.classify`'s ad-hoc decision (not calling it) — dropping any
       ``prd``-labeled issue and any issue carrying a ``Part of #<prd>`` link,
    3. **honor PRD-first ordering**: when ``prd_in_flight`` is True, only a
       ``priority:critical``/``high`` issue (:meth:`ReadyIssue.preempts`, the same rule
       :func:`retinue.lane.classify` preempts on) builds — ordinary ad-hoc work waits for
       the PRD to finish,
    4. **rank** the survivors by ``priority:<severity>`` (no-priority lowest),
    5. **classify** each survivor against GitHub truth (:meth:`AdhocGh.flight_state`) and
       partition: an in-flight issue (open PR) is skipped, a **stranded** one (pushed green
       ``issue-<N>`` branch, no open PR) goes to the PR-open-only recovery, and the rest are
       buildable,
    6. **recover** every stranded issue by driving ``open_pr`` — opening the PR for its
       already-green branch with **no rebuild** and no budget charge (opening a PR does no
       model work), so a build whose PR-open step once failed is not stranded forever,
    7. **reserve then build** — drop any buildable issue already reserved by a concurrent
       drain (``guard``), :meth:`~AdhocBuildGuard.reserve` the survivors, and **release the
       lock**. Then build each survivor **outside** the lock — materialized through
       :meth:`~retinue.adhoc_build.AdhocIssue.from_fetched_issue` (so the ``Chain-depth:``
       marker stays live) — concurrently but capped at ``config.max_parallel`` live builds,
       each metered against the **one shared** :class:`~retinue.budget.BudgetGovernor` (the
       same governor the PRD lane meters): a build that would cross the rolling-24h cap is
       skipped, so the shared budget is never overshot. The reservation is released
       (success, budget-skip, or error alike) once the builds finish.

    Args:
        repo_full_name: The target repo, e.g. ``"owner/repo"``.
        gh: The ad-hoc gh seam (lists ``ready-for-agent`` issues; answers the flight-state
            classification query).
        build: The downstream ad-hoc build+PR primitive run per buildable issue, injected
            so the drain runs with no Docker, gh, the Agent SDK, or network.
        open_pr: The PR-open-only recovery run per stranded issue — opens the PR for an
            already-green ``issue-<N>`` branch without rebuilding. Injected and faked too.
        config: The accepted repo config; ``max_parallel`` bounds the concurrent builds.
        governor: The shared service-level budget governor; each build is metered through
            it and skipped when the rolling-24h cap leaves no room.
        estimated_amount: The per-build charge metered against the shared cap.
        lock: The critical-section lock; entering it raises :class:`AdhocDrainBusyError` when
            another drain for this repo is inside its list/classify/reserve section. Held only
            around that section, not the builds. Separate from the PRD build's lock.
        guard: The per-repo in-flight :class:`AdhocBuildGuard`, reserved under the lock and
            released when the builds finish, so an overlapping drain does not double-build an
            issue. Required (no unsafe default): the worker passes the shared per-repo
            instance, so a missing wiring fails loudly rather than silently running without
            cross-drain double-build dedup (#83). A single-drain caller passes a fresh guard.
        prd_in_flight: Whether a PRD build is currently running for this repo. When True,
            PRD-first ordering holds and only preempting (``critical``/``high``) issues
            build; when False, every ranked ad-hoc issue builds.

    Returns:
        The :class:`~retinue.adhoc_build.AdhocIssue` objects actually driven through
        ``build`` (in-flight-skipped, stranded, and budget-skipped issues excluded), in
        rank order. Stranded issues whose PR was opened are not "built", so they are not
        included.

    Raises:
        AdhocDrainBusyError: Another drain for this repo holds the list/classify/reserve
            critical section (from the lock).
    """
    async with lock:
        listed = await gh.list_ready(repo_full_name=repo_full_name)
        candidates = _select_candidates(repo_full_name, prd_in_flight, listed)
        plan = await _partition_candidates(repo_full_name, candidates, gh)
        await _open_stranded_prs(repo_full_name, plan.stranded, open_pr)
        # Drop issues a concurrent drain already reserved: their builds are in flight but
        # haven't opened a PR yet, so flight-state alone would re-pick them (#83).
        buildable = [
            issue for issue in plan.buildable if issue.issue_number not in guard
        ]
        if not buildable:
            logger.info(
                "Ad-hoc drain idle: no buildable ready-for-agent issues for %s "
                "(%d stranded PR(s) recovered)",
                repo_full_name,
                len(plan.stranded),
            )
            return []

        logger.info(
            "Ad-hoc drain building up to %d issue(s) for %s (cap=%s)",
            len(buildable),
            repo_full_name,
            config.max_parallel,
        )
        reserved = [issue.issue_number for issue in buildable]
        guard.reserve(reserved)
    # Builds run OUTSIDE the lock so a concurrent critical kick can admit into the freed
    # lock; the guard (reserved above) keeps the two from double-building the same issue.
    try:
        return await _build_metered(
            repo_full_name,
            buildable,
            build=build,
            config=config,
            governor=governor,
            estimated_amount=estimated_amount,
        )
    finally:
        guard.release(reserved)


def _select_candidates(
    repo_full_name: str, prd_in_flight: bool, listed: list[ReadyIssue]
) -> list[AdhocIssue]:
    """Filter to the ad-hoc lane, honor PRD-first preemption, rank, and materialize.

    Drops non-ad-hoc issues, then — when a PRD is in flight — keeps only the preempting
    (``critical``/``high``) issues so PRD-first ordering holds. The survivors are ranked
    and materialized through :meth:`AdhocIssue.from_fetched_issue` so each one's
    ``chain_depth`` is read back from its body.
    """
    adhoc = [issue for issue in listed if issue.is_adhoc()]
    if prd_in_flight:
        adhoc = [issue for issue in adhoc if issue.preempts()]
    return [
        AdhocIssue.from_fetched_issue(repo_full_name, ready.number, ready.body)
        for ready in _rank_adhoc(adhoc)
    ]


@dataclass(frozen=True)
class _DrainPlan:
    """How the drain will act on its ranked candidates, partitioned by flight state.

    Attributes:
        buildable: :attr:`FlightState.ABSENT` issues — nothing built yet, so build them
            (metered against the shared budget). In rank order.
        stranded: :attr:`FlightState.STRANDED` issues — a green branch with no open PR, so
            open the PR without rebuilding (no budget charge). In rank order.
    """

    buildable: list[AdhocIssue]
    stranded: list[AdhocIssue]


async def _partition_candidates(
    repo_full_name: str, issues: list[AdhocIssue], gh: AdhocGh
) -> _DrainPlan:
    """Classify each candidate against GitHub truth and split build vs PR-open recovery.

    Mirrors the reconcile-style GitHub-truth source of truth (:meth:`AdhocGh.flight_state`):
    an :attr:`~FlightState.IN_FLIGHT` issue (open PR) is dropped so the drain opens no
    duplicate; a :attr:`~FlightState.STRANDED` issue (pushed green branch, no PR) routes to
    the PR-open-only recovery rather than a wasteful rebuild; the rest are buildable. Rank
    order is preserved within each bucket.
    """
    states = await _flight_states(repo_full_name, issues, gh)
    buildable: list[AdhocIssue] = []
    stranded: list[AdhocIssue] = []
    for issue in issues:
        state = states[issue.issue_number]
        if state is FlightState.IN_FLIGHT:
            logger.info(
                "Ad-hoc drain skipping issue #%d (%s): already in flight",
                issue.issue_number,
                repo_full_name,
            )
        elif state is FlightState.STRANDED:
            logger.info(
                "Ad-hoc drain recovering issue #%d (%s): green branch with no PR; "
                "opening its PR without rebuilding",
                issue.issue_number,
                repo_full_name,
            )
            stranded.append(issue)
        else:
            buildable.append(issue)
    return _DrainPlan(buildable=buildable, stranded=stranded)


async def _flight_states(
    repo_full_name: str, issues: list[AdhocIssue], gh: AdhocGh
) -> dict[int, FlightState]:
    """Resolve each candidate's :class:`FlightState`, preferring one whole-repo query.

    A seam that supports the whole-repo :class:`FlightSnapshot` (the production
    :class:`GhCli`) is queried once and every candidate is classified in memory — two ``gh``
    queries total rather than up to two subprocess spawns per issue. A seam offering only the
    per-issue :meth:`AdhocGh.flight_state` (a minimal fake) is classified one issue at a
    time. Both paths yield the identical verdicts.
    """
    if isinstance(gh, SupportsFlightSnapshot):
        snapshot = await gh.flight_snapshot(repo_full_name=repo_full_name)
        return {
            issue.issue_number: snapshot.state_for(issue.issue_number)
            for issue in issues
        }
    return {
        issue.issue_number: await gh.flight_state(
            repo_full_name=repo_full_name, issue_number=issue.issue_number
        )
        for issue in issues
    }


async def _open_stranded_prs(
    repo_full_name: str, stranded: list[AdhocIssue], open_pr: AdhocPrOpen
) -> None:
    """Open the PR for each stranded green branch, in rank order, without rebuilding.

    Opening a PR for an already-green branch does no model work, so it is not metered
    against the shared budget — a stranded build is recovered even when the cap is spent.
    """
    for issue in stranded:
        await open_pr(issue, repo_full_name=repo_full_name)


async def _build_metered(
    repo_full_name: str,
    issues: list[AdhocIssue],
    *,
    build: AdhocBuild,
    config: RepoConfig,
    governor: BudgetGovernor,
    estimated_amount: float,
) -> list[AdhocIssue]:
    """Build the issues concurrently (capped), each metered against the shared budget.

    Every build first meters its charge against the one shared
    :class:`~retinue.budget.BudgetGovernor` (the same governor the PRD lane uses); a build
    that would cross the rolling-24h cap is skipped, so the shared budget is never
    overshot. Returns the issues that actually built, in rank order.
    """
    semaphore = asyncio.Semaphore(config.max_parallel or len(issues))
    built: set[AdhocIssue] = set()

    async def build_one(issue: AdhocIssue) -> None:
        async with semaphore:
            if not await governor.meter_adhoc(amount=estimated_amount):
                logger.info(
                    "Ad-hoc drain skipping issue #%d (%s): shared budget spent",
                    issue.issue_number,
                    repo_full_name,
                )
                return
            await build(issue, repo_full_name=repo_full_name)
            built.add(issue)

    await asyncio.gather(*(build_one(issue) for issue in issues))
    # Return in rank order (``issues``); a set membership test keeps the filter O(n).
    return [issue for issue in issues if issue in built]


# A no-priority issue ranks below every labeled severity, so its rank key sits one step
# under the lowest :class:`~retinue.loopback.Severity` (``LOW``). An unknown ``priority:*``
# value parses to ``None`` too (:meth:`retinue.lane.IssueFacts.priority`), so it lands here.
_NO_PRIORITY_RANK = Severity.LOW - 1


def _rank_adhoc(issues: list[ReadyIssue]) -> list[ReadyIssue]:
    """Rank ad-hoc issues by ``priority:<severity>`` (no-priority lowest), stable on number.

    A more severe issue ranks first; a no-priority issue (or an unknown ``priority:*``
    value) ranks lowest. Ties are broken by ascending issue number so the order is
    deterministic across runs.
    """

    def rank_key(issue: ReadyIssue) -> tuple[int, int]:
        severity = issue.severity()
        return (-(severity.value if severity is not None else _NO_PRIORITY_RANK), issue.number)

    return sorted(issues, key=rank_key)
