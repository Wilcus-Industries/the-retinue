"""The scheduler drain: one pass over a repo's trigger-labeled issues (PRD #80).

This is the retinue's central mechanism — the single drain the webhook kick and the
heartbeat both invoke. One drain runs per repo, under a single-run lock, and in one
stateless pass:

1. **list** every open issue wearing ``config.trigger_label`` (number, labels, body),
2. **admit** the ones the scheduler acts on — trigger label present, ``hitl`` absent
   (the list query already scopes to the label and open state),
3. **gate on readiness** — an issue is schedulable only when every blocker is closed,
   the union of its body ``## Blocked by #N`` refs and GitHub's native relations
   (:func:`retinue.readiness.resolve_ready`); a blocked issue is invisible this pass,
4. **classify flight state** against GitHub truth (:class:`FlightState`): an issue with a
   branch *and* an open PR is in flight (skip, no duplicate), one with a pushed
   ``issue-<N>`` branch but no open PR is **stranded** (a prior green build whose PR never
   opened — open its PR without rebuilding), the rest are buildable,
5. **rank + select** the buildable set through the pure two-queue scheduler
   (:func:`retinue.scheduler.select_to_build`): ready issues split into a priority queue
   (tiers in ``config.priority_tiers``) that always drains first and a main queue, with a
   reserved priority slot when ``config.max_parallel >= 2``,
6. **build** each selected issue in a disposable container, metered against the one shared
   :class:`~retinue.budget.BudgetGovernor`; a build that would cross the rolling-24h cap
   is skipped.

Every leaf I/O — the gh queries, the readiness lookups, the budget store, the downstream
build — is injected and faked, so the whole drain runs with no real ``gh``, no Docker, and
no network. The drain is stateless per pass: each entry recomputes readiness and ranking
from GitHub truth rather than maintaining a persistent queue store.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from retinue.adhoc_build import AdhocIssue
from retinue.budget import BudgetGovernor
from retinue.gh import (
    GhBytesRunner,
    GhCliError,
    auth_env,
    parse_json_array,
    run_gh_subprocess,
)
from retinue.readiness import BlockableIssue, ReadinessGh, resolve_ready
from retinue.repo_config import RepoConfig
from retinue.scheduler import Candidate, select_to_build
from retinue.single_run import SingleRunLock
from retinue.vocab import HITL_LABEL, issue_branch

logger = logging.getLogger(__name__)


class AdhocDrainBusyError(Exception):
    """A second drain was attempted for a repo while one is already in flight.

    The single-run guarantee: :func:`run_adhoc_drain` runs inside an injected lock that
    rejects a concurrent holder rather than blocking, so the "at most one drain per repo
    at a time" contract is observable to the caller. Mirrors :class:`retinue.cron.CronBusyError`.
    """

    def __init__(self) -> None:
        super().__init__("a scheduler drain is already in flight")


class AdhocDrainLock(SingleRunLock):
    """One repo's scheduler-drain single-run lock: a second concurrent drain is rejected.

    A :class:`~retinue.single_run.SingleRunLock` raising :class:`AdhocDrainBusyError`.
    One instance guards one repo; the worker keeps a per-repo registry so two repos drain
    concurrently while a repo's own kicked and swept drains serialize through the same lock.
    """

    busy_error = AdhocDrainBusyError


# How many trigger-labeled issues to pull per drain. The cap on concurrent *builds* is
# ``config.max_parallel``; this generous-but-bounded page just keeps the visible set from
# an unbounded fetch, mirroring the cron lane's list limit.
_DEFAULT_LIST_LIMIT = 200


@dataclass(frozen=True)
class ReadyIssue:
    """One open trigger-labeled issue, as reported by the drain's gh seam.

    The body is surfaced (unlike the cron lane's backlog seam) because the drain feeds it
    to :meth:`~retinue.adhoc_build.AdhocIssue.from_fetched_issue` (which reads the
    ``Chain-depth:`` lineage marker) and to :func:`retinue.readiness.resolve_ready` (which
    scans the ``## Blocked by`` block).

    Attributes:
        number: The issue number; the build commits to the derived ``issue-<N>`` branch.
        labels: The issue's label names (carries the trigger label and, optionally, a
            ``priority:<tier>`` label the scheduler ranks on).
        body: The issue body, scanned for the ``## Blocked by`` block and the
            ``Chain-depth:`` marker.
    """

    number: int
    labels: list[str]
    body: str = ""

    def is_admissible(self, config: RepoConfig) -> bool:
        """Whether the scheduler admits this issue: trigger label present, ``hitl`` absent.

        The list query already scopes to open issues wearing ``config.trigger_label``, but
        the trigger-label check is kept as defense-in-depth; the ``hitl`` exclusion is the
        one admission decision made here — a human-escalated issue stays out of scheduling
        until the ``hitl`` label is removed (the human "resume" gesture).
        """
        if config.trigger_label not in self.labels:
            return False
        return HITL_LABEL not in self.labels


class FlightState(Enum):
    """GitHub's verdict on an issue's ``issue-<N>`` branch and open PR — the drain's truth.

    * :attr:`ABSENT` — no ``issue-<N>`` branch and no open PR: nothing was built, so build
      it fresh.
    * :attr:`STRANDED` — the ``issue-<N>`` branch exists but no open PR: a prior build
      pushed the branch (push-only-on-green, so the branch is provably green) yet never
      opened its PR. The drain opens the PR **without rebuilding**.
    * :attr:`IN_FLIGHT` — an open PR exists: a build is under way or already landed, so the
      drain skips the issue.
    """

    ABSENT = "absent"
    STRANDED = "stranded"
    IN_FLIGHT = "in_flight"


@dataclass(frozen=True)
class FlightSnapshot:
    """Whole-repo GitHub truth for flight-state classification, fetched once per drain.

    Carries the two whole-repo sets the drain classifies every candidate against in memory,
    replacing the per-issue branch-ref + open-PR spawns with a single prefetch: the head
    branch names of the repo's open PRs, and the existing ``issue-<N>`` branch names.

    Attributes:
        open_pr_heads: The head branch names of the repo's open PRs (an open PR keeps its
            head alive, so these are a subset of ``issue_branches``).
        issue_branches: The existing ``issue-<N>`` branch names on the repo.
    """

    open_pr_heads: frozenset[str]
    issue_branches: frozenset[str]

    def state_for(self, issue_number: int) -> FlightState:
        """Classify one issue from the snapshot, mirroring per-issue ``flight_state``.

        A missing ``issue-<N>`` branch is :attr:`FlightState.ABSENT`; a branch with an open
        PR is :attr:`FlightState.IN_FLIGHT`; a branch with no open PR is
        :attr:`FlightState.STRANDED` (pushed green, PR never opened).
        """
        branch = issue_branch(issue_number)
        if branch not in self.issue_branches:
            return FlightState.ABSENT
        if branch in self.open_pr_heads:
            return FlightState.IN_FLIGHT
        return FlightState.STRANDED


class AdhocGh(Protocol):
    """The gh queries behind the scheduler drain. The drain's gh seam.

    A production implementation runs ``gh issue list --label <trigger>`` (with each issue's
    labels and body) for :meth:`list_ready`, and the branch-existence + open-PR lookups for
    :meth:`flight_state`; tests inject a fake that scripts both. A production seam should
    also implement :class:`SupportsFlightSnapshot` so the drain classifies flight state with
    one whole-repo prefetch instead of a per-issue :meth:`flight_state` spawn.
    """

    async def list_ready(
        self, *, repo_full_name: str, label: str
    ) -> list[ReadyIssue]:
        """Return the repo's open ``label``-labeled issues with their labels and body."""
        ...

    async def flight_state(
        self, *, repo_full_name: str, issue_number: int
    ) -> FlightState:
        """Classify the issue against GitHub truth: absent, stranded, or in flight.

        The dedup + stranded-recovery source of truth: reads whether the issue's
        ``issue-<N>`` branch exists and whether an open PR for it exists, and returns the
        matching :class:`FlightState`. This is the *per-issue* fallback; a seam that also
        implements :class:`SupportsFlightSnapshot` is classified with one whole-repo query.
        """
        ...


@runtime_checkable
class SupportsFlightSnapshot(Protocol):
    """An optional whole-repo flight-state prefetch the drain prefers over per-issue calls.

    A gh seam that can answer the flight-state question for the *whole* repo in one shot
    implements this. The production :class:`GhCli` does — collapsing the old N-issue x
    2-spawn classification into two whole-repo ``gh`` queries — while a seam offering only
    per-issue :meth:`AdhocGh.flight_state` is classified through that fallback instead.
    """

    async def flight_snapshot(self, *, repo_full_name: str) -> FlightSnapshot:
        """Fetch the repo's whole flight-state truth in one prefetch (open PRs + branches)."""
        ...


class GhCli:
    """The production :class:`AdhocGh`: lists trigger-labeled issues via the ``gh`` CLI.

    Runs ``gh issue list --repo <repo> --label <trigger> --state open --json
    number,labels,body`` and parses the JSON into :class:`ReadyIssue` objects. The ``body``
    field is requested because the drain feeds it to
    :meth:`~retinue.adhoc_build.AdhocIssue.from_fetched_issue` (lineage marker) and to
    readiness (the ``## Blocked by`` block). Authenticates by injecting the GitHub token
    into the child env as ``GH_TOKEN``, so no token is ever placed on the command line.

    The subprocess spawn is the one impure edge, factored behind the injected ``runner``
    (the shared :data:`~retinue.gh.GhBytesRunner` seam), so command assembly, the auth env,
    and payload parsing are unit-testable without a real ``gh``, Docker, or network.

    Args:
        token: The GitHub token ``gh`` authenticates with, placed in the child env as
            ``GH_TOKEN``. ``None`` runs ``gh`` with the ambient auth.
        runner: The injected argv runner; defaults to the real subprocess spawn.
        list_limit: The max number of trigger-labeled issues to pull per drain.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        runner: GhBytesRunner | None = None,
        list_limit: int = _DEFAULT_LIST_LIMIT,
    ) -> None:
        self._token = token
        self._runner = runner or run_gh_subprocess
        self._list_limit = list_limit

    async def list_ready(
        self, *, repo_full_name: str, label: str
    ) -> list[ReadyIssue]:
        """Return the repo's open ``label``-labeled issues with their labels and body.

        Raises:
            GhCliError: ``gh`` exited non-zero (propagated from the runner).
            ValueError: ``gh`` returned a payload that did not parse as the expected
                issue listing.
        """
        argv = _list_ready_argv(repo_full_name, label=label, limit=self._list_limit)
        stdout = await self._runner(argv, auth_env(self._token))
        return [_parse_ready_issue(entry) for entry in parse_json_array(stdout)]

    async def flight_state(
        self, *, repo_full_name: str, issue_number: int
    ) -> FlightState:
        """Classify the issue from GitHub truth: absent, stranded, or in flight.

        Queries GitHub truth in two legs. First the branch-ref lookup (a missing ref 404s,
        read here as "no branch"): no branch short-circuits to :attr:`FlightState.ABSENT` —
        an open PR keeps its head alive, so a missing branch proves no open PR exists. When
        the branch exists, the open-PR list for the ``issue-<N>`` head decides.

        Raises:
            ValueError: the open-PR query returned a payload that did not parse as a JSON
                array.
        """
        branch = issue_branch(issue_number)
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
        branch ref. The drain classifies each candidate in memory
        (:meth:`FlightSnapshot.state_for`), preserving the stranded-branch recovery.

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
        stdout = await self._runner(argv, auth_env(self._token))
        return frozenset(_parse_pr_heads(stdout))

    async def _issue_branches(self, repo_full_name: str) -> frozenset[str]:
        """Every existing ``issue-<N>`` branch name on the repo (the whole branch set)."""
        argv = _issue_refs_argv(repo_full_name)
        stdout = await self._runner(argv, auth_env(self._token))
        return frozenset(_parse_issue_branches(stdout))

    async def _branch_exists(self, repo_full_name: str, branch: str) -> bool:
        """Whether ``branch`` resolves to a ref on the repo (a 404 means it does not)."""
        argv = _branch_ref_argv(repo_full_name, branch)
        try:
            await self._runner(argv, auth_env(self._token))
        except GhCliError:
            # A missing ref is a 404 / non-zero exit — not an error to propagate, just a
            # False answer to the existence question (mirrors pr_opener's staging_exists).
            return False
        return True

    async def _open_pr_exists(self, repo_full_name: str, branch: str) -> bool:
        """Whether an open PR with ``branch`` as its head exists on the repo."""
        argv = _open_pr_argv(repo_full_name, branch)
        stdout = await self._runner(argv, auth_env(self._token))
        return len(parse_json_array(stdout)) > 0


def _list_ready_argv(repo_full_name: str, *, label: str, limit: int) -> list[str]:
    """Assemble the ``gh issue list`` argv for the open ``label``-labeled issues.

    Pulls ``number``, ``labels``, and ``body`` as JSON — exactly the fields the drain needs
    to admit + rank each issue (the ``priority:*`` label), gate readiness (the
    ``## Blocked by`` block), and rebuild it through
    :meth:`~retinue.adhoc_build.AdhocIssue.from_fetched_issue` (the ``Chain-depth:`` marker).
    """
    return [
        "gh",
        "issue",
        "list",
        "--repo",
        repo_full_name,
        "--label",
        label,
        "--state",
        "open",
        "--json",
        "number,labels,body",
        "--limit",
        str(limit),
    ]


def _branch_ref_argv(repo_full_name: str, branch: str) -> list[str]:
    """Assemble the ``gh api`` argv reading one branch ref (404s when it does not exist)."""
    return [
        "gh",
        "api",
        f"repos/{repo_full_name}/git/ref/heads/{branch}",
    ]


def _open_pr_argv(repo_full_name: str, head: str) -> list[str]:
    """Assemble the ``gh pr list`` argv for the open PRs with ``head`` as their branch."""
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
    """Assemble the ``gh pr list`` argv for every open PR's head branch (whole-repo)."""
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
    for entry in parse_json_array(stdout):
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
    for entry in parse_json_array(stdout):
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


# The downstream the drain drives for each selected issue: the build+PR primitive
# (:func:`retinue.adhoc_build.build_adhoc_issue` followed by the PR step), behind one
# injected callable so the drain is exercised without Docker, gh, the Agent SDK, or
# network. The drain hands it the materialized :class:`~retinue.adhoc_build.AdhocIssue`
# (built through ``from_fetched_issue``, so its ``chain_depth`` is live) and the repo name.
AdhocBuild = Callable[..., Awaitable[None]]

# The PR-open-only recovery the drain drives for a :attr:`FlightState.STRANDED` issue: open
# the PR for an already-pushed (green) ``issue-<N>`` branch without rebuilding it. Same
# ``(issue, *, repo_full_name) -> None`` shape as :data:`AdhocBuild`, injected and faked.
AdhocPrOpen = Callable[..., Awaitable[None]]


async def run_adhoc_drain(
    *,
    repo_full_name: str,
    gh: AdhocGh,
    readiness_gh: ReadinessGh,
    build: AdhocBuild,
    open_pr: AdhocPrOpen,
    config: RepoConfig,
    governor: BudgetGovernor,
    estimated_amount: float,
    lock: AbstractAsyncContextManager[object],
) -> list[AdhocIssue]:
    """Drain the repo's scheduler work in one stateless pass: list, admit, rank, act.

    The whole drain runs under ``lock`` so two drains for the same repo never overlap (a
    second entry raises :class:`AdhocDrainBusyError`). Inside the lock:

    1. **list** the repo's open ``config.trigger_label`` issues (number, labels, body),
    2. **admit** via :meth:`ReadyIssue.is_admissible` — trigger label present, ``hitl``
       absent,
    3. **gate on readiness** — drop any issue with an open blocker (the union of body
       ``## Blocked by #N`` refs and native GitHub relations,
       :func:`retinue.readiness.resolve_ready`),
    4. **classify** each ready survivor against GitHub truth (:class:`FlightState`) and
       partition: an in-flight issue (open PR) is skipped, a **stranded** one (pushed green
       ``issue-<N>`` branch, no open PR) goes to the PR-open-only recovery, the rest are
       buildable,
    5. **recover** every stranded issue by driving ``open_pr`` — opening the PR for its
       already-green branch with **no rebuild** and no budget charge,
    6. **rank + select** the buildable set through :func:`retinue.scheduler.select_to_build`
       (``cap=config.max_parallel``): the priority queue drains first, the main queue holds
       at most ``cap-1`` slots (the reserved priority slot) when the cap is ``>= 2``,
    7. **build** each selected issue — materialized through
       :meth:`~retinue.adhoc_build.AdhocIssue.from_fetched_issue` (so ``Chain-depth:`` stays
       live) — concurrently, each metered against the **one shared**
       :class:`~retinue.budget.BudgetGovernor`: a build that would cross the rolling-24h cap
       is skipped, so the shared budget is never overshot.

    Args:
        repo_full_name: The target repo, e.g. ``"owner/repo"``.
        gh: The drain's gh seam (lists trigger-labeled issues; answers flight-state).
        readiness_gh: The readiness gh seam (native blockers + issue closed-state).
        build: The downstream build+PR primitive run per selected issue, injected so the
            drain runs with no Docker, gh, the Agent SDK, or network.
        open_pr: The PR-open-only recovery run per stranded issue.
        config: The accepted repo config; ``trigger_label``, ``severity_tiers``,
            ``priority_tiers``, and ``max_parallel`` govern admission, ranking, and the cap.
        governor: The shared service-level budget governor; each build is metered through it.
        estimated_amount: The per-build charge metered against the shared cap.
        lock: The single-run lock; entering it raises :class:`AdhocDrainBusyError` when a
            drain for this repo is already in flight.

    Returns:
        The :class:`~retinue.adhoc_build.AdhocIssue` objects actually driven through
        ``build`` (blocked, in-flight-skipped, stranded, unselected, and budget-skipped
        issues excluded), in scheduler rank order.

    Raises:
        AdhocDrainBusyError: A drain for this repo is already in flight (from the lock).
    """
    async with lock:
        listed = await gh.list_ready(
            repo_full_name=repo_full_name, label=config.trigger_label
        )
        admitted = [issue for issue in listed if issue.is_admissible(config)]
        ready = await _filter_ready(repo_full_name, admitted, readiness_gh)
        plan = await _partition_candidates(repo_full_name, ready, gh)
        await _open_stranded_prs(repo_full_name, plan.stranded, open_pr)

        selected = _select_buildable(repo_full_name, config, plan.buildable)
        if not selected:
            logger.info(
                "Scheduler drain idle: no buildable issues for %s "
                "(%d stranded PR(s) recovered)",
                repo_full_name,
                len(plan.stranded),
            )
            return []

        logger.info(
            "Scheduler drain building %d issue(s) for %s (cap=%s)",
            len(selected),
            repo_full_name,
            config.max_parallel,
        )
        return await _build_metered(
            repo_full_name,
            selected,
            build=build,
            config=config,
            governor=governor,
            estimated_amount=estimated_amount,
        )


async def _filter_ready(
    repo_full_name: str, issues: list[ReadyIssue], readiness_gh: ReadinessGh
) -> list[ReadyIssue]:
    """Keep only the issues whose every blocker is closed, preserving list order.

    Delegates to :func:`retinue.readiness.resolve_ready` — the union of body ``## Blocked
    by`` refs and native GitHub relations — so a blocked issue is invisible this pass.
    """
    ready_numbers = await resolve_ready(
        [BlockableIssue(number=issue.number, body=issue.body) for issue in issues],
        repo_full_name=repo_full_name,
        gh=readiness_gh,
    )
    return [issue for issue in issues if issue.number in ready_numbers]


def _select_buildable(
    repo_full_name: str, config: RepoConfig, buildable: list[ReadyIssue]
) -> list[AdhocIssue]:
    """Rank the buildable set and pick this pass's builds, then materialize each issue.

    Runs the pure two-queue scheduler (:func:`retinue.scheduler.select_to_build`) over the
    buildable candidates with ``cap=config.max_parallel``: the priority queue drains first,
    the main queue holds at most ``cap-1`` slots (the reserved priority slot). Each selected
    issue is materialized through :meth:`AdhocIssue.from_fetched_issue` so its
    ``chain_depth`` is read back from its body.
    """
    by_number = {issue.number: issue for issue in buildable}
    chosen = select_to_build(
        config,
        [Candidate(number=issue.number, labels=issue.labels) for issue in buildable],
        cap=config.max_parallel,
    )
    return [
        AdhocIssue.from_fetched_issue(
            repo_full_name, c.number, by_number[c.number].body
        )
        for c in chosen
    ]


@dataclass(frozen=True)
class _DrainPlan:
    """How the drain will act on its ready candidates, partitioned by flight state.

    Attributes:
        buildable: :attr:`FlightState.ABSENT` issues — nothing built yet, so rank + build
            them (metered against the shared budget).
        stranded: :attr:`FlightState.STRANDED` issues — a green branch with no open PR, so
            open the PR without rebuilding (no budget charge).
    """

    buildable: list[ReadyIssue]
    stranded: list[ReadyIssue]


async def _partition_candidates(
    repo_full_name: str, issues: list[ReadyIssue], gh: AdhocGh
) -> _DrainPlan:
    """Classify each ready candidate against GitHub truth and split build vs PR-open.

    An :attr:`~FlightState.IN_FLIGHT` issue (open PR) is dropped so the drain opens no
    duplicate; a :attr:`~FlightState.STRANDED` issue (pushed green branch, no PR) routes to
    the PR-open-only recovery rather than a wasteful rebuild; the rest are buildable.
    """
    states = await _flight_states(repo_full_name, issues, gh)
    buildable: list[ReadyIssue] = []
    stranded: list[ReadyIssue] = []
    for issue in issues:
        state = states[issue.number]
        if state is FlightState.IN_FLIGHT:
            logger.info(
                "Scheduler drain skipping issue #%d (%s): already in flight",
                issue.number,
                repo_full_name,
            )
        elif state is FlightState.STRANDED:
            logger.info(
                "Scheduler drain recovering issue #%d (%s): green branch with no PR; "
                "opening its PR without rebuilding",
                issue.number,
                repo_full_name,
            )
            stranded.append(issue)
        else:
            buildable.append(issue)
    return _DrainPlan(buildable=buildable, stranded=stranded)


async def _flight_states(
    repo_full_name: str, issues: list[ReadyIssue], gh: AdhocGh
) -> dict[int, FlightState]:
    """Resolve each candidate's :class:`FlightState`, preferring one whole-repo query.

    A seam that supports the whole-repo :class:`FlightSnapshot` (the production
    :class:`GhCli`) is queried once and every candidate is classified in memory. A seam
    offering only the per-issue :meth:`AdhocGh.flight_state` is classified one issue at a
    time. Both paths yield the identical verdicts.
    """
    if isinstance(gh, SupportsFlightSnapshot):
        snapshot = await gh.flight_snapshot(repo_full_name=repo_full_name)
        return {
            issue.number: snapshot.state_for(issue.number) for issue in issues
        }
    return {
        issue.number: await gh.flight_state(
            repo_full_name=repo_full_name, issue_number=issue.number
        )
        for issue in issues
    }


async def _open_stranded_prs(
    repo_full_name: str, stranded: list[ReadyIssue], open_pr: AdhocPrOpen
) -> None:
    """Open the PR for each stranded green branch, without rebuilding.

    Opening a PR for an already-green branch does no model work, so it is not metered
    against the shared budget — a stranded build is recovered even when the cap is spent.
    Each issue is materialized through :meth:`AdhocIssue.from_fetched_issue` so a recovered
    review-fix issue keeps its ``Chain-depth:`` lineage.
    """
    for issue in stranded:
        materialized = AdhocIssue.from_fetched_issue(
            repo_full_name, issue.number, issue.body
        )
        await open_pr(materialized, repo_full_name=repo_full_name)


async def _build_metered(
    repo_full_name: str,
    issues: list[AdhocIssue],
    *,
    build: AdhocBuild,
    config: RepoConfig,
    governor: BudgetGovernor,
    estimated_amount: float,
) -> list[AdhocIssue]:
    """Build the selected issues concurrently, each metered against the shared budget.

    The scheduler has already capped the selection to ``config.max_parallel`` (with the
    reserved priority slot), so all selected issues build concurrently; a semaphore keeps
    the concurrency bounded as a belt-and-braces guard. Every build first meters its charge
    against the one shared :class:`~retinue.budget.BudgetGovernor`; a build that would cross
    the rolling-24h cap is skipped. Returns the issues that actually built, in rank order.
    """
    semaphore = asyncio.Semaphore(config.max_parallel or len(issues))
    built: set[AdhocIssue] = set()

    async def build_one(issue: AdhocIssue) -> None:
        async with semaphore:
            if not await governor.meter_adhoc(amount=estimated_amount):
                logger.info(
                    "Scheduler drain skipping issue #%d (%s): shared budget spent",
                    issue.issue_number,
                    repo_full_name,
                )
                return
            await build(issue, repo_full_name=repo_full_name)
            built.add(issue)

    await asyncio.gather(*(build_one(issue) for issue in issues))
    # Return in rank order (``issues``); a set membership test keeps the filter O(n).
    return [issue for issue in issues if issue in built]
