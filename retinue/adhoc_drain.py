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

* **dedup via GitHub truth** — :meth:`AdhocGh.in_flight` skips any issue whose
  ``issue-<N>`` branch or open PR already exists, mirroring the reconcile-style
  source-of-truth approach (:mod:`retinue.reconcile`) so no duplicate branch/PR is opened;
* **single-run lock** — the whole drain runs under an injected lock so two drains for a
  repo never overlap (a second entry raises :class:`AdhocDrainBusyError`); the lock is
  *separate* from the orchestrator's, so the drain still runs alongside a PRD build;
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
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Protocol

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
    """A second ad-hoc drain was attempted for a repo while one is already in flight.

    The single-run guarantee: :func:`run_adhoc_drain` runs inside an injected lock that
    rejects a concurrent holder rather than blocking, so the "at most one ad-hoc drain
    per repo at a time" contract is observable to the caller. This lock is *separate* from
    the orchestrator's :class:`retinue.orchestrator.OrchestratorBusyError` lock, so an
    ad-hoc drain still runs concurrently with a PRD build. Mirrors
    :class:`retinue.cron.CronBusyError`.
    """

    def __init__(self) -> None:
        super().__init__("an ad-hoc drain is already in flight")


class AdhocDrainLock:
    """The production single-run lock: a non-blocking in-process guard for one repo's drain.

    Satisfies the ``AbstractAsyncContextManager`` :func:`run_adhoc_drain` enters: the first
    holder enters, and a *second* concurrent ``__aenter__`` raises :class:`AdhocDrainBusyError`
    rather than blocking — so the "at most one ad-hoc drain per repo at a time" contract is
    observable to the caller (mirroring the test fake's reject-don't-block behavior). One lock
    instance guards one repo; the worker keeps a per-repo registry so two repos drain
    concurrently while a repo's own kicked and swept drains serialize through the same lock.

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


class AdhocGh(Protocol):
    """The gh queries behind the ad-hoc drain. The ad-hoc lane's gh seam.

    A production implementation runs ``gh issue list --label ready-for-agent`` (with each
    issue's labels and body) for :meth:`list_ready`, and the branch-existence + open-PR
    lookups for :meth:`in_flight`; tests inject a fake that scripts both. Modeled as one
    protocol so the whole drain injects through a single collaborator, mirroring the
    gh-seam style of :mod:`retinue.cron` / :mod:`retinue.reconcile`.
    """

    async def list_ready(self, *, repo_full_name: str) -> list[ReadyIssue]:
        """Return the repo's open ``ready-for-agent`` issues with their labels and body."""
        ...

    async def in_flight(self, *, repo_full_name: str, issue_number: int) -> bool:
        """Return True when GitHub shows the issue is already being built.

        The dedup source of truth, mirroring the reconcile-style GitHub-truth approach
        (:mod:`retinue.reconcile`): an issue is in flight when its ``issue-<N>`` branch
        exists OR an open PR for it exists. Either proves a build is already under way (or
        landed), so the drain skips it rather than rebuilding a duplicate branch/PR.
        """
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

    async def in_flight(self, *, repo_full_name: str, issue_number: int) -> bool:
        """Return True when the ``issue-<N>`` branch exists or an open PR for it exists.

        Queries GitHub truth in two legs, short-circuiting on the first that proves the
        issue is already in flight: the branch-ref lookup (a missing ref 404s, surfaced by
        the runner as :class:`GhCliError`, read here as "no branch"), then the open-PR list
        for the ``issue-<N>`` head. Mirrors the reconcile-style source-of-truth dedup.

        Raises:
            ValueError: the open-PR query returned a payload that did not parse as a JSON
                array.
        """
        branch = _issue_branch(issue_number)
        if await self._branch_exists(repo_full_name, branch):
            return True
        return await self._open_pr_exists(repo_full_name, branch)

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


async def run_adhoc_drain(
    *,
    repo_full_name: str,
    gh: AdhocGh,
    build: AdhocBuild,
    config: RepoConfig,
    governor: BudgetGovernor,
    estimated_amount: float,
    lock: AbstractAsyncContextManager[object],
    prd_in_flight: bool = False,
) -> list[AdhocIssue]:
    """Drain the repo's ad-hoc work, hardened for production: list, filter, dedup, build.

    The whole drain runs under ``lock`` so two drains for the same repo never overlap (a
    second entry raises :class:`AdhocDrainBusyError`); the lock is *separate* from the
    orchestrator's, so the drain still runs concurrently with a PRD build. Inside the lock:

    1. **list** the repo's open ``ready-for-agent`` issues (number, labels, body),
    2. **filter** to the ad-hoc lane via :meth:`ReadyIssue.is_adhoc` — mirroring
       :func:`retinue.lane.classify`'s ad-hoc decision (not calling it) — dropping any
       ``prd``-labeled issue and any issue carrying a ``Part of #<prd>`` link,
    3. **honor PRD-first ordering**: when ``prd_in_flight`` is True, only a
       ``priority:critical``/``high`` issue (:meth:`ReadyIssue.preempts`, the same rule
       :func:`retinue.lane.classify` preempts on) builds — ordinary ad-hoc work waits for
       the PRD to finish,
    4. **rank** the survivors by ``priority:<severity>`` (no-priority lowest),
    5. **dedup** against GitHub truth: skip any issue already in flight (an ``issue-<N>``
       branch or open PR exists, :meth:`AdhocGh.in_flight`), so no duplicate branch/PR,
    6. **drive** the injected ``build`` for each survivor — materialized through
       :meth:`~retinue.adhoc_build.AdhocIssue.from_fetched_issue` (so the ``Chain-depth:``
       marker stays live) — concurrently but capped at ``config.max_parallel`` live builds,
       each metered against the **one shared** :class:`~retinue.budget.BudgetGovernor` (the
       same governor the PRD lane meters): a build that would cross the rolling-24h cap is
       skipped, so the shared budget is never overshot.

    Args:
        repo_full_name: The target repo, e.g. ``"owner/repo"``.
        gh: The ad-hoc gh seam (lists ``ready-for-agent`` issues; answers the in-flight
            dedup query).
        build: The downstream ad-hoc build+PR primitive run per surviving issue, injected
            so the drain runs with no Docker, gh, the Agent SDK, or network.
        config: The accepted repo config; ``max_parallel`` bounds the concurrent builds.
        governor: The shared service-level budget governor; each build is metered through
            it and skipped when the rolling-24h cap leaves no room.
        estimated_amount: The per-build charge metered against the shared cap.
        lock: The single-run lock; entering it raises :class:`AdhocDrainBusyError` when a
            drain for this repo is already in flight. Separate from the PRD build's lock.
        prd_in_flight: Whether a PRD build is currently running for this repo. When True,
            PRD-first ordering holds and only preempting (``critical``/``high``) issues
            build; when False, every ranked ad-hoc issue builds.

    Returns:
        The :class:`~retinue.adhoc_build.AdhocIssue` objects actually driven through
        ``build`` (dedup-skipped and budget-skipped issues excluded), in rank order.

    Raises:
        AdhocDrainBusyError: A drain for this repo is already in flight (from the lock).
    """
    async with lock:
        listed = await gh.list_ready(repo_full_name=repo_full_name)
        candidates = _select_candidates(repo_full_name, prd_in_flight, listed)
        buildable = await _drop_in_flight(repo_full_name, candidates, gh)
        if not buildable:
            logger.info(
                "Ad-hoc drain idle: no buildable ready-for-agent issues for %s",
                repo_full_name,
            )
            return []

        logger.info(
            "Ad-hoc drain building up to %d issue(s) for %s (cap=%s)",
            len(buildable),
            repo_full_name,
            config.max_parallel,
        )
        return await _build_metered(
            repo_full_name,
            buildable,
            build=build,
            config=config,
            governor=governor,
            estimated_amount=estimated_amount,
        )


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


async def _drop_in_flight(
    repo_full_name: str, issues: list[AdhocIssue], gh: AdhocGh
) -> list[AdhocIssue]:
    """Drop issues already in flight (an ``issue-<N>`` branch or open PR exists).

    Mirrors the reconcile-style GitHub-truth dedup: a build already under way (or landed)
    is skipped so the drain never opens a duplicate branch or PR. Order is preserved.
    """
    buildable: list[AdhocIssue] = []
    for issue in issues:
        if await gh.in_flight(
            repo_full_name=repo_full_name, issue_number=issue.issue_number
        ):
            logger.info(
                "Ad-hoc drain skipping issue #%d (%s): already in flight",
                issue.issue_number,
                repo_full_name,
            )
            continue
        buildable.append(issue)
    return buildable


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
    built: list[AdhocIssue] = []

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
            built.append(issue)

    await asyncio.gather(*(build_one(issue) for issue in issues))
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
