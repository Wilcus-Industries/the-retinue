"""Cron backlog drainer: drain loose ``backlog`` issues one at a time (issue #15).

A scheduled lane drains the loose ``backlog`` issues — the non-blocking heimdall nits
filed by :mod:`retinue.loopback` — one per tick, alongside the orchestrator's PRD builds.
The :mod:`retinue.lane` classifier routes work between the two lanes; this module is the
cron lane's per-tick driver.

Each :func:`run_cron_tick`:

1. runs under an injected single-run **lock** so at most one cron run executes at a time,
   mirroring the orchestrator's single-run lock;
2. **gates** on the shared :class:`retinue.budget.BudgetGovernor` — the *same*
   service-level governor the orchestrator shares — and **defers** when the budget is
   spent, picking nothing and running no downstream; an admitted tick's estimate is
   charged to the shared ledger at the gate (an empty backlog is checked first, so an
   idle tick never charges);
3. **picks** the next backlog issue by a weighted score (priority + age), except on every
   ``quota_every``-th tick where a **quota floor** takes the oldest low-priority issue so
   the low items provably drain rather than starving behind a steady high-priority stream;
4. runs the same downstream the orchestrator drives (build -> PR -> heimdall loopback ->
   notify) via a single injected :data:`CronBuild` callable.

The clock is injected (:class:`retinue.budget.Clock`) for age-weighting and the tick
counter is passed in, so nothing reads the wall clock. The gh backlog query, the budget
governor, the single-run lock, and the downstream build are all injected and faked, so a
tick runs with no real ``gh``, no Docker, and no network.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from retinue.budget import BudgetGovernor, Clock
from retinue.container import ContainerRuntime
from retinue.container_build import Implementer, Slice
from retinue.done_check import DEFAULT_IMAGE, ReportSink, SecretResolver
from retinue.gh import GhBytesRunner, auth_env, parse_json_array, run_gh_subprocess
from retinue.github_app import InstallationAuth
from retinue.orchestrator import (
    BuildResult,
    GitOps,
    build_slice,
)
from retinue.repo_config import RepoConfig
from retinue.vocab import BACKLOG_LABEL, Severity, parse_priority

logger = logging.getLogger(__name__)

# A day's age is worth this much weighted score; one severity step is worth a day's worth
# multiplied by this lever. Keeping a severity step strictly larger than any realistic age
# contribution makes priority dominate, while age still breaks ties within a severity. The
# quota floor (below) is what stops a low item from starving when priority always wins.
_AGE_WEIGHT_PER_DAY = 1.0
_PRIORITY_WEIGHT = 10_000.0

# A backlog issue with no parsable ``priority:*`` label is scored as LOW so it still ranks
# and is still swept up by the low-priority quota floor.
_DEFAULT_SEVERITY = Severity.LOW

# Below this severity an issue is "low priority" for the quota floor — the items the
# every-Nth tick deliberately drains so they never starve behind higher-priority work.
_LOW_PRIORITY_CEILING = Severity.MEDIUM


class CronBusyError(Exception):
    """A second cron tick was attempted while one is already in flight.

    The single-run guarantee: :func:`run_cron_tick` runs inside an injected lock that
    rejects a concurrent holder rather than blocking, so the "at most one cron run at a
    time" contract is observable to the caller.
    """

    def __init__(self) -> None:
        super().__init__("a cron tick is already in flight")


class CronLock:
    """The production single-run lock for the backlog cron tick: a non-blocking guard.

    Satisfies the ``AbstractAsyncContextManager`` :func:`run_cron_tick` enters: the first
    holder enters, and a *second* concurrent ``__aenter__`` raises :class:`CronBusyError`
    rather than blocking — so the "at most one cron tick at a time" contract is observable
    to the caller. The worker keeps a per-repo registry so two repos tick concurrently
    while a repo's own ticks serialize through the same lock. Mirrors
    :class:`retinue.adhoc_drain.AdhocDrainLock`.

    The guard is a plain in-process flag (no real wall-clock, Redis, or file lock), correct
    because the whole tick runs inside a single worker process; a cross-process lock is out
    of scope for the single-worker deployment.
    """

    def __init__(self) -> None:
        self._held = False

    async def __aenter__(self) -> CronLock:
        if self._held:
            raise CronBusyError
        self._held = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        self._held = False


@dataclass(frozen=True)
class BacklogIssue:
    """One loose ``backlog`` issue, as reported by the backlog gh seam.

    Attributes:
        number: The issue number.
        labels: The issue's label names (carries ``backlog`` and a ``priority:<severity>``).
        created_at: When the issue was opened; the age input to the weighted score.
    """

    number: int
    labels: list[str]
    created_at: datetime

    def severity(self) -> Severity:
        """The issue's ``priority:<severity>`` as a :class:`Severity` (LOW when absent)."""
        severity = parse_priority(self.labels)
        return _DEFAULT_SEVERITY if severity is None else severity


class CronGh(Protocol):
    """The gh query behind the backlog drain. The cron lane's gh seam.

    A production implementation runs ``gh issue list --label backlog`` (with each issue's
    labels and ``createdAt``); tests inject a fake that returns scripted issues. Modeled
    as a protocol so the whole tick injects through a single collaborator, mirroring the
    gh-seam style of :mod:`retinue.reconcile` / :mod:`retinue.handoff`.
    """

    async def list_backlog(self, *, repo_full_name: str) -> list[BacklogIssue]:
        """Return the repo's open ``backlog`` issues with their labels and ages."""
        ...


# How many backlog issues to pull per tick. The drainer only ever picks one, but it
# scores across the visible set, so a generous-but-bounded page keeps the score honest
# without an unbounded fetch.
_DEFAULT_LIST_LIMIT = 200

class GhCli:
    """The production :class:`CronGh`: lists ``backlog`` issues via the ``gh`` CLI.

    Runs ``gh issue list --repo <repo> --label backlog --state open --json
    number,labels,createdAt`` and parses the JSON into :class:`BacklogIssue` objects.
    Authenticates by injecting the GitHub token into the child env as ``GH_TOKEN`` (the
    same variable the ``gh`` CLI reads), so no token is ever placed on the command line.

    The actual subprocess spawn is the one impure edge, factored behind the injected
    ``runner`` so command assembly, the auth env, and payload parsing are unit-testable
    without a real ``gh``, Docker, or network. Production leaves ``runner`` defaulted to
    :func:`retinue.gh.run_gh_subprocess`.

    Args:
        token: The GitHub token ``gh`` authenticates with, placed in the child env as
            ``GH_TOKEN``. ``None`` runs ``gh`` with the ambient auth (e.g. a logged-in
            CLI), useful for local runs.
        runner: The injected argv runner; defaults to the real subprocess spawn.
        list_limit: The max number of backlog issues to pull per tick.
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

    async def list_backlog(self, *, repo_full_name: str) -> list[BacklogIssue]:
        """Return the repo's open ``backlog`` issues with their labels and ages.

        Assembles the ``gh issue list`` argv, runs it through the injected runner with
        the auth env, and parses the JSON payload into :class:`BacklogIssue` objects.

        Raises:
            GhCliError: ``gh`` exited non-zero (propagated from the runner).
            ValueError: ``gh`` returned a payload that did not parse as the expected
                issue listing.
        """
        argv = _list_backlog_argv(repo_full_name, limit=self._list_limit)
        stdout = await self._runner(argv, auth_env(self._token))
        return _parse_backlog(stdout)


def _list_backlog_argv(repo_full_name: str, *, limit: int) -> list[str]:
    """Assemble the ``gh issue list`` argv for the open ``backlog`` issues of a repo.

    Pulls ``number``, ``labels``, and ``createdAt`` as JSON — exactly the fields
    :class:`BacklogIssue` needs for its severity + age weighting.
    """
    return [
        "gh",
        "issue",
        "list",
        "--repo",
        repo_full_name,
        "--label",
        BACKLOG_LABEL,
        "--state",
        "open",
        "--json",
        "number,labels,createdAt",
        "--limit",
        str(limit),
    ]


def _parse_backlog(stdout: bytes) -> list[BacklogIssue]:
    """Parse a ``gh issue list --json`` payload into :class:`BacklogIssue` objects.

    Each entry carries ``number``, ``createdAt`` (an ISO-8601 timestamp, ``Z``-suffixed
    UTC), and ``labels`` (a list of ``{"name": ...}`` objects). A malformed payload raises
    :class:`ValueError` rather than silently dropping issues.
    """
    return [_parse_issue(entry) for entry in parse_json_array(stdout)]


def _parse_issue(entry: object) -> BacklogIssue:
    """Parse one ``gh`` issue object into a :class:`BacklogIssue`."""
    if not isinstance(entry, dict):
        raise ValueError(f"gh issue entry is not an object: {entry!r}")
    try:
        number = int(entry["number"])
        labels = [str(label["name"]) for label in entry["labels"]]
        created_at = _parse_timestamp(str(entry["createdAt"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"gh issue entry is malformed: {entry!r} ({exc})") from exc
    return BacklogIssue(number=number, labels=labels, created_at=created_at)


def _parse_timestamp(raw: str) -> datetime:
    """Parse gh's ISO-8601 ``createdAt`` (``Z``-suffixed UTC) into an aware datetime."""
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


# The downstream a cron tick drives for the picked issue: the same build -> PR ->
# heimdall loopback -> notify chain the orchestrator runs, behind one injected callable so
# the tick is exercised without Docker, gh, or network. Production wires it to the
# orchestrator build + pr_opener + loopback + handoff chain.
CronBuild = Callable[..., Awaitable[None]]


# A loose backlog nit has no parent PRD — it is filed standalone by the heimdall loopback.
# The cron lane drains each onto its own integration target so a nit's build never collides
# with another's; the per-issue PRD number is the issue number itself, giving the dedicated
# integration branch ``retinue/prd-<issue>`` (see :func:`retinue.orchestrator.integration_branch`).
def _slice_for_backlog_issue(repo_full_name: str, issue_number: int) -> Slice:
    """Assemble the standalone :class:`Slice` the cron lane builds for a backlog nit.

    A loose backlog issue carries no parent PRD, so it drains onto its own integration
    target: the per-issue PRD number is the issue number itself, yielding the dedicated
    integration branch ``retinue/prd-<issue>``. The assembly is pure — no gh, Docker, or
    network — so it is unit-testable in isolation.
    """
    return Slice(
        repo_full_name=repo_full_name,
        issue_number=issue_number,
        prd_number=issue_number,
    )


# The side-effecting build of one assembled slice: the orchestrator's spawn -> done-check
# -> merge chain (:func:`retinue.orchestrator.build_slice`). Injected so the slice
# assembly + downstream wiring of :class:`SliceBuilder` are exercised without the Agent
# SDK, Docker, gh, or network, mirroring the injected-runner style of :class:`GhCli`. The
# default (:func:`_run_build_slice`) calls the real orchestrator.
SliceRunner = Callable[[Slice], Awaitable[BuildResult]]


class SliceBuilder:
    """The production :class:`CronBuild`: drains one backlog nit through ``build_slice``.

    Each cron tick hands this the picked backlog ``issue_number``; the builder assembles
    the standalone :class:`Slice` for it (:func:`_slice_for_backlog_issue`) and drives the
    same downstream the orchestrator runs — spawn the implementer, gate on the done-check,
    merge the green branch — via :func:`retinue.orchestrator.build_slice`.

    All of that downstream's side-effecting collaborators (implementer, git, auth,
    container runtime, secret resolver, report sink) are carried here and threaded into a
    single injected ``runner``, so the cron-side decision and slice assembly are
    unit-testable without the Agent SDK, Docker, gh, or network. Production leaves
    ``runner`` defaulted to :func:`_run_build_slice`, which calls the real orchestrator
    with the carried collaborators.

    Args:
        config: The accepted repo config (its ``staging_branch`` bases a new integration
            branch, its ``secrets`` feed the done-check).
        claude_md: The repo's ``CLAUDE.md`` text carrying the done-check command.
        implementer: The Agent SDK seam that builds the slice on its ``issue-<N>`` branch.
        git: The integration-branch git operations (the merge seam).
        auth: Mints the installation token used to clone (the auth seam).
        runtime: Spawns the disposable container the done-check runs in (Docker seam).
        resolve_secret: Resolves the config's declared secret names/refs to values.
        report: Sink the done-check outcome is posted to.
        image: Container image the done-check runs in.
        runner: The injected slice runner; defaults to the real ``build_slice`` call.
    """

    def __init__(
        self,
        *,
        config: RepoConfig,
        claude_md: str,
        implementer: Implementer,
        git: GitOps,
        auth: InstallationAuth,
        runtime: ContainerRuntime,
        resolve_secret: SecretResolver,
        report: ReportSink,
        image: str = DEFAULT_IMAGE,
        runner: SliceRunner | None = None,
    ) -> None:
        self._config = config
        self._claude_md = claude_md
        self._implementer = implementer
        self._git = git
        self._auth = auth
        self._runtime = runtime
        self._resolve_secret = resolve_secret
        self._report = report
        self._image = image
        self._runner = runner or self._default_runner

    async def __call__(self, *, repo_full_name: str, issue_number: int) -> None:
        """Drain one backlog nit: assemble its slice and run the orchestrator downstream.

        Matches the :data:`CronBuild` protocol invoked by :func:`run_cron_tick`. Assembles
        the standalone slice for ``issue_number`` and hands it to the injected runner; the
        tick does not read a return value, it gates on the budget governor up front.
        """
        slice_ = _slice_for_backlog_issue(repo_full_name, issue_number)
        result = await self._runner(slice_)
        logger.info(
            "Cron drained backlog issue #%d -> %s (%s)",
            issue_number,
            result.outcome.value,
            result.integration_branch,
        )

    async def _default_runner(self, slice_: Slice) -> BuildResult:
        """Run the real orchestrator ``build_slice`` for ``slice_`` with the carried deps.

        The one impure edge, factored behind the :data:`SliceRunner` seam so the assembly
        above is testable without it.
        """
        return await build_slice(
            slice_,
            self._config,
            self._claude_md,
            implementer=self._implementer,
            git=self._git,
            auth=self._auth,
            runtime=self._runtime,
            resolve_secret=self._resolve_secret,
            report=self._report,
            image=self._image,
        )


class CronOutcome(enum.Enum):
    """Why a cron tick ran a build, deferred, or found nothing to do."""

    RAN = "ran"
    DEFERRED = "deferred"
    IDLE = "idle"


@dataclass(frozen=True)
class CronTickResult:
    """Outcome of one cron tick.

    Attributes:
        outcome: ``RAN`` when an issue was picked and its downstream ran; ``DEFERRED``
            when the shared budget was spent; ``IDLE`` when the backlog was empty.
        issue_number: The drained issue on ``RAN``; ``None`` otherwise.
        defer_until: When the budget window frees on ``DEFERRED``; ``None`` otherwise.
    """

    outcome: CronOutcome
    issue_number: int | None = None
    defer_until: datetime | None = None


async def run_cron_tick(
    *,
    repo_full_name: str,
    gh: CronGh,
    governor: BudgetGovernor,
    clock: Clock,
    build: CronBuild,
    tick_number: int,
    estimated_amount: float,
    lock: AbstractAsyncContextManager[object],
    quota_every: int = 5,
) -> CronTickResult:
    """Drain one backlog issue: gate on budget, pick by score/quota, run the downstream.

    Runs under ``lock`` so at most one cron tick executes at a time. An empty backlog is
    ``IDLE`` and never touches the budget; otherwise the tick gates on the shared
    ``governor`` — which *charges* an admitted estimate to the shared rolling-24h ledger —
    and **defers** (picking nothing, running and charging nothing) when the budget is
    spent. An admitted tick picks the next backlog issue — the highest weighted score
    (priority + age), except on every ``quota_every``-th tick where the oldest low-priority
    issue is taken so low items provably drain — and runs its downstream ``build``.

    Args:
        repo_full_name: The target repo, e.g. "owner/repo".
        gh: The backlog gh seam (lists ``backlog`` issues with labels + ages).
        governor: The shared service-level budget governor; its ``gate`` defers the tick
            or charges the admitted estimate to the shared ledger.
        clock: The injected time source for age-weighting (no wall-clock).
        build: The downstream chain run for the picked issue (build -> PR -> loopback ->
            notify), injected so the tick runs with no Docker, gh, or network.
        tick_number: This tick's sequence number; ``tick_number % quota_every == 0`` is a
            quota tick that forces the oldest low-priority issue through.
        estimated_amount: The tick's estimated charge, gated against — and recorded on —
            the rolling-24h ledger when the tick is admitted.
        lock: The single-run lock; entering it raises :class:`CronBusyError` when a tick
            is already in flight.
        quota_every: Take the oldest low-priority issue on every Nth tick (default 5).

    Returns:
        A :class:`CronTickResult`: ``RAN`` with the drained issue, ``DEFERRED`` with a
        ``defer_until``, or ``IDLE`` when the backlog is empty.

    Raises:
        CronBusyError: A tick is already in flight (from the injected lock).
    """
    async with lock:
        # The empty-backlog check must precede the gate: the gate *charges* an admitted
        # estimate to the shared ledger, so gating an idle tick would fill the rolling
        # window with phantom spend and defer real work.
        issues = await gh.list_backlog(repo_full_name=repo_full_name)
        if not issues:
            logger.info("Cron tick %d idle: no backlog issues", tick_number)
            return CronTickResult(outcome=CronOutcome.IDLE)

        gate = await governor.gate(estimated_amount=estimated_amount)
        if gate.deferred:
            logger.info(
                "Cron tick %d deferred: budget spent, defer until %s",
                tick_number,
                gate.defer_until,
            )
            return CronTickResult(
                outcome=CronOutcome.DEFERRED, defer_until=gate.defer_until
            )

        picked = _pick_issue(
            issues, now=clock.now(), tick_number=tick_number, quota_every=quota_every
        )
        logger.info(
            "Cron tick %d draining backlog issue #%d (%s)",
            tick_number,
            picked.number,
            repo_full_name,
        )
        await build(repo_full_name=repo_full_name, issue_number=picked.number)
        return CronTickResult(outcome=CronOutcome.RAN, issue_number=picked.number)


def _pick_issue(
    issues: list[BacklogIssue],
    *,
    now: datetime,
    tick_number: int,
    quota_every: int,
) -> BacklogIssue:
    """Pick the backlog issue this tick drains: the quota floor or the weighted score.

    On a quota tick (``tick_number`` is a positive multiple of ``quota_every``) the oldest
    low-priority issue is taken so the low backlog provably drains rather than starving
    behind higher-priority work. When there is no low-priority issue, the quota tick falls
    back to the ordinary weighted-score pick.
    """
    if quota_every > 0 and tick_number > 0 and tick_number % quota_every == 0:
        floor = _oldest_low_priority(issues)
        if floor is not None:
            return floor
    return max(issues, key=lambda issue: _weighted_score(issue, now=now))


def _oldest_low_priority(issues: list[BacklogIssue]) -> BacklogIssue | None:
    """The oldest issue below the low-priority ceiling, or ``None`` when there is none."""
    low = [issue for issue in issues if issue.severity() < _LOW_PRIORITY_CEILING]
    if not low:
        return None
    return min(low, key=lambda issue: issue.created_at)


def _weighted_score(issue: BacklogIssue, *, now: datetime) -> float:
    """The issue's selection score: priority dominates, age breaks ties within a priority.

    A severity step is worth far more than any realistic age contribution, so a more
    severe issue always outranks a less severe one; among equal-severity issues the older
    one (more accumulated age) scores higher.
    """
    age_days = max((now - issue.created_at) / timedelta(days=1), 0.0)
    return issue.severity() * _PRIORITY_WEIGHT + age_days * _AGE_WEIGHT_PER_DAY
