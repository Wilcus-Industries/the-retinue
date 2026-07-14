"""Heimdall verdict loopback: fix / rebuild / converge on the staging PR (issue #11).

After the PR opener lands ``retinue/prd-<n>`` -> staging (:mod:`retinue.pr_opener`),
heimdall posts a bot review on that PR. This module subscribes to that
``pull_request_review`` event and reasons about the verdict plus a *persisted* per-PR
round count, then carries out one of three things:

* **rebuild** — heimdall raised **blocking** findings (severity at/above the
  threshold) and there is round budget left (below ``RepoConfig.retry_cap``). Each
  blocking finding becomes a fix-issue (``ready-for-agent`` + ``Part of #<prd>``,
  reusing :mod:`retinue.slicer`'s create-issue seam) that rebuilds onto the **same**
  integration branch and re-triggers heimdall review. The round count is recorded so
  the loop survives a worker restart and is bounded at the cap,
* **converge** — heimdall raised **no** blocking findings. The PR is good; the flow
  proceeds to handoff. Any non-blocking nits are still filed as ``backlog`` issues
  carrying heimdall severity mapped 1:1 to a ``priority:<severity>`` label,
* **escalate** — the round budget is spent while still blocked. The flow stops: it
  comments the PRD, applies the ``hitl`` label, and notifies through the shared
  :class:`retinue.notify.Notifier`, leaving the PR open for a human.

The persisted round count mirrors the durable-SQLite style of
:class:`retinue.impl_retry.ImplRetryStore`. Every collaborator — the heimdall verdict
input, the gh issue creator, the rebuild-onto-same-branch trigger, the handoff, and
the notifier sinks — is injected, so the whole flow is unit-tested with no real gh,
heimdall, push service, or network.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import aiosqlite

from retinue.notify import Notification, Notifier
from retinue.repo_config import RepoConfig
from retinue.slicer import HITL_LABEL, READY_LABEL, CreatedIssue, IssueCreator, IssueDraft

logger = logging.getLogger(__name__)

BACKLOG_LABEL = "backlog"


class Severity(enum.IntEnum):
    """A heimdall finding's severity, ordered so a blocking threshold is a comparison.

    The integer order encodes "more severe is greater", so a finding is *blocking*
    when its severity is at or above the configured threshold (default
    :attr:`Severity.HIGH`). The member *name* (lower-cased) is what maps 1:1 to a
    ``priority:<severity>`` label for a backlog nit.
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# Heimdall's blocking threshold: a finding at or above this severity blocks the PR.
# Below it, the finding is a non-blocking nit routed to the backlog.
_BLOCKING_THRESHOLD = Severity.HIGH


def priority_label(severity: Severity) -> str:
    """Return the backlog ``priority:<severity>`` label for a heimdall severity.

    The mapping is 1:1 with the severity name, so heimdall's own severity vocabulary
    survives onto the filed backlog issue without translation.

    Args:
        severity: The finding's heimdall severity.

    Returns:
        ``"priority:low"`` / ``"priority:medium"`` / ``"priority:high"`` /
        ``"priority:critical"``.
    """
    return f"priority:{severity.name.lower()}"


class ReviewState(enum.Enum):
    """The state of heimdall's bot review on the PR (the gh review ``state``)."""

    APPROVED = "approved"
    COMMENTED = "commented"
    REQUEST_CHANGES = "request_changes"


@dataclass(frozen=True)
class HeimdallFinding:
    """One finding heimdall raised in its review.

    Attributes:
        summary: The finding's what/why — carried into the fix-issue or backlog body.
        severity: The finding's severity; at/above the threshold it blocks the PR,
            below it is a non-blocking nit (see :data:`_BLOCKING_THRESHOLD`).
    """

    summary: str
    severity: Severity


@dataclass(frozen=True)
class HeimdallReview:
    """A parsed heimdall bot review on the ``retinue/prd-<n>`` -> staging PR.

    The ``pull_request_review`` webhook payload (heimdall's bot review) is parsed into
    this before the loopback reasons about it.

    Attributes:
        repo_full_name: e.g. "owner/repo"; keys the round count and targets gh.
        pr_number: The reviewed PR; part of the persisted round-count identity.
        prd_number: The parent PRD; the integration branch is ``retinue/prd-<n>`` and
            fix/backlog issues link back via ``Part of #<prd>``.
        prd_issue_number: The PRD's tracking issue, where an escalation comments/labels.
        integration_branch: The branch the fix-issues rebuild onto (the SAME branch).
        state: Heimdall's review state.
        findings: The findings heimdall raised; split into blocking vs nits here.
    """

    repo_full_name: str
    pr_number: int
    prd_number: int
    prd_issue_number: int
    integration_branch: str
    state: ReviewState
    findings: list[HeimdallFinding]

    @property
    def blocking(self) -> list[HeimdallFinding]:
        """Findings at or above the blocking threshold."""
        return [f for f in self.findings if f.severity >= _BLOCKING_THRESHOLD]

    @property
    def nits(self) -> list[HeimdallFinding]:
        """Findings below the blocking threshold — non-blocking backlog nits."""
        return [f for f in self.findings if f.severity < _BLOCKING_THRESHOLD]


@dataclass(frozen=True)
class RebuildRequest:
    """Payload handed to the rebuild seam: rebuild the fix-issues onto the same branch.

    Attributes:
        repo_full_name: The target repo.
        integration_branch: The SAME ``retinue/prd-<n>`` branch the PR is built from;
            the fix-issues are built and merged here, re-triggering heimdall review.
        prd_number: The parent PRD number.
        pr_number: The PR whose review loop this rebuild continues.
        fix_issues: The fix-issue numbers just filed for this round, to build.
    """

    repo_full_name: str
    integration_branch: str
    prd_number: int
    pr_number: int
    fix_issues: list[int]


class VerdictDecision(enum.Enum):
    """What the loopback decided about a heimdall verdict and the round count."""

    REBUILD = "rebuild"
    CONVERGED = "converged"
    ESCALATE = "escalate"
    NO_VERDICT = "no_verdict"


class VerdictOutcome(enum.Enum):
    """The terminal outcome of processing one heimdall review."""

    REBUILT = "rebuilt"
    CONVERGED = "converged"
    ESCALATED = "escalated"
    IGNORED = "ignored"


@dataclass(frozen=True)
class VerdictResult:
    """Outcome of processing one heimdall review.

    Attributes:
        outcome: ``REBUILT`` when fix-issues were filed and rebuilt, ``CONVERGED`` when
            the PR was clean and handed off, ``ESCALATED`` when the round budget was
            spent while still blocked.
        filed_issues: Issue numbers filed this round (fix-issues and/or backlog nits),
            in finding order.
        pr_left_open: True on the escalate path — the PR is deliberately left open for
            a human; False otherwise.
    """

    outcome: VerdictOutcome
    filed_issues: list[int] = field(default_factory=list)
    pr_left_open: bool = False


# Injected seams. ``rebuild`` re-runs the build of the fix-issues onto the same branch
# and re-triggers heimdall review; ``handoff`` proceeds past a converged PR. Both async
# and faked in tests — no gh, no heimdall, no network.
Rebuilder = Callable[[RebuildRequest], Awaitable[None]]
Handoff = Callable[..., Awaitable[None]]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS heimdall_rounds (
    pr_key TEXT PRIMARY KEY,
    rounds INTEGER NOT NULL DEFAULT 0
)
"""


def round_key(repo_full_name: str, pr_number: int) -> str:
    """Return the round-count identity of a PR: its repo and PR number.

    Args:
        repo_full_name: e.g. "owner/repo".
        pr_number: The reviewed PR number.

    Returns:
        A stable ``"owner/repo#<pr>"`` key.
    """
    return f"{repo_full_name}#{pr_number}"


class HeimdallRoundStore:
    """Durable per-PR heimdall rebuild-round counter over a SQLite file.

    The count is the number of rebuild rounds already triggered for a PR. The loopback
    reads it to decide whether another rebuild is within ``RepoConfig.retry_cap``, and
    records a round before each rebuild so the budget is consumed even across worker
    restarts — a doomed PR cannot loop forever. Mirrors the durable-SQLite style of
    :class:`retinue.impl_retry.ImplRetryStore`.

    Args:
        db_path: Path to the SQLite database file. Created on first use; parent
            directories are created if missing.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)

    async def count(self, key: str) -> int:
        """Return the number of rebuild rounds recorded for ``key`` (zero if unseen).

        Args:
            key: The PR round key (see :func:`round_key`).

        Returns:
            The persisted round count, or ``0`` for a PR never recorded.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_SCHEMA)
            async with db.execute(
                "SELECT rounds FROM heimdall_rounds WHERE pr_key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    async def record_round(self, key: str) -> int:
        """Atomically increment ``key``'s round count and return the new value.

        The upsert is atomic on the primary key, so concurrent runs cannot lose a
        round increment.

        Args:
            key: The PR round key (see :func:`round_key`).

        Returns:
            The round count after this increment.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_SCHEMA)
            await db.execute(
                """
                INSERT INTO heimdall_rounds (pr_key, rounds) VALUES (?, 1)
                ON CONFLICT(pr_key) DO UPDATE SET rounds = rounds + 1
                """,
                (key,),
            )
            await db.commit()
            async with db.execute(
                "SELECT rounds FROM heimdall_rounds WHERE pr_key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0


def decide_verdict(
    *, state: ReviewState, blocking: int, rounds: int, cap: int
) -> VerdictDecision:
    """Decide rebuild / converge / escalate from the review state and the round count.

    The review ``state`` is the authoritative verdict — the parsed finding count only
    refines a rejection. The reasoning, in order:

    1. ``APPROVED`` converges (proceed to handoff), whatever the parsed findings or
       round count — an approval is never rebuilt.
    2. ``COMMENTED`` carries no verdict (heimdall's progress notes): the loopback
       neither converges nor rebuilds on it.
    3. ``REQUEST_CHANGES`` with **no** parseable blocking findings escalates — the PR
       was explicitly rejected, so converging on an unparseable body would hand off
       rejected work.
    4. Otherwise, while the persisted round count is below ``cap`` there is budget for
       another rebuild round; with the budget exhausted the flow escalates.

    Args:
        state: Heimdall's review state — the authoritative verdict.
        blocking: The number of blocking findings (severity at/above the threshold).
        rounds: Rebuild rounds already persisted for this PR (the spent budget).
        cap: The round cap (``RepoConfig.retry_cap``); ``0`` means no rebuild rounds.

    Returns:
        The :class:`VerdictDecision` to carry out.
    """
    if state is ReviewState.APPROVED:
        return VerdictDecision.CONVERGED
    if state is ReviewState.COMMENTED:
        return VerdictDecision.NO_VERDICT
    if blocking == 0:
        return VerdictDecision.ESCALATE
    if rounds < cap:
        return VerdictDecision.REBUILD
    return VerdictDecision.ESCALATE


async def process_review(
    review: HeimdallReview,
    config: RepoConfig,
    *,
    round_store: HeimdallRoundStore,
    create_issue: IssueCreator,
    rebuild: Rebuilder,
    handoff: Handoff,
    notifier: Notifier,
) -> VerdictResult:
    """Process one heimdall review: rebuild on blocking findings, else converge/escalate.

    Feeds the review state (the authoritative verdict), the blocking-finding count,
    and the *persisted* round count to :func:`decide_verdict`. ``REBUILD`` files each
    blocking finding as a fix-issue (``ready-for-agent`` + ``Part of #<prd>``),
    records a round, and triggers a rebuild onto the SAME integration branch
    (re-triggering heimdall review). ``CONVERGED`` files the remaining findings as
    ``backlog`` + ``priority:<severity>`` issues and proceeds to handoff.
    ``ESCALATE`` comments the PRD, applies ``hitl``, notifies, and leaves the PR
    open. ``NO_VERDICT`` (a plain comment) is ignored. On every verdict path the
    non-blocking nits are filed as backlog.

    Args:
        review: The parsed heimdall bot review.
        config: The accepted repo config; ``retry_cap`` bounds the rebuild rounds.
        round_store: Persisted per-PR rebuild-round counter bounding the loop.
        create_issue: The gh issue creator (slicer's seam) for fix and backlog issues.
        rebuild: The rebuild seam: build the fix-issues onto the same branch + re-review.
        handoff: The handoff seam invoked when the PR converges.
        notifier: Shared notify primitive for the escalate path.

    Returns:
        A :class:`VerdictResult` recording the terminal outcome and filed issues.
    """
    key = round_key(review.repo_full_name, review.pr_number)
    rounds = await round_store.count(key)
    decision = decide_verdict(
        state=review.state,
        blocking=len(review.blocking),
        rounds=rounds,
        cap=config.retry_cap,
    )

    if decision is VerdictDecision.NO_VERDICT:
        logger.info(
            "Heimdall comment on PR #%d (%s) carries no verdict; ignoring",
            review.pr_number,
            review.repo_full_name,
        )
        return VerdictResult(outcome=VerdictOutcome.IGNORED)
    if decision is VerdictDecision.CONVERGED:
        return await _converge(review, create_issue, handoff)
    if decision is VerdictDecision.REBUILD:
        return await _rebuild(review, round_store, key, create_issue, rebuild, notifier)
    reason = (
        _CAP_EXHAUSTED_REASON
        if review.blocking
        else _UNPARSEABLE_REJECTION_REASON
    )
    return await _escalate(review, create_issue, notifier, reason=reason)


async def _converge(
    review: HeimdallReview,
    create_issue: IssueCreator,
    handoff: Handoff,
) -> VerdictResult:
    """Converge: park every finding as backlog, then hand off the approved PR.

    An approval converges whatever the parsed findings say, so any blocking-severity
    lines in an approved body are parked as backlog (carrying their priority label)
    rather than dropped or rebuilt.
    """
    filed = await _file_backlog(review, review.findings, create_issue)
    logger.info(
        "Heimdall converged on PR #%d (%s); proceeding to handoff",
        review.pr_number,
        review.repo_full_name,
    )
    await handoff(repo_full_name=review.repo_full_name, pr_number=review.pr_number)
    return VerdictResult(outcome=VerdictOutcome.CONVERGED, filed_issues=filed)


async def _rebuild(
    review: HeimdallReview,
    round_store: HeimdallRoundStore,
    key: str,
    create_issue: IssueCreator,
    rebuild: Rebuilder,
    notifier: Notifier,
) -> VerdictResult:
    """File fix-issues for the blocking findings, then rebuild onto the same branch.

    The round is recorded *before* the rebuild trigger so a doomed PR cannot loop
    unbounded across retries. A failed trigger therefore escalates (the round is
    already spent and the fix-issues already filed — retrying the whole job would
    double-file them) instead of raising.
    """
    fix_issues: list[int] = []
    for finding in review.blocking:
        created = await _file_fix_issue(finding, review, create_issue)
        fix_issues.append(created.issue_number)

    backlog = await _file_backlog(review, review.nits, create_issue)

    round_number = await round_store.record_round(key)
    logger.info(
        "Heimdall round %d on PR #%d (%s): rebuilding %d fix-issue(s) onto %s",
        round_number,
        review.pr_number,
        review.repo_full_name,
        len(fix_issues),
        review.integration_branch,
    )
    try:
        await rebuild(
            RebuildRequest(
                repo_full_name=review.repo_full_name,
                integration_branch=review.integration_branch,
                prd_number=review.prd_number,
                pr_number=review.pr_number,
                fix_issues=fix_issues,
            )
        )
    except (GhCommandError, ValueError) as exc:
        logger.error(
            "Heimdall rebuild trigger failed on PR #%d (%s): %s",
            review.pr_number,
            review.repo_full_name,
            exc,
        )
        await _notify_escalation(
            review,
            notifier,
            reason=(
                f"The fix-issues for round {round_number} were filed "
                f"({', '.join(f'#{n}' for n in fix_issues)}) but the heimdall "
                f"re-review request failed ({exc}). The loop is stalled until a "
                f"human re-requests the review or resolves the PR."
            ),
        )
        return VerdictResult(
            outcome=VerdictOutcome.ESCALATED,
            filed_issues=fix_issues + backlog,
            pr_left_open=True,
        )
    return VerdictResult(outcome=VerdictOutcome.REBUILT, filed_issues=fix_issues + backlog)


_CAP_EXHAUSTED_REASON = (
    "Heimdall still raised blocking findings after the rebuild budget (retry_cap) "
    "was spent. The retinue is stopping the loopback; the PR is left open for a "
    "human to resolve."
)

_UNPARSEABLE_REJECTION_REASON = (
    "Heimdall requested changes but no blocking findings could be parsed from the "
    "review body, so the retinue cannot file fix-issues. The PR is left open for a "
    "human to read the review and resolve it."
)


async def _escalate(
    review: HeimdallReview,
    create_issue: IssueCreator,
    notifier: Notifier,
    *,
    reason: str,
) -> VerdictResult:
    """Escalate a blocked PR to a human; park any parsed nits as backlog first."""
    filed = await _file_backlog(review, review.nits, create_issue)
    await _notify_escalation(review, notifier, reason=reason)
    return VerdictResult(
        outcome=VerdictOutcome.ESCALATED, filed_issues=filed, pr_left_open=True
    )


async def _notify_escalation(
    review: HeimdallReview, notifier: Notifier, *, reason: str
) -> None:
    """Comment the PRD, apply ``hitl``, and push-notify that a human is needed."""
    await notifier.notify(
        Notification(
            repo_full_name=review.repo_full_name,
            issue_number=review.prd_issue_number,
            title=f"Retinue needs a human on PR #{review.pr_number}",
            body=f"{reason} (PR #{review.pr_number})",
            label=HITL_LABEL,
        )
    )
    logger.warning(
        "Heimdall loopback escalated PR #%d (%s): %s",
        review.pr_number,
        review.repo_full_name,
        reason,
    )


async def _file_fix_issue(
    finding: HeimdallFinding,
    review: HeimdallReview,
    create_issue: IssueCreator,
) -> CreatedIssue:
    """File one blocking finding as a ready-for-agent, PRD-linked fix-issue."""
    draft = IssueDraft(
        title=f"Heimdall fix: {finding.summary}",
        body=(
            f"Heimdall raised a blocking finding ({finding.severity.name.lower()}) on "
            f"PR #{review.pr_number}:\n\n{finding.summary}\n\n"
            f"The fix targets the round's integration branch "
            f"`{review.integration_branch}`.\n\nPart of #{review.prd_number}"
        ),
        labels=[READY_LABEL],
    )
    return await create_issue(draft)


async def _file_backlog(
    review: HeimdallReview,
    findings: list[HeimdallFinding],
    create_issue: IssueCreator,
) -> list[int]:
    """File each finding as a backlog issue carrying its priority label."""
    filed: list[int] = []
    for finding in findings:
        draft = IssueDraft(
            title=f"Heimdall nit: {finding.summary}",
            body=(
                f"Heimdall raised a non-blocking finding on PR #{review.pr_number}:\n\n"
                f"{finding.summary}\n\nPart of #{review.prd_number}"
            ),
            labels=[BACKLOG_LABEL, priority_label(finding.severity)],
        )
        created = await create_issue(draft)
        filed.append(created.issue_number)
    return filed


# --- production gh-cli Rebuilder --------------------------------------------------
#
# :func:`process_review` depends only on the :data:`Rebuilder` protocol. Production wires
# the concrete :class:`GhCliRebuilder` below; tests inject a fake that records the
# :class:`RebuildRequest`. The fix-issues were already filed by the loopback before the
# trigger fires, so the rebuild's single side effect is re-triggering heimdall's bot
# review on the same PR — re-requesting review through an injected :class:`GhRunner`
# (the only process-spawn seam), authenticated with a ``GH_TOKEN`` bearer. The filed
# ``ready-for-agent`` fix-issues re-enter the build lane through the ordinary issue
# routing.
#
# The adapter never shells out itself: every pure/parseable part — the auth-env build, the
# ``gh pr edit --add-reviewer`` command assembly, and parsing the re-review payload — is a
# free function tested with a recording fake runner, never a live ``gh``/heimdall/network.
# The local :class:`GhRunner`/:class:`GhResult` mirror the gh-seam shape used in
# :mod:`retinue.slicer` / :mod:`retinue.pr_opener`; each module keeps its own copy so the
# layers stay edit-isolated.

@dataclass(frozen=True)
class GhResult:
    """Captured result of a single ``gh`` invocation.

    Attributes:
        exit_code: ``gh``'s process exit status; ``0`` means success.
        stdout: Captured standard output (the re-review payload parsed below).
        stderr: Captured standard error (surfaced in the error on failure).
    """

    exit_code: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        """True when ``gh`` exited successfully (exit code 0)."""
        return self.exit_code == 0


class GhRunner(Protocol):
    """Runs a single ``gh`` command. The process-spawn seam under :class:`GhCliRebuilder`.

    A production implementation spawns ``gh`` as a subprocess with ``env`` merged into its
    environment (so ``GH_TOKEN`` authenticates the call) and returns the captured
    :class:`GhResult`; tests inject a fake that records each ``(args, env)`` and returns a
    canned result. ``args`` never includes the leading ``"gh"`` — the runner owns the
    executable name.
    """

    async def run(self, args: list[str], *, env: dict[str, str]) -> GhResult:
        """Run ``gh <args>`` with ``env`` in the environment and capture the result."""
        ...


class GhCommandError(RuntimeError):
    """A ``gh`` invocation exited non-zero. Carries the args and stderr for debugging."""

    def __init__(self, command: list[str], result: GhResult) -> None:
        self.command = command
        self.result = result
        super().__init__(
            f"gh {' '.join(command)} exited {result.exit_code}: {result.stderr.strip()}"
        )


def _auth_env(token: str) -> dict[str, str]:
    """Build the env that authenticates ``gh``: a ``GH_TOKEN`` bearer for the API.

    ``gh`` reads ``GH_TOKEN`` and sends it as ``Authorization: Bearer <token>`` on every
    REST/GraphQL call, so the adapter never assembles a header itself — it injects the
    token here and lets ``gh`` own the wire format.
    """
    return {"GH_TOKEN": token}


def _re_review_args(request: RebuildRequest, reviewer_login: str) -> list[str]:
    """Assemble the ``gh pr edit`` argv that re-requests a heimdall review (no ``"gh"``).

    Re-adding the heimdall bot (``reviewer_login``) as a reviewer on the rebuilt PR is what
    re-triggers a fresh bot review of the integration branch the fix-issues were just built
    onto. The login is the centralized ``Settings.heimdall_bot_login`` — the same value the
    webhook filters inbound reviews by — so the re-request and the inbound filter cannot
    drift. ``--repo`` targets the request's repo and the trailing positional is the PR
    number. The argv is assembled purely so it is unit-testable without a live ``gh``.
    """
    return [
        "pr",
        "edit",
        str(request.pr_number),
        "--repo",
        request.repo_full_name,
        "--add-reviewer",
        reviewer_login,
    ]


def _parse_review_requested(stdout: str) -> int:
    """Parse the PR number back from ``gh pr edit``'s re-review payload.

    ``gh pr edit`` echoes the edited PR's URL (e.g.
    ``https://github.com/owner/repo/pull/42``) on success. The number is the trailing path
    segment; it is read back as a confirmation that the review re-request landed on the
    expected PR. Raises :class:`ValueError` when the output carries no trailing integer, so
    a malformed response fails loudly rather than silently dropping the re-review.
    """
    tail = stdout.strip().rstrip("/").rsplit("/", 1)[-1]
    try:
        return int(tail)
    except ValueError as exc:
        raise ValueError(f"gh pr edit returned no PR number: {stdout!r}") from exc


class GhCliRebuilder:
    """Production :data:`Rebuilder`: re-triggers heimdall's review on the rebuilt PR.

    An instance is callable as ``await rebuilder(request)`` — it satisfies the
    :data:`Rebuilder` protocol via :meth:`__call__`, so it drops straight in where the fake
    rebuilder sits in tests and at the wiring site (``process_review``). The loopback has
    already filed the round's fix-issues before this trigger fires (re-filing them here
    would double every finding), so the trigger's single job is re-requesting the heimdall
    bot review on the same PR: it dispatches the assembled ``gh pr edit`` argv
    (:func:`_re_review_args`) through the injected :class:`GhRunner`, authenticated with a
    ``GH_TOKEN`` bearer (:func:`_auth_env`), and confirms the PR number echoed back
    (:func:`_parse_review_requested`).

    The runner is the only side-effecting seam, which keeps the command assembly and
    payload parsing unit-testable with no live ``gh``/heimdall/network
    (``external_dep none``).

    Args:
        runner: The process-spawn seam that runs each ``gh`` command.
        token: The installation/access token ``gh`` authenticates with.
        reviewer_login: The bot login re-requested as a reviewer to re-trigger heimdall's
            review — the centralized ``Settings.heimdall_bot_login`` the webhook also
            filters inbound reviews by, so the re-request and the filter cannot drift.
    """

    def __init__(
        self,
        runner: GhRunner,
        *,
        token: str,
        reviewer_login: str,
    ) -> None:
        self._runner = runner
        self._token = token
        self._reviewer_login = reviewer_login

    async def __call__(self, request: RebuildRequest) -> None:
        """Re-request the heimdall review on the rebuilt PR."""
        args = _re_review_args(request, self._reviewer_login)
        result = await self._runner.run(args, env=_auth_env(self._token))
        if not result.ok:
            raise GhCommandError(args, result)
        pr_number = _parse_review_requested(result.stdout)
        logger.info(
            "Re-requested heimdall review on PR #%d (%s) after rebuilding %d fix-issue(s) "
            "onto %s",
            pr_number,
            request.repo_full_name,
            len(request.fix_issues),
            request.integration_branch,
        )
