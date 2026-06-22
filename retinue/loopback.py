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


class VerdictOutcome(enum.Enum):
    """The terminal outcome of processing one heimdall review."""

    REBUILT = "rebuilt"
    CONVERGED = "converged"
    ESCALATED = "escalated"


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


def decide_verdict(*, blocking: int, rounds: int, cap: int) -> VerdictDecision:
    """Decide rebuild / converge / escalate from the verdict and the round count.

    The reasoning, in order:

    1. No blocking findings means the PR is good — converge (proceed to handoff),
       whatever the round count.
    2. Otherwise, while the persisted round count is below ``cap`` there is budget for
       another rebuild round.
    3. With the round budget exhausted while still blocked, the flow escalates.

    Args:
        blocking: The number of blocking findings (severity at/above the threshold).
        rounds: Rebuild rounds already persisted for this PR (the spent budget).
        cap: The round cap (``RepoConfig.retry_cap``); ``0`` means no rebuild rounds.

    Returns:
        The :class:`VerdictDecision` to carry out.
    """
    if blocking == 0:
        return VerdictDecision.CONVERGED
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

    Splits the review's findings into blocking (severity at/above the threshold) and
    non-blocking nits, then feeds the blocking count plus the *persisted* round count to
    :func:`decide_verdict`. ``REBUILD`` files each blocking finding as a fix-issue
    (``ready-for-agent`` + ``Part of #<prd>``), records a round, and triggers a rebuild
    onto the SAME integration branch (re-triggering heimdall review). ``CONVERGED``
    files the nits as ``backlog`` + ``priority:<severity>`` issues and proceeds to
    handoff. ``ESCALATE`` comments the PRD, applies ``hitl``, notifies, and leaves the
    PR open. Nits are always filed as backlog regardless of decision.

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
        blocking=len(review.blocking), rounds=rounds, cap=config.retry_cap
    )

    if decision is VerdictDecision.CONVERGED:
        return await _converge(review, create_issue, handoff)
    if decision is VerdictDecision.REBUILD:
        return await _rebuild(review, round_store, key, create_issue, rebuild)
    return await _escalate(review, notifier)


async def _converge(
    review: HeimdallReview,
    create_issue: IssueCreator,
    handoff: Handoff,
) -> VerdictResult:
    """Converge: file any nits as backlog, then hand off the clean PR."""
    filed = await _file_backlog_nits(review, create_issue)
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
) -> VerdictResult:
    """File fix-issues for the blocking findings, then rebuild onto the same branch."""
    fix_issues: list[int] = []
    for finding in review.blocking:
        created = await _file_fix_issue(finding, review, create_issue)
        fix_issues.append(created.issue_number)

    backlog = await _file_backlog_nits(review, create_issue)

    round_number = await round_store.record_round(key)
    logger.info(
        "Heimdall round %d on PR #%d (%s): rebuilding %d fix-issue(s) onto %s",
        round_number,
        review.pr_number,
        review.repo_full_name,
        len(fix_issues),
        review.integration_branch,
    )
    await rebuild(
        RebuildRequest(
            repo_full_name=review.repo_full_name,
            integration_branch=review.integration_branch,
            prd_number=review.prd_number,
            pr_number=review.pr_number,
            fix_issues=fix_issues,
        )
    )
    return VerdictResult(outcome=VerdictOutcome.REBUILT, filed_issues=fix_issues + backlog)


async def _escalate(review: HeimdallReview, notifier: Notifier) -> VerdictResult:
    """Escalate a still-blocked PR with the round budget spent; leave the PR open."""
    body = (
        f"Heimdall still raised blocking findings on PR #{review.pr_number} after the "
        f"rebuild budget (retry_cap) was spent. The retinue is stopping the loopback; "
        f"the PR is left open for a human to resolve."
    )
    await notifier.notify(
        Notification(
            repo_full_name=review.repo_full_name,
            issue_number=review.prd_issue_number,
            title=f"Retinue needs a human on PR #{review.pr_number}",
            body=body,
            label=HITL_LABEL,
        )
    )
    logger.warning(
        "Heimdall loopback escalated PR #%d (%s): budget spent, PR left open",
        review.pr_number,
        review.repo_full_name,
    )
    return VerdictResult(outcome=VerdictOutcome.ESCALATED, pr_left_open=True)


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
            f"PR #{review.pr_number}:\n\n{finding.summary}\n\nPart of #{review.prd_number}"
        ),
        labels=[READY_LABEL],
    )
    return await create_issue(draft)


async def _file_backlog_nits(
    review: HeimdallReview,
    create_issue: IssueCreator,
) -> list[int]:
    """File each non-blocking nit as a backlog issue carrying its priority label."""
    filed: list[int] = []
    for nit in review.nits:
        draft = IssueDraft(
            title=f"Heimdall nit: {nit.summary}",
            body=(
                f"Heimdall raised a non-blocking nit on PR #{review.pr_number}:\n\n"
                f"{nit.summary}\n\nPart of #{review.prd_number}"
            ),
            labels=[BACKLOG_LABEL, priority_label(nit.severity)],
        )
        created = await create_issue(draft)
        filed.append(created.issue_number)
    return filed
