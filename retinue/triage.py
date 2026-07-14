"""Reasoning failure triage for the orchestrator (issue #8).

The implementer seam (the Agent SDK subagent) does not always cleanly build a
slice: it can *fail* (raise) or *return notes* describing why it could not finish
or that the slice is mis-scoped. A blind retry loop is wrong here — the
orchestrator must *reason* about the signal plus how many attempts it has already
spent, and decide one of three things:

* **retry** — try the implementer again, while the persisted attempt count is below
  the cap (default :attr:`retinue.repo_config.RepoConfig.retry_cap`). The bound is a
  *persisted* count (:class:`retinue.impl_retry.ImplRetryStore`) so a doomed slice
  cannot retry forever and the budget survives a worker restart,
* **reslice** — when the notes say the slice is mis-scoped, file an adjusted slice
  through the gh ``create_issue`` seam (reusing :mod:`retinue.slicer`'s creator),
* **escalate** — hand the slice to a human by fanning a
  :class:`retinue.notify.Notification` out through the shared
  :class:`retinue.notify.Notifier` (push + comment + the ``hitl`` label).

The decision itself is the pure :func:`decide_triage`; :func:`triage_implementer`
drives the implementer, persists the retry count, and carries out the decision.
Both the failure path and the notes path reach the reasoned decision — notes are
never silently dropped. Every collaborator (implementer, notifier sinks, the gh
issue creator, the SQLite store) is injected so the flow is unit-tested with no
Agent SDK, no gh, no push service, and no network.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Protocol

from retinue.container import Container
from retinue.impl_retry import ImplRetryStore, impl_retry_key
from retinue.notify import Notification, Notifier
from retinue.orchestrator import Slice
from retinue.repo_config import RepoConfig
from retinue.slicer import HITL_LABEL, READY_LABEL, CreatedIssue, IssueCreator, IssueDraft

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImplementerNotes:
    """Why an implementer could not cleanly finish, returned instead of raising.

    A triage-aware implementer returns this (rather than ``None``) when it built
    something but wants the orchestrator to reason about a problem — a flaky run it
    wants retried, or a slice it believes is mis-scoped and should be reshaped.

    Attributes:
        summary: Human-readable reasoning, carried into the reslice body or the
            escalation comment so the decision is auditable.
        reslice: True when the implementer judges the slice mis-scoped and a reshaped
            slice should be filed rather than retried.
    """

    summary: str
    reslice: bool = False


class TriageImplementer(Protocol):
    """The implementer seam, triage-aware: it may raise, return notes, or finish.

    Extends the orchestrator's :class:`retinue.orchestrator.Implementer` contract:
    a clean build returns ``None`` (so existing implementers satisfy it unchanged),
    while a build that needs reasoning returns :class:`ImplementerNotes`. A hard
    failure raises, as before. Like the base contract, it builds the slice inside the
    disposable build ``container`` the orchestrator owns.
    """

    async def implement(
        self, slice_: Slice, *, container: Container
    ) -> ImplementerNotes | None:
        """Build ``slice_`` in ``container``; return notes to triage, ``None`` if clean."""
        ...

    def auth_env(self) -> dict[str, str]:
        """The agent's credential env, merged into the build container at start."""
        ...


class TriageDecision(enum.Enum):
    """What the orchestrator decided to do about a failure or returned notes."""

    RETRY = "retry"
    RESLICE = "reslice"
    ESCALATE = "escalate"


class TriageOutcome(enum.Enum):
    """The terminal outcome of triaging one slice's implementer."""

    BUILT = "built"
    RESLICED = "resliced"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class TriageResult:
    """Outcome of triaging one slice.

    Attributes:
        outcome: ``BUILT`` when the implementer finished (possibly after a retry),
            ``RESLICED`` when an adjusted slice was filed, ``ESCALATED`` when the
            slice was handed to a human.
        resliced_issue: The new issue number filed on the reslice path, else ``None``.
    """

    outcome: TriageOutcome
    resliced_issue: int | None = None


@dataclass
class _Signal:
    """One implementer attempt's result: a hard failure or returned notes (or clean).

    ``budget_exhausted_before_run`` marks the pre-run escalation where a prior run had
    already spent the whole retry budget: the implementer never ran this session, so the
    escalation reason must not claim a fresh failure.
    """

    failed: bool = False
    notes: ImplementerNotes | None = None
    budget_exhausted_before_run: bool = False

    @property
    def clean(self) -> bool:
        """A clean build: neither a failure nor notes to reason about."""
        return not self.failed and self.notes is None


def decide_triage(
    *, failed: bool, notes: ImplementerNotes | None, attempts: int, cap: int
) -> TriageDecision:
    """Decide retry / reslice / escalate from a signal and the persisted attempt count.

    The reasoning, in order:

    1. Notes that explicitly request a reslice reshape the slice — a mis-scoped
       slice is not fixed by retrying it.
    2. Otherwise, while the persisted attempt count is below ``cap`` there is budget
       to retry.
    3. With the budget exhausted, the slice escalates to a human.

    Args:
        failed: True when the implementer raised a hard failure.
        notes: Notes the implementer returned, or ``None``.
        attempts: Attempts already persisted for this slice (the spent budget).
        cap: The retry cap (``RepoConfig.retry_cap``); ``0`` means no retries.

    Returns:
        The :class:`TriageDecision` to carry out.
    """
    if notes is not None and notes.reslice:
        return TriageDecision.RESLICE
    if attempts < cap:
        return TriageDecision.RETRY
    return TriageDecision.ESCALATE


async def triage_implementer(
    slice_: Slice,
    config: RepoConfig,
    *,
    implementer: TriageImplementer,
    notifier: Notifier,
    create_issue: IssueCreator,
    retry_store: ImplRetryStore,
    container: Container,
) -> TriageResult:
    """Run the implementer for ``slice_``, reasoning about any failure or notes.

    Invokes the implementer in the build ``container``; a clean build is
    :attr:`TriageOutcome.BUILT`. A hard failure or returned notes is fed — with the
    *persisted* attempt count — to :func:`decide_triage`. ``RETRY`` records an attempt
    against the persisted cap and re-runs the implementer (in the same container);
    ``RESLICE`` files an adjusted slice via ``create_issue``; ``ESCALATE`` notifies a
    human and applies the ``hitl`` label. Retries are bounded by the persisted count, so
    a slice that has spent its budget in an earlier run escalates without re-running.

    Args:
        slice_: The slice whose implementer to drive and triage.
        config: The accepted repo config; ``retry_cap`` bounds the retries.
        implementer: The triage-aware implementer seam (Agent SDK).
        notifier: Shared notify primitive for the escalate path.
        create_issue: The gh issue creator (slicer's seam) for the reslice path.
        retry_store: Persisted per-slice attempt counter bounding the retries.
        container: The disposable build container the implementer execs in.

    Returns:
        A :class:`TriageResult` recording the terminal outcome.
    """
    key = impl_retry_key(slice_)
    attempts = await retry_store.count(key)
    if attempts > 0 and attempts >= config.retry_cap:
        # A prior run already spent the whole retry budget on this slice; do not
        # burn another doomed attempt — escalate straight to a human.
        return await _escalate(
            slice_, _Signal(budget_exhausted_before_run=True), notifier
        )

    while True:
        signal = await _run_implementer(slice_, implementer, container)
        if signal.clean:
            return TriageResult(outcome=TriageOutcome.BUILT)

        decision = decide_triage(
            failed=signal.failed,
            notes=signal.notes,
            attempts=attempts,
            cap=config.retry_cap,
        )
        if decision is TriageDecision.RETRY:
            attempts = await retry_store.record_attempt(key)
            logger.info(
                "Retrying implementer for %s (attempt %d/%d)",
                key,
                attempts,
                config.retry_cap,
            )
            continue
        if decision is TriageDecision.RESLICE:
            return await _reslice(slice_, signal.notes, create_issue)
        return await _escalate(slice_, signal, notifier)


async def _run_implementer(
    slice_: Slice, implementer: TriageImplementer, container: Container
) -> _Signal:
    """Run one implementer attempt in ``container``, capturing a failure or notes.

    A raised exception becomes a failure signal rather than propagating, so the
    triage gets to reason about it. ``KeyboardInterrupt``/``SystemExit`` are not
    caught — only ``Exception`` — so process control still propagates.
    """
    try:
        notes = await implementer.implement(slice_, container=container)
    except Exception:
        logger.warning("Implementer failed for %s", impl_retry_key(slice_), exc_info=True)
        return _Signal(failed=True)
    return _Signal(notes=notes)


async def _reslice(
    slice_: Slice, notes: ImplementerNotes | None, create_issue: IssueCreator
) -> TriageResult:
    """File an adjusted slice carrying the implementer's reasoning."""
    reason = notes.summary if notes is not None else "adjusted after a failed build"
    draft = IssueDraft(
        title=f"Reslice of #{slice_.issue_number}",
        body=(
            f"Adjusted slice of #{slice_.issue_number}, reshaped after the implementer "
            f"reported it was mis-scoped.\n\n{reason}"
        ),
        labels=[READY_LABEL],
    )
    created: CreatedIssue = await create_issue(draft)
    logger.info("Resliced %s into #%d", impl_retry_key(slice_), created.issue_number)
    return TriageResult(outcome=TriageOutcome.RESLICED, resliced_issue=created.issue_number)


async def _escalate(slice_: Slice, signal: _Signal, notifier: Notifier) -> TriageResult:
    """Escalate the slice to a human: notify (push + comment) and apply ``hitl``."""
    reason = _escalation_reason(signal)
    await notifier.notify(
        Notification(
            repo_full_name=slice_.repo_full_name,
            issue_number=slice_.issue_number,
            title=f"Retinue needs a human on #{slice_.issue_number}",
            body=reason,
            label=HITL_LABEL,
        )
    )
    logger.warning("Escalated %s to hitl: %s", impl_retry_key(slice_), reason)
    return TriageResult(outcome=TriageOutcome.ESCALATED)


def _escalation_reason(signal: _Signal) -> str:
    """Build the human-readable escalation body from the triage signal."""
    if signal.budget_exhausted_before_run:
        return (
            "Retry budget was already exhausted by prior runs; the implementer was not "
            "re-run this session. A human needs to look at this slice."
        )
    if signal.notes is not None:
        return (
            "The implementer could not finish this slice and the retry budget is spent. "
            f"Its notes: {signal.notes.summary}"
        )
    return (
        "The implementer failed repeatedly and the retry budget is spent. "
        "A human needs to look at this slice."
    )
