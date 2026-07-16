"""Tests for reasoning failure triage (issue #8).

When an implementer *fails* or *returns notes*, the orchestrator reasons about the
signal plus the persisted retry count and returns a typed decision — RETRY,
RESLICE, or ESCALATE — never a silent drop. The triage:

* **retries** while the persisted attempt count is below the cap (default from
  ``RepoConfig.retry_cap``), bounded by that persisted count so a doomed slice
  cannot retry forever and the budget survives a restart,
* **reslices** by filing an adjusted slice through the gh ``create_issue`` seam,
* **escalates** to ``hitl`` by fanning a :class:`retinue.notify.Notification` out
  through the shared :class:`retinue.notify.Notifier` (push + comment + label).

Every side-effecting collaborator is faked: a scripted implementer that raises or
returns notes, recording sinks for the notifier, a fake issue creator for the
reslice path, and an on-disk SQLite retry store in a tmp dir. No Agent SDK, no gh,
no push service, no network.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from retinue.container import RunResult
from retinue.container_build import Slice
from retinue.impl_retry import ImplRetryStore
from retinue.notify import (
    CommentRequest,
    LabelRequest,
    Notifier,
    PushRequest,
)
from retinue.repo_config import RepoConfig
from retinue.slicer import CreatedIssue, IssueDraft
from retinue.triage import (
    ImplementerNotes,
    TriageDecision,
    TriageOutcome,
    decide_triage,
    triage_implementer,
)


class _NoContainer:
    """A do-nothing build container the triage threads to the implementer (unused here)."""

    async def run_command(
        self, command: list[str], *, env: Mapping[str, str] | None = None
    ) -> RunResult:
        return RunResult(exit_code=0)

    async def destroy(self) -> None:
        return None


_CONTAINER = _NoContainer()


class _RecordingSinks:
    """Captures notifier sink calls so a test can assert the escalation fired."""

    def __init__(self) -> None:
        self.pushes: list[PushRequest] = []
        self.comments: list[CommentRequest] = []
        self.labels: list[LabelRequest] = []

    async def push(self, request: PushRequest) -> None:
        self.pushes.append(request)

    async def comment(self, request: CommentRequest) -> None:
        self.comments.append(request)

    async def label(self, request: LabelRequest) -> None:
        self.labels.append(request)


class _RecordingCreator:
    """A fake gh issue creator for the reslice path; hands back ascending numbers."""

    def __init__(self, start: int = 100) -> None:
        self.drafts: list[IssueDraft] = []
        self._next = start

    async def __call__(self, draft: IssueDraft) -> CreatedIssue:
        self.drafts.append(draft)
        number = self._next
        self._next += 1
        return CreatedIssue(issue_number=number)


class FailingImplementer:
    """An implementer that always raises; records how many times it was invoked."""

    def __init__(self) -> None:
        self.calls = 0

    async def implement(self, slice_: Slice, *, container: object) -> None:
        self.calls += 1
        raise RuntimeError("implementer blew up")

    def auth_env(self) -> dict[str, str]:
        return {}


class NotesImplementer:
    """An implementer that completes but returns notes instead of raising."""

    def __init__(self, notes: ImplementerNotes) -> None:
        self.calls = 0
        self._notes = notes

    async def implement(self, slice_: Slice, *, container: object) -> ImplementerNotes:
        self.calls += 1
        return self._notes

    def auth_env(self) -> dict[str, str]:
        return {}


class OkImplementer:
    """An implementer that completes cleanly with no notes."""

    def __init__(self) -> None:
        self.calls = 0

    async def implement(self, slice_: Slice, *, container: object) -> None:
        self.calls += 1

    def auth_env(self) -> dict[str, str]:
        return {}


def _slice(issue_number: int = 7) -> Slice:
    return Slice(repo_full_name="owner/repo", issue_number=issue_number, prd_number=1)


def _notifier(sinks: _RecordingSinks) -> Notifier:
    return Notifier(push=sinks.push, comment=sinks.comment, label=sinks.label)


# --- the pure decision function --------------------------------------------------


def test_decide_retry_while_under_cap() -> None:
    """Below the cap, the next decision is RETRY."""
    assert decide_triage(failed=True, notes=None, attempts=0, cap=3) is TriageDecision.RETRY
    assert decide_triage(failed=True, notes=None, attempts=2, cap=3) is TriageDecision.RETRY


def test_decide_escalate_at_cap() -> None:
    """Once attempts reach the cap, a hard failure escalates rather than retrying."""
    assert decide_triage(failed=True, notes=None, attempts=3, cap=3) is TriageDecision.ESCALATE


def test_decide_zero_cap_escalates_immediately() -> None:
    """A retry_cap of zero means no retries: the first failure escalates."""
    assert decide_triage(failed=True, notes=None, attempts=0, cap=0) is TriageDecision.ESCALATE


def test_decide_notes_requesting_reslice_reslices() -> None:
    """Notes that ask for a reslice reach a RESLICE decision, not a silent drop."""
    notes = ImplementerNotes(summary="scope too big", reslice=True)
    assert decide_triage(failed=False, notes=notes, attempts=0, cap=3) is TriageDecision.RESLICE


def test_decide_notes_without_reslice_retries_under_cap() -> None:
    """Plain notes under the cap are a reasoned RETRY, never silently dropped."""
    notes = ImplementerNotes(summary="flaky, please retry")
    assert decide_triage(failed=False, notes=notes, attempts=0, cap=3) is TriageDecision.RETRY


def test_decide_notes_at_cap_escalate() -> None:
    """Notes with the retry budget exhausted escalate to a human."""
    notes = ImplementerNotes(summary="still stuck")
    assert decide_triage(failed=False, notes=notes, attempts=3, cap=3) is TriageDecision.ESCALATE


# --- the orchestrated triage: failure retried to cap then escalated --------------


@pytest.mark.asyncio
async def test_failure_retried_up_to_cap_then_escalates(tmp_path: Path) -> None:
    """An injected failure retries up to the persisted cap, then escalates (notify+label)."""
    store = ImplRetryStore(tmp_path / "retry.sqlite3")
    implementer = FailingImplementer()
    sinks = _RecordingSinks()
    creator = _RecordingCreator()
    config = RepoConfig(retry_cap=2)

    result = await triage_implementer(
        _slice(),
        config,
        implementer=implementer,
        notifier=_notifier(sinks),
        create_issue=creator,
        retry_store=store,
        container=_CONTAINER,
    )

    # One initial attempt + retries up to the cap of 2 = 3 invocations total.
    assert implementer.calls == 3
    assert result.outcome is TriageOutcome.ESCALATED
    # Escalation fired: a comment + the hitl label landed (push is best-effort).
    assert len(sinks.comments) == 1
    assert sinks.labels[0].label == "hitl"
    # No reslice issue was filed on the escalate path.
    assert creator.drafts == []
    # The persisted count was consumed up to the cap.
    assert await store.count("owner/repo#7") == 2


@pytest.mark.asyncio
async def test_failure_with_zero_cap_escalates_without_retry(tmp_path: Path) -> None:
    """retry_cap=0 escalates the first failure with no retry."""
    store = ImplRetryStore(tmp_path / "retry.sqlite3")
    implementer = FailingImplementer()
    sinks = _RecordingSinks()
    config = RepoConfig(retry_cap=0)

    result = await triage_implementer(
        _slice(),
        config,
        implementer=implementer,
        notifier=_notifier(sinks),
        create_issue=_RecordingCreator(),
        retry_store=store,
        container=_CONTAINER,
    )

    assert implementer.calls == 1
    assert result.outcome is TriageOutcome.ESCALATED
    assert len(sinks.comments) == 1


@pytest.mark.asyncio
async def test_persisted_count_bounds_retries_across_runs(tmp_path: Path) -> None:
    """A prior run's attempts count against the cap: the budget is not reset."""
    db = tmp_path / "retry.sqlite3"
    # A previous run already consumed the whole budget for this slice.
    await ImplRetryStore(db).record_attempt("owner/repo#7")
    await ImplRetryStore(db).record_attempt("owner/repo#7")

    implementer = FailingImplementer()
    sinks = _RecordingSinks()
    config = RepoConfig(retry_cap=2)

    result = await triage_implementer(
        _slice(),
        config,
        implementer=implementer,
        notifier=_notifier(sinks),
        create_issue=_RecordingCreator(),
        retry_store=ImplRetryStore(db),
        container=_CONTAINER,
    )

    # Budget already spent: the slice escalates without another implement attempt.
    assert implementer.calls == 0
    assert result.outcome is TriageOutcome.ESCALATED


@pytest.mark.asyncio
async def test_prior_exhausted_budget_escalates_with_honest_reason(tmp_path: Path) -> None:
    """A budget spent by prior runs escalates with a reason that does not claim a fresh failure."""
    db = tmp_path / "retry.sqlite3"
    await ImplRetryStore(db).record_attempt("owner/repo#7")
    await ImplRetryStore(db).record_attempt("owner/repo#7")
    implementer = FailingImplementer()
    sinks = _RecordingSinks()

    result = await triage_implementer(
        _slice(),
        RepoConfig(retry_cap=2),
        implementer=implementer,
        notifier=_notifier(sinks),
        create_issue=_RecordingCreator(),
        retry_store=ImplRetryStore(db),
        container=_CONTAINER,
    )

    assert implementer.calls == 0
    assert result.outcome is TriageOutcome.ESCALATED
    body = sinks.comments[0].body.lower()
    # Honest: the implementer never ran this session, so it did not "fail repeatedly".
    assert "already exhausted" in body or "already spent" in body
    assert "failed repeatedly" not in body


class _CountingRetryStore(ImplRetryStore):
    """An ImplRetryStore that records how many times ``count`` is called."""

    def __init__(self, db_path: Path) -> None:
        super().__init__(db_path)
        self.count_calls = 0

    async def count(self, key: str) -> int:
        self.count_calls += 1
        return await super().count(key)


@pytest.mark.asyncio
async def test_retry_count_is_read_once_per_triage(tmp_path: Path) -> None:
    """The persisted retry count is read once, not re-read on every loop iteration."""
    store = _CountingRetryStore(tmp_path / "retry.sqlite3")
    implementer = FailingImplementer()
    sinks = _RecordingSinks()

    await triage_implementer(
        _slice(),
        RepoConfig(retry_cap=2),
        implementer=implementer,
        notifier=_notifier(sinks),
        create_issue=_RecordingCreator(),
        retry_store=store,
        container=_CONTAINER,
    )

    assert store.count_calls == 1


# --- the orchestrated triage: a retry that then succeeds -------------------------


class FlakyImplementer:
    """Fails the first ``fail_times`` invocations, then succeeds."""

    def __init__(self, fail_times: int) -> None:
        self.calls = 0
        self._fail_times = fail_times

    async def implement(self, slice_: Slice, *, container: object) -> None:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise RuntimeError("transient failure")

    def auth_env(self) -> dict[str, str]:
        return {}


@pytest.mark.asyncio
async def test_transient_failure_retried_then_succeeds(tmp_path: Path) -> None:
    """A failure that clears on retry yields a BUILT outcome, no escalation."""
    store = ImplRetryStore(tmp_path / "retry.sqlite3")
    implementer = FlakyImplementer(fail_times=1)
    sinks = _RecordingSinks()
    config = RepoConfig(retry_cap=3)

    result = await triage_implementer(
        _slice(),
        config,
        implementer=implementer,
        notifier=_notifier(sinks),
        create_issue=_RecordingCreator(),
        retry_store=store,
        container=_CONTAINER,
    )

    assert implementer.calls == 2
    assert result.outcome is TriageOutcome.BUILT
    assert sinks.comments == []


# --- the orchestrated triage: clean build is a no-op -----------------------------


@pytest.mark.asyncio
async def test_clean_build_is_built_with_no_side_effects(tmp_path: Path) -> None:
    """An implementer that finishes cleanly with no notes is BUILT, untriaged."""
    store = ImplRetryStore(tmp_path / "retry.sqlite3")
    implementer = OkImplementer()
    sinks = _RecordingSinks()
    creator = _RecordingCreator()

    result = await triage_implementer(
        _slice(),
        RepoConfig(),
        implementer=implementer,
        notifier=_notifier(sinks),
        create_issue=creator,
        retry_store=store,
        container=_CONTAINER,
    )

    assert implementer.calls == 1
    assert result.outcome is TriageOutcome.BUILT
    assert sinks.comments == []
    assert creator.drafts == []


# --- the orchestrated triage: returned notes reach a reasoned decision -----------


@pytest.mark.asyncio
async def test_notes_requesting_reslice_files_adjusted_issue(tmp_path: Path) -> None:
    """Notes asking for a reslice file an adjusted issue, not a silent drop."""
    store = ImplRetryStore(tmp_path / "retry.sqlite3")
    notes = ImplementerNotes(
        summary="This slice is two slices; split out the schema migration.",
        reslice=True,
    )
    implementer = NotesImplementer(notes)
    sinks = _RecordingSinks()
    creator = _RecordingCreator(start=200)

    result = await triage_implementer(
        _slice(),
        RepoConfig(),
        implementer=implementer,
        notifier=_notifier(sinks),
        create_issue=creator,
        retry_store=store,
        container=_CONTAINER,
    )

    assert result.outcome is TriageOutcome.RESLICED
    assert result.resliced_issue == 200
    # The adjusted slice was filed with the implementer's reasoning in the body.
    assert len(creator.drafts) == 1
    assert "schema migration" in creator.drafts[0].body
    # A reslice is not an escalation: no hitl label on the original.
    assert sinks.labels == []


@pytest.mark.asyncio
async def test_notes_retry_then_clean_is_built(tmp_path: Path) -> None:
    """Plain notes under the cap trigger a reasoned retry that can then succeed."""
    store = ImplRetryStore(tmp_path / "retry.sqlite3")

    class NotesThenClean:
        def __init__(self) -> None:
            self.calls = 0

        async def implement(
            self, slice_: Slice, *, container: object
        ) -> ImplementerNotes | None:
            self.calls += 1
            if self.calls == 1:
                return ImplementerNotes(summary="needed another pass")
            return None

        def auth_env(self) -> dict[str, str]:
            return {}

    implementer = NotesThenClean()
    sinks = _RecordingSinks()

    result = await triage_implementer(
        _slice(),
        RepoConfig(retry_cap=3),
        implementer=implementer,
        notifier=_notifier(sinks),
        create_issue=_RecordingCreator(),
        retry_store=store,
        container=_CONTAINER,
    )

    assert implementer.calls == 2
    assert result.outcome is TriageOutcome.BUILT
    assert sinks.comments == []


@pytest.mark.asyncio
async def test_notes_never_silently_dropped_when_budget_exhausted(tmp_path: Path) -> None:
    """Notes with the budget spent escalate (notify + label), never vanish silently."""
    store = ImplRetryStore(tmp_path / "retry.sqlite3")
    notes = ImplementerNotes(summary="still cannot finish this")
    implementer = NotesImplementer(notes)
    sinks = _RecordingSinks()
    config = RepoConfig(retry_cap=0)

    result = await triage_implementer(
        _slice(),
        config,
        implementer=implementer,
        notifier=_notifier(sinks),
        create_issue=_RecordingCreator(),
        retry_store=store,
        container=_CONTAINER,
    )

    assert result.outcome is TriageOutcome.ESCALATED
    assert len(sinks.comments) == 1
    assert sinks.labels[0].label == "hitl"
    # The implementer's reasoning is carried into the escalation comment.
    assert "still cannot finish" in sinks.comments[0].body
