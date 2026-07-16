"""Tests for the heimdall verdict loopback (issue #11).

When heimdall posts a bot review on the ``retinue/prd-<n>`` -> staging PR, the
loopback reads the verdict and reasons about it plus the *persisted* per-PR round
count:

* **blocking findings** (severity at/above the threshold) become fix-issues
  (``ready-for-agent`` + ``Part of #<prd>``) that rebuild onto the **same**
  integration branch and re-trigger heimdall review — bounded at ``retry_cap``=3
  rounds (persisted),
* **non-blocking nits** become ``backlog`` issues carrying heimdall severity mapped
  1:1 to a ``priority:<severity>`` label,
* **zero blocking findings** = CONVERGED, proceeding to handoff,
* **cap-hit while still blocked** = ESCALATE: comment the PRD, label, notify, and
  leave the PR open.

Every collaborator — the heimdall verdict input, the gh issue creator, the
rebuild-onto-same-branch trigger, the handoff, the notifier sinks, and the SQLite
round store — is faked/injected. No real gh, heimdall, push service, or network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from retinue.gh import GhCommandError, GhResult
from retinue.loopback import (
    BACKLOG_LABEL,
    GhCliRebuilder,
    HeimdallFinding,
    HeimdallReview,
    HeimdallRoundStore,
    RebuildRequest,
    ReviewState,
    Severity,
    VerdictDecision,
    VerdictOutcome,
    VerdictResult,
    _parse_review_requested,
    _re_review_args,
    decide_verdict,
    priority_label,
    process_review,
)
from retinue.notify import CommentRequest, LabelRequest, Notifier, PushRequest
from retinue.repo_config import RepoConfig
from retinue.slicer import READY_LABEL, CreatedIssue, IssueDraft


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
    """A fake gh issue creator; records drafts and hands back ascending numbers."""

    def __init__(self, start: int = 100) -> None:
        self.drafts: list[IssueDraft] = []
        self._next = start

    async def __call__(self, draft: IssueDraft) -> CreatedIssue:
        self.drafts.append(draft)
        number = self._next
        self._next += 1
        return CreatedIssue(issue_number=number)


class _RecordingRebuilder:
    """Records rebuild-onto-same-branch triggers (re-running the build + re-review)."""

    def __init__(self) -> None:
        self.requests: list[RebuildRequest] = []

    async def __call__(self, request: RebuildRequest) -> None:
        self.requests.append(request)


class _RecordingHandoff:
    """Records the handoff call fired when a PR converges."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def __call__(self, *, repo_full_name: str, pr_number: int) -> None:
        self.calls.append((repo_full_name, pr_number))


def _notifier(sinks: _RecordingSinks) -> Notifier:
    return Notifier(push=sinks.push, comment=sinks.comment, label=sinks.label)


def _review(
    state: ReviewState,
    findings: list[HeimdallFinding],
    *,
    clean_pass: bool = False,
) -> HeimdallReview:
    return HeimdallReview(
        repo_full_name="owner/repo",
        pr_number=42,
        prd_number=1,
        prd_issue_number=1,
        integration_branch="retinue/prd-1",
        state=state,
        findings=findings,
        clean_pass=clean_pass,
    )


def _blocking(severity: Severity = Severity.HIGH) -> HeimdallFinding:
    return HeimdallFinding(summary="null deref in handler", severity=severity)


def _nit(severity: Severity = Severity.LOW) -> HeimdallFinding:
    return HeimdallFinding(summary="rename this variable", severity=severity)


async def _run(
    review: HeimdallReview,
    *,
    config: RepoConfig,
    store: HeimdallRoundStore,
    sinks: _RecordingSinks,
    creator: _RecordingCreator | None = None,
    rebuilder: _RecordingRebuilder | None = None,
    handoff: _RecordingHandoff | None = None,
) -> tuple[_RecordingCreator, _RecordingRebuilder, _RecordingHandoff, VerdictResult]:
    creator = creator or _RecordingCreator()
    rebuilder = rebuilder or _RecordingRebuilder()
    handoff = handoff or _RecordingHandoff()
    result = await process_review(
        review,
        config,
        round_store=store,
        create_issue=creator,
        rebuild=rebuilder,
        handoff=handoff,
        notifier=_notifier(sinks),
    )
    return creator, rebuilder, handoff, result


# --- severity -> priority label mapping (unit-tested 1:1) ------------------------


def test_priority_label_maps_severity_one_to_one() -> None:
    """Every heimdall severity maps 1:1 to a ``priority:<severity>`` label."""
    assert priority_label(Severity.LOW) == "priority:low"
    assert priority_label(Severity.MEDIUM) == "priority:medium"
    assert priority_label(Severity.HIGH) == "priority:high"
    assert priority_label(Severity.CRITICAL) == "priority:critical"


# --- the pure verdict decision ---------------------------------------------------


def test_decide_approved_converges_regardless_of_findings_or_rounds() -> None:
    """An APPROVED review converges — the authoritative state beats parsed findings."""
    approved = ReviewState.APPROVED
    assert (
        decide_verdict(state=approved, blocking=0, carries_verdict=True, rounds=0, cap=3)
        is VerdictDecision.CONVERGED
    )
    # Even with blocking-looking parsed lines or the budget spent: approved is approved.
    assert (
        decide_verdict(state=approved, blocking=2, carries_verdict=True, rounds=3, cap=3)
        is VerdictDecision.CONVERGED
    )


def test_decide_verdictless_comment_is_ignored() -> None:
    """A COMMENTED review with no verdict (heimdall's "review failed" note, progress
    prose) neither converges nor rebuilds."""
    commented = ReviewState.COMMENTED
    assert (
        decide_verdict(state=commented, blocking=0, carries_verdict=False, rounds=0, cap=3)
        is VerdictDecision.NO_VERDICT
    )


def test_decide_verdict_carrying_comment_converges() -> None:
    """Heimdall never submits APPROVED: its clean pass is a COMMENTED review (the
    "no concerns" body or nits-only findings), so a verdict-carrying COMMENT converges
    — even with blocking-severity parsed lines, the state (not blocking) is
    authoritative, matching the APPROVED path."""
    commented = ReviewState.COMMENTED
    assert (
        decide_verdict(state=commented, blocking=0, carries_verdict=True, rounds=0, cap=3)
        is VerdictDecision.CONVERGED
    )
    assert (
        decide_verdict(state=commented, blocking=2, carries_verdict=True, rounds=0, cap=3)
        is VerdictDecision.CONVERGED
    )


def test_decide_rebuild_while_under_cap() -> None:
    """Blocking findings under the round cap drive a REBUILD."""
    changes = ReviewState.REQUEST_CHANGES
    assert (
        decide_verdict(state=changes, blocking=2, carries_verdict=True, rounds=0, cap=3)
        is VerdictDecision.REBUILD
    )
    assert (
        decide_verdict(state=changes, blocking=1, carries_verdict=True, rounds=2, cap=3)
        is VerdictDecision.REBUILD
    )


def test_decide_escalate_at_cap_while_blocked() -> None:
    """Blocking findings with the round budget spent escalates."""
    changes = ReviewState.REQUEST_CHANGES
    assert (
        decide_verdict(state=changes, blocking=1, carries_verdict=True, rounds=3, cap=3)
        is VerdictDecision.ESCALATE
    )


def test_decide_zero_cap_escalates_first_blocking_round() -> None:
    """retry_cap=0 means no rebuild rounds: the first blocking verdict escalates."""
    changes = ReviewState.REQUEST_CHANGES
    assert (
        decide_verdict(state=changes, blocking=1, carries_verdict=True, rounds=0, cap=0)
        is VerdictDecision.ESCALATE
    )


def test_decide_request_changes_without_parseable_findings_escalates() -> None:
    """REQUEST_CHANGES with zero parsed blocking findings escalates, never converges.

    Heimdall explicitly rejected the PR; if its body didn't parse into actionable
    findings the loopback must not silently hand the PR off as clean.
    """
    changes = ReviewState.REQUEST_CHANGES
    assert (
        decide_verdict(state=changes, blocking=0, carries_verdict=True, rounds=0, cap=3)
        is VerdictDecision.ESCALATE
    )


# --- REQUEST_CHANGES -> fix-issues that rebuild onto the same branch -------------


@pytest.mark.asyncio
async def test_request_changes_files_fix_issues_and_rebuilds(tmp_path: Path) -> None:
    """A REQUEST_CHANGES verdict files fix-issues that rebuild onto the same branch."""
    store = HeimdallRoundStore(tmp_path / "rounds.sqlite3")
    sinks = _RecordingSinks()
    review = _review(ReviewState.REQUEST_CHANGES, [_blocking(Severity.HIGH)])

    creator, rebuilder, handoff, result = await _run(
        review, config=RepoConfig(retry_cap=3), store=store, sinks=sinks
    )

    assert result.outcome is VerdictOutcome.REBUILT
    # A fix-issue was filed, ready-for-agent + Part of #<prd>, not backlog.
    assert len(creator.drafts) == 1
    fix = creator.drafts[0]
    assert READY_LABEL in fix.labels
    assert BACKLOG_LABEL not in fix.labels
    assert "Part of #1" in fix.body
    # The rebuild was triggered onto the SAME integration branch.
    assert len(rebuilder.requests) == 1
    assert rebuilder.requests[0].integration_branch == "retinue/prd-1"
    assert rebuilder.requests[0].fix_issues == [100]
    # Converging handoff did not fire on a rebuild.
    assert handoff.calls == []
    # One round was persisted.
    assert await store.count("owner/repo#42") == 1


@pytest.mark.asyncio
async def test_loop_bounded_at_retry_cap_persisted(tmp_path: Path) -> None:
    """The rebuild loop is bounded at retry_cap=3 across persisted rounds."""
    db = tmp_path / "rounds.sqlite3"
    config = RepoConfig(retry_cap=3)
    review = _review(ReviewState.REQUEST_CHANGES, [_blocking()])

    # Three blocking rounds each rebuild and persist a round.
    for expected_round in (1, 2, 3):
        creator, rebuilder, _, result = await _run(
            review, config=config, store=HeimdallRoundStore(db), sinks=_RecordingSinks()
        )
        assert result.outcome is VerdictOutcome.REBUILT
        assert len(rebuilder.requests) == 1
        assert await HeimdallRoundStore(db).count("owner/repo#42") == expected_round

    # The fourth blocking verdict is over the cap: escalate, no further rebuild.
    sinks = _RecordingSinks()
    creator, rebuilder, _, result = await _run(
        review, config=config, store=HeimdallRoundStore(db), sinks=sinks
    )
    assert result.outcome is VerdictOutcome.ESCALATED
    assert rebuilder.requests == []
    assert creator.drafts == []
    # The round count is not bumped past the cap on the escalate path.
    assert await HeimdallRoundStore(db).count("owner/repo#42") == 3


# --- non-blocking nits -> backlog issues with priority labels --------------------


@pytest.mark.asyncio
async def test_non_blocking_nits_become_backlog_priority_issues(tmp_path: Path) -> None:
    """Non-blocking nits file backlog issues with priority:<severity> labels, no rebuild."""
    store = HeimdallRoundStore(tmp_path / "rounds.sqlite3")
    sinks = _RecordingSinks()
    review = _review(
        ReviewState.APPROVED,
        [_nit(Severity.LOW), _nit(Severity.MEDIUM)],
    )

    creator, rebuilder, handoff, result = await _run(
        review, config=RepoConfig(), store=store, sinks=sinks
    )

    # Approved -> converged, and the nits are filed as backlog.
    assert result.outcome is VerdictOutcome.CONVERGED
    assert handoff.calls == [("owner/repo", 42)]
    assert len(creator.drafts) == 2
    labels = {label for draft in creator.drafts for label in draft.labels}
    assert BACKLOG_LABEL in labels
    assert "priority:low" in labels
    assert "priority:medium" in labels
    # Backlog issues still link to the PRD but are not rebuilt.
    assert all("Part of #1" in draft.body for draft in creator.drafts)
    assert rebuilder.requests == []


@pytest.mark.asyncio
async def test_verdictless_comment_is_ignored(tmp_path: Path) -> None:
    """A verdict-less COMMENTED review neither hands off nor rebuilds nor files issues.

    Heimdall posts verdict-less COMMENT notes (the "review failed" note, progress
    prose) that parse zero findings and carry no clean-pass marker; converging on one
    would hand off a PR heimdall never actually reviewed.
    """
    store = HeimdallRoundStore(tmp_path / "rounds.sqlite3")
    sinks = _RecordingSinks()
    review = _review(ReviewState.COMMENTED, [])

    creator, rebuilder, handoff, result = await _run(
        review, config=RepoConfig(), store=store, sinks=sinks
    )

    assert result.outcome is VerdictOutcome.IGNORED
    assert handoff.calls == []
    assert creator.drafts == []
    assert rebuilder.requests == []
    assert sinks.comments == []
    assert await store.count("owner/repo#42") == 0


@pytest.mark.asyncio
async def test_clean_pass_comment_converges_with_no_issues(tmp_path: Path) -> None:
    """Heimdall's clean pass — a COMMENTED "no concerns" review — converges to handoff.

    Heimdall never submits APPROVED; its clean verdict is a COMMENT with the
    no-concerns body, so that review must hand off rather than be ignored.
    """
    store = HeimdallRoundStore(tmp_path / "rounds.sqlite3")
    sinks = _RecordingSinks()
    review = _review(ReviewState.COMMENTED, [], clean_pass=True)

    creator, rebuilder, handoff, result = await _run(
        review, config=RepoConfig(), store=store, sinks=sinks
    )

    assert result.outcome is VerdictOutcome.CONVERGED
    assert handoff.calls == [("owner/repo", 42)]
    assert creator.drafts == []
    assert rebuilder.requests == []


@pytest.mark.asyncio
async def test_nits_only_comment_converges_and_files_backlog(tmp_path: Path) -> None:
    """A COMMENTED review with parsed findings converges and parks them as backlog.

    Heimdall COMMENTs (rather than REQUEST_CHANGES) when no finding crosses its
    blocking threshold, so the findings are nits: file them as backlog and hand off.
    """
    store = HeimdallRoundStore(tmp_path / "rounds.sqlite3")
    sinks = _RecordingSinks()
    review = _review(ReviewState.COMMENTED, [_nit(), _nit(Severity.MEDIUM)])

    creator, rebuilder, handoff, result = await _run(
        review, config=RepoConfig(), store=store, sinks=sinks
    )

    assert result.outcome is VerdictOutcome.CONVERGED
    assert handoff.calls == [("owner/repo", 42)]
    assert len(creator.drafts) == 2
    labels = {label for draft in creator.drafts for label in draft.labels}
    assert BACKLOG_LABEL in labels
    assert rebuilder.requests == []


@pytest.mark.asyncio
async def test_approved_with_blocking_lines_still_converges(tmp_path: Path) -> None:
    """APPROVED beats parsed blocking-looking lines: converge, park them as backlog."""
    store = HeimdallRoundStore(tmp_path / "rounds.sqlite3")
    sinks = _RecordingSinks()
    review = _review(ReviewState.APPROVED, [_blocking(Severity.HIGH)])

    creator, rebuilder, handoff, result = await _run(
        review, config=RepoConfig(retry_cap=3), store=store, sinks=sinks
    )

    assert result.outcome is VerdictOutcome.CONVERGED
    assert handoff.calls == [("owner/repo", 42)]
    assert rebuilder.requests == []
    # The finding isn't lost: it's parked as backlog carrying its severity.
    assert len(creator.drafts) == 1
    assert BACKLOG_LABEL in creator.drafts[0].labels
    assert "priority:high" in creator.drafts[0].labels


@pytest.mark.asyncio
async def test_blocking_and_nits_together_rebuild_and_file_backlog(tmp_path: Path) -> None:
    """A mixed verdict rebuilds on the blocking findings and backlogs the nits."""
    store = HeimdallRoundStore(tmp_path / "rounds.sqlite3")
    sinks = _RecordingSinks()
    review = _review(
        ReviewState.REQUEST_CHANGES,
        [_blocking(Severity.CRITICAL), _nit(Severity.LOW)],
    )

    creator, rebuilder, _, result = await _run(
        review, config=RepoConfig(retry_cap=3), store=store, sinks=sinks
    )

    assert result.outcome is VerdictOutcome.REBUILT
    # The blocking finding became a fix-issue; the nit became a backlog issue.
    fix_drafts = [d for d in creator.drafts if BACKLOG_LABEL not in d.labels]
    backlog_drafts = [d for d in creator.drafts if BACKLOG_LABEL in d.labels]
    assert len(fix_drafts) == 1
    assert len(backlog_drafts) == 1
    assert "priority:low" in backlog_drafts[0].labels
    # Only the fix-issue (filed first, number 100) is fed to the rebuild; the
    # backlog nit is not rebuilt.
    assert rebuilder.requests[0].fix_issues == [100]


# --- zero blocking findings = converged -> handoff -------------------------------


@pytest.mark.asyncio
async def test_approved_with_no_findings_converges_to_handoff(tmp_path: Path) -> None:
    """An APPROVED verdict with no findings converges straight to handoff."""
    store = HeimdallRoundStore(tmp_path / "rounds.sqlite3")
    sinks = _RecordingSinks()
    review = _review(ReviewState.APPROVED, [])

    creator, rebuilder, handoff, result = await _run(
        review, config=RepoConfig(), store=store, sinks=sinks
    )

    assert result.outcome is VerdictOutcome.CONVERGED
    assert handoff.calls == [("owner/repo", 42)]
    assert creator.drafts == []
    assert rebuilder.requests == []
    assert sinks.comments == []


# --- cap-hit while still blocked = escalate, PR left open ------------------------


@pytest.mark.asyncio
async def test_cap_exhaustion_escalates_and_leaves_pr_open(tmp_path: Path) -> None:
    """With the round budget spent and still blocked: comment + label + notify, PR open."""
    db = tmp_path / "rounds.sqlite3"
    # Three prior rebuild rounds already consumed the budget.
    for _ in range(3):
        await HeimdallRoundStore(db).record_round("owner/repo#42")

    sinks = _RecordingSinks()
    review = _review(ReviewState.REQUEST_CHANGES, [_blocking(), _nit(Severity.LOW)])

    creator, rebuilder, handoff, result = await _run(
        review,
        config=RepoConfig(retry_cap=3),
        store=HeimdallRoundStore(db),
        sinks=sinks,
    )

    assert result.outcome is VerdictOutcome.ESCALATED
    # Escalation fired against the PRD: a comment + a label landed.
    assert len(sinks.comments) == 1
    assert sinks.comments[0].issue_number == 1
    assert sinks.labels[0].issue_number == 1
    assert sinks.labels[0].label == "hitl"
    # No rebuild, no fix-issue, no handoff: the PR is left open for a human.
    assert rebuilder.requests == []
    assert handoff.calls == []
    # The nit is still parked as backlog even on the escalate path.
    assert len(creator.drafts) == 1
    assert BACKLOG_LABEL in creator.drafts[0].labels
    assert result.filed_issues == [100]
    # result advertises the PR was left open.
    assert result.pr_left_open is True


@pytest.mark.asyncio
async def test_severity_below_threshold_escalates_a_rejection(tmp_path: Path) -> None:
    """REQUEST_CHANGES whose findings all parse below the threshold escalates.

    Heimdall rejected the PR but nothing actionable parsed as blocking — silently
    converging would hand off a rejected PR, so a human is pulled in instead. The
    parsed nit is still parked as backlog.
    """
    store = HeimdallRoundStore(tmp_path / "rounds.sqlite3")
    sinks = _RecordingSinks()
    # Threshold defaults to HIGH; a LOW finding is non-blocking.
    review = _review(ReviewState.REQUEST_CHANGES, [_nit(Severity.LOW)])

    creator, rebuilder, handoff, result = await _run(
        review, config=RepoConfig(), store=store, sinks=sinks
    )

    assert result.outcome is VerdictOutcome.ESCALATED
    assert rebuilder.requests == []
    assert handoff.calls == []
    assert len(creator.drafts) == 1
    assert BACKLOG_LABEL in creator.drafts[0].labels
    assert sinks.labels[0].label == "hitl"
    assert result.pr_left_open is True


@pytest.mark.asyncio
async def test_rebuild_trigger_failure_escalates_instead_of_raising(
    tmp_path: Path,
) -> None:
    """A failed rebuild trigger (gh error) escalates with the round already counted.

    The round is recorded before the trigger so a doomed PR cannot loop unbounded;
    when the re-review request then fails, raising would just retry the whole job
    and double-file the fix-issues — instead a human is notified and the PR is left
    open with the fix-issues already filed.
    """

    class _FailingRebuilder:
        async def __call__(self, request: RebuildRequest) -> None:
            raise GhCommandError(
                ["pr", "edit"], GhResult(exit_code=1, stderr="boom")
            )

    store = HeimdallRoundStore(tmp_path / "rounds.sqlite3")
    sinks = _RecordingSinks()
    review = _review(ReviewState.REQUEST_CHANGES, [_blocking()])
    creator = _RecordingCreator()

    result = await process_review(
        review,
        RepoConfig(retry_cap=3),
        round_store=store,
        create_issue=creator,
        rebuild=_FailingRebuilder(),
        handoff=_RecordingHandoff(),
        notifier=_notifier(sinks),
    )

    assert result.outcome is VerdictOutcome.ESCALATED
    assert result.pr_left_open is True
    # The fix-issue was filed and the round consumed before the trigger failed.
    assert len(creator.drafts) == 1
    assert await store.count("owner/repo#42") == 1
    # A human was pulled in.
    assert sinks.labels[0].label == "hitl"
    assert "re-review" in sinks.comments[0].body


# --- the production GhCliRebuilder: pure/parseable parts (no live gh/network) ------


class _RecordingRunner:
    """A fake :class:`GhRunner`; records each ``(args, env)`` and returns a canned result."""

    def __init__(self, result: GhResult) -> None:
        self.result = result
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    async def run(self, args: list[str], *, env: dict[str, str]) -> GhResult:
        self.calls.append((args, env))
        return self.result


def _rebuild_request() -> RebuildRequest:
    return RebuildRequest(
        repo_full_name="owner/repo",
        integration_branch="retinue/prd-1",
        prd_number=1,
        pr_number=42,
        fix_issues=[100, 101],
    )


def test_re_review_args_assemble_a_heimdall_review_re_request() -> None:
    """The re-review argv re-adds the configured heimdall bot as a reviewer on the PR."""
    args = _re_review_args(_rebuild_request(), "heimdall[bot]")
    assert args[:5] == ["pr", "edit", "42", "--repo", "owner/repo"]
    # The heimdall bot is re-requested as a reviewer (this re-triggers its bot review).
    add_at = args.index("--add-reviewer")
    assert args[add_at + 1] == "heimdall[bot]"


def test_re_review_args_use_the_configured_reviewer_login() -> None:
    """The re-review re-requests whatever login is configured, not a hardcoded one."""
    args = _re_review_args(_rebuild_request(), "watcher[bot]")
    add_at = args.index("--add-reviewer")
    assert args[add_at + 1] == "watcher[bot]"


def test_parse_review_requested_reads_the_pr_number_from_the_url() -> None:
    """The re-review payload (the edited PR URL) parses back to its PR number."""
    assert (
        _parse_review_requested("https://github.com/owner/repo/pull/42\n") == 42
    )
    assert _parse_review_requested("https://github.com/owner/repo/pull/42/") == 42


def test_parse_review_requested_raises_on_a_numberless_payload() -> None:
    """A malformed re-review payload fails loudly rather than dropping the re-review."""
    with pytest.raises(ValueError, match="no PR number"):
        _parse_review_requested("not-a-url")


@pytest.mark.asyncio
async def test_gh_cli_rebuilder_re_requests_review_without_refiling() -> None:
    """The real rebuilder only re-requests the heimdall review — no duplicate issues.

    The fix-issues were already filed by the loopback before the rebuild trigger;
    re-filing them here would double every finding as a second ready-for-agent issue.
    """
    runner = _RecordingRunner(
        GhResult(exit_code=0, stdout="https://github.com/owner/repo/pull/42")
    )
    rebuilder = GhCliRebuilder(
        runner, token="ghs_tok", reviewer_login="watcher[bot]"
    )

    await rebuilder(_rebuild_request())

    # Exactly one gh call: the heimdall re-review, GH_TOKEN-authenticated, re-requesting
    # the *configured* reviewer login (not a hardcoded one).
    assert len(runner.calls) == 1
    args, env = runner.calls[0]
    assert args == _re_review_args(_rebuild_request(), "watcher[bot]")
    assert env == {"GH_TOKEN": "ghs_tok"}


@pytest.mark.asyncio
async def test_gh_cli_rebuilder_raises_on_a_failed_re_review() -> None:
    """A non-zero ``gh`` re-review surfaces a GhCommandError carrying the stderr."""
    runner = _RecordingRunner(GhResult(exit_code=1, stderr="no such PR"))
    rebuilder = GhCliRebuilder(
        runner,
        token="ghs_tok",
        reviewer_login="heimdall[bot]",
    )

    with pytest.raises(GhCommandError, match="no such PR"):
        await rebuilder(_rebuild_request())
