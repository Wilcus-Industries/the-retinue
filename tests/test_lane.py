"""Tests for the lane classifier (issues #15, #26).

The classifier routes a GitHub issue to one of three lanes:

* a slice carrying ``ready-for-agent`` + ``Part of #<prd>`` -> the orchestrator lane,
* a ``ready-for-agent`` issue with *no* ``Part of #<prd>`` link -> the ad-hoc lane,
* a ``backlog``-labeled issue -> the cron lane.

PRD work runs first by default, but a *standalone* ``priority:critical`` /
``priority:high`` issue preempts the PRD-first ordering onto the orchestrator lane. The
classifier is pure: it reads only the issue's labels (no gh, no network).
"""

from __future__ import annotations

from retinue.lane import IssueFacts, Lane, classify


def _facts(*labels: str, body: str = "") -> IssueFacts:
    return IssueFacts(labels=list(labels), body=body)


# --- PRD slices -> orchestrator lane ---------------------------------------------


def test_ready_prd_slice_routes_to_orchestrator() -> None:
    """A ``ready-for-agent`` + ``Part of #<prd>`` slice goes to the orchestrator lane."""
    facts = _facts("ready-for-agent", body="Implements the thing.\n\nPart of #1\n")
    assert classify(facts).lane is Lane.ORCHESTRATOR
    assert classify(facts).prd_number == 1


def test_ready_slice_with_part_of_is_orchestrator_not_adhoc() -> None:
    """``ready-for-agent`` *with* a ``Part of #<prd>`` link stays on the orchestrator lane."""
    decision = classify(_facts("ready-for-agent", body="Implements it.\n\nPart of #3\n"))
    assert decision.lane is Lane.ORCHESTRATOR
    assert decision.prd_number == 3


# --- ad-hoc issues -> ad-hoc lane (issue #26) ------------------------------------


def test_ready_without_part_of_routes_to_adhoc() -> None:
    """``ready-for-agent`` with no ``Part of #<prd>`` link routes to the ad-hoc lane."""
    decision = classify(_facts("ready-for-agent"))
    assert decision.lane is Lane.ADHOC
    assert decision.prd_number is None
    assert decision.preempts is False


def test_ready_with_unrelated_body_still_adhoc() -> None:
    """A ``ready-for-agent`` issue whose body has no Part-of link is still ad-hoc."""
    facts = _facts("ready-for-agent", body="Do the thing. No parent link here.")
    assert classify(facts).lane is Lane.ADHOC


# --- backlog issues -> cron lane -------------------------------------------------


def test_backlog_issue_routes_to_cron() -> None:
    """A loose ``backlog`` issue (no PRD link) goes to the cron lane."""
    assert classify(_facts("backlog")).lane is Lane.CRON


def test_low_priority_backlog_still_cron() -> None:
    """A ``backlog`` issue at a sub-high priority stays on the cron lane."""
    facts = _facts("backlog", "priority:low")
    assert classify(facts).lane is Lane.CRON


# --- preemption: standalone critical/high jumps the PRD-first order ---------------


def test_standalone_critical_preempts() -> None:
    """A standalone ``priority:critical`` issue preempts onto the orchestrator lane."""
    decision = classify(_facts("priority:critical"))
    assert decision.lane is Lane.ORCHESTRATOR
    assert decision.preempts is True


def test_standalone_high_preempts() -> None:
    """A standalone ``priority:high`` issue preempts onto the orchestrator lane."""
    decision = classify(_facts("priority:high"))
    assert decision.lane is Lane.ORCHESTRATOR
    assert decision.preempts is True


def test_standalone_medium_does_not_preempt() -> None:
    """A standalone ``priority:medium`` issue does not preempt (below the threshold)."""
    decision = classify(_facts("priority:medium"))
    assert decision.preempts is False
    assert decision.lane is Lane.NONE


def test_critical_backlog_preempts_over_cron() -> None:
    """A ``backlog`` issue that is also ``priority:critical`` preempts to the orchestrator.

    The drainer routes loose backlog to cron, but a critical standalone must jump the
    PRD-first ordering rather than wait its turn in the slow cron drain.
    """
    decision = classify(_facts("backlog", "priority:critical"))
    assert decision.lane is Lane.ORCHESTRATOR
    assert decision.preempts is True


def test_ready_prd_slice_does_not_report_preempt() -> None:
    """An ordinary ready PRD slice routes to the orchestrator without preemption."""
    decision = classify(_facts("ready-for-agent", body="Part of #2"))
    assert decision.lane is Lane.ORCHESTRATOR
    assert decision.preempts is False


# --- unroutable ------------------------------------------------------------------


def test_unlabeled_issue_is_unroutable() -> None:
    """An issue with no routing label lands in neither lane."""
    assert classify(_facts()).lane is Lane.NONE
