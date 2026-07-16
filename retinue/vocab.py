"""Shared retinue vocabulary: labels, severities, and branch naming.

The bottom layer of the package — every lane and workflow module (slicer, loopback,
lane, cron, ad-hoc drain, handoff, webhook) speaks this vocabulary, so it lives here
rather than in any one of them. This module imports nothing from the rest of
:mod:`retinue`; anything here must stay importable by every other module without
creating a cycle.
"""

from __future__ import annotations

import enum

# The single "build me" trigger: an issue the retinue picks up and builds.
READY_LABEL = "ready-for-agent"

# Human-in-the-loop escalation: the retinue stops and leaves the issue for a human.
HITL_LABEL = "hitl"

# Provenance marker: a slice the slicer filed off a PRD. Stamped alongside
# ``ready-for-agent`` so a PRD slice is distinguishable from ad-hoc ``ready-for-agent``
# work at the label layer (the lane router reads the ``Part of #<prd>`` link, not this
# label, but downstream tooling and humans can tell a slice apart from pickup). See
# :mod:`retinue.lane`.
PRD_SLICE_LABEL = "prd-slice"

# A PRD tracking issue — the retinue's slicing entry point (:mod:`retinue.webhook`
# gates on it; the ad-hoc drain excludes it).
PRD_LABEL = "prd"

# A loose backlog nit (the non-blocking heimdall findings :mod:`retinue.loopback`
# files), drained by the cron lane.
BACKLOG_LABEL = "backlog"

# The "test & merge" handoff is a heads-up for a human, not an escalation. It carries a
# findable label so the converged-but-unmerged PRs are routable, but it never merges.
TEST_AND_MERGE_LABEL = "test-and-merge"

# A backlog nit (or any preempting standalone) carries its severity as the
# ``priority:<severity>`` label :func:`priority_label` emits; :func:`parse_priority`
# reads it back.
PRIORITY_LABEL_PREFIX = "priority:"


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


def parse_priority(labels: list[str]) -> Severity | None:
    """Parse an issue's ``priority:<severity>`` label back into a :class:`Severity`.

    The inverse of :func:`priority_label`. Unknown ``priority:*`` values are skipped
    (later labels are still tried) rather than raising, so a stray label never breaks
    routing; callers that want a floor (e.g. the cron lane's LOW default) map ``None``
    themselves.

    Args:
        labels: The issue's label names.

    Returns:
        The first parsable ``priority:<severity>``, or ``None`` when the issue carries
        no (parsable) priority label.
    """
    for label in labels:
        if label.startswith(PRIORITY_LABEL_PREFIX):
            name = label[len(PRIORITY_LABEL_PREFIX) :].upper()
            try:
                return Severity[name]
            except KeyError:
                continue
    return None


def issue_branch(issue_number: int) -> str:
    """The ``issue-<N>`` branch an issue's build commits to (also the dedup branch name)."""
    return f"issue-{issue_number}"
