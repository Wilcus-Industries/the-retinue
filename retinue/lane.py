"""Lane classifier: route an issue to a work lane from its labels (issues #15, #26).

``ready-for-agent`` is the single "build me" trigger; the ``Part of #<prd>`` body link is
what tells provenance (a PRD slice) apart from pickup (ad-hoc work). The retinue has three
work lanes:

* the **orchestrator** lane — PRD slices, filed by the slicer with ``ready-for-agent``
  *and* a ``Part of #<prd>`` body link (:mod:`retinue.slicer`), built by
  :func:`retinue.orchestrator.build_prd`,
* the **ad-hoc** lane — a ``ready-for-agent`` issue with **no** ``Part of #<prd>`` link:
  standalone work picked up directly, not a slice of any PRD,
* the **cron** lane — loose ``backlog`` issues (the non-blocking heimdall nits filed by
  :mod:`retinue.loopback`), drained one at a time by :mod:`retinue.cron`.

PRD work runs first by default, but a **standalone** ``priority:critical`` /
``priority:high`` issue *preempts* that ordering onto the orchestrator lane — it must not
wait its turn in the slow backlog drain. The classifier is pure: it reads only the
issue's labels and body (the ``Part of #<prd>`` link), with no ``gh`` and no network, so
it is exhaustively unit-tested. The priority vocabulary is the same
``priority:<severity>`` labels :func:`retinue.vocab.priority_label` emits.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field

from retinue.vocab import BACKLOG_LABEL, READY_LABEL, Severity, parse_priority

# The slicer renders the parent link as ``Part of #<prd>`` in the issue body. Matched
# loosely (any line, surrounding text tolerated) so a real body with extra prose routes.
_PART_OF_RE = re.compile(r"Part of #(\d+)")

# At or above this severity, a standalone (non-slice) issue preempts the PRD-first order.
_PREEMPT_THRESHOLD = Severity.HIGH


def preempts_prd_first(priority: Severity | None) -> bool:
    """Whether a ``priority:<severity>`` jumps the PRD-first order (``critical``/``high``).

    The single source of truth for the preemption rule, shared by :func:`classify` (which
    routes a standalone preempting issue onto the orchestrator lane) and the ad-hoc drain
    (which builds a preempting issue ahead of PRD-first ordering). A ``None`` priority —
    no or unknown ``priority:*`` label — never preempts.

    Args:
        priority: The issue's parsed severity, or ``None`` when it carries no priority.

    Returns:
        True when ``priority`` is at or above the preemption threshold (``high``).
    """
    return priority is not None and priority >= _PREEMPT_THRESHOLD


class Lane(enum.Enum):
    """Which work lane an issue is routed to.

    ``ORCHESTRATOR`` is the PRD-build lane; ``ADHOC`` is the standalone ``ready-for-agent``
    (no Part-of link) lane; ``CRON`` is the backlog drainer lane; ``NONE`` means the issue
    carries no routing signal the retinue acts on.
    """

    ORCHESTRATOR = "orchestrator"
    ADHOC = "adhoc"
    CRON = "cron"
    NONE = "none"


@dataclass(frozen=True)
class IssueFacts:
    """The routing-relevant facts of one GitHub issue.

    Attributes:
        labels: The issue's label names (e.g. ``ready-for-agent``, ``backlog``,
            ``priority:critical``).
        body: The issue body, scanned for the slicer's ``Part of #<prd>`` link.
    """

    labels: list[str] = field(default_factory=list)
    body: str = ""

    def has_label(self, label: str) -> bool:
        """Whether the issue carries ``label``."""
        return label in self.labels

    def prd_link(self) -> int | None:
        """The PRD number from a ``Part of #<prd>`` body link, or ``None`` when absent."""
        match = _PART_OF_RE.search(self.body)
        return int(match.group(1)) if match else None

    def priority(self) -> Severity | None:
        """The issue's ``priority:<severity>`` as a :class:`Severity`, or ``None``.

        Unknown ``priority:*`` values are ignored (treated as no priority) rather than
        raising, so a stray label never breaks routing.
        """
        return parse_priority(self.labels)


@dataclass(frozen=True)
class LaneDecision:
    """The classifier's verdict for one issue.

    Attributes:
        lane: The routed lane.
        preempts: True when a standalone ``priority:critical``/``high`` issue jumped the
            PRD-first ordering onto the orchestrator lane.
        prd_number: The parent PRD number for an orchestrator-lane slice, else ``None``.
    """

    lane: Lane
    preempts: bool = False
    prd_number: int | None = None


def classify(facts: IssueFacts) -> LaneDecision:
    """Route an issue to a lane from its labels and ``Part of #<prd>`` link.

    Order of precedence:

    1. **preempt** — a standalone ``priority:critical``/``high`` issue (at or above the
       :data:`_PREEMPT_THRESHOLD`) jumps onto the orchestrator lane regardless of any
       ``backlog`` label, so a critical never waits its turn in the slow cron drain.
    2. **PRD slice** — ``ready-for-agent`` plus a ``Part of #<prd>`` body link routes to
       the orchestrator lane.
    3. **ad-hoc** — ``ready-for-agent`` with no ``Part of #<prd>`` link routes to the
       ad-hoc lane: standalone work to build directly, not a slice of any PRD.
    4. **backlog** — a loose ``backlog`` issue routes to the cron lane.
    5. otherwise ``Lane.NONE``.

    Args:
        facts: The routing-relevant facts of the issue.

    Returns:
        A :class:`LaneDecision` carrying the lane, whether it preempted, and the parent
        PRD number for an orchestrator-lane slice.
    """
    if preempts_prd_first(facts.priority()):
        return LaneDecision(
            lane=Lane.ORCHESTRATOR, preempts=True, prd_number=facts.prd_link()
        )

    if facts.has_label(READY_LABEL):
        prd_number = facts.prd_link()
        if prd_number is not None:
            return LaneDecision(lane=Lane.ORCHESTRATOR, prd_number=prd_number)
        return LaneDecision(lane=Lane.ADHOC)

    if facts.has_label(BACKLOG_LABEL):
        return LaneDecision(lane=Lane.CRON)

    return LaneDecision(lane=Lane.NONE)
