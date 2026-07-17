"""The severity-pivot scheduler's queue model: two queues, tier ranking, reserved slot.

This is the pure heart of the unified scheduler (PRD #80). A drain pass has already
reduced the repo's trigger-labeled, unblocked, open issues to a set of candidates; this
module decides their **order** and **which build this pass**:

* **Two queues by tier.** Each candidate's tier is its ``priority:<tier>`` label matched
  against ``config.severity_tiers`` (:meth:`retinue.repo_config.RepoConfig.tier_of`).
  Candidates whose tier is in the configured top range (``config.priority_tiers``) form
  the **priority queue**; the rest form the **main queue**. Both are ranked by tier
  (earlier tier = more urgent), untiered issues last, ties broken by ascending number.
* **Reserved priority slot.** With a parallel-build cap ``N ≥ 2``, the main queue may
  occupy at most ``N-1`` slots so one is always free for priority-queue work — a critical
  arrival is never stuck behind a full queue of nits. The priority queue always drains
  first. At ``N == 1`` there is no slot to reserve, so strict priority-first ordering
  governs (priority queue, then main). An unset cap (``None``) builds every candidate.

The scheduler never cancels in-flight builds; it only decides what to *start* this pass.
Everything here is pure — no ``gh``, no I/O — so the ranking and reserved-slot allocation
are exhaustively unit-tested, and the drain (:mod:`retinue.adhoc_drain`) calls it.
"""

from __future__ import annotations

from dataclasses import dataclass

from retinue.repo_config import RepoConfig


@dataclass(frozen=True)
class Candidate:
    """One schedulable issue: its number and the labels its tier is read from.

    Attributes:
        number: The issue number; the build commits to the derived ``issue-<N>`` branch.
        labels: The issue's label names, scanned for the ``priority:<tier>`` queue tier.
    """

    number: int
    labels: list[str]


def _rank_key(config: RepoConfig, candidate: Candidate) -> tuple[int, int]:
    """Sort key ``(tier_rank, number)``: more urgent tier first, then lower number.

    ``tier_rank`` places an earlier ``severity_tiers`` entry first and an untiered issue
    last (:meth:`RepoConfig.tier_rank`); the number tiebreak makes the order deterministic.
    """
    tier = config.tier_of(candidate.labels)
    return (config.tier_rank(tier), candidate.number)


def partition_queues(
    config: RepoConfig, candidates: list[Candidate]
) -> tuple[list[Candidate], list[Candidate]]:
    """Split candidates into the ranked ``(priority, main)`` queues.

    Priority-queue membership is ``config.is_priority_tier`` on the candidate's tier; both
    queues come back tier-ranked (ties by ascending number).
    """
    ranked = sorted(candidates, key=lambda c: _rank_key(config, c))
    priority = [c for c in ranked if config.is_priority_tier(config.tier_of(c.labels))]
    main = [c for c in ranked if not config.is_priority_tier(config.tier_of(c.labels))]
    return priority, main


def select_to_build(
    config: RepoConfig, candidates: list[Candidate], *, cap: int | None
) -> list[Candidate]:
    """Choose the candidates to start this pass, honoring the reserved priority slot.

    The priority queue drains first; the main queue is held to at most ``cap-1`` slots so
    one is always free for priority work (the reserved slot). At ``cap == 1`` strict
    priority-first ordering governs — the single slot goes to the top priority-queue item,
    or the top main item when the priority queue is empty. An unset ``cap`` (``None``)
    returns every candidate (priority first).

    Args:
        config: The repo config providing the tier vocabulary and priority range.
        candidates: This pass's ready candidates (any order).
        cap: The parallel-build cap (``config.max_parallel``); ``None`` means no cap.

    Returns:
        The candidates to build this pass, priority-first then main, in rank order.
    """
    priority, main = partition_queues(config, candidates)
    if cap is None:
        return priority + main
    if cap <= 1:
        # No slot to reserve: strict priority-first, take as many as the cap allows.
        return (priority + main)[: max(cap, 0)]
    p_take = priority[:cap]
    remaining = cap - len(p_take)
    # The main queue never occupies more than cap-1 slots (the reserved priority slot),
    # and never more than the slots priority left free.
    main_cap = min(cap - 1, remaining)
    return p_take + main[:main_cap]
