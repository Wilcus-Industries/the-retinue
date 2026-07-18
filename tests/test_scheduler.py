"""Tests for the pure two-queue ranking and reserved-slot selection."""

from __future__ import annotations

from retinue.repo_config import RepoConfig
from retinue.scheduler import Candidate, partition_queues, select_to_build

# Default vocabulary: critical/high/medium/low, priority range critical+high.
CONFIG = RepoConfig()


def _c(number: int, tier: str | None = None) -> Candidate:
    labels = ["ready-for-agent"]
    if tier is not None:
        labels.append(f"priority:{tier}")
    return Candidate(number=number, labels=labels)


def _nums(cands: list[Candidate]) -> list[int]:
    return [c.number for c in cands]


def test_partition_splits_and_ranks_by_tier() -> None:
    cands = [_c(5, "low"), _c(3, "critical"), _c(4, "high"), _c(2, "medium"), _c(9)]
    priority, main = partition_queues(CONFIG, cands)
    assert _nums(priority) == [3, 4]  # critical before high
    # main: medium before low before untiered; #9 untiered ranks last.
    assert _nums(main) == [2, 5, 9]


def test_rank_ties_break_on_number() -> None:
    priority, _ = partition_queues(CONFIG, [_c(7, "high"), _c(2, "high")])
    assert _nums(priority) == [2, 7]


def test_reserved_slot_holds_one_for_priority_when_main_full() -> None:
    """cap=2, only main work: main is held to 1 (N-1), one slot reserved."""
    built = select_to_build(CONFIG, [_c(1, "low"), _c(2, "low"), _c(3, "low")], cap=2)
    assert _nums(built) == [1]


def test_priority_and_main_share_the_cap() -> None:
    """cap=2, one priority + main: priority takes its slot, main gets the other."""
    built = select_to_build(CONFIG, [_c(1, "critical"), _c(2, "low"), _c(3, "low")], cap=2)
    assert _nums(built) == [1, 2]


def test_priority_can_saturate_the_cap() -> None:
    """cap=2 with three priority items: priority fills both slots, main gets none."""
    built = select_to_build(
        CONFIG, [_c(1, "critical"), _c(2, "high"), _c(3, "critical"), _c(9, "low")], cap=2
    )
    assert _nums(built) == [1, 3]  # critical #1,#3 before high #2; main #9 excluded


def test_cap_three_holds_main_to_two() -> None:
    built = select_to_build(
        CONFIG, [_c(1, "low"), _c(2, "low"), _c(3, "low"), _c(4, "low")], cap=3
    )
    assert _nums(built) == [1, 2]  # main capped at N-1 = 2, one slot reserved


def test_cap_one_is_strict_priority_first() -> None:
    """At cap 1 the single slot goes to the top priority item..."""
    built = select_to_build(CONFIG, [_c(5, "low"), _c(3, "critical")], cap=1)
    assert _nums(built) == [3]
    # ...or the top main item when there is no priority work (slot not stranded).
    built_main = select_to_build(CONFIG, [_c(5, "low"), _c(6, "medium")], cap=1)
    assert _nums(built_main) == [6]


def test_no_cap_builds_everything_priority_first() -> None:
    built = select_to_build(CONFIG, [_c(5, "low"), _c(3, "critical"), _c(4, "high")], cap=None)
    assert _nums(built) == [3, 4, 5]


def test_custom_vocabulary() -> None:
    """A repo's own tiers and priority range drive the split."""
    config = RepoConfig(
        severity_tiers=["urgent", "normal", "whenever"], priority_tiers=["urgent"]
    )
    cands = [
        Candidate(number=1, labels=["priority:normal"]),
        Candidate(number=2, labels=["priority:urgent"]),
        Candidate(number=3, labels=["priority:whenever"]),
    ]
    priority, main = partition_queues(config, cands)
    assert _nums(priority) == [2]
    assert _nums(main) == [1, 3]
