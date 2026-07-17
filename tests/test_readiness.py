"""Tests for blocked-by readiness: the union source and the closed-only rule."""

from __future__ import annotations

import pytest

from retinue.readiness import (
    BlockableIssue,
    ReadinessGh,
    parse_body_blockers,
    resolve_ready,
)


class FakeReadinessGh:
    """Scripted :class:`ReadinessGh`: native blockers per issue, closed-set of numbers."""

    def __init__(
        self, *, native: dict[int, list[int]] | None = None, closed: set[int] | None = None
    ) -> None:
        self._native = native or {}
        self._closed = closed or set()
        self.native_calls: list[int] = []
        self.closed_calls: list[int] = []

    async def native_blockers(
        self, *, repo_full_name: str, issue_number: int
    ) -> list[int]:
        self.native_calls.append(issue_number)
        return list(self._native.get(issue_number, []))

    async def is_closed(self, *, repo_full_name: str, issue_number: int) -> bool:
        self.closed_calls.append(issue_number)
        return issue_number in self._closed


def test_parse_body_blockers_reads_heading_and_lines() -> None:
    """Refs on the heading line and the lines below it are all collected, deduped."""
    body = "Do the thing.\n\n## Blocked by\n#12\n#13\n\n## Notes\n#99 (unrelated)\n"
    assert parse_body_blockers(body) == [12, 13]
    assert parse_body_blockers("## Blocked by #7 and #8\n") == [7, 8]
    assert parse_body_blockers("no block here #5") == []
    assert parse_body_blockers("## Blocked by\n#4\n#4\n") == [4]


@pytest.mark.asyncio
async def test_no_blockers_is_ready() -> None:
    gh = FakeReadinessGh()
    ready = await resolve_ready(
        [BlockableIssue(number=1, body="nothing blocks me")],
        repo_full_name="o/r",
        gh=gh,
    )
    assert ready == {1}


@pytest.mark.asyncio
async def test_open_body_blocker_blocks() -> None:
    gh = FakeReadinessGh(closed=set())  # blocker #2 is open
    ready = await resolve_ready(
        [BlockableIssue(number=1, body="## Blocked by\n#2\n")],
        repo_full_name="o/r",
        gh=gh,
    )
    assert ready == set()


@pytest.mark.asyncio
async def test_closed_body_blocker_is_satisfied() -> None:
    gh = FakeReadinessGh(closed={2})
    ready = await resolve_ready(
        [BlockableIssue(number=1, body="## Blocked by\n#2\n")],
        repo_full_name="o/r",
        gh=gh,
    )
    assert ready == {1}


@pytest.mark.asyncio
async def test_native_and_body_blockers_are_unioned() -> None:
    """A native relation blocks even with no body ref; both must be closed to be ready."""
    gh = FakeReadinessGh(native={1: [3]}, closed={2})  # body #2 closed, native #3 open
    ready = await resolve_ready(
        [BlockableIssue(number=1, body="## Blocked by\n#2\n")],
        repo_full_name="o/r",
        gh=gh,
    )
    assert ready == set()

    gh2 = FakeReadinessGh(native={1: [3]}, closed={2, 3})  # both closed
    ready2 = await resolve_ready(
        [BlockableIssue(number=1, body="## Blocked by\n#2\n")],
        repo_full_name="o/r",
        gh=gh2,
    )
    assert ready2 == {1}


@pytest.mark.asyncio
async def test_self_reference_is_ignored() -> None:
    gh = FakeReadinessGh(native={1: [1]})
    ready = await resolve_ready(
        [BlockableIssue(number=1, body="## Blocked by\n#1\n")],
        repo_full_name="o/r",
        gh=gh,
    )
    assert ready == {1}


@pytest.mark.asyncio
async def test_shared_blocker_state_fetched_once() -> None:
    """Two candidates blocked by the same issue only query its state once."""
    gh = FakeReadinessGh(closed={9})
    ready = await resolve_ready(
        [
            BlockableIssue(number=1, body="## Blocked by\n#9\n"),
            BlockableIssue(number=2, body="## Blocked by\n#9\n"),
        ],
        repo_full_name="o/r",
        gh=gh,
    )
    assert ready == {1, 2}
    assert gh.closed_calls == [9]


def test_fake_satisfies_protocol() -> None:
    assert isinstance(FakeReadinessGh(), ReadinessGh)
