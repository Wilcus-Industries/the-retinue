"""Tests for the cross-process run-ledger (issue #89).

The ledger holds one row per ``(repo, issue)``: the worker upserts a coarse run-state at
the drain's choke points and the API reads the rows back. These tests pin the store's
contract — upsert on the key, distinct keys tracked independently, the url round-trip, an
empty unseen store, and cross-store persistence on one file (a second process reading the
first's writes).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from retinue.run_ledger import RunLedgerStore, RunState


@pytest.mark.asyncio
async def test_record_upserts_on_repo_and_issue(db_path: Path) -> None:
    """Re-recording the same ``(repo, issue)`` overwrites its state (one current row)."""
    store = RunLedgerStore(db_path)
    await store.record(repo_full_name="owner/repo", issue=7, state=RunState.QUEUED)
    await store.record(repo_full_name="owner/repo", issue=7, state=RunState.BUILDING)

    rows = await store.rows()
    assert len(rows) == 1
    assert rows[0].issue == 7
    assert rows[0].state == RunState.BUILDING.value


@pytest.mark.asyncio
async def test_distinct_keys_are_tracked_independently(db_path: Path) -> None:
    """Two different issues yield two rows, each with its own state."""
    store = RunLedgerStore(db_path)
    await store.record(repo_full_name="owner/repo", issue=7, state=RunState.BUILDING)
    await store.record(repo_full_name="owner/repo", issue=8, state=RunState.QUEUED)

    states = {r.issue: r.state for r in await store.rows()}
    assert states == {7: RunState.BUILDING.value, 8: RunState.QUEUED.value}


@pytest.mark.asyncio
async def test_url_round_trips_and_defaults_to_none(db_path: Path) -> None:
    """A recorded url is read back verbatim; the default record leaves url None."""
    store = RunLedgerStore(db_path)
    await store.record(
        repo_full_name="owner/repo",
        issue=7,
        state=RunState.PR_OPENED,
        url="https://github.com/owner/repo/pull/1",
    )
    await store.record(repo_full_name="owner/repo", issue=8, state=RunState.QUEUED)

    by_issue = {r.issue: r for r in await store.rows()}
    assert by_issue[7].url == "https://github.com/owner/repo/pull/1"
    assert by_issue[8].url is None


@pytest.mark.asyncio
async def test_unseen_store_has_no_rows(db_path: Path) -> None:
    """A store over a fresh path reads back an empty list."""
    assert await RunLedgerStore(db_path).rows() == []


@pytest.mark.asyncio
async def test_recorded_rows_survive_a_fresh_store(db_path: Path) -> None:
    """A row written via one store is read back by a fresh store on the same file.

    This is the cross-process guarantee: the worker writes, the web reader (a second
    ``RunLedgerStore`` over the same path) sees it.
    """
    await RunLedgerStore(db_path).record(
        repo_full_name="owner/repo", issue=7, state=RunState.BUILDING
    )

    rows = await RunLedgerStore(db_path).rows()
    assert [(r.repo, r.issue, r.state) for r in rows] == [
        ("owner/repo", 7, RunState.BUILDING.value)
    ]
