"""Tests for SQLite-backed PRD-event deduplication.

The dedupe store records every PRD it has accepted, keyed by repo + issue, so a
redelivered or duplicate ``issues`` event for an already-processed PRD is ignored.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from retinue.dedupe import PrdDedupeStore, prd_dedupe_key
from retinue.queue import PrdJob


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """An on-disk SQLite path inside the test's tmp dir."""
    return tmp_path / "dedupe.sqlite3"


@pytest_asyncio.fixture()
async def store(db_path: Path) -> AsyncIterator[PrdDedupeStore]:
    """A dedupe store closed at teardown so no worker thread outlives the test."""
    dedupe = PrdDedupeStore(db_path)
    yield dedupe
    await dedupe.close()


def test_key_is_repo_and_issue() -> None:
    """The dedupe key identifies a PRD by repo and issue, not by action."""
    opened = PrdJob(repo_full_name="owner/repo", issue_number=7, action="opened")
    labeled = PrdJob(repo_full_name="owner/repo", issue_number=7, action="labeled")
    assert prd_dedupe_key(opened) == prd_dedupe_key(labeled)
    assert prd_dedupe_key(opened) != prd_dedupe_key(
        PrdJob(repo_full_name="owner/repo", issue_number=8, action="opened")
    )


@pytest.mark.asyncio
async def test_first_claim_succeeds_duplicate_is_ignored(store: PrdDedupeStore) -> None:
    """The first claim of a PRD wins; a second claim of the same PRD is rejected."""
    key = "owner/repo#7"
    assert await store.claim(key) is True
    assert await store.claim(key) is False


@pytest.mark.asyncio
async def test_distinct_prds_each_claim(store: PrdDedupeStore) -> None:
    """Different PRDs do not collide with one another."""
    assert await store.claim("owner/repo#1") is True
    assert await store.claim("owner/repo#2") is True
    assert await store.claim("other/repo#1") is True


@pytest.mark.asyncio
async def test_dedupe_persists_across_store_instances(db_path: Path) -> None:
    """A claim survives a fresh store on the same DB file (worker restart)."""
    first = PrdDedupeStore(db_path)
    assert await first.claim("owner/repo#7") is True
    await first.close()
    # A brand-new store object pointed at the same file must see the prior claim.
    second = PrdDedupeStore(db_path)
    assert await second.claim("owner/repo#7") is False
    await second.close()


@pytest.mark.asyncio
async def test_release_lets_a_burned_prd_be_reclaimed(store: PrdDedupeStore) -> None:
    """Releasing a claim deletes the row so a crashed-mid-flight PRD can retry.

    A worker that claims a PRD then dies before its run state persists must not lose
    the PRD forever; the failure path releases the claim so a redelivery re-claims it.
    """
    assert await store.claim("owner/repo#7") is True
    await store.release("owner/repo#7")
    assert await store.claim("owner/repo#7") is True


@pytest.mark.asyncio
async def test_release_of_unclaimed_key_is_a_noop(store: PrdDedupeStore) -> None:
    """Releasing a key that was never claimed neither raises nor claims it."""
    await store.release("never/claimed#1")  # must not raise
    assert await store.claim("never/claimed#1") is True


@pytest.mark.asyncio
async def test_store_reuses_a_single_connection(store: PrdDedupeStore) -> None:
    """The store opens one long-lived connection, not a fresh one per call."""
    await store.claim("owner/repo#1")
    connection = store._db
    assert connection is not None
    await store.claim("owner/repo#2")
    assert store._db is connection


@pytest.mark.asyncio
async def test_connection_thread_is_a_daemon(store: PrdDedupeStore) -> None:
    """The connection's worker thread must not block interpreter exit.

    aiosqlite runs each connection on a thread; a store leaked without close() (worker
    crash path, test teardown) would otherwise hang process shutdown forever on the
    non-daemon thread join.
    """
    await store.claim("owner/repo#1")
    assert store._db is not None
    assert store._db._thread.daemon is True


@pytest.mark.asyncio
async def test_concurrent_claims_of_one_key_have_a_single_winner(
    store: PrdDedupeStore,
) -> None:
    """The per-store lock keeps concurrent claims of one key to exactly one winner."""
    results = await asyncio.gather(*[store.claim("owner/repo#7") for _ in range(25)])
    assert sum(results) == 1
