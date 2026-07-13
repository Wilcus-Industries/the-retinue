"""Tests for the SQLite-backed implementer-retry counter.

The triage loop bounds implementer retries by a *persisted* count so a retry
budget survives a worker restart and cannot be reset by re-running the
orchestrator. :class:`ImplRetryStore` mirrors the durable-SQLite style of
:class:`retinue.dedupe.PrdDedupeStore`: one row per slice, keyed by repo + issue,
holding the number of attempts recorded so far.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import retinue.impl_retry as impl_retry_module
from retinue.impl_retry import ImplRetryStore, impl_retry_key
from retinue.orchestrator import Slice


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """An on-disk SQLite path inside the test's tmp dir."""
    return tmp_path / "impl-retry.sqlite3"


def _slice(issue_number: int = 7) -> Slice:
    return Slice(repo_full_name="owner/repo", issue_number=issue_number, prd_number=1)


def test_key_is_repo_and_issue() -> None:
    """The retry key identifies a slice by repo and issue number."""
    assert impl_retry_key(_slice(7)) == "owner/repo#7"
    assert impl_retry_key(_slice(7)) != impl_retry_key(_slice(8))


@pytest.mark.asyncio
async def test_unseen_slice_starts_at_zero(db_path: Path) -> None:
    """A slice never recorded has a count of zero."""
    store = ImplRetryStore(db_path)
    assert await store.count("owner/repo#7") == 0


@pytest.mark.asyncio
async def test_record_attempt_increments_and_returns_new_count(db_path: Path) -> None:
    """Each recorded attempt increments the persisted count and returns it."""
    store = ImplRetryStore(db_path)
    assert await store.record_attempt("owner/repo#7") == 1
    assert await store.record_attempt("owner/repo#7") == 2
    assert await store.count("owner/repo#7") == 2


@pytest.mark.asyncio
async def test_distinct_slices_track_independently(db_path: Path) -> None:
    """Different slices keep separate retry counters."""
    store = ImplRetryStore(db_path)
    await store.record_attempt("owner/repo#1")
    assert await store.count("owner/repo#1") == 1
    assert await store.count("owner/repo#2") == 0


@pytest.mark.asyncio
async def test_count_persists_across_store_instances(db_path: Path) -> None:
    """A recorded attempt survives a fresh store on the same DB file (restart)."""
    await ImplRetryStore(db_path).record_attempt("owner/repo#7")
    # A brand-new store object on the same file must see the prior attempt.
    assert await ImplRetryStore(db_path).count("owner/repo#7") == 1


@pytest.mark.asyncio
async def test_schema_init_is_cached_per_db_path(db_path: Path) -> None:
    """Schema/mkdir/WAL setup runs once per db-path, not on every call.

    A fresh store is built per build binding, so re-running CREATE TABLE and mkdir on
    every call is pure churn. The one-time init is memoised on the db-path.
    """
    impl_retry_module._initialized_paths.discard(db_path)
    store = ImplRetryStore(db_path)
    await store.count("owner/repo#7")
    assert db_path in impl_retry_module._initialized_paths


@pytest.mark.asyncio
async def test_concurrent_record_attempts_count_each(db_path: Path) -> None:
    """Concurrent increments through per-call connections each land (WAL, atomic upsert)."""
    store = ImplRetryStore(db_path)
    await asyncio.gather(*[store.record_attempt("owner/repo#7") for _ in range(15)])
    assert await store.count("owner/repo#7") == 15
