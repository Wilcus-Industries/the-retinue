"""Tests for SQLite-backed PRD-event deduplication.

The dedupe store records every PRD it has accepted, keyed by repo + issue, so a
redelivered or duplicate ``issues`` event for an already-processed PRD is ignored.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from retinue.dedupe import PrdDedupeStore, prd_dedupe_key
from retinue.queue import PrdJob


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """An on-disk SQLite path inside the test's tmp dir."""
    return tmp_path / "dedupe.sqlite3"


def test_key_is_repo_and_issue() -> None:
    """The dedupe key identifies a PRD by repo and issue, not by action."""
    opened = PrdJob(repo_full_name="owner/repo", issue_number=7, action="opened")
    labeled = PrdJob(repo_full_name="owner/repo", issue_number=7, action="labeled")
    assert prd_dedupe_key(opened) == prd_dedupe_key(labeled)
    assert prd_dedupe_key(opened) != prd_dedupe_key(
        PrdJob(repo_full_name="owner/repo", issue_number=8, action="opened")
    )


@pytest.mark.asyncio
async def test_first_claim_succeeds_duplicate_is_ignored(db_path: Path) -> None:
    """The first claim of a PRD wins; a second claim of the same PRD is rejected."""
    store = PrdDedupeStore(db_path)
    key = "owner/repo#7"
    assert await store.claim(key) is True
    assert await store.claim(key) is False


@pytest.mark.asyncio
async def test_distinct_prds_each_claim(db_path: Path) -> None:
    """Different PRDs do not collide with one another."""
    store = PrdDedupeStore(db_path)
    assert await store.claim("owner/repo#1") is True
    assert await store.claim("owner/repo#2") is True
    assert await store.claim("other/repo#1") is True


@pytest.mark.asyncio
async def test_dedupe_persists_across_store_instances(db_path: Path) -> None:
    """A claim survives a fresh store on the same DB file (worker restart)."""
    assert await PrdDedupeStore(db_path).claim("owner/repo#7") is True
    # A brand-new store object pointed at the same file must see the prior claim.
    assert await PrdDedupeStore(db_path).claim("owner/repo#7") is False
