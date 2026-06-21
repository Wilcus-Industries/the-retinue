"""Tests for the PRD opt-in gate in the worker.

The gate decides, per dequeued PRD: is the repo opted in (a valid
``.github/retinue.yml`` present), and is this PRD new (not already deduped)? It
fetches the config text through an injected async callable so the GitHub fetch is
mocked out; the dedupe store is a real SQLite file in a tmp dir.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from retinue.dedupe import PrdDedupeStore
from retinue.queue import PrdJob
from retinue.worker import GateOutcome, gate_prd

VALID_CONFIG = "staging_branch: release\nretry_cap: 2\n"


@pytest.fixture()
def store(tmp_path: Path) -> PrdDedupeStore:
    return PrdDedupeStore(tmp_path / "dedupe.sqlite3")


@pytest.fixture()
def job() -> PrdJob:
    return PrdJob(repo_full_name="owner/repo", issue_number=7, action="opened")


async def _fetch_valid(repo_full_name: str) -> str | None:
    return VALID_CONFIG


async def _fetch_missing(repo_full_name: str) -> str | None:
    return None


async def _fetch_malformed(repo_full_name: str) -> str | None:
    return "staging_branch: [unclosed\n"


@pytest.mark.asyncio
async def test_valid_config_is_accepted_and_parsed(
    job: PrdJob, store: PrdDedupeStore
) -> None:
    """A repo with a valid config is accepted and the parsed config is returned."""
    result = await gate_prd(job, fetch_config=_fetch_valid, dedupe=store)
    assert result.outcome is GateOutcome.ACCEPTED
    assert result.config is not None
    assert result.config.staging_branch == "release"
    assert result.config.retry_cap == 2


@pytest.mark.asyncio
async def test_missing_config_is_skipped(
    job: PrdJob, store: PrdDedupeStore, caplog: pytest.LogCaptureFixture
) -> None:
    """A repo with no config file is an observable skip with no parsed config."""
    with caplog.at_level(logging.INFO, logger="retinue.worker"):
        result = await gate_prd(job, fetch_config=_fetch_missing, dedupe=store)
    assert result.outcome is GateOutcome.NOT_OPTED_IN
    assert result.config is None
    assert "not opted in" in caplog.text.lower()


@pytest.mark.asyncio
async def test_malformed_config_is_skipped_not_crashing(
    job: PrdJob, store: PrdDedupeStore, caplog: pytest.LogCaptureFixture
) -> None:
    """A malformed config is skipped and logged, never raising out of the gate."""
    with caplog.at_level(logging.WARNING, logger="retinue.worker"):
        result = await gate_prd(job, fetch_config=_fetch_malformed, dedupe=store)
    assert result.outcome is GateOutcome.MALFORMED_CONFIG
    assert result.config is None
    assert "malformed" in caplog.text.lower()


@pytest.mark.asyncio
async def test_duplicate_prd_is_skipped(
    job: PrdJob, store: PrdDedupeStore, caplog: pytest.LogCaptureFixture
) -> None:
    """A second gate of the same PRD is a dedupe skip; the first is accepted."""
    first = await gate_prd(job, fetch_config=_fetch_valid, dedupe=store)
    assert first.outcome is GateOutcome.ACCEPTED

    redelivery = PrdJob(repo_full_name="owner/repo", issue_number=7, action="labeled")
    with caplog.at_level(logging.INFO, logger="retinue.worker"):
        second = await gate_prd(redelivery, fetch_config=_fetch_valid, dedupe=store)
    assert second.outcome is GateOutcome.DUPLICATE
    assert second.config is None
    assert "duplicate" in caplog.text.lower()


@pytest.mark.asyncio
async def test_malformed_config_does_not_consume_dedupe_claim(
    job: PrdJob, store: PrdDedupeStore
) -> None:
    """A malformed config must not burn the dedupe slot, so a later fix can run.

    If the repo fixes its config and the PRD is redelivered, the gate should be
    able to accept it — the earlier malformed attempt must not have claimed the key.
    """
    first = await gate_prd(job, fetch_config=_fetch_malformed, dedupe=store)
    assert first.outcome is GateOutcome.MALFORMED_CONFIG

    retry = await gate_prd(job, fetch_config=_fetch_valid, dedupe=store)
    assert retry.outcome is GateOutcome.ACCEPTED
