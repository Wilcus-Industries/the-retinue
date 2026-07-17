"""Tests for the durable run-state store and the PR-state gh query.

The scheduler drain is stateless per pass, but the ad-hoc PR flow persists the PR<->issue
mapping in the SQLite :class:`RunStateStore` so a later merge webhook resolves the PR back
to the issue it closes. The gh seam (:class:`GhCliReconcile`) reads a PR's lifecycle state.
Every gh touch is faked and the run-state lives in a temp SQLite file — no real ``gh`` or
network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from retinue.reconcile import (
    GhCliReconcile,
    PrState,
    RunStateStore,
    gh_env,
)

# --- run-state store: persistence across restart ---------------------------------


@pytest.mark.asyncio
async def test_recorded_slices_survive_a_fresh_store(db_path: Path) -> None:
    """The owned issue set survives a fresh store on the same file (restart)."""
    await RunStateStore(db_path).record_slices(
        repo_full_name="owner/repo", prd_number=1, issue_numbers=[2, 3, 4]
    )
    # A brand-new store object on the same file must see the recorded issues.
    reloaded = await RunStateStore(db_path).slices_of(
        repo_full_name="owner/repo", prd_number=1
    )
    assert reloaded == [2, 3, 4]


@pytest.mark.asyncio
async def test_unseen_prd_has_no_recorded_slices(db_path: Path) -> None:
    """A key never recorded reports an empty set, not an error."""
    store = RunStateStore(db_path)
    assert await store.slices_of(repo_full_name="owner/repo", prd_number=9) == []


@pytest.mark.asyncio
async def test_recording_slices_is_idempotent(db_path: Path) -> None:
    """Re-recording the same key's issues does not duplicate or crash (re-run)."""
    store = RunStateStore(db_path)
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=1, issue_numbers=[2, 3]
    )
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=1, issue_numbers=[2, 3]
    )
    assert await store.slices_of(repo_full_name="owner/repo", prd_number=1) == [2, 3]


@pytest.mark.asyncio
async def test_pr_mapping_survives_a_fresh_store(db_path: Path) -> None:
    """The PR<->issue mapping survives a fresh store on the same file (restart)."""
    await RunStateStore(db_path).record_pr(
        repo_full_name="owner/repo", prd_number=1, pr_number=101
    )
    assert (
        await RunStateStore(db_path).pr_of(repo_full_name="owner/repo", prd_number=1)
        == 101
    )


@pytest.mark.asyncio
async def test_unseen_prd_has_no_pr(db_path: Path) -> None:
    """A key with no recorded PR reports None, not an error."""
    store = RunStateStore(db_path)
    assert await store.pr_of(repo_full_name="owner/repo", prd_number=9) is None


@pytest.mark.asyncio
async def test_round_for_pr_reverse_lookup(db_path: Path) -> None:
    """round_for_pr maps a PR number back to its issue and owned set."""
    store = RunStateStore(db_path)
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=7, issue_numbers=[100, 101]
    )
    await store.record_pr(repo_full_name="owner/repo", prd_number=7, pr_number=99)

    assert await store.round_for_pr(repo_full_name="owner/repo", pr_number=99) == (
        7,
        [100, 101],
    )


@pytest.mark.asyncio
async def test_round_for_pr_unknown_pr_is_none(db_path: Path) -> None:
    """A PR number the store never recorded reverse-resolves to None."""
    store = RunStateStore(db_path)
    await store.record_pr(repo_full_name="owner/repo", prd_number=7, pr_number=99)
    assert await store.round_for_pr(repo_full_name="owner/repo", pr_number=5) is None


@pytest.mark.asyncio
async def test_distinct_prds_track_independently(db_path: Path) -> None:
    """Different keys keep separate issue sets and PR mappings."""
    store = RunStateStore(db_path)
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=1, issue_numbers=[2]
    )
    await store.record_pr(repo_full_name="owner/repo", prd_number=1, pr_number=101)
    assert await store.slices_of(repo_full_name="owner/repo", prd_number=2) == []
    assert await store.pr_of(repo_full_name="owner/repo", prd_number=2) is None


# --- run-state store: cleanup ----------------------------------------------------


@pytest.mark.asyncio
async def test_delete_round_removes_the_row(db_path: Path) -> None:
    """delete_round clears the issue set and the PR mapping for a key."""
    store = RunStateStore(db_path)
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=1, issue_numbers=[2, 3]
    )
    await store.record_pr(repo_full_name="owner/repo", prd_number=1, pr_number=101)

    await store.delete_round(repo_full_name="owner/repo", prd_number=1)

    assert await store.slices_of(repo_full_name="owner/repo", prd_number=1) == []
    assert await store.round_for_pr(repo_full_name="owner/repo", pr_number=101) is None


@pytest.mark.asyncio
async def test_delete_round_leaves_other_rounds_alone(db_path: Path) -> None:
    """Deleting one key does not disturb a sibling key's state."""
    store = RunStateStore(db_path)
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=1, issue_numbers=[2, 3]
    )
    await store.record_slices(
        repo_full_name="owner/repo", prd_number=9, issue_numbers=[40]
    )
    await store.delete_round(repo_full_name="owner/repo", prd_number=1)

    assert await store.slices_of(repo_full_name="owner/repo", prd_number=9) == [40]


@pytest.mark.asyncio
async def test_delete_round_of_unknown_prd_is_a_noop(db_path: Path) -> None:
    """Deleting a key never recorded is a safe no-op."""
    store = RunStateStore(db_path)
    await store.delete_round(repo_full_name="owner/repo", prd_number=9)
    assert await store.slices_of(repo_full_name="owner/repo", prd_number=9) == []


# --- real gh-cli adapter: pure env build + PR-state query ------------------------


class RecordingRunner:
    """A fake :class:`GhRunner`: returns a canned stdout and records the argv it ran."""

    def __init__(self, stdout: str = "") -> None:
        self._stdout = stdout
        self.calls: list[list[str]] = []

    async def __call__(self, argv: list[str]) -> str:
        self.calls.append(argv)
        return self._stdout


def test_gh_env_carries_the_token_and_disables_prompts() -> None:
    """The gh child env authenticates via GH_TOKEN and never prompts interactively."""
    env = gh_env("ghs_secret", {"PATH": "/usr/bin", "GH_TOKEN": "stale"})

    assert env["GH_TOKEN"] == "ghs_secret"  # the auth credential gh sends as Bearer
    assert env["GH_PROMPT_DISABLED"] == "1"
    assert env["PATH"] == "/usr/bin"  # base env preserved


def test_gh_env_does_not_mutate_the_base_env() -> None:
    """Building the gh env copies rather than mutating the caller's environment."""
    base = {"PATH": "/usr/bin"}
    gh_env("tok", base)
    assert "GH_TOKEN" not in base


@pytest.mark.asyncio
async def test_pr_state_reads_the_pr_lifecycle_state() -> None:
    """pr_state reads gh pr view --json state and maps it to the PrState enum."""
    runner = RecordingRunner('{"state": "MERGED"}')
    gh = GhCliReconcile(runner)

    state = await gh.pr_state(repo_full_name="owner/repo", pr_number=101)

    assert state is PrState.MERGED
    assert runner.calls == [
        ["pr", "view", "101", "--repo", "owner/repo", "--json", "state"]
    ]
