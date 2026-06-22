"""Tests for the worker's pipeline wiring: process_prd + the review/reap tasks.

The worker tasks read their collaborators from the Arq ``ctx`` (populated by
``on_startup``). These tests inject fakes into ``ctx`` — a recording pipeline, a config
fetcher, a PRD-body fetcher — so the dispatch and parsing are exercised with no real gh,
Anthropic, Docker, or network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from retinue.dedupe import PrdDedupeStore
from retinue.handoff import MergedPullRequest, ReapOutcome, ReapResult
from retinue.loopback import (
    HeimdallReview,
    ReviewState,
    Severity,
    VerdictOutcome,
    VerdictResult,
)
from retinue.pipeline import PrdJobResult
from retinue.repo_config import RepoConfig
from retinue.worker import (
    parse_heimdall_review,
    process_prd,
    process_review_job,
    reap_pr_job,
)

_CONFIG_YAML = "staging_branch: staging\nretry_cap: 2\n"


@dataclass
class _RecordingPipeline:
    """A fake Pipeline recording every call the worker tasks make against it."""

    prd_calls: list[dict[str, Any]] = field(default_factory=list)
    reviews: list[HeimdallReview] = field(default_factory=list)
    reaps: list[MergedPullRequest] = field(default_factory=list)
    pr_round: tuple[int, list[int]] | None = (7, [100, 101])

    async def process_prd_job(
        self, *, repo_full_name: str, prd_number: int, prd_body: str
    ) -> PrdJobResult:
        self.prd_calls.append(
            {"repo": repo_full_name, "prd": prd_number, "body": prd_body}
        )
        return PrdJobResult(sliced=True, pr_opened=True)

    async def process_review(self, review: HeimdallReview) -> VerdictResult:
        self.reviews.append(review)
        return VerdictResult(outcome=VerdictOutcome.CONVERGED)

    async def reap_pr(self, merged: MergedPullRequest) -> ReapResult:
        self.reaps.append(merged)
        return ReapResult(outcome=ReapOutcome.REAPED, prd_closed=True)

    async def round_for_pr(
        self, *, repo_full_name: str, pr_number: int
    ) -> tuple[int, list[int]] | None:
        return self.pr_round


def _ctx(tmp_path: Path, pipeline: _RecordingPipeline, *, body: str = "") -> dict[str, Any]:
    async def fetch_config(repo_full_name: str) -> str | None:
        return _CONFIG_YAML

    async def fetch_body(repo_full_name: str, issue_number: int) -> str:
        return body

    async def factory(repo_full_name: str, config: RepoConfig) -> _RecordingPipeline:
        return pipeline

    return {
        "fetch_config": fetch_config,
        "fetch_prd_body": fetch_body,
        "pipeline_factory": factory,
        "dedupe": PrdDedupeStore(tmp_path / "dedupe.sqlite3"),
    }


# --- process_prd ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_prd_drives_pipeline_with_fetched_body(tmp_path: Path) -> None:
    """An accepted PRD reaches the pipeline with its fetched issue body."""
    pipeline = _RecordingPipeline()
    body = "Implement the thing with enough body text to slice it responsibly here."
    ctx = _ctx(tmp_path, pipeline, body=body)

    await process_prd(ctx, repo_full_name="owner/repo", issue_number=7, action="opened")

    assert pipeline.prd_calls == [{"repo": "owner/repo", "prd": 7, "body": body}]


@pytest.mark.asyncio
async def test_process_prd_without_pipeline_is_a_noop(tmp_path: Path) -> None:
    """With no pipeline_factory wired the accepted PRD stops after the gate."""
    pipeline = _RecordingPipeline()
    ctx = _ctx(tmp_path, pipeline)
    del ctx["pipeline_factory"]

    await process_prd(ctx, repo_full_name="owner/repo", issue_number=7, action="opened")

    assert pipeline.prd_calls == []


# --- review loopback ------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_review_job_parses_and_drives_loopback(tmp_path: Path) -> None:
    """A review job resolves the PRD, parses findings, and drives the loopback."""
    pipeline = _RecordingPipeline(pr_round=(7, [100]))
    ctx = _ctx(tmp_path, pipeline)

    await process_review_job(
        ctx,
        repo_full_name="owner/repo",
        pr_number=99,
        review_state="changes_requested",
        review_body="high: a blocking problem\nlow: a nit",
    )

    assert len(pipeline.reviews) == 1
    review = pipeline.reviews[0]
    assert review.pr_number == 99
    assert review.prd_number == 7
    assert review.integration_branch == "retinue/prd-7"
    assert review.state is ReviewState.REQUEST_CHANGES
    assert [f.severity for f in review.findings] == [Severity.HIGH, Severity.LOW]


@pytest.mark.asyncio
async def test_process_review_job_skips_unknown_pr(tmp_path: Path) -> None:
    """A review on a PR not in run-state is skipped (not the retinue's PR)."""
    pipeline = _RecordingPipeline(pr_round=None)
    ctx = _ctx(tmp_path, pipeline)

    await process_review_job(
        ctx, repo_full_name="owner/repo", pr_number=5, review_state="approved"
    )

    assert pipeline.reviews == []


# --- reap -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reap_pr_job_resolves_slices_and_reaps(tmp_path: Path) -> None:
    """A merge job resolves the PRD + slice issues from run-state and reaps."""
    pipeline = _RecordingPipeline(pr_round=(7, [100, 101]))
    ctx = _ctx(tmp_path, pipeline)

    await reap_pr_job(ctx, repo_full_name="owner/repo", pr_number=99)

    assert pipeline.reaps == [
        MergedPullRequest(
            repo_full_name="owner/repo",
            pr_number=99,
            prd_number=7,
            slice_issues=[100, 101],
        )
    ]


@pytest.mark.asyncio
async def test_reap_pr_job_skips_unknown_pr(tmp_path: Path) -> None:
    """A merge of a PR the retinue never opened is skipped, not reaped."""
    pipeline = _RecordingPipeline(pr_round=None)
    ctx = _ctx(tmp_path, pipeline)

    await reap_pr_job(ctx, repo_full_name="owner/repo", pr_number=5)

    assert pipeline.reaps == []


# --- parse_heimdall_review ------------------------------------------------------


def test_parse_heimdall_review_maps_state_and_findings() -> None:
    """The review parser maps gh state and reads severity:summary finding lines."""
    review = parse_heimdall_review(
        repo_full_name="owner/repo",
        pr_number=99,
        prd_number=7,
        review_state="approved",
        review_body="critical: data loss\nnot a finding line\nmedium: slow path",
    )
    assert review.state is ReviewState.APPROVED
    assert review.integration_branch == "retinue/prd-7"
    assert [(f.severity, f.summary) for f in review.findings] == [
        (Severity.CRITICAL, "data loss"),
        (Severity.MEDIUM, "slow path"),
    ]


def test_parse_heimdall_review_unknown_state_is_commented() -> None:
    """An unrecognised gh review state reads as a plain comment (no verdict)."""
    review = parse_heimdall_review(
        repo_full_name="owner/repo",
        pr_number=1,
        prd_number=2,
        review_state="dismissed",
        review_body="",
    )
    assert review.state is ReviewState.COMMENTED
    assert review.findings == []
