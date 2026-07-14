"""Tests for retinue.level: the level-label resolution entry point.

Exercises the human-override policy against a fake classifier callable and a fake
GitHub label sink (modeling gh's add-only label semantics) — no network, no gh.
"""

from __future__ import annotations

import logging

import pytest

from retinue.classifier import ClassifyInput, ClassifyResult
from retinue.level import LEVEL_LABEL_PREFIX, LevelResolution, level_label, resolve_level
from retinue.notify import LabelRequest
from retinue.repo_config import RoutingConfig, RoutingLevel


def _routing() -> RoutingConfig:
    return RoutingConfig(
        default="standard",
        levels={
            "trivial": RoutingLevel(description="cheap work"),
            "standard": RoutingLevel(description="everything else"),
        },
    )


def _issue(labels: list[str]) -> ClassifyInput:
    # Copy so a caller's later mutation of `labels` (e.g. _FakeGhLabels.add
    # appending in place) can never retroactively change what was passed here.
    return ClassifyInput(title="t", body="b", labels=list(labels))


class _FakeGhLabels:
    """Models a fake GitHub issue's label set: add-only, mirrors GhLabelSink."""

    def __init__(self, initial: list[str]) -> None:
        self.current: list[str] = list(initial)
        self.calls: list[LabelRequest] = []

    async def add(self, request: LabelRequest) -> None:
        self.calls.append(request)
        if request.label not in self.current:
            self.current.append(request.label)


class _FailingLabelSink:
    async def add(self, request: LabelRequest) -> None:
        raise RuntimeError("gh exited 1")


class _RecordingClassifier:
    """Canned classifier: returns the scripted result, records every call."""

    def __init__(self, result: ClassifyResult) -> None:
        self._result = result
        self.calls: list[ClassifyInput] = []

    async def __call__(self, issue: ClassifyInput) -> ClassifyResult:
        self.calls.append(issue)
        return self._result


class _UnusedClassifier:
    """Fails the test if ever invoked — for the skip-classifier branches."""

    async def __call__(self, issue: ClassifyInput) -> ClassifyResult:
        raise AssertionError("classifier must not be called")


@pytest.mark.asyncio
async def test_single_known_level_label_skips_classifier() -> None:
    routing = _routing()
    labels = _FakeGhLabels(["level:trivial", "ready-for-agent"])

    result = await resolve_level(
        _issue(labels.current),
        routing,
        classify=_UnusedClassifier(),
        label_sink=labels.add,
        repo_full_name="owner/repo",
        issue_number=1,
    )

    assert result == LevelResolution(level="trivial", classified=False)
    assert labels.calls == []


@pytest.mark.asyncio
async def test_unknown_level_label_classifies_and_adds_additively() -> None:
    routing = _routing()
    labels = _FakeGhLabels(["level:bogus"])
    classifier = _RecordingClassifier(ClassifyResult(level="standard"))

    result = await resolve_level(
        _issue(labels.current),
        routing,
        classify=classifier,
        label_sink=labels.add,
        repo_full_name="owner/repo",
        issue_number=2,
    )

    assert result == LevelResolution(level="standard", classified=True)
    assert classifier.calls == [_issue(["level:bogus"])]
    assert labels.calls == [
        LabelRequest(repo_full_name="owner/repo", issue_number=2, label="level:standard")
    ]
    # additive: the stale/unknown label is still present alongside the new one.
    assert labels.current == ["level:bogus", "level:standard"]


@pytest.mark.asyncio
async def test_no_level_label_classifies_and_applies() -> None:
    routing = _routing()
    labels = _FakeGhLabels(["ready-for-agent"])
    classifier = _RecordingClassifier(ClassifyResult(level="trivial"))

    result = await resolve_level(
        _issue(labels.current),
        routing,
        classify=classifier,
        label_sink=labels.add,
        repo_full_name="owner/repo",
        issue_number=3,
    )

    assert result == LevelResolution(level="trivial", classified=True)
    assert labels.current == ["ready-for-agent", "level:trivial"]


@pytest.mark.asyncio
async def test_multiple_known_level_labels_warn_and_use_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    routing = _routing()
    labels = _FakeGhLabels(["level:trivial", "level:standard"])

    with caplog.at_level(logging.WARNING):
        result = await resolve_level(
            _issue(labels.current),
            routing,
            classify=_UnusedClassifier(),
            label_sink=labels.add,
            repo_full_name="owner/repo",
            issue_number=4,
        )

    assert result == LevelResolution(level="standard", classified=False)
    assert labels.calls == []
    assert any("known level labels" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_label_application_failure_warns_and_keeps_computed_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    routing = _routing()
    classifier = _RecordingClassifier(ClassifyResult(level="trivial"))

    with caplog.at_level(logging.WARNING):
        result = await resolve_level(
            _issue([]),
            routing,
            classify=classifier,
            label_sink=_FailingLabelSink().add,
            repo_full_name="owner/repo",
            issue_number=5,
        )

    assert result == LevelResolution(level="trivial", classified=True)
    assert any("Failed to apply" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_classifier_failure_falls_back_to_default_without_labeling() -> None:
    routing = _routing()
    labels = _FakeGhLabels([])
    classifier = _RecordingClassifier(ClassifyResult(level=None))

    result = await resolve_level(
        _issue([]),
        routing,
        classify=classifier,
        label_sink=labels.add,
        repo_full_name="owner/repo",
        issue_number=6,
    )

    assert result == LevelResolution(level="standard", classified=True, failed=True)
    assert labels.calls == []


def test_level_label_formats_the_prefix() -> None:
    assert level_label("trivial") == f"{LEVEL_LABEL_PREFIX}trivial"
