"""Level-label resolution: the single entry point answering "what level is this issue?"

The human-override policy for PRD #58's routing table (issue #62): a pre-existing
``level:<name>`` label naming a level the repo's routing table declares wins over
classification — human or a prior run. Exactly one known label short-circuits the
classifier entirely; zero known labels (whether none at all, or only unknown/typo
labels) triggers a fresh classification and an additive ``level:<name>`` label; more
than one known label is ambiguous and falls back to the table's ``default`` level with
a warning, no classifier call.

Reuses :class:`~retinue.notify.LabelSink` (``gh issue edit --add-label``) as the
label-application seam — the same adapter fakes out in tests and drives GitHub live.
Labels are never removed by this module; ``--add-label`` is additive by construction.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from retinue.classifier import ClassifyInput, ClassifyResult
from retinue.notify import LabelRequest, LabelSink
from retinue.repo_config import RoutingConfig

logger = logging.getLogger(__name__)

LEVEL_LABEL_PREFIX = "level:"

# Matches ClaudeIssueClassifier.__call__'s signature so a fake can be injected in
# tests without importing the Messages-API adapter itself.
Classifier = Callable[[ClassifyInput], Awaitable[ClassifyResult]]


def level_label(name: str) -> str:
    """Return the ``level:<name>`` GitHub label for a routing-table level name."""
    return f"{LEVEL_LABEL_PREFIX}{name}"


def _known_level_labels(labels: list[str], routing: RoutingConfig) -> list[str]:
    """Return the level names named by ``labels`` that the table actually declares.

    Parses every ``level:<name>`` label (labels with no ``level:`` prefix are not
    level labels at all) and keeps only names present in ``routing.levels`` — an
    unknown name (typo, or a table that no longer declares it) is dropped so it is
    treated the same as carrying no level label.
    """
    names = (
        label.removeprefix(LEVEL_LABEL_PREFIX)
        for label in labels
        if label.startswith(LEVEL_LABEL_PREFIX)
    )
    return [name for name in names if name in routing.levels]


@dataclass(frozen=True)
class LevelResolution:
    """The outcome of resolving one issue's routing level.

    Attributes:
        level: The resolved level name — always one of the table's declared levels.
        classified: True when the classifier actually ran (zero known pre-existing
            labels). False when a pre-existing label (exactly one known, or more
            than one known) decided the level without a classifier call.
        failed: True when the classifier ran but returned no level (both attempts
            failed, per :attr:`~retinue.classifier.ClassifyResult.failed`) — ``level``
            is then the table's default. Always False when ``classified`` is False.
    """

    level: str
    classified: bool
    failed: bool = False


async def resolve_level(
    issue: ClassifyInput,
    routing: RoutingConfig,
    *,
    classify: Classifier,
    label_sink: LabelSink,
    repo_full_name: str,
    issue_number: int,
) -> LevelResolution:
    """Resolve one issue's routing level, honoring any pre-existing ``level:`` label.

    Four cases, checked in order:

    1. Exactly one of ``issue.labels`` names a known level: the classifier is
       skipped and that level wins (human override or a prior run).
    2. More than one known level label: ambiguous — a warning is logged and the
       table's ``default`` level is used; the classifier is still skipped.
    3/4. Zero known level labels (none at all, or only unknown/typo ones):
       ``classify`` runs. On success the resulting ``level:<name>`` label is
       applied via ``label_sink`` (existing labels, known or unknown, are left
       alone — additive only). On classifier failure (both attempts exhausted,
       see :class:`~retinue.classifier.ClassifyResult`) the table's ``default``
       level is used and no label is applied; the caller is signalled via
       :attr:`LevelResolution.failed` (posting an explanatory comment on that
       path is a later PRD #58 slice's responsibility, not this entry point's).

    A label-application failure (case 3/4 success path) never fails resolution: it
    is logged as a warning and the already-computed level is returned unchanged.

    Args:
        issue: The issue to resolve, including its current label set.
        routing: The repo's validated routing table.
        classify: The classifier seam — ``ClaudeIssueClassifier.__call__`` in
            production, a canned fake in tests.
        label_sink: The label-application seam — :class:`~retinue.notify.GhLabelSink`
            in production, a recording fake in tests.
        repo_full_name: ``"owner/repo"``, forwarded to the label sink.
        issue_number: The issue number, forwarded to the label sink.

    Returns:
        The resolved :class:`LevelResolution`.
    """
    known = _known_level_labels(issue.labels, routing)

    if len(known) == 1:
        return LevelResolution(level=known[0], classified=False)

    if len(known) > 1:
        logger.warning(
            "%s#%d carries %d known level labels %s; using default level %r",
            repo_full_name,
            issue_number,
            len(known),
            sorted(known),
            routing.default,
        )
        return LevelResolution(level=routing.default, classified=False)

    result = await classify(issue)
    if result.failed:
        return LevelResolution(level=routing.default, classified=True, failed=True)

    level = result.level
    assert level is not None  # narrows for mypy: `.failed` is False above
    await _apply_label(
        label_sink,
        repo_full_name=repo_full_name,
        issue_number=issue_number,
        level=level,
    )
    return LevelResolution(level=level, classified=True)


async def _apply_label(
    label_sink: LabelSink, *, repo_full_name: str, issue_number: int, level: str
) -> None:
    """Apply the resolved level's label; log and continue on any failure.

    A lost label is not fatal — routing already has the computed level in hand —
    so unlike :class:`~retinue.notify.Notifier`'s comment/label sinks (whose
    failure must propagate, since they carry the durable escalation record) this
    failure is caught broadly and swallowed, mirroring the identical best-effort
    contract already established for the push sink in
    :meth:`~retinue.notify.Notifier._try_push`.
    """
    try:
        await label_sink(
            LabelRequest(
                repo_full_name=repo_full_name,
                issue_number=issue_number,
                label=level_label(level),
            )
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning(
            "Failed to apply %r to %s#%d; continuing with computed level %r",
            level_label(level),
            repo_full_name,
            issue_number,
            level,
            exc_info=True,
        )
