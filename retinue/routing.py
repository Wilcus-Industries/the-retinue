"""Per-issue implementer-model routing for the PRD build lane.

The PRD build lane's only per-issue agent role is the implementer. This module resolves
that implementer's *model* per slice at the slice's first build: it fetches the issue's
facts (title/body/labels), classifies the slice to a routing level via
:func:`retinue.level.resolve_level` (honoring any pre-existing ``level:`` label), and
returns a :class:`~retinue.orchestrator.ContainerImplementer` carrying that level's
implementer model. Two slices of one PRD can therefore launch on different models.

The router is only constructed when the repo declares a ``routing:`` table
(:attr:`~retinue.repo_config.RepoConfig.routing`); a table-less repo keeps the single
injected implementer for every slice and makes zero classifier calls. Each classifier
call that actually runs is metered after the fact on the shared budget ledger via
:meth:`~retinue.budget.BudgetGovernor.record_charge` (a pre-existing-label short-circuit
runs no classifier and records nothing). A classification failure builds the slice at the
table's ``default`` level and posts an explanatory issue comment naming that level.

The classify-one-issue-to-a-level hop is factored into :func:`resolve_issue_level`, the
single seam both this PRD-lane router and the ad-hoc lane
(:func:`retinue.pipeline._resolve_adhoc_level`) route through, so a PRD slice and a
standalone ad-hoc issue classify, meter, label, and comment identically.

The two seams — :data:`IssueFactsSource` (fetch one issue's :class:`ClassifyInput`) and
:data:`PerIssueImplementer` (resolve the implementer for one slice) — are injected so the
production ``gh`` adapter fakes out in tests. No import of :mod:`retinue.wiring` or
:mod:`retinue.pipeline`, so there is no cycle.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from retinue.budget import BudgetGovernor
from retinue.classifier import ClassifyInput
from retinue.level import Classifier, resolve_level
from retinue.notify import CommentRequest, CommentSink, LabelSink
from retinue.orchestrator import ContainerImplementer, Implementer, Slice
from retinue.reconcile import GhRunner
from retinue.repo_config import RepoConfig
from retinue.roles import Role, resolve_model

logger = logging.getLogger(__name__)

# Fetch one issue's classification facts (title/body/labels) — the router's read seam.
IssueFactsSource = Callable[[str, int], Awaitable[ClassifyInput]]

# Resolve the implementer for one slice — the build lane's per-slice implementer seam.
PerIssueImplementer = Callable[[Slice], Awaitable[Implementer]]

# The comment posted when classification fails and the slice falls back to the table's
# default level; it must name the applied level so the record explains what was built.
_FAILURE_COMMENT = (
    "Retinue could not classify this slice's routing level after retries; building it "
    "at the routing table's default level '{level}'."
)


def _issue_facts_argv(repo_full_name: str, issue_number: int) -> list[str]:
    """Build the ``gh issue view`` argv fetching one issue's title/body/labels JSON."""
    return [
        "issue",
        "view",
        str(issue_number),
        "--repo",
        repo_full_name,
        "--json",
        "title,body,labels",
    ]


def _parse_issue_facts(stdout: str) -> ClassifyInput:
    """Parse ``gh issue view --json title,body,labels`` stdout into a :class:`ClassifyInput`.

    Tolerates missing keys — a payload lacking ``title``/``body``/``labels`` yields empty
    defaults rather than raising. ``gh`` returns labels as objects with a ``name`` field;
    only that name is read. ``prd_body`` is left ``None`` (no second GitHub fetch).
    """
    payload = json.loads(stdout)
    labels = [label["name"] for label in payload.get("labels", [])]
    return ClassifyInput(
        title=payload.get("title", ""),
        body=payload.get("body", ""),
        labels=labels,
        prd_body=None,
    )


@dataclass(frozen=True)
class GhCliIssueFacts:
    """Production :data:`IssueFactsSource`: fetch one issue's facts via ``gh issue view``.

    Runs the ``gh`` argv through the shared :class:`~retinue.reconcile.GhRunner` seam and
    parses the JSON stdout into a :class:`ClassifyInput`.

    Attributes:
        runner: The ``gh``-subprocess seam (``__call__(argv) -> stdout``).
    """

    runner: GhRunner

    async def __call__(
        self, repo_full_name: str, issue_number: int
    ) -> ClassifyInput:
        stdout = await self.runner(_issue_facts_argv(repo_full_name, issue_number))
        return _parse_issue_facts(stdout)


async def resolve_issue_level(
    repo_full_name: str,
    issue_number: int,
    config: RepoConfig,
    *,
    classify: Classifier,
    label_sink: LabelSink,
    comment_sink: CommentSink,
    issue_facts: IssueFactsSource,
    governor: BudgetGovernor,
    classifier_charge: float,
) -> str:
    """Classify one issue to a routing level, the shared per-issue routing hop.

    Fetches the issue's facts, resolves the level (honoring a pre-existing ``level:``
    label via :func:`retinue.level.resolve_level`), meters each classifier call that
    actually ran on the shared ledger, and posts the failure comment on a classification
    failure. Returns the resolved level name — always one of the table's declared levels.

    The PRD build lane (:class:`PerIssueImplementerRouter`) and the ad-hoc lane
    (:func:`retinue.pipeline._resolve_adhoc_level`) both route through this one hop, so a
    slice and a standalone issue classify and meter identically. A facts-fetch, label, or
    comment error propagates; each lane's caller decides the fallback (both fall back to
    the table's ``default`` level rather than escalating).

    Args:
        repo_full_name: ``"owner/repo"``.
        issue_number: The issue number being routed.
        config: The repo config; its ``routing:`` table supplies the level set.
        classify: The classifier seam (``ClaudeIssueClassifier.__call__`` in production).
        label_sink: Applies the resolved ``level:`` label (best-effort).
        comment_sink: Posts the classification-failure explanation.
        issue_facts: Fetches one issue's :class:`ClassifyInput`.
        governor: The shared budget governor each classifier charge is recorded on.
        classifier_charge: The estimated charge one classifier call meters.

    Returns:
        The resolved routing level name.
    """
    routing = config.routing
    # The helper is only reached for routing repos; assert narrows for mypy and fails
    # loudly on misuse.
    assert routing is not None
    facts = await issue_facts(repo_full_name, issue_number)
    resolution = await resolve_level(
        facts,
        routing,
        classify=classify,
        label_sink=label_sink,
        repo_full_name=repo_full_name,
        issue_number=issue_number,
    )
    if resolution.classified:
        await governor.record_charge(amount=classifier_charge)
    if resolution.failed:
        await comment_sink(
            CommentRequest(
                repo_full_name=repo_full_name,
                issue_number=issue_number,
                body=_FAILURE_COMMENT.format(level=resolution.level),
            )
        )
    return resolution.level


@dataclass(frozen=True)
class PerIssueImplementerRouter:
    """A :data:`PerIssueImplementer`: classify one slice and route its implementer model.

    Only constructed for a repo with a ``routing:`` table. For one slice it fetches the
    issue facts, resolves the routing level (honoring a pre-existing ``level:`` label),
    meters each classifier call that actually ran, posts an explanatory comment on a
    classification failure, and returns the base implementer with the level's implementer
    model swapped in via :func:`dataclasses.replace` (same credential/auth_mode/max_turns).

    Any runtime failure along that path — a gh flake fetching facts, malformed JSON, a
    label object missing ``name``, or a failed failure-comment post — is caught and
    swallowed: the router logs a warning and returns the injected base implementer
    unchanged rather than propagating. Propagating would surface out of the triage
    wrapper before its retry logic, escalating the slice with zero retries; the base
    implementer (the table's default level) is a sound fallback, mirroring the
    best-effort label contract in :func:`retinue.level.resolve_level`.
    :class:`AssertionError` (router misuse: no routing table) is the exception — a
    programming error propagates loudly instead of hiding behind the fallback.

    Attributes:
        base_implementer: The template implementer whose model is replaced per slice.
        config: The repo config; its ``routing:`` table supplies the level's model.
        classify: The classifier seam (``ClaudeIssueClassifier.__call__`` in production).
        label_sink: Applies the resolved ``level:`` label (best-effort).
        comment_sink: Posts the classification-failure explanation.
        issue_facts: Fetches one issue's :class:`ClassifyInput`.
        governor: The shared budget governor each classifier charge is recorded on.
        classifier_charge: The estimated charge one classifier call meters.
    """

    base_implementer: ContainerImplementer
    config: RepoConfig
    classify: Classifier
    label_sink: LabelSink
    comment_sink: CommentSink
    issue_facts: IssueFactsSource
    governor: BudgetGovernor
    classifier_charge: float

    async def __call__(self, slice_: Slice) -> Implementer:
        # A per-slice resolution failure (a gh flake fetching facts, malformed JSON, a
        # label object missing 'name', or a failed failure-comment post) must never
        # propagate: it would surface out of the triage wrapper *before* its retry
        # logic, escalating the slice with zero retries and skipping its dependent
        # subtree. Instead fall back to the injected base implementer — the table's
        # default level is a good fallback — mirroring resolve_level's best-effort
        # label contract. AssertionError is excluded: it marks router misuse (no
        # routing table), a programming error that must fail loudly, not a flake.
        try:
            return await self._resolve(slice_)
        except (asyncio.CancelledError, AssertionError):
            raise
        except Exception:
            logger.warning(
                "Routing resolution failed for %s#%d; falling back to the base "
                "implementer",
                slice_.repo_full_name,
                slice_.issue_number,
                exc_info=True,
            )
            return self.base_implementer

    async def _resolve(self, slice_: Slice) -> Implementer:
        """Classify one slice and route its implementer model; may raise on any hop."""
        level = await resolve_issue_level(
            slice_.repo_full_name,
            slice_.issue_number,
            self.config,
            classify=self.classify,
            label_sink=self.label_sink,
            comment_sink=self.comment_sink,
            issue_facts=self.issue_facts,
            governor=self.governor,
            classifier_charge=self.classifier_charge,
        )
        model = resolve_model(Role.IMPLEMENTER, self.config, level=level)
        # The implementer runs silently in its container, so this line is the only
        # production surface showing which model a slice was routed to.
        logger.info(
            "Routed %s#%d at level %r: implementer model %s",
            slice_.repo_full_name,
            slice_.issue_number,
            level,
            model,
        )
        return replace(self.base_implementer, model=model)
