"""Open the staging PR for a fully built PRD, behind a heimdall precheck.

Once a PRD's ready set drains (the full-PRD build in :mod:`retinue.orchestrator`
completes), the integration branch ``retinue/prd-<n>`` is ready to land. This module
opens exactly one PR ``retinue/prd-<n>`` -> ``staging`` for it, gated by two prechecks
applied in order:

1. **heimdall installed** — the repo must have the heimdall check installed. A repo
   without it escalates through :class:`retinue.notify.Notifier` (push + comment +
   label) and opens no PR — landing into ``staging`` without the gate is unsafe.
2. **staging exists** — the target ``staging`` branch (``config.staging_branch``) must
   exist. A missing one escalates on its own path and opens no PR.

When both pass, the integration branch is brought up to date with ``staging`` and
exactly one PR is opened. Every gh-touching collaborator — the heimdall precheck, the
staging-branch existence check, the bring-up-to-date, and the open-PR action — is an
injected :class:`PrOps` seam, mirroring the gh-seam style of
:mod:`retinue.github_app` / :mod:`retinue.orchestrator`. Tests inject a fake, so the
whole flow runs with no real ``gh`` and no network. Escalations reuse the shared
:class:`~retinue.notify.Notifier` fan-out rather than re-implementing notification.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Protocol

from retinue.notify import Notification, Notifier
from retinue.orchestrator import integration_branch
from retinue.repo_config import RepoConfig

logger = logging.getLogger(__name__)

# The label every PR-opener escalation carries: a human must intervene (install
# heimdall, create the staging branch) before the PR can open.
_ESCALATION_LABEL = "hitl"


@dataclass(frozen=True)
class OpenPrRequest:
    """Payload handed to the open-PR seam.

    Attributes:
        repo_full_name: The target repo, e.g. "owner/repo".
        head: The source branch, ``retinue/prd-<n>``.
        base: The target branch, ``config.staging_branch`` (default ``staging``).
        title: The PR title.
        body: The PR body.
    """

    repo_full_name: str
    head: str
    base: str
    title: str
    body: str


@dataclass(frozen=True)
class PullRequest:
    """An opened pull request as reported back by the open-PR seam.

    Attributes:
        number: The PR number GitHub assigned.
        url: The PR's web URL.
    """

    number: int
    url: str


class PrOps(Protocol):
    """The gh operations behind opening the staging PR. The PR-opener gh seam.

    A production implementation runs ``gh`` against the target repo (the heimdall
    check lookup, a branch-existence query, a fast-forward/merge of ``staging`` into
    the integration branch, and ``gh pr create``); tests inject a fake that scripts
    the prechecks and records the sync + open-PR calls. Modeled as one protocol so the
    whole PR-opener flow injects through a single collaborator.
    """

    async def heimdall_installed(self, repo_full_name: str) -> bool:
        """Return True when the heimdall check is installed on ``repo_full_name``."""
        ...

    async def staging_exists(self, *, repo_full_name: str, branch: str) -> bool:
        """Return True when ``branch`` (the staging branch) exists on the repo."""
        ...

    async def bring_up_to_date(self, *, branch: str, base: str) -> None:
        """Bring ``branch`` up to date with ``base`` before the PR is opened."""
        ...

    async def open_pr(self, request: OpenPrRequest) -> PullRequest:
        """Open the PR described by ``request`` and return the created pull request."""
        ...


class PrOpenOutcome(enum.Enum):
    """Why the PR-opener opened a PR or escalated instead."""

    OPENED = "opened"
    HEIMDALL_MISSING = "heimdall_missing"
    STAGING_MISSING = "staging_missing"


@dataclass(frozen=True)
class PrOpenResult:
    """Outcome of attempting to open the staging PR for a built PRD.

    Attributes:
        outcome: ``OPENED`` when exactly one PR was opened; ``HEIMDALL_MISSING`` or
            ``STAGING_MISSING`` when a precheck failed and the run escalated.
        integration_branch: The source branch the PR opens from, ``retinue/prd-<n>``.
        pull_request: The opened PR on ``OPENED``; ``None`` on an escalation.
    """

    outcome: PrOpenOutcome
    integration_branch: str
    pull_request: PullRequest | None

    @property
    def opened(self) -> bool:
        """True only when a PR was actually opened."""
        return self.outcome is PrOpenOutcome.OPENED


async def open_staging_pr(
    *,
    repo_full_name: str,
    prd_number: int,
    prd_issue_number: int,
    config: RepoConfig,
    ops: PrOps,
    notifier: Notifier,
) -> PrOpenResult:
    """Open exactly one PR ``retinue/prd-<n>`` -> ``staging`` behind a heimdall precheck.

    Applies two prechecks in order: heimdall must be installed on the repo, then the
    staging branch must exist. A failed precheck escalates through ``notifier`` (push +
    comment + label) and opens no PR. When both pass, the integration branch is brought
    up to date with the staging branch and exactly one PR is opened.

    Args:
        repo_full_name: The target repo, e.g. "owner/repo".
        prd_number: The PRD number; the source branch is ``retinue/prd-<prd_number>``.
        prd_issue_number: The PRD's tracking issue, where an escalation comments/labels.
        config: The accepted repo config; ``staging_branch`` is the PR base and the
            sync base.
        ops: The injected gh seam (heimdall precheck, staging check, sync, open-PR).
        notifier: The shared escalation fan-out used on either precheck-failure path.

    Returns:
        A :class:`PrOpenResult`: ``OPENED`` with the created PR, or an escalation
        outcome (``HEIMDALL_MISSING`` / ``STAGING_MISSING``) with no PR.

    Raises:
        Whatever ``ops`` raises on a real gh failure, and whatever ``notifier`` raises
        when the durable comment/label record cannot be written.
    """
    branch = integration_branch(prd_number)
    staging = config.staging_branch

    if not await ops.heimdall_installed(repo_full_name):
        return await _escalate(
            repo_full_name=repo_full_name,
            prd_issue_number=prd_issue_number,
            branch=branch,
            notifier=notifier,
            outcome=PrOpenOutcome.HEIMDALL_MISSING,
            title="Retinue cannot open the staging PR: heimdall not installed",
            body=(
                f"PRD #{prd_issue_number} built `{branch}`, but heimdall is not "
                f"installed on `{repo_full_name}`. Install the heimdall check, then "
                "the retinue can open the PR into staging. No PR was opened."
            ),
        )

    if not await ops.staging_exists(repo_full_name=repo_full_name, branch=staging):
        return await _escalate(
            repo_full_name=repo_full_name,
            prd_issue_number=prd_issue_number,
            branch=branch,
            notifier=notifier,
            outcome=PrOpenOutcome.STAGING_MISSING,
            title="Retinue cannot open the staging PR: staging branch missing",
            body=(
                f"PRD #{prd_issue_number} built `{branch}`, but the staging branch "
                f"`{staging}` does not exist on `{repo_full_name}`. Create it, then "
                "the retinue can open the PR. No PR was opened."
            ),
        )

    return await _bring_up_to_date_and_open(
        repo_full_name=repo_full_name,
        prd_issue_number=prd_issue_number,
        branch=branch,
        staging=staging,
        ops=ops,
    )


async def _bring_up_to_date_and_open(
    *,
    repo_full_name: str,
    prd_issue_number: int,
    branch: str,
    staging: str,
    ops: PrOps,
) -> PrOpenResult:
    """Sync the integration branch with staging, then open exactly one PR."""
    await ops.bring_up_to_date(branch=branch, base=staging)
    pull_request = await ops.open_pr(
        OpenPrRequest(
            repo_full_name=repo_full_name,
            head=branch,
            base=staging,
            title=f"Retinue: land {branch} into {staging}",
            body=f"Automated PR for PRD #{prd_issue_number}: merge `{branch}` into "
            f"`{staging}` after a full build.",
        )
    )
    logger.info(
        "Opened PR #%d for %s: %s -> %s",
        pull_request.number,
        repo_full_name,
        branch,
        staging,
    )
    return PrOpenResult(
        outcome=PrOpenOutcome.OPENED,
        integration_branch=branch,
        pull_request=pull_request,
    )


async def _escalate(
    *,
    repo_full_name: str,
    prd_issue_number: int,
    branch: str,
    notifier: Notifier,
    outcome: PrOpenOutcome,
    title: str,
    body: str,
) -> PrOpenResult:
    """Escalate a failed precheck through the shared notifier and open no PR."""
    logger.warning(
        "Escalating PR-opener for %s (%s): %s", repo_full_name, outcome.value, title
    )
    await notifier.notify(
        Notification(
            repo_full_name=repo_full_name,
            issue_number=prd_issue_number,
            title=title,
            body=body,
            label=_ESCALATION_LABEL,
        )
    )
    return PrOpenResult(
        outcome=outcome, integration_branch=branch, pull_request=None
    )
