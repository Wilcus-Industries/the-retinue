"""Open the ``issue-<N>`` -> target-branch PR for a green build.

Once a build pushes a green ``issue-<N>`` branch, this module opens exactly one PR
``issue-<N>`` -> ``config.require_target_branch()`` for it, gated by one precheck: the
target branch must exist. A missing one escalates through
:class:`retinue.notify.Notifier` (push + comment + label) and opens no PR.

When the precheck passes, the head branch is brought up to date with the target branch and
exactly one PR is opened. Every gh-touching collaborator — the target-branch existence
check, the bring-up-to-date, and the open-PR action — is an injected :class:`PrOps` seam.
Tests inject a fake, so the whole flow runs with no real ``gh`` and no network. Escalations
reuse the shared :class:`~retinue.notify.Notifier` fan-out rather than re-implementing
notification.
"""

from __future__ import annotations

import enum
import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

from retinue.gh import GhCommandError, GhResult, GhRunner, auth_env
from retinue.notify import Notification, Notifier
from retinue.repo_config import RepoConfig

logger = logging.getLogger(__name__)

# The label every PR-opener escalation carries: a human must intervene (create the target
# branch) before the PR can open.
_ESCALATION_LABEL = "hitl"


@dataclass(frozen=True)
class OpenPrRequest:
    """Payload handed to the open-PR seam.

    Attributes:
        repo_full_name: The target repo, e.g. "owner/repo".
        head: The source branch, ``issue-<N>``.
        base: The target branch, ``config.require_target_branch()``.
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
    """The gh operations behind opening the PR. The PR-opener gh seam.

    A production implementation runs ``gh`` against the target repo (a branch-existence
    query, a merge of the target branch into the head, and ``gh pr create``); tests inject
    a fake that scripts the precheck and records the sync + open-PR calls. Modeled as one
    protocol so the whole PR-opener flow injects through a single collaborator.
    """

    async def staging_exists(self, *, repo_full_name: str, branch: str) -> bool:
        """Return True when ``branch`` (the staging branch) exists on the repo."""
        ...

    async def existing_open_pr(
        self, *, repo_full_name: str, head: str, base: str
    ) -> PullRequest | None:
        """Return the already-open ``head`` -> ``base`` PR, or None when none exists."""
        ...

    async def bring_up_to_date(
        self, *, repo_full_name: str, branch: str, base: str
    ) -> None:
        """Bring ``branch`` up to date with ``base`` before the PR is opened."""
        ...

    async def open_pr(self, request: OpenPrRequest) -> PullRequest:
        """Open the PR described by ``request`` and return the created pull request."""
        ...


class PrOpenOutcome(enum.Enum):
    """Why the PR-opener opened a PR or escalated instead."""

    OPENED = "opened"
    STAGING_MISSING = "staging_missing"


@dataclass(frozen=True)
class PrOpenResult:
    """Outcome of attempting to open the ``issue-<N>`` -> target-branch PR.

    Attributes:
        outcome: ``OPENED`` when exactly one PR was opened; ``STAGING_MISSING`` when the
            target-branch precheck failed and the run escalated.
        integration_branch: The source branch the PR opens from, ``issue-<N>``.
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
    head: str,
) -> PrOpenResult:
    """Open exactly one PR ``<head>`` -> target-branch behind a branch-existence precheck.

    The target branch (``config.require_target_branch()``) must exist; a missing one
    escalates through ``notifier`` (push + comment + label) and opens no PR. When the
    precheck passes, the head branch is brought up to date with the target branch and
    exactly one PR is opened.

    The head is the built ``issue-<N>`` branch, so a standalone build opens its PR straight
    into the target branch with no integration branch.

    Args:
        repo_full_name: The target repo, e.g. "owner/repo".
        prd_number: The issue number the PR maps to (recorded by the caller's mapping).
        prd_issue_number: The tracking issue an escalation comments/labels.
        config: The accepted repo config; ``require_target_branch()`` is the PR base and
            the sync base.
        ops: The injected gh seam (target-branch check, sync, open-PR).
        notifier: The shared escalation fan-out used on the precheck-failure path.
        head: The source ``issue-<N>`` branch to open from.

    Returns:
        A :class:`PrOpenResult`: ``OPENED`` with the created PR, or ``STAGING_MISSING``
        with no PR.

    Raises:
        Whatever ``ops`` raises on a real gh failure, and whatever ``notifier`` raises
        when the durable comment/label record cannot be written.
    """
    branch = head
    staging = config.require_target_branch()

    if not await ops.staging_exists(repo_full_name=repo_full_name, branch=staging):
        return await _escalate(
            repo_full_name=repo_full_name,
            prd_issue_number=prd_issue_number,
            branch=branch,
            notifier=notifier,
            outcome=PrOpenOutcome.STAGING_MISSING,
            title="Retinue cannot open the PR: target branch missing",
            body=(
                f"Issue #{prd_issue_number} built `{branch}`, but the target branch "
                f"`{staging}` does not exist on `{repo_full_name}`. Create it, then "
                "the retinue can open the PR. No PR was opened."
            ),
        )

    existing = await ops.existing_open_pr(
        repo_full_name=repo_full_name, head=branch, base=staging
    )
    if existing is not None:
        # Idempotency: a webhook redelivery, arq retry, or startup-sweep double-resume
        # must never stack a second PR onto staging for the same head.
        logger.info(
            "PR #%d already open for %s: %s -> %s; not opening another",
            existing.number,
            repo_full_name,
            branch,
            staging,
        )
        return PrOpenResult(
            outcome=PrOpenOutcome.OPENED,
            integration_branch=branch,
            pull_request=existing,
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
    await ops.bring_up_to_date(
        repo_full_name=repo_full_name, branch=branch, base=staging
    )
    pull_request = await ops.open_pr(
        OpenPrRequest(
            repo_full_name=repo_full_name,
            head=branch,
            base=staging,
            title=f"Retinue: land {branch} into {staging}",
            body=f"Automated PR for issue #{prd_issue_number}: merge `{branch}` into "
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


# --- production gh-cli PrOps -------------------------------------------------------
#
# The flow above depends only on the :class:`PrOps` protocol. Production wires the
# concrete :class:`GhCliPrOps` below; tests inject a fake. ``GhCliPrOps`` itself does
# not shell out — it assembles ``gh`` invocations and parses their output, delegating
# the actual process spawn to an injected :class:`~retinue.gh.GhRunner`. That keeps
# every pure/parseable part (auth-header build, command assembly, payload parsing)
# testable with a recording fake runner, never a live ``gh``/network. The runner shape,
# result shape, error, and auth env all come from the shared :mod:`retinue.gh` seam.


def _pr_create_args(request: OpenPrRequest) -> list[str]:
    """Assemble the ``gh pr create`` argv for ``request`` (no leading ``"gh"``)."""
    return [
        "pr",
        "create",
        "--repo",
        request.repo_full_name,
        "--base",
        request.base,
        "--head",
        request.head,
        "--title",
        request.title,
        "--body",
        request.body,
    ]


# ``gh pr create`` prints the created PR's URL to stdout; it has no ``--json`` flag
# (json output belongs to list/view — passing it exits 1 with "unknown flag", which
# broke the first live PR open). The URL line is the wire contract to parse.
_PR_URL_RE = re.compile(r"https://[^\s/]+/[^\s/]+/[^\s/]+/pull/(\d+)")


def _parse_pr(stdout: str) -> PullRequest:
    """Parse the PR URL ``gh pr create`` prints into a :class:`PullRequest`.

    Scans stdout for the ``.../pull/<number>`` URL (create may print warnings around
    it) and derives the number from its last path segment. Raises :class:`ValueError`
    when no PR URL is present, so a malformed response fails loudly rather than
    yielding a bogus PR handle.
    """
    match = _PR_URL_RE.search(stdout)
    if match is None:
        raise ValueError(f"gh pr create output carried no PR URL: {stdout!r}")
    return PullRequest(number=int(match.group(1)), url=match.group(0))


class GhCliPrOps:
    """Production :class:`PrOps`: drives the PR through ``gh`` and ``git``.

    Every gh-touching step the flow needs — the target-branch existence query, the
    bring-up-to-date, and ``gh pr create`` — is assembled here and dispatched through the
    injected :class:`~retinue.gh.GhRunner`, authenticated with a ``GH_TOKEN`` bearer (see
    :func:`retinue.gh.auth_env`). The runner is the only side-effecting seam, which keeps
    command assembly and payload parsing unit-testable.

    Args:
        runner: The process-spawn seam that runs each ``gh`` command.
        token: The installation/access token ``gh`` authenticates with.
    """

    def __init__(self, runner: GhRunner, *, token: str) -> None:
        self._runner = runner
        self._token = token

    async def _gh(self, args: list[str]) -> GhResult:
        """Run one ``gh`` command authenticated with the token, raising on failure."""
        result = await self._runner.run(args, env=auth_env(self._token))
        if not result.ok:
            raise GhCommandError(args, result)
        return result

    async def staging_exists(self, *, repo_full_name: str, branch: str) -> bool:
        """Return True when ``branch`` resolves to a ref on ``repo_full_name``."""
        result = await self._runner.run(
            ["api", f"repos/{repo_full_name}/branches/{branch}"],
            env=auth_env(self._token),
        )
        # A missing branch is a 404, which gh reports as a non-zero exit — not an
        # error to raise on, just a False answer to the existence question.
        return result.ok

    async def existing_open_pr(
        self, *, repo_full_name: str, head: str, base: str
    ) -> PullRequest | None:
        """Return the open ``head`` -> ``base`` PR via ``gh pr list``, or None."""
        result = await self._gh(
            [
                "pr",
                "list",
                "--repo",
                repo_full_name,
                "--head",
                head,
                "--base",
                base,
                "--state",
                "open",
                "--json",
                "number,url",
            ]
        )
        payload = json.loads(result.stdout or "[]")
        if not payload:
            return None
        return PullRequest(
            number=int(payload[0]["number"]), url=str(payload[0]["url"])
        )

    async def bring_up_to_date(
        self, *, repo_full_name: str, branch: str, base: str
    ) -> None:
        """Merge ``base`` into ``branch`` server-side so the PR opens up to date.

        The merges API path is built explicitly from ``repo_full_name`` — never from
        ``gh``'s ``{owner}/{repo}`` placeholders, which resolve from the process's
        working-directory git remote. The worker runs in the retinue source, not a
        clone of the target repo, so cwd-relative resolution would target the wrong
        repo (or fail).
        """
        await self._gh(
            [
                "api",
                "--method",
                "POST",
                f"repos/{repo_full_name}/merges",
                "-f",
                f"base={branch}",
                "-f",
                f"head={base}",
            ]
        )

    async def open_pr(self, request: OpenPrRequest) -> PullRequest:
        """Open the PR via ``gh pr create`` and return the parsed PR handle."""
        result = await self._gh(_pr_create_args(request))
        return _parse_pr(result.stdout)
