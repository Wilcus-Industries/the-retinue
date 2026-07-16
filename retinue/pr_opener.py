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
import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

from retinue.gh import GhCommandError, GhResult, GhRunner, auth_env
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
    head: str | None = None,
) -> PrOpenResult:
    """Open exactly one PR ``<head>`` -> ``staging`` behind a heimdall precheck.

    Applies two prechecks in order: heimdall must be installed on the repo, then the
    staging branch must exist. A failed precheck escalates through ``notifier`` (push +
    comment + label) and opens no PR. When both pass, the head branch is brought up to
    date with the staging branch and exactly one PR is opened.

    The head defaults to the PRD lane's integration branch ``retinue/prd-<prd_number>``;
    the ad-hoc lane passes its own ``issue-<N>`` branch as ``head`` so a standalone build
    opens its PR straight into staging with **no** integration branch — the only ad-hoc
    difference at this seam. Both lanes share every precheck, the bring-up-to-date, and
    the single open-PR action.

    Args:
        repo_full_name: The target repo, e.g. "owner/repo".
        prd_number: The PRD (or ad-hoc issue) number; names the default integration head.
        prd_issue_number: The tracking issue an escalation comments/labels (the PRD's, or
            the ad-hoc issue itself).
        config: The accepted repo config; ``staging_branch`` is the PR base and the
            sync base.
        ops: The injected gh seam (heimdall precheck, staging check, sync, open-PR).
        notifier: The shared escalation fan-out used on either precheck-failure path.
        head: The source branch to open from; defaults to ``retinue/prd-<prd_number>``.
            The ad-hoc lane passes ``issue-<N>`` to open straight into staging.

    Returns:
        A :class:`PrOpenResult`: ``OPENED`` with the created PR, or an escalation
        outcome (``HEIMDALL_MISSING`` / ``STAGING_MISSING``) with no PR.

    Raises:
        Whatever ``ops`` raises on a real gh failure, and whatever ``notifier`` raises
        when the durable comment/label record cannot be written.
    """
    branch = head if head is not None else integration_branch(prd_number)
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
    """Production :class:`PrOps`: drives the staging PR through ``gh`` and ``git``.

    Every gh-touching step the flow needs — the heimdall check lookup, the staging
    branch-existence query, the bring-up-to-date, and ``gh pr create`` — is assembled
    here and dispatched through the injected :class:`~retinue.gh.GhRunner`,
    authenticated with a ``GH_TOKEN`` bearer (see :func:`retinue.gh.auth_env`). The
    runner is the only side-effecting
    seam, which keeps command assembly and payload parsing unit-testable.

    Args:
        runner: The process-spawn seam that runs each ``gh`` command.
        token: The installation/access token ``gh`` authenticates with.
        heimdall_check_name: The check-run name that marks heimdall as installed.
    """

    def __init__(
        self,
        runner: GhRunner,
        *,
        token: str,
        heimdall_check_name: str = "heimdall",
    ) -> None:
        self._runner = runner
        self._token = token
        self._heimdall_check_name = heimdall_check_name

    async def _gh(self, args: list[str]) -> GhResult:
        """Run one ``gh`` command authenticated with the token, raising on failure."""
        result = await self._runner.run(args, env=auth_env(self._token))
        if not result.ok:
            raise GhCommandError(args, result)
        return result

    async def heimdall_installed(self, repo_full_name: str) -> bool:
        """Return True when a check named ``heimdall`` is required by any repo ruleset.

        The repo-rulesets *list* endpoint omits each ruleset's ``rules`` (it returns only
        summaries: id, name, target, enforcement), so the required-check contexts must be
        read from each ruleset's *detail* endpoint. List the ruleset ids, then membership-
        test the heimdall context across each ruleset's rules, short-circuiting on the first
        hit. (Querying ``.rules`` on the list response always yields nothing — the prior bug
        that left ``heimdall_installed`` permanently False, so no PR ever opened.)

        A 403 from the list endpoint ("Upgrade to GitHub Pro or make this repository
        public to enable this feature.") means the repo has no rulesets feature at all —
        a private repo on a free plan — so no ruleset can require the heimdall check:
        that is a False answer (the HEIMDALL_MISSING escalation path), not a gh failure
        to raise on. Any other non-zero exit (auth, network) still raises.
        """
        listed = await self._runner.run(
            ["api", f"repos/{repo_full_name}/rulesets", "--jq", "[.[].id]"],
            env=auth_env(self._token),
        )
        if not listed.ok:
            if "HTTP 403" in listed.stderr:
                logger.info(
                    "Rulesets feature unavailable on %s (HTTP 403); reading "
                    "heimdall as not installed",
                    repo_full_name,
                )
                return False
            raise GhCommandError(
                ["api", f"repos/{repo_full_name}/rulesets", "--jq", "[.[].id]"], listed
            )
        for ruleset_id in json.loads(listed.stdout or "[]"):
            detail = await self._gh(
                [
                    "api",
                    f"repos/{repo_full_name}/rulesets/{ruleset_id}",
                    "--jq",
                    "[.rules[]?.parameters.required_status_checks[]?.context]",
                ]
            )
            if self._heimdall_check_name in json.loads(detail.stdout or "[]"):
                return True
        return False

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
