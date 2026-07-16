"""Convergence handoff + merge reap (issue #12): notify, then reap on the human merge.

This module owns the two ends of the retinue's "the retinue never merges" contract:

1. **convergence handoff** (:func:`announce_handoff`) — when heimdall converges on the
   staging PR (:mod:`retinue.loopback` reaches ``CONVERGED`` and calls its ``Handoff``
   seam), the retinue fires a single "test & merge" notification: a push (the
   out-of-band heads-up) plus a PR comment (the durable, in-repo record) telling a human
   to test and merge. There is **no** merge collaborator — the retinue hands off and a
   human merges. The signature is the loopback ``Handoff`` shape (``repo_full_name=`` +
   ``pr_number=``) so it wires straight in.
2. **merge reap** (:func:`reap_merged_pr`) — subscribing to a ``pull_request``
   closed+merged signal, on the human's merge the retinue closes the PR's slice issues
   and then *reaps* the PRD: it closes the PRD IFF every non-``hitl`` child issue is
   closed. An open ``hitl`` child (a deliberately human-only slice) does not block the
   reap; an open non-``hitl`` child does.

The push + PR comment reuse the shared :class:`retinue.notify.Notifier` fan-out. The
gh issue-close and PRD child-enumeration are an injected :class:`Handoff` gh seam,
mirroring the gh-seam style of :mod:`retinue.pr_opener` / :mod:`retinue.orchestrator`,
so both flows run with no real ``gh``, push service, or network in a unit test.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Protocol

from retinue.gh import GhTextRunner, run_gh
from retinue.notify import Notification, Notifier
from retinue.slicer import HITL_LABEL

logger = logging.getLogger(__name__)

# The "test & merge" handoff is a heads-up for a human, not an escalation. It carries a
# findable label so the converged-but-unmerged PRs are routable, but it never merges.
TEST_AND_MERGE_LABEL = "test-and-merge"


async def announce_handoff(
    *,
    repo_full_name: str,
    pr_number: int,
    notifier: Notifier,
) -> None:
    """Announce a converged PR for a human to test & merge — the retinue never merges.

    Fires one notification through the shared :class:`~retinue.notify.Notifier`: a push
    heads-up plus a PR comment (and a findable label) telling a human the PR is clean and
    ready to test and merge. No merge happens here and no merge seam is accepted — the
    human merges, and :func:`reap_merged_pr` then reacts to that merge.

    The keyword signature matches the loopback ``Handoff`` seam
    (:data:`retinue.loopback.Handoff`), so the converge path wires this in directly.

    Args:
        repo_full_name: e.g. "owner/repo"; targets the PR comment and label.
        pr_number: The converged staging PR to hand off; the comment lands on it.
        notifier: The shared push + comment + label fan-out.

    Raises:
        Whatever ``notifier`` raises when the durable comment/label record cannot be
        written (a push-sink failure is logged and swallowed by the notifier).
    """
    logger.info(
        "Handing off converged PR #%d (%s) for a human to test & merge",
        pr_number,
        repo_full_name,
    )
    await notifier.notify(
        Notification(
            repo_full_name=repo_full_name,
            issue_number=pr_number,
            title=f"Retinue: test & merge PR #{pr_number}",
            body=(
                f"Heimdall converged on PR #{pr_number}: it is clean and ready. "
                "Please test and merge it — the retinue does not merge. Once you merge, "
                "the retinue will close the slice issues and reap the PRD."
            ),
            label=TEST_AND_MERGE_LABEL,
        )
    )


@dataclass(frozen=True)
class MergedPullRequest:
    """A ``pull_request`` closed+merged signal: the human merged the staging PR.

    Parsed from the ``pull_request`` webhook payload (action ``closed`` with
    ``merged == true``) before the reap reasons about it.

    Attributes:
        repo_full_name: e.g. "owner/repo"; targets the issue closes and the child query.
        pr_number: The merged PR number, for logging and the reap record.
        prd_number: The parent PRD whose children gate the reap; children reference it
            via ``Part of #<prd>`` (the slicer convention).
        slice_issues: The PR's slice issue numbers to close on merge.
    """

    repo_full_name: str
    pr_number: int
    prd_number: int
    slice_issues: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ChildIssue:
    """One child issue of a PRD, as reported by the child-enumeration gh seam.

    A child is any issue carrying ``Part of #<prd>`` (the slicer convention). The reap
    gate looks at the non-``hitl`` children only.

    Attributes:
        number: The child issue number.
        closed: True when the issue is already closed.
        hitl: True when the issue carries the ``hitl`` label — a deliberately human-only
            slice that does NOT block the reap.
    """

    number: int
    closed: bool
    hitl: bool


class Handoff(Protocol):
    """The gh operations behind the merge reap. The reap gh seam.

    A production implementation runs ``gh`` against the target repo (``gh issue close``
    and a child-enumeration query that finds issues carrying ``Part of #<prd>`` with
    their closed-state and ``hitl`` label); tests inject a fake that records the closes
    and scripts the children. There is deliberately **no** merge method — the retinue
    never merges; the human does, and this seam only reacts to that merge.
    """

    async def close_issue(self, *, repo_full_name: str, issue_number: int) -> None:
        """Close ``issue_number`` on ``repo_full_name`` (a ``gh issue close``)."""
        ...

    async def children_of(
        self, *, repo_full_name: str, prd_number: int
    ) -> list[ChildIssue]:
        """Return the PRD's children (issues carrying ``Part of #<prd>``)."""
        ...


class ReapOutcome(enum.Enum):
    """Whether the merge reap closed the PRD or left it open."""

    REAPED = "reaped"
    KEPT_OPEN = "kept_open"


@dataclass(frozen=True)
class ReapResult:
    """Outcome of reacting to a merged PR.

    Attributes:
        outcome: ``REAPED`` when the PRD was closed (all non-``hitl`` children closed),
            ``KEPT_OPEN`` when an open non-``hitl`` child still blocks it.
        closed_slice_issues: The slice issue numbers closed *successfully* on this merge,
            in the input order (a slice whose close failed is omitted).
        prd_closed: True only when the PRD itself was closed.
    """

    outcome: ReapOutcome
    closed_slice_issues: list[int] = field(default_factory=list)
    prd_closed: bool = False


async def reap_merged_pr(merged: MergedPullRequest, *, gh: Handoff) -> ReapResult:
    """React to a human-merged PR: close its slice issues, then reap the PRD.

    Closes the PR's slice issues first, then reaps the PRD — closing it IFF every
    non-``hitl`` child is closed. Slice issues are closed *before* the child gate is
    evaluated so a slice that the children query still reports open is already accounted
    for. An open ``hitl`` child is excluded from the gate (a deliberately human-only
    slice must not hold the PRD open); an open non-``hitl`` child keeps the PRD open.

    Args:
        merged: The parsed ``pull_request`` closed+merged signal.
        gh: The injected reap gh seam (issue close + child enumeration).

    Returns:
        A :class:`ReapResult`: ``REAPED`` with ``prd_closed`` true, or ``KEPT_OPEN``.

    Raises:
        Whatever ``gh`` raises on a real gh failure.
    """
    closed = await _close_slice_issues(merged, gh)
    closed_in_order = [n for n in merged.slice_issues if n in closed]
    logger.info(
        "Merged PR #%d (%s): closed %d of %d slice issue(s)",
        merged.pr_number,
        merged.repo_full_name,
        len(closed),
        len(merged.slice_issues),
    )

    if not await _all_non_hitl_children_closed(merged, gh, just_closed=closed):
        logger.info(
            "PRD #%d (%s) kept open: a non-hitl child is still open",
            merged.prd_number,
            merged.repo_full_name,
        )
        return ReapResult(
            outcome=ReapOutcome.KEPT_OPEN,
            closed_slice_issues=closed_in_order,
        )

    await gh.close_issue(
        repo_full_name=merged.repo_full_name, issue_number=merged.prd_number
    )
    logger.info(
        "Reaped PRD #%d (%s): every non-hitl child is closed",
        merged.prd_number,
        merged.repo_full_name,
    )
    return ReapResult(
        outcome=ReapOutcome.REAPED,
        closed_slice_issues=closed_in_order,
        prd_closed=True,
    )


async def _close_slice_issues(merged: MergedPullRequest, gh: Handoff) -> set[int]:
    """Close every slice issue concurrently; return the ones that closed cleanly.

    The closes are fanned out with :func:`asyncio.gather` (``return_exceptions=True``) so a
    transient per-issue ``gh`` failure does not abort the closes for the other slices — each
    failure is logged with context and skipped. Only the issues that closed without error
    are returned, since a failed close means the issue is genuinely still open and must not
    be credited to the reap gate.
    """
    results = await asyncio.gather(
        *(
            gh.close_issue(
                repo_full_name=merged.repo_full_name, issue_number=issue_number
            )
            for issue_number in merged.slice_issues
        ),
        return_exceptions=True,
    )
    closed: set[int] = set()
    for issue_number, result in zip(merged.slice_issues, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning(
                "Failed to close slice issue #%d (%s) on merge of PR #%d; continuing",
                issue_number,
                merged.repo_full_name,
                merged.pr_number,
                exc_info=result,
            )
        else:
            closed.add(issue_number)
    return closed


async def _all_non_hitl_children_closed(
    merged: MergedPullRequest, gh: Handoff, *, just_closed: set[int]
) -> bool:
    """Whether every non-``hitl`` child of the PRD is closed (the reap gate).

    A ``hitl`` child is excluded from the gate — it is a deliberately human-only slice
    that must not hold the PRD open. A child in ``just_closed`` (a slice this merge closed
    *successfully*) is treated as closed even if the children query still reports it open,
    since the close already ran; a slice whose close *failed* is not in that set, so it
    correctly keeps the PRD open.
    """
    children = await gh.children_of(
        repo_full_name=merged.repo_full_name, prd_number=merged.prd_number
    )
    return all(
        child.closed or child.number in just_closed
        for child in children
        if not child.hitl
    )


# --- production gh-cli reap seam --------------------------------------------------

# The child-enumeration query asks ``gh`` for every issue (open or closed) carrying the
# slicer's ``Part of #<prd>`` marker, returning just the fields the reap gate reads.
_CHILD_JSON_FIELDS = "number,state,labels"

# The only parent-environment variables the ``gh`` child needs: ``PATH`` to locate the
# ``gh`` executable (and the ``git`` it shells out to) and ``HOME`` for its config / the
# host's own ``gh auth`` state. The rest of the worker's environment — Anthropic
# credentials and the like — is deliberately NOT forwarded to a ``gh`` subprocess.
_GH_PASSTHROUGH_ENV = ("PATH", "HOME")


def _auth_env(token: str | None) -> dict[str, str]:
    """Build the minimal ``gh`` subprocess environment, injecting the token when supplied.

    Only the variables ``gh`` actually needs are forwarded — ``PATH`` (to locate the
    ``gh`` executable) and ``HOME`` (its config and the host's own ``gh auth`` state) — so
    the worker's wider environment (its Anthropic credentials, etc.) never leaks into the
    child. ``gh`` reads its credential from ``GH_TOKEN`` (preferred over the host's
    ``gh auth`` state), so a per-call installation token is threaded in via that env var
    rather than a literal ``Authorization`` header. With no token the host's own ``gh``
    auth is used, found via ``HOME``.
    """
    env = {
        name: os.environ[name] for name in _GH_PASSTHROUGH_ENV if name in os.environ
    }
    if token is not None:
        env["GH_TOKEN"] = token
    return env


def _close_issue_argv(*, repo_full_name: str, issue_number: int) -> list[str]:
    """Assemble the ``gh issue close`` argv (without the leading ``gh``)."""
    return [
        "issue",
        "close",
        str(issue_number),
        "--repo",
        repo_full_name,
    ]


def _children_query_argv(*, repo_full_name: str, prd_number: int) -> list[str]:
    """Assemble the child-enumeration argv: every issue marked ``Part of #<prd>``.

    Uses ``gh issue list`` with the ``Part of #<prd>`` search term (the slicer's child
    marker) and ``--state all`` so closed children are returned too, emitting the
    closed-state + labels the reap gate parses.
    """
    return [
        "issue",
        "list",
        "--repo",
        repo_full_name,
        "--state",
        "all",
        "--search",
        f"Part of #{prd_number} in:body",
        "--json",
        _CHILD_JSON_FIELDS,
    ]


def _parse_children(stdout: str) -> list[ChildIssue]:
    """Parse ``gh issue list --json`` stdout into :class:`ChildIssue` records.

    Each issue's ``state`` ("OPEN"/"CLOSED", case-insensitive) becomes ``closed`` and
    the presence of the ``hitl`` label becomes ``hitl``. Empty/whitespace stdout (``gh``
    prints nothing when there are no matches) parses to an empty child list.
    """
    text = stdout.strip()
    if not text:
        return []
    issues = json.loads(text)
    children: list[ChildIssue] = []
    for issue in issues:
        labels = {label["name"] for label in issue.get("labels", [])}
        children.append(
            ChildIssue(
                number=int(issue["number"]),
                closed=str(issue["state"]).upper() == "CLOSED",
                hitl=HITL_LABEL in labels,
            )
        )
    return children


@dataclass(frozen=True)
class HandoffGh:
    """The production gh-cli :class:`Handoff`: close issues + enumerate PRD children.

    Backs :func:`reap_merged_pr` against a live ``gh`` CLI. ``close_issue`` runs
    ``gh issue close``; ``children_of`` runs ``gh issue list`` for the slicer's
    ``Part of #<prd>`` marker and parses the JSON into :class:`ChildIssue` records.
    There is deliberately **no** merge method — the retinue never merges.

    The command assembly (:func:`_close_issue_argv`, :func:`_children_query_argv`), the
    auth env build (:func:`_auth_env`), and the payload parse (:func:`_parse_children`)
    are pure and unit-tested directly. The subprocess itself is the injected ``runner``
    (default: :func:`retinue.gh.run_gh`), so the seam is exercisable without a network.

    Attributes:
        token: An optional ``gh`` token (e.g. a GitHub App installation token), threaded
            in via ``GH_TOKEN``; ``None`` falls back to the host's own ``gh`` auth.
        runner: The subprocess runner; defaults to a real ``gh`` invocation.
    """

    token: str | None = None
    runner: GhTextRunner = run_gh

    async def close_issue(self, *, repo_full_name: str, issue_number: int) -> None:
        """Close ``issue_number`` on ``repo_full_name`` via ``gh issue close``."""
        argv = _close_issue_argv(
            repo_full_name=repo_full_name, issue_number=issue_number
        )
        await self.runner(argv, _auth_env(self.token))

    async def children_of(
        self, *, repo_full_name: str, prd_number: int
    ) -> list[ChildIssue]:
        """Return the PRD's children (issues carrying ``Part of #<prd>``)."""
        argv = _children_query_argv(
            repo_full_name=repo_full_name, prd_number=prd_number
        )
        stdout = await self.runner(argv, _auth_env(self.token))
        return _parse_children(stdout)
