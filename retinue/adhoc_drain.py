"""Ad-hoc drain: list + rank ready-for-agent non-PRD issues, build up to the cap (#32).

One ad-hoc drain runs per repo. It lists every open ``ready-for-agent`` issue via the gh
seam, keeps only the ones the **ad-hoc** lane decision claims (dropping any
``prd``-labeled issue and any issue carrying a ``Part of #<prd>`` link — those route to the
orchestrator lane), ranks the survivors by ``priority:<severity>`` (no-priority lowest),
and drives the ad-hoc build+PR primitive for each up to the concurrency cap
(``config.max_parallel``). The lane filter **mirrors** :func:`retinue.lane.classify`'s
ad-hoc decision (reusing :class:`~retinue.lane.IssueFacts`) but deliberately does **not**
call ``classify``: routing standalone ``priority:critical``/``high`` issues through
classify would preempt them onto the orchestrator lane and exclude them from the drain.

Each surviving issue is materialized into an :class:`~retinue.adhoc_build.AdhocIssue`
through :meth:`~retinue.adhoc_build.AdhocIssue.from_fetched_issue` — fed the issue body the
gh seam surfaces — never the bare constructor. ``from_fetched_issue`` parses the
``Chain-depth:`` lineage marker out of the body into
:attr:`~retinue.adhoc_build.AdhocIssue.chain_depth`; building the issue by hand would
default every hop to depth 0 and silently make the #39/#40 review-fix chain bound inert.
The gh list seam therefore surfaces each issue's ``body`` alongside its labels.

No dedup/locking/budget hardening yet — that is the next slice (#33). The gh query and the
downstream build are injected and faked, so the whole drain runs with no real ``gh``, no
Docker, and no network — mirroring the injected-seam style of :mod:`retinue.cron`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from retinue.adhoc_build import AdhocIssue
from retinue.cron import GhRunner, _parse_json_array, _run_gh_subprocess
from retinue.lane import IssueFacts
from retinue.loopback import Severity
from retinue.repo_config import RepoConfig
from retinue.slicer import READY_LABEL
from retinue.webhook import PRD_LABEL

logger = logging.getLogger(__name__)

# How many ready-for-agent issues to pull per drain. The cap on concurrent *builds* is
# ``config.max_parallel``; this generous-but-bounded page just keeps the visible set from
# an unbounded fetch, mirroring the cron lane's list limit.
_DEFAULT_LIST_LIMIT = 200


@dataclass(frozen=True)
class ReadyIssue:
    """One open ``ready-for-agent`` issue, as reported by the ad-hoc gh seam.

    The body is surfaced (unlike the cron lane's backlog seam) because the drain feeds it
    to :meth:`~retinue.adhoc_build.AdhocIssue.from_fetched_issue`, which reads the
    ``Chain-depth:`` lineage marker out of it — and because :meth:`is_adhoc` scans it for
    the ``Part of #<prd>`` link (mirroring :func:`retinue.lane.classify`'s decision, not
    calling it) to split orchestrator-lane slices from ad-hoc work.

    Attributes:
        number: The issue number; the build commits to the derived ``issue-<N>`` branch.
        labels: The issue's label names (carries ``ready-for-agent`` and, optionally, a
            ``priority:<severity>`` or the ``prd`` label).
        body: The issue body, scanned for the ``Part of #<prd>`` link and the
            ``Chain-depth:`` marker.
    """

    number: int
    labels: list[str]
    body: str = ""

    def _facts(self) -> IssueFacts:
        """The routing-relevant facts of this issue, for the lane decision and ranking."""
        return IssueFacts(labels=list(self.labels), body=self.body)

    def is_adhoc(self) -> bool:
        """Whether this issue is ad-hoc work the drain builds directly.

        Mirrors :func:`retinue.lane.classify`'s ad-hoc decision (reusing
        :class:`~retinue.lane.IssueFacts` for the ``Part of #<prd>`` scan): a
        ``ready-for-agent`` issue is ad-hoc unless it carries the ``prd`` label or a
        ``Part of #<prd>`` link — both of which route to the orchestrator lane and are
        excluded here. Unlike a raw ``classify`` call this does *not* fold in classify's
        priority **preemption**, which reorders a standalone ``priority:critical``/``high``
        onto the orchestrator lane: that is an ordering optimization, not a statement that
        the work is not ad-hoc, and the drain ranks those at the top itself.
        """
        facts = self._facts()
        if not facts.has_label(READY_LABEL):
            return False
        if facts.has_label(PRD_LABEL):
            return False
        return facts.prd_link() is None

    def severity(self) -> Severity | None:
        """The issue's ``priority:<severity>`` as a :class:`Severity`, or ``None``.

        Reuses :meth:`retinue.lane.IssueFacts.priority`, so an unknown ``priority:*`` value
        is treated as no priority (ranked lowest) rather than raising.
        """
        return self._facts().priority()


class AdhocGh(Protocol):
    """The gh query behind the ad-hoc drain. The ad-hoc lane's gh seam.

    A production implementation runs ``gh issue list --label ready-for-agent`` (with each
    issue's labels and body); tests inject a fake that returns scripted issues. Modeled as
    a protocol so the whole drain injects through a single collaborator, mirroring the
    gh-seam style of :mod:`retinue.cron` / :mod:`retinue.reconcile`.
    """

    async def list_ready(self, *, repo_full_name: str) -> list[ReadyIssue]:
        """Return the repo's open ``ready-for-agent`` issues with their labels and body."""
        ...


class GhCli:
    """The production :class:`AdhocGh`: lists ``ready-for-agent`` issues via the ``gh`` CLI.

    Runs ``gh issue list --repo <repo> --label ready-for-agent --state open --json
    number,labels,body`` and parses the JSON into :class:`ReadyIssue` objects. The ``body``
    field is requested — unlike the cron lane's backlog query — because the drain feeds it
    to :meth:`~retinue.adhoc_build.AdhocIssue.from_fetched_issue` (lineage marker) and
    scans it in :meth:`ReadyIssue.is_adhoc` for the ``Part of #<prd>`` link (mirroring
    :func:`retinue.lane.classify`'s decision, not calling it). Authenticates by injecting
    the GitHub token into the child env as ``GH_TOKEN``, so no token is ever placed on the
    command line.

    The subprocess spawn is the one impure edge, factored behind the injected ``runner``
    (the cron lane's :data:`~retinue.cron.GhRunner` and :func:`~retinue.cron._run_gh_subprocess`,
    reused) so command assembly, the auth env, and payload parsing are unit-testable
    without a real ``gh``, Docker, or network.

    Args:
        token: The GitHub token ``gh`` authenticates with, placed in the child env as
            ``GH_TOKEN``. ``None`` runs ``gh`` with the ambient auth (e.g. a logged-in CLI).
        runner: The injected argv runner; defaults to the real subprocess spawn.
        list_limit: The max number of ready-for-agent issues to pull per drain.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        runner: GhRunner | None = None,
        list_limit: int = _DEFAULT_LIST_LIMIT,
    ) -> None:
        self._token = token
        self._runner = runner or _run_gh_subprocess
        self._list_limit = list_limit

    async def list_ready(self, *, repo_full_name: str) -> list[ReadyIssue]:
        """Return the repo's open ``ready-for-agent`` issues with their labels and body.

        Raises:
            GhCliError: ``gh`` exited non-zero (propagated from the runner).
            ValueError: ``gh`` returned a payload that did not parse as the expected
                issue listing.
        """
        argv = _list_ready_argv(repo_full_name, limit=self._list_limit)
        stdout = await self._runner(argv, self._auth_env())
        return [_parse_ready_issue(entry) for entry in _parse_json_array(stdout)]

    def _auth_env(self) -> Mapping[str, str]:
        """The child-process env carrying the token as ``GH_TOKEN`` (empty when none).

        The token goes in the env, never on the argv, so it never lands in a process
        listing or a log of the command.
        """
        return {"GH_TOKEN": self._token} if self._token else {}


def _list_ready_argv(repo_full_name: str, *, limit: int) -> list[str]:
    """Assemble the ``gh issue list`` argv for the open ``ready-for-agent`` issues.

    Pulls ``number``, ``labels``, and ``body`` as JSON — exactly the fields the drain needs
    to classify (the ``Part of #<prd>`` link), rank (the ``priority:*`` label), and rebuild
    each issue through :meth:`~retinue.adhoc_build.AdhocIssue.from_fetched_issue` (the
    ``Chain-depth:`` marker).
    """
    return [
        "gh",
        "issue",
        "list",
        "--repo",
        repo_full_name,
        "--label",
        READY_LABEL,
        "--state",
        "open",
        "--json",
        "number,labels,body",
        "--limit",
        str(limit),
    ]


def _parse_ready_issue(entry: object) -> ReadyIssue:
    """Parse one ``gh`` issue object into a :class:`ReadyIssue`.

    A malformed entry raises :class:`ValueError` rather than silently dropping the issue.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"gh issue entry is not an object: {entry!r}")
    try:
        number = int(entry["number"])
        labels = [str(label["name"]) for label in entry["labels"]]
        body = str(entry.get("body", "") or "")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"gh issue entry is malformed: {entry!r} ({exc})") from exc
    return ReadyIssue(number=number, labels=labels, body=body)


# The downstream the drain drives for each ranked ad-hoc issue: the ad-hoc build primitive
# (:func:`retinue.adhoc_build.build_adhoc_issue`) followed by the PR step
# (:meth:`retinue.pipeline.Pipeline.process_adhoc_pr`), behind one injected callable so the
# drain is exercised without Docker, gh, the Agent SDK, or network. The drain hands it the
# materialized :class:`~retinue.adhoc_build.AdhocIssue` (built through
# ``from_fetched_issue``, so its ``chain_depth`` is live) and the repo name.
AdhocBuild = Callable[..., Awaitable[None]]


async def run_adhoc_drain(
    *,
    repo_full_name: str,
    gh: AdhocGh,
    build: AdhocBuild,
    config: RepoConfig,
) -> list[AdhocIssue]:
    """Drain the repo's ad-hoc work: list, filter, rank, then build up to the cap.

    1. **list** the repo's open ``ready-for-agent`` issues (number, labels, body),
    2. **filter** to the ad-hoc lane via :meth:`ReadyIssue.is_adhoc` — which mirrors
       :func:`retinue.lane.classify`'s ad-hoc decision (not calling it) — dropping any
       ``prd``-labeled issue and any issue carrying a ``Part of #<prd>`` link, since those
       route to the orchestrator lane,
    3. **rank** the survivors by ``priority:<severity>`` (no-priority lowest), and
    4. **drive** the injected ``build`` for each — materializing the issue through
       :meth:`~retinue.adhoc_build.AdhocIssue.from_fetched_issue` (fed the fetched body, so
       the ``Chain-depth:`` marker is read back and the #39/#40 chain bound stays live) —
       concurrently across the ranked set but capped at ``config.max_parallel`` live builds.

    No dedup/locking/budget hardening in this slice (#33).

    Args:
        repo_full_name: The target repo, e.g. ``"owner/repo"``.
        gh: The ad-hoc gh seam (lists ``ready-for-agent`` issues with labels + body).
        build: The downstream ad-hoc build+PR primitive run per ranked issue, injected so
            the drain runs with no Docker, gh, the Agent SDK, or network.
        config: The accepted repo config; ``max_parallel`` bounds the concurrent builds.

    Returns:
        The :class:`~retinue.adhoc_build.AdhocIssue` objects driven through ``build``, in
        rank order (highest priority first) — the drain's observable surface.
    """
    listed = await gh.list_ready(repo_full_name=repo_full_name)
    ranked = _rank_adhoc([issue for issue in listed if issue.is_adhoc()])
    issues = [
        AdhocIssue.from_fetched_issue(repo_full_name, ready.number, ready.body)
        for ready in ranked
    ]
    if not issues:
        logger.info("Ad-hoc drain idle: no ready-for-agent issues for %s", repo_full_name)
        return []

    logger.info(
        "Ad-hoc drain building %d issue(s) for %s (cap=%s)",
        len(issues),
        repo_full_name,
        config.max_parallel,
    )
    semaphore = asyncio.Semaphore(config.max_parallel or len(issues))

    async def build_one(issue: AdhocIssue) -> None:
        async with semaphore:
            await build(issue, repo_full_name=repo_full_name)

    await asyncio.gather(*(build_one(issue) for issue in issues))
    return issues


# A no-priority issue ranks below every labeled severity, so its rank key sits one step
# under the lowest :class:`~retinue.loopback.Severity` (``LOW``). An unknown ``priority:*``
# value parses to ``None`` too (:meth:`retinue.lane.IssueFacts.priority`), so it lands here.
_NO_PRIORITY_RANK = Severity.LOW - 1


def _rank_adhoc(issues: list[ReadyIssue]) -> list[ReadyIssue]:
    """Rank ad-hoc issues by ``priority:<severity>`` (no-priority lowest), stable on number.

    A more severe issue ranks first; a no-priority issue (or an unknown ``priority:*``
    value) ranks lowest. Ties are broken by ascending issue number so the order is
    deterministic across runs.
    """

    def rank_key(issue: ReadyIssue) -> tuple[int, int]:
        severity = issue.severity()
        return (-(severity.value if severity is not None else _NO_PRIORITY_RANK), issue.number)

    return sorted(issues, key=rank_key)
