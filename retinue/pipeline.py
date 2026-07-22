"""The ad-hoc pipeline: tie the real adapters together behind one orchestration object.

The scheduler drain drives the real pipeline through a :class:`Pipeline`. The pipeline is
the single seam the worker injects: production builds one from
:class:`retinue.config.Settings` via :func:`build_pipeline_factory`, wiring each step to
its real adapter; tests inject a fake ``Pipeline`` (or a real one over recording-fake
collaborators).

The pipeline owns two responsibilities:

1. **open the ad-hoc PR** (:meth:`Pipeline.process_adhoc_pr`) — after a green build, open
   exactly one ``issue-<N>`` -> target-branch PR and record the PR<->issue mapping so a
   later merge webhook can reap it.
2. **reap a merged PR** (:meth:`Pipeline.reap_pr`) — on the human's merge, close the PR's
   issue(s) and reap the parent through :func:`retinue.handoff.reap_merged_pr`.

Both choke points also record the cross-process run-ledger's terminal states (issue #91):
``escalated`` on a blocking review gate, ``pr_opened`` when the ad-hoc PR opens, ``failed``
on a red build, and ``merged`` on the reap — so ``GET /api/runs`` and ``GET
/api/escalations`` see the pipeline's own outcomes, not just the drain's ``queued`` /
``building``.

The build+PR primitive the scheduler drain drives per issue, and the PR-open-only stranded
recovery, are bound here too (:func:`bind_adhoc_build` / :func:`bind_adhoc_pr_open`) so the
drain runs the production lane.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from retinue.adhoc_build import (
    AdhocBuildResult,
    AdhocIssue,
    ContainerPlanner,
    ReviewGateOutcome,
    build_adhoc_issue,
)
from retinue.budget import (
    CLASSIFIER_ESTIMATED_AMOUNT,
    BudgetGovernor,
)
from retinue.classifier import ClaudeIssueClassifier
from retinue.config import state_dir
from retinue.container import DockerRuntime
from retinue.container_build import ContainerImplementer
from retinue.done_check import (
    EnvSecretResolver,
    GhReportSink,
    ReportSink,
    SecretResolver,
)
from retinue.gh import SubprocessGhRunner
from retinue.handoff import (
    Handoff,
    HandoffGh,
    MergedPullRequest,
    ReapResult,
    reap_merged_pr,
)
from retinue.issues import GhCliIssueCreator, IssueCreator, IssueDraft
from retinue.messages_api import HttpxTransport
from retinue.notify import (
    GhCommentSink,
    GhLabelSink,
    Notification,
    Notifier,
    build_push_sink,
)
from retinue.pr_opener import GhCliPrOps, PrOpenResult, PrOps, open_staging_pr
from retinue.reconcile import ReconcileGhRunner, RunStateStore
from retinue.repo_config import RepoConfig
from retinue.reviewer import AgentSdkReviewGenerator, ReviewFinding
from retinue.roles import Role, resolve_effort, resolve_model
from retinue.routing import GhCliIssueFacts, resolve_issue_level
from retinue.run_ledger import RunLedgerStore, RunState, run_ledger_store_path
from retinue.vocab import BACKLOG_LABEL, HITL_LABEL, issue_web_url, priority_label

if TYPE_CHECKING:
    from retinue.config import Settings
    from retinue.github_app import InstallationAuth

logger = logging.getLogger(__name__)


@dataclass
class Pipeline:
    """Ties the real adapters into the ad-hoc PR open + merge-reap flow.

    The collaborators are the already-built real adapters (or recording fakes in a test);
    the SQLite-backed run-state store is constructed lazily from its path so one pipeline
    owns one durable file.

    Attributes:
        config: The accepted repo config (its resolved ``target_branch`` is the PR base).
        claude_md: The repo's ``CLAUDE.md`` text carrying the done-check command.
        governor: The shared service-level budget governor.
        notifier: The shared escalation fan-out (push + comment + label).
        create_issue: The gh issue creator reused across the advisory review filing.
        pr_ops: The PR-opener gh seam (target-branch check, sync, open).
        reap_gh: The reap gh seam (issue close + child enumeration).
        retry_store_path: SQLite file backing the per-issue implementer-retry counter.
        run_state_path: SQLite file backing the per-issue run-state (PR mapping).
        run_ledger: The cross-process run-ledger store; the pipeline records the terminal
            lifecycle states (``escalated``, ``pr_opened``, ``failed``, ``merged``) into it
            at its own choke points (:meth:`process_adhoc_pr`, :meth:`reap_pr`).
    """

    config: RepoConfig
    claude_md: str
    governor: BudgetGovernor
    notifier: Notifier
    create_issue: IssueCreator
    pr_ops: PrOps
    reap_gh: Handoff
    retry_store_path: Path
    run_state_path: Path
    run_ledger: RunLedgerStore

    _run_state: RunStateStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._run_state = RunStateStore(self.run_state_path)

    async def process_adhoc_pr(
        self, issue: AdhocIssue, build: AdhocBuildResult
    ) -> PrOpenResult | None:
        """Consume the review gate, then open the PR for a green ad-hoc build.

        After a green build, the in-session review gate's outcome routes the issue:

        * **blocking findings** (severity at or above the threshold, or a fix-pass
          regression): the issue is escalated — one :class:`~retinue.notify.Notification`
          fans a ``hitl``-labeled comment + push out — and **no PR opens**. The green
          branch stays pushed for a human to pick up.
        * **backlog findings**: each is filed as a ``backlog`` + ``priority:<severity>``
          follow-up issue, then the PR opens normally.
        * **clean gate** (or no gate, e.g. stranded recovery): the PR opens straight.

        The PR ``issue-<N>`` -> target-branch is opened with the ``issue-<N>`` branch as
        its head (no integration branch), and the PR<->issue mapping is recorded keyed by
        the single issue so a human merge reaps it through :meth:`reap_pr`.

        A red build pushed nothing, so there is no head to open a PR from: the step is
        skipped and ``None`` is returned.

        Args:
            issue: The ad-hoc issue that was built (carries the ``issue-<N>`` branch).
            build: The build outcome; only a ``passed`` build opens a PR, and its ``gate``
                carries the review findings to escalate or file.

        Returns:
            The :class:`PrOpenResult` when the PR step ran, or ``None`` when the red build,
            or a blocking review gate, skipped it.
        """
        if not build.passed:
            logger.info("Ad-hoc build for %s failed; opening no PR", issue.branch)
            await self.run_ledger.record(
                repo_full_name=issue.repo_full_name,
                issue=issue.issue_number,
                state=RunState.FAILED,
            )
            return None
        gate = build.gate
        if gate is not None and gate.blocking:
            await self._escalate_blocking(issue, gate)
            return None
        if gate is not None:
            await self._file_backlog(issue, gate.backlog)
        return await self._open_pr(
            issue.repo_full_name, issue.issue_number, head=build.branch
        )

    async def _escalate_blocking(
        self, issue: AdhocIssue, gate: ReviewGateOutcome
    ) -> None:
        """Escalate a blocked ad-hoc issue: one hitl notification, no PR.

        The single :meth:`Notifier.notify` fans out to push + comment + label, so the
        blocking findings land as a durable ``hitl`` comment on the issue and a push
        heads-up. The green branch is left pushed for the human to take over.
        """
        logger.info(
            "Ad-hoc review gate for %s blocked the PR (%d finding(s)); escalating",
            issue.branch,
            len(gate.blocking),
        )
        await self.notifier.notify(
            Notification(
                repo_full_name=issue.repo_full_name,
                issue_number=issue.issue_number,
                title=f"Review gate blocked issue #{issue.issue_number}",
                body=_render_blocking_body(issue, gate),
                label=HITL_LABEL,
            )
        )
        await self.run_ledger.record(
            repo_full_name=issue.repo_full_name,
            issue=issue.issue_number,
            state=RunState.ESCALATED,
            url=issue_web_url(issue.repo_full_name, issue.issue_number),
        )

    async def _file_backlog(
        self, issue: AdhocIssue, backlog: list[ReviewFinding]
    ) -> None:
        """File each sub-threshold gate finding as a ``priority:<severity>`` backlog nit."""
        for finding in backlog:
            await self.create_issue(
                IssueDraft(
                    title=finding.title,
                    body=finding.body,
                    labels=[BACKLOG_LABEL, priority_label(finding.severity)],
                )
            )
        if backlog:
            logger.info(
                "Ad-hoc review gate for %s filed %d backlog nit(s)",
                issue.branch,
                len(backlog),
            )

    async def reap_pr(self, merged: MergedPullRequest) -> ReapResult:
        """React to a human-merged PR: close its issue(s), then reap the parent.

        The merge is the terminal event, so its run-state row is deleted — otherwise a
        stale mapping would linger for a finished PR.
        """
        result = await reap_merged_pr(merged, gh=self.reap_gh)
        await self._run_state.delete_round(
            repo_full_name=merged.repo_full_name, prd_number=merged.prd_number
        )
        await self.run_ledger.record(
            repo_full_name=merged.repo_full_name,
            issue=merged.prd_number,
            state=RunState.MERGED,
        )
        return result

    async def round_for_pr(
        self, *, repo_full_name: str, pr_number: int
    ) -> tuple[int, list[int]] | None:
        """Resolve a PR to its ``(issue_number, owned_issues)`` from the run-state store.

        The webhook routes merge events by PR number, but the reap needs the issue the PR
        maps to; the mapping recorded when the PR opened (:meth:`_open_pr`) is the source.
        Returns ``None`` for a PR the retinue never opened.
        """
        return await self._run_state.round_for_pr(
            repo_full_name=repo_full_name, pr_number=pr_number
        )

    async def _open_pr(
        self, repo_full_name: str, prd_number: int, *, head: str
    ) -> PrOpenResult:
        """Open the ``issue-<N>`` -> target-branch PR and record the PR<->issue mapping."""
        result = await open_staging_pr(
            repo_full_name=repo_full_name,
            prd_number=prd_number,
            prd_issue_number=prd_number,
            config=self.config,
            ops=self.pr_ops,
            notifier=self.notifier,
            head=head,
        )
        if result.opened and result.pull_request is not None:
            await self._run_state.record_pr(
                repo_full_name=repo_full_name,
                prd_number=prd_number,
                pr_number=result.pull_request.number,
            )
            await self.run_ledger.record(
                repo_full_name=repo_full_name,
                issue=prd_number,
                state=RunState.PR_OPENED,
                url=result.pull_request.url,
            )
        return result


def _render_blocking_body(issue: AdhocIssue, gate: ReviewGateOutcome) -> str:
    """Render the hitl escalation comment body listing a gate's blocking findings.

    Leads with why the PR was held, then one bullet per blocking finding (title +
    severity + body). A fix-pass regression reads the same way — its single synthetic
    finding explains the done-check turned red and the fix was not pushed.
    """
    lead = (
        f"The in-session review gate blocked issue #{issue.issue_number} from opening a "
        f"PR: the reviewer found {len(gate.blocking)} blocking finding(s) it still saw "
        "after one fix pass. The green branch is pushed and left for a human.\n"
    )
    bullets = "\n".join(
        f"- **{f.title}** ({f.severity.name.lower()})\n\n  {f.body.rstrip()}"
        for f in gate.blocking
    )
    return f"{lead}\n{bullets}"


# --- production wiring: build the real pipeline from Settings ----------------------


def run_state_store_path(settings: Settings) -> Path:
    """The run-state SQLite file the factory's pipelines record into.

    Args:
        settings: The runtime settings locating the worker's durable state directory.

    Returns:
        The run-state database path.
    """
    return state_dir(settings) / "run-state.sqlite3"


# Builds a :class:`Pipeline` for an accepted repo. Async so the production factory can mint
# a per-repo installation token before constructing the gh adapters. Injected onto the Arq
# context by :func:`retinue.worker.on_startup` so the worker tasks stay testable with a
# fake factory; production binds it to :func:`build_pipeline_factory`'s factory.
PipelineFactory = Callable[[str, RepoConfig], Awaitable[Pipeline]]


def build_pipeline_factory(
    settings: Settings,
    auth: InstallationAuth,
    *,
    governor: BudgetGovernor,
    fetch_claude_md: ClaudeMdFetcher | None = None,
) -> PipelineFactory:
    """Build the production pipeline factory over the real adapters.

    Returns an async ``(repo_full_name, config) -> Pipeline`` that mints a per-repo
    installation token, then constructs every gh/push adapter against it: the gh issue
    creator, the PR-opener gh ops, the reap gh seam, and the shared notifier (push +
    comment + label).

    Args:
        settings: The runtime settings carrying budget, Anthropic, and push config.
        auth: The GitHub App installation auth used to mint per-repo tokens.
        governor: The one service-level budget governor, shared across the lanes.
        fetch_claude_md: Reads the target repo's ``CLAUDE.md`` text (the done-check command
            source); ``None`` falls back to empty text.

    Returns:
        An async pipeline factory keyed by repo and config.
    """
    push = build_push_sink(settings)
    store_dir = state_dir(settings)
    retry_store_path = store_dir / "impl-retries.sqlite3"
    gh_runner = SubprocessGhRunner()

    async def factory(repo_full_name: str, config: RepoConfig) -> Pipeline:
        token = (await auth.installation_token(repo_full_name)).token
        notifier = Notifier(
            push=push,
            comment=GhCommentSink(token=token),
            label=GhLabelSink(token=token),
        )
        create_issue = GhCliIssueCreator(
            gh_runner, token=token, repo_full_name=repo_full_name
        )
        pr_ops = GhCliPrOps(gh_runner, token=token)
        reap_gh = HandoffGh(token=token)
        return Pipeline(
            config=config,
            claude_md=await _fetch_claude_md(fetch_claude_md, repo_full_name),
            governor=governor,
            notifier=notifier,
            create_issue=create_issue,
            pr_ops=pr_ops,
            reap_gh=reap_gh,
            retry_store_path=retry_store_path,
            run_state_path=run_state_store_path(settings),
            run_ledger=RunLedgerStore(run_ledger_store_path(settings)),
        )

    return factory


# Reads the target repo's ``CLAUDE.md`` text given its full name. Injected (over the GitHub
# contents API in production) so the build's done-check command is parsed from the real
# repo text rather than an empty string; absent, the factory falls back to "".
ClaudeMdFetcher = Callable[[str], Awaitable[str]]


async def _fetch_claude_md(
    fetch_claude_md: ClaudeMdFetcher | None, repo_full_name: str
) -> str:
    """Read the target repo's ``CLAUDE.md`` text, or "" when no fetcher is wired."""
    if fetch_claude_md is None:
        return ""
    return await fetch_claude_md(repo_full_name)


async def _resolve_adhoc_level(
    issue: AdhocIssue,
    config: RepoConfig,
    *,
    classify: ClaudeIssueClassifier,
    label_sink: GhLabelSink,
    comment_sink: GhCommentSink,
    issue_facts: GhCliIssueFacts,
    governor: BudgetGovernor,
) -> str | None:
    """Resolve one ad-hoc issue's routing level once, best-effort.

    Returns ``None`` for a table-less repo (every role falls through to the registry
    default, and zero classifier calls). Otherwise classifies the issue via
    :func:`retinue.routing.resolve_issue_level` (honoring a pre-existing ``level:`` label,
    metering, applying the label, commenting on a classification failure) and returns the
    level name. Any error along the classify path is logged and falls back to the table's
    ``default`` level, so a gh flake never crashes the build.

    Args:
        issue: The ad-hoc issue being routed.
        config: The repo config; its ``routing:`` table supplies the level set.
        classify: The classifier seam.
        label_sink: Applies the resolved ``level:`` label.
        comment_sink: Posts the classification-failure explanation.
        issue_facts: Fetches the issue's classification facts.
        governor: The shared budget governor each classifier charge is metered on.

    Returns:
        The resolved level name, or ``None`` for a table-less repo.
    """
    if config.routing is None:
        return None
    try:
        return await resolve_issue_level(
            issue.repo_full_name,
            issue.issue_number,
            config,
            classify=classify,
            label_sink=label_sink,
            comment_sink=comment_sink,
            issue_facts=issue_facts,
            governor=governor,
            classifier_charge=CLASSIFIER_ESTIMATED_AMOUNT,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning(
            "Ad-hoc routing resolution failed for %s#%d; building at the default level",
            issue.repo_full_name,
            issue.issue_number,
            exc_info=True,
        )
        return config.routing.default


def bind_adhoc_build(
    settings: Settings,
    auth: InstallationAuth,
    *,
    pipeline: Pipeline,
    repo_full_name: str,
    token: str,
    config: RepoConfig,
    claude_md: str,
) -> Callable[..., Awaitable[None]]:
    """Bind the real ad-hoc build+PR primitive for one repo (the drain's downstream).

    Returns an async ``(issue, *, repo_full_name) -> None`` — the
    :data:`retinue.adhoc_drain.AdhocBuild` shape — that drives one ad-hoc issue end to end:
    :func:`retinue.adhoc_build.build_adhoc_issue` (plan -> implement -> done-check -> push,
    then the in-session review gate) in a fresh disposable container, then the *already
    constructed* repo :class:`Pipeline`'s :meth:`~Pipeline.process_adhoc_pr`, which consumes
    the gate outcome: it escalates blocking findings (no PR), files backlog nits, and opens
    the ``issue-<N>`` -> target-branch PR on a clean-or-backlog gate. A red build pushes
    nothing and opens no PR (``process_adhoc_pr`` skips it).

    Each ad-hoc issue is **classified once** at build start via
    :func:`_resolve_adhoc_level`, and its planner, implementer, and reviewer are then
    constructed at that resolved level. The classifier and its collaborators are constructed
    only when the repo declares a ``routing:`` table; a table-less repo makes zero
    classifier calls and builds all three roles at the registry defaults.

    Args:
        settings: Carries the Anthropic credential/auth mode.
        auth: Mints the installation token the build clones over.
        pipeline: The repo's constructed pipeline; its ``process_adhoc_pr`` consumes the
            review gate, opens the PR and records the mapping, and its ``create_issue``
            files the gate's backlog nits.
        repo_full_name: The target repo the build runs against.
        token: The minted installation token the gh report sink files under.
        config: The repo's validated config (resolved target branch, secrets, overrides).
        claude_md: The repo's ``CLAUDE.md`` text carrying the done-check command.

    Returns:
        The bound :data:`retinue.adhoc_drain.AdhocBuild` callable the drain runs per issue.
    """
    runtime = DockerRuntime()
    resolve_secret: SecretResolver = EnvSecretResolver()
    report: ReportSink = GhReportSink(token=token)
    # The issue-facts fetch is unconditional: every implementer bakes the issue's
    # title/body into its prompt (the build container cannot reach GitHub itself). The
    # classifier (and its sinks) is constructed only when the repo declares a routing
    # table, so a table-less repo makes zero classifier calls.
    issue_facts = GhCliIssueFacts(ReconcileGhRunner(token))
    classifier: ClaudeIssueClassifier | None = None
    label_sink: GhLabelSink | None = None
    comment_sink: GhCommentSink | None = None
    if config.routing is not None:
        classifier = ClaudeIssueClassifier(
            credential=settings.anthropic_credential,
            transport=HttpxTransport(),
            routing=config.routing,
        )
        label_sink = GhLabelSink(token=token)
        comment_sink = GhCommentSink(token=token)

    async def build(issue: AdhocIssue, *, repo_full_name: str) -> None:
        level: str | None = None
        if classifier is not None:
            assert label_sink is not None
            assert comment_sink is not None
            level = await _resolve_adhoc_level(
                issue,
                config,
                classify=classifier,
                label_sink=label_sink,
                comment_sink=comment_sink,
                issue_facts=issue_facts,
                governor=pipeline.governor,
            )
        planner = ContainerPlanner(
            credential=settings.anthropic_credential,
            auth_mode=settings.auth_mode,
            model=resolve_model(Role.PLANNER, config, level=level),
        )
        implementer = ContainerImplementer(
            credential=settings.anthropic_credential,
            auth_mode=settings.auth_mode,
            model=resolve_model(Role.IMPLEMENTER, config, level=level),
            max_turns=settings.implement_max_turns,
            issue_facts=issue_facts,
        )
        review_generate = AgentSdkReviewGenerator(
            credential=settings.anthropic_credential,
            transport=HttpxTransport(),
            model=resolve_model(Role.REVIEWER, config, level=level),
            effort=resolve_effort(Role.REVIEWER, config, level=level),
        )
        result = await build_adhoc_issue(
            issue,
            config,
            claude_md,
            planner=planner,
            implementer=implementer,
            auth=auth,
            runtime=runtime,
            resolve_secret=resolve_secret,
            report=report,
            review_generate=review_generate,
        )
        await pipeline.process_adhoc_pr(issue, result)

    return build


def bind_adhoc_pr_open(pipeline: Pipeline) -> Callable[..., Awaitable[None]]:
    """Bind the PR-open-only recovery for a stranded green ``issue-<N>`` branch.

    Returns an async ``(issue, *, repo_full_name) -> None`` — the
    :data:`retinue.adhoc_drain.AdhocPrOpen` shape — the drain drives for a
    :attr:`~retinue.adhoc_drain.FlightState.STRANDED` issue: a branch a prior build pushed
    (push-only-on-green, so it is provably green) but whose PR never opened. It synthesizes
    the green :class:`AdhocBuildResult` for that branch and drives the *same*
    :meth:`Pipeline.process_adhoc_pr` the build's PR step uses — so the PR opens and the
    mapping is recorded identically, with no rebuild and no done-check.

    Args:
        pipeline: The repo's constructed pipeline; its ``process_adhoc_pr`` opens the PR.

    Returns:
        The bound :data:`retinue.adhoc_drain.AdhocPrOpen` callable the drain runs per
        stranded issue.
    """

    async def open_pr(issue: AdhocIssue, *, repo_full_name: str) -> None:
        await pipeline.process_adhoc_pr(
            issue, AdhocBuildResult(branch=issue.branch, passed=True)
        )

    return open_pr
