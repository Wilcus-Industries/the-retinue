"""Orchestrator: build one ready slice, or build a full PRD in dependency order.

The single-slice primitive (issue #6) is :func:`build_slice`. For one ready slice the
orchestrator runs the whole build inside **one disposable container** that is destroyed
on every path (the shared :func:`retinue.container_build.build_issue_in_container`
lifecycle, which the ad-hoc lane also runs):

1. **clone + branch** — the container clones the repo over the installation token and
   checks out a fresh ``issue-<N>`` branch off the integration branch ``retinue/prd-<n>``
   (created off the config's ``staging_branch`` on origin *before* the build starts), so a
   later-round slice builds on the sibling work an earlier round already merged onto it,
2. **implement** — one implementer subagent (the Agent SDK seam) execs the headless
   ``claude`` CLI *inside that container* and commits the slice to ``issue-<N>``; the AI
   step is confined to the throwaway container, off the worker host and its docker.sock,
3. **done-check** — the repo's done-check runs in the *same* container, over the real
   changes, and the outcome is posted to the report sink,
4. **push** — only on a green done-check the ``issue-<N>`` branch is pushed to origin, so
   the merge seam can fetch it. A red done-check pushes nothing and **blocks** the merge.

The merge then advances the integration branch ``retinue/prd-<n>`` (ensured to exist on
origin before the build) by merging the pushed ``issue-<N>`` into it. No red slice is ever
merged.

The full-PRD driver (issue #7) is :func:`build_prd`. It wraps the single-slice
primitive: pick the ready set (every ``blocked_by`` ref merged/closed), fan out
implementers in parallel bounded by ``config.max_parallel``, merge the green branches
in topological order under the done-check (resolving a conflict or escalating), and
loop rounds until the ready set drains — all under a single-run lock so at most one
orchestrator run executes at a time.

Every side-effecting collaborator is injected — the implementer spawn, the container
runtime, the auth, the secret resolver, the report sink, the git operations, the
conflict resolver, and the single-run lock — so the whole flow is exercised in tests
with no Agent SDK, no Docker, no gh, no network, and no concurrency races.
"""

from __future__ import annotations

import asyncio
import enum
import heapq
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import httpx

from retinue.classifier import ClassifyInput
from retinue.container import Container, ContainerRuntime, RunResult
from retinue.container_build import (
    GIT_AUTHOR_EMAIL,
    GIT_AUTHOR_NAME,
    GitOpsError,
    Implementer,
    ImplementError,
    Slice,
    build_issue_in_container,
    create_branch_commands,
    implement_env,
    push_branch_command,
)
from retinue.done_check import (
    DEFAULT_IMAGE,
    ReportSink,
    SecretResolver,
)
from retinue.github_app import InstallationAuth
from retinue.repo_config import RepoConfig
from retinue.reviewer import ReviewGenerationError
from retinue.roles import (
    Role,
    resolve_model,
)

logger = logging.getLogger(__name__)


def integration_branch(prd_number: int) -> str:
    """The integration branch a PRD's slices are merged onto: ``retinue/prd-<n>``."""
    return f"retinue/prd-{prd_number}"


# --- real container-exec implementer (production adapter behind the Implementer seam) ---
#
# The production :class:`Implementer` execs the headless ``claude`` CLI *inside the
# disposable build container* the orchestrator owns — the "shell out to claude" discipline
# the PRD cites. The repo is already cloned and the ``issue-<N>`` branch checked out in that
# container, so the agent edits files and commits over the real tree the done-check then
# runs against. Confining the autonomous AI step to a throwaway container keeps it off the
# worker host and its mounted ``docker.sock``. The one side effect is the exec, taken behind
# the injected :class:`~retinue.container.Container`, so the flow is exercisable without a
# live model, the CLI, Docker, gh, or network. The bug-prone pure parts — prompt assembly,
# argv construction, the api_key-vs-subscription env auth routing, and reading the CLI result
# — are factored into the free functions below so they are unit-tested in isolation.
#
# Auth mirrors :class:`retinue.config.Settings`: ``auth_mode="api_key"`` threads the
# credential to the CLI as ``ANTHROPIC_API_KEY``; ``auth_mode="subscription"`` threads it as
# ``CLAUDE_CODE_OAUTH_TOKEN`` (the subscription OAuth env var). The credential rides the
# container env (fixed at ``start``), so the implementer exposes it via :meth:`auth_env` for
# the orchestrator to merge in, rather than passing it per-exec. The contract is the commit
# on ``slice.branch``; the orchestrator gates on the done-check that follows, so a run the
# CLI finishes "successfully" but that fails to satisfy the repo is still caught downstream.
# A non-zero CLI exit (or a json result flagged ``is_error``) raises :class:`ImplementError`.
#
# The implementing model and effort tier come from the :data:`~retinue.roles.Role.IMPLEMENTER`
# registry entry (Sonnet 4.6 at the ``high`` tier by default), resolved at construction time
# so a repo's routing level can swap the model at the wiring site. The in-container
# ``claude`` CLI carries no effort flag today, so the ``high`` tier is registry metadata that
# records the PRD's intent without changing the wire.

# The implementer's brief, appended to the CLI's system prompt. Frozen (no per-slice
# interpolation) so the prefix is stable; the slice specifics ride in the per-slice prompt.
_IMPLEMENT_SYSTEM = (
    "You are an autonomous implementer. Build the requested GitHub issue inside the "
    "repository you are running in. Default to test-driven development: when the change "
    "has testable behavior, write or update a failing test first, then write code until "
    "it passes. A documentation- or config-only change has nothing to test — make the "
    "change directly rather than inventing a test for it. Either way, ensure the repo's "
    "own checks pass before you commit. Make the smallest change that satisfies the "
    "issue; do not refactor unrelated code. When the work is complete and committed to "
    "the issue's branch, stop."
)

# Hard cap on the implementer's agent loop. Without it the headless ``claude`` run is
# bounded only by the arq job_timeout, which cancels the *whole* job (container and all)
# mid-implement — so a thrashing run (e.g. a doc task re-running the full check suite each
# turn) is killed before the done-check ever runs. The cap makes the agent stop and lets
# the done-check report on whatever was committed. Tunable via ``implement_max_turns``.
_DEFAULT_IMPLEMENT_MAX_TURNS = 80


# Fetch one issue's facts (title/body/labels) — the seam the implementer bakes the issue
# content into its prompt through. Defined here rather than in :mod:`retinue.routing`
# (which imports this module and re-exports the alias) so :class:`ContainerImplementer`
# can carry it without an import cycle.
IssueFactsSource = Callable[[str, int], Awaitable[ClassifyInput]]


def _implement_prompt(
    slice_: Slice, *, plan_path: str | None = None, facts: ClassifyInput | None = None
) -> str:
    """Assemble the per-slice prompt: which issue to build, on which branch.

    Names the target repo, the issue number to implement, and the ``issue-<N>`` branch the
    work must be committed to, so the spawned subagent commits where the orchestrator's
    merge seam expects to find it. When ``plan_path`` is given (the ad-hoc lane), the prompt
    leads with an instruction to read that materialized plan first; with no ``plan_path``
    (the PRD lane) the prompt is unchanged, so ``build_slice``/``build_prd`` are unaffected.

    When ``facts`` is given, the issue's title and body are appended as the authoritative
    spec. This is load-bearing: the build container has no ``gh`` and no GitHub token in
    its env (the installation token only rides the clone URL), so the agent cannot read
    the issue itself — without the baked content it no-ops and the slice builds hollow.
    """
    plan_preamble = (
        f"Read the implementation plan at '{plan_path}' first, then implement it. "
        if plan_path is not None
        else ""
    )
    facts_section = (
        ""
        if facts is None
        else (
            "\n\nThe issue's title and body follow; they are the authoritative "
            "specification for this change. This container cannot reach GitHub, so "
            "work from them rather than trying to fetch the issue.\n\n"
            f"Issue title: {facts.title}\n\n"
            f"Issue body:\n{facts.body}"
        )
    )
    return (
        f"{plan_preamble}"
        f"Implement issue #{slice_.issue_number} of {slice_.repo_full_name}. "
        f"Commit your work to the '{slice_.branch}' branch (already checked out). "
        "Implement it test-driven when the change has testable behavior; a documentation- "
        "or config-only change needs no test. Ensure the repo's checks pass before "
        f"committing.{facts_section}"
    )


def _claude_argv(*, prompt: str, model: str, max_turns: int) -> list[str]:
    """Assemble the headless ``claude`` CLI argv for one in-container implement run.

    Runs non-interactively (``-p`` print mode), pins the implementing ``model``, and runs
    with ``--permission-mode bypassPermissions``: ``acceptEdits`` only auto-accepts *file
    edits*, leaving every Bash call — ``git commit``, the repo's checks — blocked pending
    an approval a headless run can never give, so the agent edits its whole run and exits
    0 with zero commits (a hollow-implement cause, seen live). The container is
    disposable and isolated, so bypassing permissions is the intended trade. Caps the
    agent loop at ``max_turns`` so a runaway/thrashing run stops instead of being killed
    mid-implement by the arq job_timeout, appends the frozen implementer brief to the
    system prompt, and emits a machine-readable json result so the exit can be
    cross-checked. The CLI runs in the container's working dir (the cloned repo), so no
    cwd flag is needed.
    """
    return [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--permission-mode",
        "bypassPermissions",
        "--max-turns",
        str(max_turns),
        "--append-system-prompt",
        _IMPLEMENT_SYSTEM,
        "--output-format",
        "json",
    ]


def _claude_result_is_error(stdout: str) -> bool:
    """Whether the CLI's ``--output-format json`` result flags the run as errored.

    The headless CLI emits a json object carrying an ``is_error`` boolean. A non-json or
    empty stdout is not treated as an error here — the exit code is the primary signal —
    so this only catches a run that exited 0 yet reported an internal error. An
    unparseable or empty stdout is unexpected given ``--output-format json`` was
    requested, so it is logged as a warning (the exit code still decides) rather than
    silently passing.
    """
    try:
        result = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        logger.warning(
            "Implementer CLI stdout was not parseable JSON despite --output-format "
            "json (exit code stays authoritative): %r",
            stdout,
        )
        return False
    return bool(isinstance(result, dict) and result.get("is_error"))


# The result text can carry the agent's whole closing message; the log keeps enough to
# diagnose a wrong-but-clean run without flooding the line.
_RESULT_SNIPPET_CHARS = 500


def _claude_result_summary(stdout: str) -> str:
    """A log-ready summary of the CLI's json result: turn count + result snippet.

    Returns an empty string when the stdout is not the expected json object, so the
    completion log line degrades gracefully rather than raising over telemetry.
    """
    try:
        result = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(result, dict):
        return ""
    turns = result.get("num_turns")
    text = str(result.get("result", ""))[:_RESULT_SNIPPET_CHARS]
    return f" ({turns} turns): {text}"


@dataclass(frozen=True)
class ContainerImplementer:
    """Real :class:`Implementer`: build a slice by exec-ing ``claude`` in the build container.

    Satisfies the implementer protocol ``implement(slice_, *, container) -> None`` so it
    drops in where the fake implementer sits in tests and at the wiring site. It execs the
    headless ``claude`` CLI inside the already-cloned, branch-checked-out container, hands it
    the per-slice prompt, and lets it implement TDD-first and commit to ``slice.branch``. The
    orchestrator gates on the done-check that follows, so the contract here is only that the
    exec ran and committed; a non-zero exit (or an ``is_error`` json result) raises
    :class:`ImplementError`.

    Attributes:
        credential: The Anthropic credential (API key or subscription OAuth token).
        auth_mode: ``"api_key"`` (credential rides ``ANTHROPIC_API_KEY``) or
            ``"subscription"`` (credential rides ``CLAUDE_CODE_OAUTH_TOKEN``).
        model: The implementing model id; defaults to the
            :data:`~retinue.roles.Role.IMPLEMENTER` registry entry (Sonnet 4.6), which a
            repo's routing level can replace at the wiring site.
        max_turns: Hard cap on the agent loop, threaded to ``claude --max-turns`` so a
            runaway implement stops itself rather than being killed (with its done-check)
            by the arq job_timeout. The wiring site passes ``settings.implement_max_turns``.
        issue_facts: Fetches the issue's title/body on the worker (which has ``gh`` and
            the installation token) so they are baked into the prompt — the container
            cannot reach GitHub itself. ``None`` keeps the bare prompt.
    """

    credential: str
    auth_mode: str = "api_key"
    model: str = field(default_factory=lambda: resolve_model(Role.IMPLEMENTER))
    max_turns: int = _DEFAULT_IMPLEMENT_MAX_TURNS
    issue_facts: IssueFactsSource | None = None

    async def implement(
        self, slice_: Slice, *, container: Container, plan_path: str | None = None
    ) -> None:
        """Exec ``claude`` in ``container`` to build ``slice_``; raise on an errored run.

        ``plan_path``, when given, names a materialized plan the per-slice prompt instructs
        the subagent to read before building (the ad-hoc lane); the PRD lane passes nothing.
        """
        facts: ClassifyInput | None = None
        if self.issue_facts is not None:
            facts = await self.issue_facts(
                slice_.repo_full_name, slice_.issue_number
            )
        prompt = _implement_prompt(slice_, plan_path=plan_path, facts=facts)
        argv = _claude_argv(prompt=prompt, model=self.model, max_turns=self.max_turns)
        # The runner container execs as root, and the CLI refuses bypassPermissions
        # under root unless IS_SANDBOX=1 marks the env as a disposable sandbox —
        # which this container is.
        result = await container.run_command(argv, env={"IS_SANDBOX": "1"})
        if not result.ok:
            raise ImplementError(
                f"implementer for {slice_.branch} exited {result.exit_code}: "
                f"{result.stderr}"
            )
        if _claude_result_is_error(result.stdout):
            raise ImplementError(
                f"implementer for {slice_.branch} reported an error: {result.stdout}"
            )
        # The CLI's stdout is consumed here and the container is destroyed after the
        # build, so this line is the only forensic trace of what the agent reported —
        # without it a clean-but-wrong run (e.g. "I could not commit") is invisible.
        logger.info(
            "Implementer for %s completed in-container%s",
            slice_.branch,
            _claude_result_summary(result.stdout),
        )

    def auth_env(self) -> dict[str, str]:
        """The credential env the orchestrator merges into the build container at start."""
        return implement_env(self.credential, self.auth_mode)


class MergeConflict(Exception):
    """A merge could not complete because of a conflict.

    The git seam raises this (or a subclass) instead of returning a sentinel, so a
    half-merged slice is never reported as merged. The full-PRD driver catches it to
    hand the conflict to the resolver; the single-slice primitive lets it propagate.
    Carries the source/target branches for the resolver and the report.
    """

    def __init__(self, source: str, into: str) -> None:
        super().__init__(f"merge conflict merging {source} into {into}")
        self.source = source
        self.into = into


class GitOps(Protocol):
    """Git operations on the integration branch. The merge seam.

    A production implementation runs ``git`` inside the disposable container against
    the cloned repo; tests inject a fake that records branch creation and merges. A
    merge that cannot complete (a conflict) raises :class:`MergeConflict` rather than
    returning a sentinel, so a half-merged slice is never reported as merged.
    """

    async def ensure_integration_branch(self, *, branch: str, base: str) -> None:
        """Ensure ``branch`` exists, creating it off ``base`` when it is absent."""
        ...

    async def merge(self, *, source: str, into: str) -> None:
        """Merge ``source`` into ``into``; raise :class:`MergeConflict` on a conflict."""
        ...


# --- real container-backed GitOps adapter ----------------------------------------
#
# The production seam runs ``git`` inside the disposable container that already holds
# the cloned repo (the same workspace ``build_slice``/``build_prd`` clone into). No git
# SDK and no network of its own: ``external_dep none`` — the only collaborator is the
# already-injected :class:`~retinue.container.Container`. The bug-prone, pure parts —
# argv assembly and classifying a failed merge as a conflict vs. a hard git error — are
# factored into the free functions below so they are tested without a live container.

# Per-command git identity for the merge commit ``git`` records. Merges are
# non-interactive, so a committer identity must be configured or ``git commit`` refuses
# to run; it is set per-command via ``-c`` rather than mutating global config in the
# shared workspace. The identity itself is the shared retinue committer.
_GIT_IDENTITY_FLAGS = [
    "-c",
    f"user.name={GIT_AUTHOR_NAME}",
    "-c",
    f"user.email={GIT_AUTHOR_EMAIL}",
]

# Substrings git prints to stdout/stderr when a merge stops on a content conflict, as
# opposed to a hard error (unknown ref, not a repo, …). Matched case-insensitively.
_CONFLICT_MARKERS = (
    "conflict",
    "automatic merge failed",
    "fix conflicts and then commit",
)


def _remote_branch_exists_command(branch: str) -> list[str]:
    """Argv that exits 0 iff ``branch`` already exists on ``origin``.

    The integration branch lives on ``origin`` (a prior run may have created it and a
    fresh merge-container clone has no local ref for it), so its existence is checked
    against the remote, not the local ``refs/heads``.
    """
    return ["git", "ls-remote", "--exit-code", "origin", f"refs/heads/{branch}"]


def _merge_commands(source: str, into: str) -> list[list[str]]:
    """Argv list that checks out ``into`` and merges ``source`` into it.

    The merge is a non-fast-forward, no-edit commit (``--no-ff --no-edit``) under a
    fixed committer identity, so every integration is an explicit, attributable merge
    commit. ``source`` is taken from the remote (``origin/<source>``) to merge the tip
    the implementer pushed.
    """
    return [
        ["git", "checkout", into],
        ["git", "fetch", "origin", source],
        [
            "git",
            *_GIT_IDENTITY_FLAGS,
            "merge",
            "--no-ff",
            "--no-edit",
            f"origin/{source}",
        ],
    ]


_ABORT_MERGE_COMMAND = ["git", "merge", "--abort"]


def _branch_diff_command(branch: str, base: str) -> list[str]:
    """Argv for the diff a merged ``branch`` contributed over the integration ``base``.

    Uses the three-dot form (``base...origin/branch``) so the diff is the branch's own
    changes since it diverged from the integration branch — the slice's contribution —
    rather than also folding in whatever else advanced ``base`` in parallel. ``branch``
    is taken from the remote tip the implementer pushed.
    """
    return ["git", "diff", f"{base}...origin/{branch}"]


def _is_merge_conflict(result: RunResult) -> bool:
    """Whether a failed ``git merge`` stopped on a content conflict (vs. a hard error).

    A conflict is recoverable by the resolver; a hard error (bad ref, not a repo) is
    not. Git signals a conflict with exit code 1 *and* a conflict marker in its output,
    so both are required — a different non-zero exit, or a code-1 failure without a
    marker, is treated as a hard error and surfaced as such rather than as a conflict.
    """
    if result.exit_code != 1:
        return False
    blob = f"{result.stdout}\n{result.stderr}".lower()
    return any(marker in blob for marker in _CONFLICT_MARKERS)


class ContainerGitOps:
    """Production :class:`GitOps` that runs ``git`` in the cloned-repo container.

    The integration-branch merges happen inside a disposable merge container that has
    cloned the repo, so this seam has no external dependency of its own beyond the
    injected :class:`~retinue.container.Container`. A successful merge is pushed to
    ``origin`` so the opened staging PR has a real remote head to land. A merge that
    stops on a content conflict is aborted (to leave the workspace clean for the
    resolver) and surfaced as :class:`MergeConflict`; any other ``git`` failure is a hard
    :class:`GitOpsError`, never silently swallowed.
    """

    def __init__(self, container: Container) -> None:
        self._container = container

    async def ensure_integration_branch(self, *, branch: str, base: str) -> None:
        """Ensure ``branch`` exists on ``origin``, creating it off ``origin/<base>`` if absent.

        Checked before the implementers run so they can root issue-<N> on the integration
        branch's tip: a fresh branch is created off ``origin/<base>`` *and pushed to origin*
        so each build container can ``git fetch`` it as its branch base. An existing branch
        is reused untouched (a no-op fetch already keeps later rounds on its merged tip).
        """
        exists = await self._container.run_command(
            _remote_branch_exists_command(branch)
        )
        if exists.ok:
            logger.info("Integration branch %s already exists on origin", branch)
            return
        for command in create_branch_commands(branch, base):
            await self._run_checked(command, action=f"create {branch} off {base}")
        await self._run_checked(
            push_branch_command(branch), action=f"push {branch} to origin"
        )
        logger.info("Created integration branch %s off %s and pushed to origin", branch, base)

    async def merge(self, *, source: str, into: str) -> None:
        """Merge ``source`` into ``into`` and push ``into``; raise on a conflict.

        Runs checkout + fetch + merge, then pushes ``into`` to ``origin`` so the staging
        PR opened off it has a real remote head. A merge that stops on a content conflict
        is aborted to leave the workspace clean, then raised as :class:`MergeConflict`;
        any other non-zero ``git`` exit (merge or push) is a :class:`GitOpsError`.
        """
        commands = _merge_commands(source, into)
        for command in commands[:-1]:
            await self._run_checked(command, action=f"prepare merge of {source}")
        result = await self._container.run_command(commands[-1])
        if result.ok:
            await self._run_checked(
                push_branch_command(into), action=f"push {into} after merge"
            )
            logger.info("Merged %s into %s and pushed %s", source, into, into)
            return
        if _is_merge_conflict(result):
            await self._container.run_command(_ABORT_MERGE_COMMAND)
            raise MergeConflict(source=source, into=into)
        raise GitOpsError(
            f"git merge of {source} into {into} failed "
            f"(exit {result.exit_code}): {result.stderr}"
        )

    async def round_diff(self, *, merged_branches: list[str], base: str) -> str:
        """Return the merged diff of a round's ``merged_branches`` over the ``base``.

        Concatenates each merged ``issue-<N>`` branch's own contribution since it diverged
        from the integration ``base`` (the three-dot diff), giving the internal reviewer
        the round's merged surface to review. Only already-merged branches are diffed, and
        :meth:`merge` fetched each one's tip from origin moments earlier in this same
        container, so ``origin/<branch>`` is already current — no re-fetch is issued here.
        An empty branch list yields an empty diff (nothing merged, nothing to review).
        """
        sections: list[str] = []
        for branch in merged_branches:
            result = await self._run_checked(
                _branch_diff_command(branch, base), action=f"diff {branch}"
            )
            sections.append(result.stdout)
        return "\n".join(sections)

    async def _run_checked(self, command: list[str], *, action: str) -> RunResult:
        """Run ``command`` in the container; raise :class:`GitOpsError` on failure."""
        result = await self._container.run_command(command)
        if not result.ok:
            raise GitOpsError(
                f"git failed to {action} (exit {result.exit_code}): {result.stderr}"
            )
        return result


class ConflictResolution(enum.Enum):
    """Whether a conflict resolver believes it fixed the merge."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class ConflictResolver(Protocol):
    """Attempts to resolve a merge conflict. The conflict-resolving merger seam.

    A production implementation re-runs the merge with a conflict-resolving agent
    inside the container; tests inject a fake that scripts the outcome. Returning
    ``RESOLVED`` means the resolver staged a resolution and the merge should be
    retried; ``UNRESOLVED`` means it gave up and the slice must escalate. A claimed
    resolution is still verified — the retried merge is the real gate, so a resolver
    that lies (leaves the conflict in place) escalates the slice rather than merging.
    """

    async def __call__(self, *, source: str, into: str) -> ConflictResolution:
        """Attempt to resolve the conflict merging ``source`` into ``into``."""
        ...


class BuildOutcome(enum.Enum):
    """Why the orchestrator merged a slice or blocked it."""

    MERGED = "merged"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class BuildResult:
    """Result of building one ready slice.

    Attributes:
        outcome: ``MERGED`` when the green slice was merged into the integration
            branch, ``BLOCKED`` when a red done-check stopped the merge.
        integration_branch: The integration branch the slice targets,
            ``retinue/prd-<n>`` — merged into on MERGED, left untouched on BLOCKED.
    """

    outcome: BuildOutcome
    integration_branch: str

    @property
    def merged(self) -> bool:
        """True only when the slice was actually merged into the integration branch."""
        return self.outcome is BuildOutcome.MERGED


async def build_slice(
    slice_: Slice,
    config: RepoConfig,
    claude_md: str,
    *,
    implementer: Implementer,
    git: GitOps,
    auth: InstallationAuth,
    runtime: ContainerRuntime,
    resolve_secret: SecretResolver,
    report: ReportSink,
    image: str = DEFAULT_IMAGE,
) -> BuildResult:
    """Build one ready slice in its own container, gate on the done-check, then merge.

    The integration branch ``retinue/prd-<n>`` is ensured on origin first (created off
    ``config.staging_branch`` when absent), then the whole build runs in one disposable
    container (the shared :func:`retinue.container_build.build_issue_in_container`
    lifecycle): clone, check out a fresh ``issue-<N>``
    branch off the integration branch's tip, let the implementer exec ``claude`` inside it
    and commit, run the repo's done-check over the real changes, and push ``issue-<N>``
    only when green. The done-check result gates the merge: a green check merges the pushed
    ``issue-<N>`` into ``retinue/prd-<n>``, while a red check pushes nothing and blocks the
    merge so no failing slice is ever integrated.

    Args:
        slice_: The ready slice to build (repo, issue number, PRD number).
        config: The accepted repo config; its ``staging_branch`` is the integration-branch
            base (and the integration branch is the slice-branch base), and its ``secrets``
            are injected into the container.
        claude_md: The repo's ``CLAUDE.md`` text, carrying the done-check command.
        implementer: Execs the implementer subagent in the container (the Agent SDK seam).
        git: Integration-branch git operations (the merge seam).
        auth: Mints the installation token used to clone (the auth seam).
        runtime: Spawns the disposable build container (the Docker seam).
        resolve_secret: Resolves the config's declared secret names/refs to values.
        report: Sink the done-check outcome is posted to (commit status / comment).
        image: Container image the build runs in.

    Returns:
        A :class:`BuildResult`: ``MERGED`` when the green slice was merged, or
        ``BLOCKED`` when a red done-check stopped it.

    Raises:
        Propagates whatever the build container raises (e.g. a missing secret or a clone
        failure), and any merge error the git seam raises on a conflict.
    """
    branch = integration_branch(slice_.prd_number)

    # The integration branch is created off staging *before* the implementer runs, so
    # the build container can root issue-<N> on its tip (PRD: implementers branch off the
    # integration branch, not staging).
    await git.ensure_integration_branch(branch=branch, base=config.staging_branch)

    passed = await build_issue_in_container(
        slice_,
        config,
        claude_md,
        base=branch,
        implementer=implementer,
        auth=auth,
        runtime=runtime,
        resolve_secret=resolve_secret,
        report=report,
        image=image,
    )

    if not passed:
        # A red slice is never merged: nothing was pushed, leave the branch untouched.
        logger.info(
            "Blocking merge of %s into %s: done-check failed",
            slice_.branch,
            branch,
        )
        return BuildResult(outcome=BuildOutcome.BLOCKED, integration_branch=branch)

    await _merge_green_slice(slice_, branch, git=git)
    return BuildResult(outcome=BuildOutcome.MERGED, integration_branch=branch)


async def _merge_green_slice(slice_: Slice, branch: str, *, git: GitOps) -> None:
    """Merge a green slice onto the integration branch.

    The integration branch is ensured once up front (by :func:`build_slice` and
    :func:`build_prd` before any slice is merged), so this merges straight onto it
    rather than re-ensuring — a per-merge re-ensure is a redundant ``git ls-remote``
    round-trip against origin on every merged slice.
    """
    await git.merge(source=slice_.branch, into=branch)
    logger.info("Merged %s into %s after green done-check", slice_.branch, branch)


# --- full-PRD driver (issue #7) --------------------------------------------------


@dataclass(frozen=True)
class PrdSlice(Slice):
    """A PRD slice: a :class:`Slice` plus the issue numbers it is blocked by.

    Attributes:
        blocked_by: Issue numbers this slice depends on. A slice is *ready* only once
            every blocker is merged in this run (or is absent from the PRD's slice
            set, meaning it was already merged/closed before the run began).
    """

    blocked_by: list[int] = field(default_factory=list)


class RoundReviewer(Protocol):
    """Reviews a merged round and enqueues review-fix slices. The reviewer seam.

    After each round's merge, :func:`build_prd` hands the round's merged issue numbers
    (in merge order) to ``review``. A production implementation reads the round's merged
    diff, drives :func:`retinue.reviewer.review_round` to file ``review-fix`` follow-up
    issues (and wire them into dependents' ``## Blocked by``), and returns one
    :class:`PrdSlice` per filed issue — wired with the in-round dependents it blocks — so
    the fix enters a *subsequent* round's ready set and is built in a later round of the
    same run. A clean review returns an empty list; an absent seam runs no review at all.
    Tests inject a fake that records the merged set without the Agent SDK, gh, or network.
    """

    async def review(self, *, merged_issues: list[int]) -> list[PrdSlice]:
        """Review the round's ``merged_issues``; return review-fix slices to enqueue."""
        ...


@dataclass(frozen=True)
class PrdBuildResult:
    """Outcome of a full-PRD build, partitioned by what happened to each slice.

    Every input slice lands in exactly one bucket — none is dropped from all of them.

    Attributes:
        integration_branch: The integration branch every slice targeted.
        merged_issues: Issue numbers merged into the integration branch, in merge
            (topological) order.
        blocked_issues: Issue numbers whose done-check was red, so they were not
            merged.
        escalated_issues: Issue numbers whose merge hit a conflict that could not be
            resolved under the done-check.
        skipped_issues: Issue numbers that never became ready because an upstream
            slice they transitively depend on was blocked or escalated, pruning the
            subtree. Reported here rather than silently dropped.
        deferred: True when the budget gate held the run back — nothing built, every
            bucket empty. Carried so the caller can re-enqueue rather than lose the run.
        defer_until: When the budget window frees on a deferred run; ``None`` otherwise.
    """

    integration_branch: str
    merged_issues: list[int]
    blocked_issues: list[int]
    escalated_issues: list[int]
    skipped_issues: list[int]
    deferred: bool = False
    defer_until: datetime | None = None


async def build_prd(
    slices: list[PrdSlice],
    config: RepoConfig,
    claude_md: str,
    *,
    implementer: Implementer,
    git: GitOps,
    auth: InstallationAuth,
    runtime: ContainerRuntime,
    resolve_secret: SecretResolver,
    report: ReportSink,
    lock: AbstractAsyncContextManager[object],
    resolve_conflict: ConflictResolver | None = None,
    review_round: RoundReviewer | None = None,
    image: str = DEFAULT_IMAGE,
) -> PrdBuildResult:
    """Build a full PRD: ready set -> parallel fan-out -> topological merge -> review -> loop.

    Runs under ``lock`` so at most one orchestrator run executes at a time. Each round
    picks the ready set (every ``blocked_by`` merged this run or already merged/closed
    before it), fans the slices out to implementers bounded by ``config.max_parallel``,
    then merges the green branches in dependency order under the done-check — resolving
    a conflict or escalating. After the round's merge, the injected ``review_round`` (when
    present) reviews the round's merged diff and files review-fix follow-up issues, which
    re-enter as slices and build in a *subsequent* round of the same run. Rounds repeat
    until no ready slice remains.

    Args:
        slices: The PRD's slices with their ``blocked_by`` graph.
        config: The accepted repo config; ``max_parallel`` bounds the fan-out and
            ``staging_branch`` is the base for a new integration branch.
        claude_md: The repo's ``CLAUDE.md`` text, carrying the done-check command.
        implementer: Spawns the implementer subagent (the Agent SDK seam).
        git: Integration-branch git operations (the merge seam).
        auth: Mints the installation token used to clone (the auth seam).
        runtime: Spawns the disposable container the done-check runs in (Docker seam).
        resolve_secret: Resolves the config's declared secret names/refs to values.
        report: Sink the done-check outcome is posted to.
        lock: The single-run lock; entering it raises when a run is already in
            flight, so a concurrent second run is rejected rather than serialized.
        resolve_conflict: Attempts to resolve a merge conflict; absent means any
            conflict escalates.
        review_round: The internal reviewer run after each round's merge (the reviewer
            seam); absent means no per-round review and no review-fix slices.
        image: Container image the done-check runs in.

    Returns:
        A :class:`PrdBuildResult` partitioning the slices into
        merged/blocked/escalated/skipped — a subtree pruned by a failed upstream
        slice lands in ``skipped``, so every slice is accounted for. Review-fix slices
        the reviewer filed and built mid-run land in ``merged`` alongside the originals.

    Raises:
        Exception: Whatever the injected lock raises when a run is already in flight.
    """
    branch = integration_branch(slices[0].prd_number) if slices else integration_branch(0)
    async with lock:
        # One integration branch off staging per PRD, created up front so every round's
        # implementers branch off its tip — and a later round builds on the sibling work
        # the earlier round already merged onto it. An empty PRD creates nothing.
        if slices:
            await git.ensure_integration_branch(
                branch=branch, base=config.staging_branch
            )
        state = _PrdState(slices)
        while True:
            ready = state.ready_set()
            if not ready:
                break
            built = await _build_round(
                ready,
                config,
                claude_md,
                base=branch,
                implementer=implementer,
                auth=auth,
                runtime=runtime,
                resolve_secret=resolve_secret,
                report=report,
                image=image,
            )
            merged_this_round = await _merge_round(
                built,
                branch,
                git=git,
                resolve_conflict=resolve_conflict,
                state=state,
            )
            # After the round merges, the internal reviewer reviews the round's merged
            # diff and files review-fix follow-ups; each comes back as a slice that
            # enters a later round's ready set and builds in the same run.
            if review_round is not None and merged_this_round:
                fixes = await _review_round_advisory(
                    review_round, merged_this_round, slices[0].prd_number
                )
                state.enqueue(fixes)
        # The ready set has drained; any still-pending slice was pruned by a failed
        # upstream slice. Surface it as skipped rather than dropping it silently.
        state.drain_skipped()

    return PrdBuildResult(
        integration_branch=branch,
        merged_issues=state.merged,
        blocked_issues=state.blocked,
        escalated_issues=state.escalated,
        skipped_issues=state.skipped,
    )


async def _review_round_advisory(
    review_round: RoundReviewer, merged_issues: list[int], prd_number: int
) -> list[PrdSlice]:
    """Run the per-round reviewer, swallowing its failure so it never aborts the build.

    The reviewer is advisory — it only files review-fix follow-ups and never edits code —
    but it runs *after* the round's slices are merged and pushed, so letting a reviewer
    failure (an HTTP 400 from the Messages API, a leaked httpx transport error) propagate
    would fail the whole arq job and discard an already-merged build. A failure is logged
    against the PRD and treated as "no follow-ups this round" rather than raised. Only the
    reviewer's own error types are caught; anything else still surfaces.
    """
    try:
        return await review_round.review(merged_issues=merged_issues)
    except (ReviewGenerationError, httpx.HTTPError):
        logger.warning(
            "Advisory round review for PRD #%d failed; continuing with no review-fixes",
            prd_number,
            exc_info=True,
        )
        return []


class _PrdState:
    """Tracks ready-set selection and per-slice outcomes across rounds.

    A slice is *ready* when it is still pending and every blocker is either merged
    this run or absent from the PRD's slice set (already merged/closed beforehand). A
    blocked or escalated slice is terminal: its dependents never become ready, so a
    failure naturally prunes the subtree below it. That pruning is made observable —
    once the ready set drains, every still-pending slice is recorded as *skipped* via
    :meth:`drain_skipped`, so the abandoned subtree is reported, not silently dropped.
    """

    def __init__(self, slices: list[PrdSlice]) -> None:
        self._pending = list(slices)
        self._slice_numbers = {s.issue_number for s in slices}
        self.merged: list[int] = []
        self.blocked: list[int] = []
        self.escalated: list[int] = []
        self.skipped: list[int] = []

    def ready_set(self) -> list[PrdSlice]:
        """Slices whose every blocker is merged (or was already merged/closed)."""
        return [s for s in self._pending if self._is_ready(s)]

    def _is_ready(self, slice_: PrdSlice) -> bool:
        return all(
            blocker in self.merged or blocker not in self._slice_numbers
            for blocker in slice_.blocked_by
        )

    def record(self, slice_: PrdSlice, bucket: list[int]) -> None:
        """Move ``slice_`` out of pending into a terminal bucket."""
        self._pending.remove(slice_)
        bucket.append(slice_.issue_number)

    def enqueue(self, slices: list[PrdSlice]) -> None:
        """Add review-fix slices to the pending set so a later round builds them.

        The internal reviewer files review-fix follow-ups after a round's merge; each
        comes back as a :class:`PrdSlice` enqueued here so it joins the ready-set
        selection and builds in a subsequent round of the same run. A slice whose number
        is already known is ignored, so a re-reviewed round never double-enqueues a fix.
        """
        for slice_ in slices:
            if slice_.issue_number in self._slice_numbers:
                continue
            self._pending.append(slice_)
            self._slice_numbers.add(slice_.issue_number)

    def drain_skipped(self) -> None:
        """Record every still-pending slice as skipped once no slice can become ready.

        Called after the ready set drains: any slice still pending was pruned by a
        blocked or escalated upstream slice and can never become ready. Recording it
        here surfaces the abandoned subtree instead of letting it vanish from the
        result. Issue order is preserved so the bucket is deterministic.
        """
        for slice_ in list(self._pending):
            self.record(slice_, self.skipped)


# Default cap on concurrent per-slice build containers when a repo sets no
# ``max_parallel``. Without a bound a default-config repo runs *every* ready slice's
# container at once, so a large ready set can exhaust the worker host's docker/CPU. An
# explicit ``config.max_parallel`` still wins; this only backs the unset case.
_DEFAULT_MAX_PARALLEL = 3


class _SliceBuildOutcome(enum.Enum):
    """How one slice's build ended, before the round's serial merge phase.

    Distinguishes a red done-check (``BLOCKED``) from a build that raised
    (``ERRORED``): the first is a clean-but-failing slice, the second a transient
    build failure (clone/implement/Docker error) that must not cancel its siblings.
    ``GREEN`` slices proceed to the merge.
    """

    GREEN = "green"
    BLOCKED = "blocked"
    ERRORED = "errored"


async def _build_round(
    ready: list[PrdSlice],
    config: RepoConfig,
    claude_md: str,
    *,
    base: str,
    implementer: Implementer,
    auth: InstallationAuth,
    runtime: ContainerRuntime,
    resolve_secret: SecretResolver,
    report: ReportSink,
    image: str,
) -> list[tuple[PrdSlice, _SliceBuildOutcome]]:
    """Fan out the ready slices, bounded by ``max_parallel``; return per-slice outcomes.

    Each slice's build runs in its own disposable container (clone → branch off ``base``
    → implement → done-check → push-on-green), concurrently across the round but capped at
    ``config.max_parallel`` live builds (falling back to :data:`_DEFAULT_MAX_PARALLEL`
    when unset, never more than the ready count). ``base`` is the integration branch, so a
    later round branches off the sibling work an earlier round already merged onto it.

    One slice's build raising (a transient Docker/clone/implement error) must not cancel
    its siblings, so the fan-out gathers with ``return_exceptions=True``: a raised build
    is logged and mapped to ``ERRORED`` (escalated in the merge phase), while the green
    siblings still merge. Only the build phase runs here; merges are serialized afterwards.
    """
    bound = config.max_parallel or min(len(ready), _DEFAULT_MAX_PARALLEL) or 1
    semaphore = asyncio.Semaphore(bound)

    async def build_one(slice_: PrdSlice) -> _SliceBuildOutcome:
        async with semaphore:
            passed = await build_issue_in_container(
                slice_,
                config,
                claude_md,
                base=base,
                implementer=implementer,
                auth=auth,
                runtime=runtime,
                resolve_secret=resolve_secret,
                report=report,
                image=image,
            )
            return _SliceBuildOutcome.GREEN if passed else _SliceBuildOutcome.BLOCKED

    results = await asyncio.gather(
        *(build_one(s) for s in ready), return_exceptions=True
    )
    built: list[tuple[PrdSlice, _SliceBuildOutcome]] = []
    for slice_, result in zip(ready, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning(
                "Slice %s build raised; escalating (siblings unaffected)",
                slice_.branch,
                exc_info=result,
            )
            built.append((slice_, _SliceBuildOutcome.ERRORED))
        else:
            built.append((slice_, result))
    return built


async def _merge_round(
    built: list[tuple[PrdSlice, _SliceBuildOutcome]],
    branch: str,
    *,
    git: GitOps,
    resolve_conflict: ConflictResolver | None,
    state: _PrdState,
) -> list[int]:
    """Merge a round's green slices in dependency order; return the issues merged.

    Merges are serialized (a shared integration branch) and ordered by a Kahn
    topological sort over the round's ``blocked_by`` graph, so an in-round blocker
    always merges before its dependent; issue number only breaks ties between
    mutually independent slices, keeping the order deterministic. A red slice is
    recorded blocked, a build that raised is recorded escalated, and a merge conflict
    is resolved under the done-check or escalated. The returned list is *this* round's
    merged issue numbers in merge order — the surface the per-round reviewer reviews.
    """
    outcome_by_number = {slice_.issue_number: outcome for slice_, outcome in built}
    merged_this_round: list[int] = []
    for slice_ in _topo_merge_order([slice_ for slice_, _ in built]):
        outcome = outcome_by_number[slice_.issue_number]
        if outcome is _SliceBuildOutcome.BLOCKED:
            state.record(slice_, state.blocked)
            continue
        if outcome is _SliceBuildOutcome.ERRORED:
            state.record(slice_, state.escalated)
            continue
        merged = await _merge_or_escalate(
            slice_, branch, git=git, resolve_conflict=resolve_conflict
        )
        state.record(slice_, state.merged if merged else state.escalated)
        if merged:
            merged_this_round.append(slice_.issue_number)
    return merged_this_round


def _topo_merge_order(slices: list[PrdSlice]) -> list[PrdSlice]:
    """Order a round's slices topologically over their ``blocked_by`` graph (Kahn).

    Only edges between slices *in this round* constrain the order — a blocker outside
    the round was already merged/closed, so it imposes no in-round ordering. Among
    mutually independent slices the lowest issue number is emitted first, so the order
    is fully deterministic. The input is assumed acyclic (the slicer publishes a DAG);
    if a cycle ever slipped in, its members are simply left out of the result.
    """
    by_number = {s.issue_number: s for s in slices}
    blockers = {
        s.issue_number: [b for b in s.blocked_by if b in by_number] for s in slices
    }
    dependents: dict[int, list[int]] = {n: [] for n in by_number}
    for number, deps in blockers.items():
        for blocker in deps:
            dependents[blocker].append(number)

    indegree = {number: len(deps) for number, deps in blockers.items()}
    ready = [number for number, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)

    ordered: list[PrdSlice] = []
    while ready:
        number = heapq.heappop(ready)
        ordered.append(by_number[number])
        for dependent in dependents[number]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)
    return ordered


async def _merge_or_escalate(
    slice_: PrdSlice,
    branch: str,
    *,
    git: GitOps,
    resolve_conflict: ConflictResolver | None,
) -> bool:
    """Merge a green slice; on a conflict, resolve-and-retry once or escalate.

    Returns True when the slice merged, False when it escalated (an unresolvable
    conflict, no resolver, or a retry that still conflicts).
    """
    try:
        await _merge_green_slice(slice_, branch, git=git)
        return True
    except MergeConflict as conflict:
        if resolve_conflict is None:
            logger.warning("Escalating %s: merge conflict, no resolver", slice_.branch)
            return False
        resolution = await resolve_conflict(source=conflict.source, into=conflict.into)
        if resolution is ConflictResolution.UNRESOLVED:
            logger.warning("Escalating %s: conflict unresolved", slice_.branch)
            return False
        return await _retry_merge_after_resolution(slice_, branch, git=git)


async def _retry_merge_after_resolution(
    slice_: PrdSlice, branch: str, *, git: GitOps
) -> bool:
    """Retry a merge after a claimed resolution; escalate if it still conflicts."""
    try:
        await _merge_green_slice(slice_, branch, git=git)
        return True
    except MergeConflict:
        logger.warning("Escalating %s: still conflicts after resolution", slice_.branch)
        return False
