"""Orchestrator: build one ready slice, or build a full PRD in dependency order.

The single-slice primitive (issue #6) is :func:`build_slice`. For one ready slice the
orchestrator runs the whole build inside **one disposable container** that is destroyed
on every path (:func:`_build_slice_in_container`):

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
import base64
import enum
import heapq
import json
import logging
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

from retinue.container import Container, ContainerRuntime, RunResult
from retinue.done_check import (
    DEFAULT_IMAGE,
    DoneCheckReport,
    ReportSink,
    SecretResolver,
    parse_done_check,
    resolve_secrets_or_escalate,
    run_done_check_commands,
)
from retinue.github_app import InstallationAuth
from retinue.repo_config import RepoConfig
from retinue.roles import Role, resolve_effort, resolve_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Slice:
    """One ready slice: a single issue an implementer builds on its own branch.

    Attributes:
        repo_full_name: The target repo, e.g. "owner/repo".
        issue_number: The slice's GitHub issue number; the implementer commits to
            the ``issue-<N>`` branch derived from it.
        prd_number: The parent PRD number; the integration branch is
            ``retinue/prd-<prd_number>``.
    """

    repo_full_name: str
    issue_number: int
    prd_number: int

    @property
    def branch(self) -> str:
        """The branch the implementer commits the slice to: ``issue-<N>``."""
        return f"issue-{self.issue_number}"


def integration_branch(prd_number: int) -> str:
    """The integration branch a PRD's slices are merged onto: ``retinue/prd-<n>``."""
    return f"retinue/prd-{prd_number}"


class Implementer(Protocol):
    """Spawns one implementer subagent that builds a slice. The Agent SDK seam.

    A production implementation execs the headless ``claude`` CLI *inside the disposable
    build container* the orchestrator passes in; the subagent implements TDD-first and
    commits to the slice's ``issue-<N>`` branch already checked out there. Tests inject a
    fake that records the request (and may mark the container log) without any real spawn.
    The contract is the commit on ``slice.branch``; the orchestrator does not read a
    return value, it gates on the done-check that follows.
    """

    async def implement(
        self, slice_: Slice, *, container: Container, plan_path: str | None = None
    ) -> None:
        """Build ``slice_`` in ``container``, committing to its ``issue-<N>`` branch.

        ``plan_path`` is the in-container path of a materialized implementation plan the
        subagent must read before building. The PRD lane passes nothing (``None``), so its
        prompt is unchanged; the ad-hoc lane passes its ``PLAN_FILE`` so the subagent is
        pointed at the plan the planner wrote.
        """
        ...

    def auth_env(self) -> dict[str, str]:
        """The env the agent authenticates with, merged into the container at start.

        Returned by the implementer (which owns the Anthropic credential) so the
        orchestrator can inject it into the build container's environment at ``start``
        without knowing how the credential is routed. A fake that needs no credential
        returns an empty mapping.
        """
        ...


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
# so a repo's ``models`` override can swap the model at the wiring site. The in-container
# ``claude`` CLI carries no effort flag today, so the ``high`` tier is registry metadata that
# records the PRD's intent without changing the wire.

# The implementer's brief, appended to the CLI's system prompt. Frozen (no per-slice
# interpolation) so the prefix is stable; the slice specifics ride in the per-slice prompt.
_IMPLEMENT_SYSTEM = (
    "You are an autonomous implementer. Build the requested GitHub issue inside the "
    "repository you are running in, test-driven: write or update a failing test first, "
    "then write code until it passes and the repo's own checks are green. Make the "
    "smallest change that satisfies the issue; do not refactor unrelated code. When the "
    "work is complete and committed to the issue's branch, stop."
)


def _implement_prompt(slice_: Slice, *, plan_path: str | None = None) -> str:
    """Assemble the per-slice prompt: which issue to build, on which branch.

    Names the target repo, the issue number to implement, and the ``issue-<N>`` branch the
    work must be committed to, so the spawned subagent commits where the orchestrator's
    merge seam expects to find it. When ``plan_path`` is given (the ad-hoc lane), the prompt
    leads with an instruction to read that materialized plan first; with no ``plan_path``
    (the PRD lane) the prompt is unchanged, so ``build_slice``/``build_prd`` are unaffected.
    """
    plan_preamble = (
        f"Read the implementation plan at '{plan_path}' first, then implement it. "
        if plan_path is not None
        else ""
    )
    return (
        f"{plan_preamble}"
        f"Implement issue #{slice_.issue_number} of {slice_.repo_full_name}. "
        f"Commit your work to the '{slice_.branch}' branch (already checked out). "
        "Implement test-driven and ensure the repo's checks pass before committing."
    )


def _implement_env(credential: str, auth_mode: str) -> dict[str, str]:
    """Build the env the ``claude`` CLI authenticates with, routing the credential by mode.

    ``api_key`` mode threads the credential as ``ANTHROPIC_API_KEY``; ``subscription`` mode
    threads it as ``CLAUDE_CODE_OAUTH_TOKEN`` (the Claude subscription OAuth env var the
    headless CLI reads). Only the credential env var is set here — the orchestrator merges
    it into the build container's environment at ``start``.
    """
    if auth_mode == "subscription":
        return {"CLAUDE_CODE_OAUTH_TOKEN": credential}
    return {"ANTHROPIC_API_KEY": credential}


def _claude_argv(*, prompt: str, model: str) -> list[str]:
    """Assemble the headless ``claude`` CLI argv for one in-container implement run.

    Runs non-interactively (``-p`` print mode), pins the implementing ``model``, accepts
    edits without a human in the loop (``--permission-mode acceptEdits`` — the run is
    autonomous), appends the frozen implementer brief to the system prompt, and emits a
    machine-readable json result so the exit can be cross-checked. The CLI runs in the
    container's working dir (the cloned repo), so no cwd flag is needed.
    """
    return [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--permission-mode",
        "acceptEdits",
        "--append-system-prompt",
        _IMPLEMENT_SYSTEM,
        "--output-format",
        "json",
    ]


def _claude_result_is_error(stdout: str) -> bool:
    """Whether the CLI's ``--output-format json`` result flags the run as errored.

    The headless CLI emits a json object carrying an ``is_error`` boolean. A non-json or
    empty stdout is not treated as an error here — the exit code is the primary signal —
    so this only catches a run that exited 0 yet reported an internal error.
    """
    try:
        result = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return False
    return bool(isinstance(result, dict) and result.get("is_error"))


class ImplementError(RuntimeError):
    """The container-exec implementer run ended in an error rather than a clean build.

    Distinct from a *clean-but-insufficient* build, which the orchestrator catches via the
    done-check that follows: this is the ``claude`` CLI exec itself failing (a non-zero exit
    code, or a json result flagged ``is_error``), so the slice surfaces the failure rather
    than proceeding to a done-check over a half-built tree.
    """


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
            repo's ``models`` override can replace at the wiring site.
    """

    credential: str
    auth_mode: str = "api_key"
    model: str = field(default_factory=lambda: resolve_model(Role.IMPLEMENTER))

    async def implement(
        self, slice_: Slice, *, container: Container, plan_path: str | None = None
    ) -> None:
        """Exec ``claude`` in ``container`` to build ``slice_``; raise on an errored run.

        ``plan_path``, when given, names a materialized plan the per-slice prompt instructs
        the subagent to read before building (the ad-hoc lane); the PRD lane passes nothing.
        """
        prompt = _implement_prompt(slice_, plan_path=plan_path)
        argv = _claude_argv(prompt=prompt, model=self.model)
        result = await container.run_command(argv)
        if not result.ok:
            raise ImplementError(
                f"implementer for {slice_.branch} exited {result.exit_code}: "
                f"{result.stderr}"
            )
        if _claude_result_is_error(result.stdout):
            raise ImplementError(
                f"implementer for {slice_.branch} reported an error: {result.stdout}"
            )
        logger.info("Implementer for %s completed in-container", slice_.branch)

    def auth_env(self) -> dict[str, str]:
        """The credential env the orchestrator merges into the build container at start."""
        return _implement_env(self.credential, self.auth_mode)


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

# Identity used for the merge commit ``git`` records. Merges are non-interactive, so a
# committer identity must be configured or ``git commit`` refuses to run; it is set
# per-command via ``-c`` rather than mutating global config in the shared workspace.
_GIT_AUTHOR_NAME = "the-retinue"
_GIT_AUTHOR_EMAIL = "retinue@users.noreply.github.com"
_GIT_IDENTITY_FLAGS = [
    "-c",
    f"user.name={_GIT_AUTHOR_NAME}",
    "-c",
    f"user.email={_GIT_AUTHOR_EMAIL}",
]

# The committer identity injected into the build container's env so the *agent's* own
# ``git commit`` (and the push) run non-interactively. The container env is fixed at
# ``start``, so the identity must ride it there rather than per-command ``-c`` flags the
# agent would not use; git reads these four vars without any repo config.
_GIT_COMMITTER_ENV = {
    "GIT_AUTHOR_NAME": _GIT_AUTHOR_NAME,
    "GIT_AUTHOR_EMAIL": _GIT_AUTHOR_EMAIL,
    "GIT_COMMITTER_NAME": _GIT_AUTHOR_NAME,
    "GIT_COMMITTER_EMAIL": _GIT_AUTHOR_EMAIL,
}

# Substrings git prints to stdout/stderr when a merge stops on a content conflict, as
# opposed to a hard error (unknown ref, not a repo, …). Matched case-insensitively.
_CONFLICT_MARKERS = (
    "conflict",
    "automatic merge failed",
    "fix conflicts and then commit",
)


def _clone_command(clone_url: str) -> list[str]:
    """Argv that clones the repo (over the installation-token URL) into the workspace."""
    return ["git", "clone", clone_url, "."]


def _push_branch_command(branch: str) -> list[str]:
    """Argv that pushes ``branch`` to ``origin`` (authenticated by the cloned remote URL)."""
    return ["git", "push", "origin", branch]


def _remote_branch_exists_command(branch: str) -> list[str]:
    """Argv that exits 0 iff ``branch`` already exists on ``origin``.

    The integration branch lives on ``origin`` (a prior run may have created it and a
    fresh merge-container clone has no local ref for it), so its existence is checked
    against the remote, not the local ``refs/heads``.
    """
    return ["git", "ls-remote", "--exit-code", "origin", f"refs/heads/{branch}"]


def _create_branch_commands(branch: str, base: str) -> list[list[str]]:
    """Argv list that creates ``branch`` off ``base`` and checks it out.

    ``base`` is referenced via ``origin/<base>`` so the branch is rooted on the freshly
    cloned remote tip rather than whatever happens to be checked out, then a local
    ``branch`` is created at that point and made current.
    """
    return [
        ["git", "fetch", "origin", base],
        ["git", "checkout", "-B", branch, f"origin/{base}"],
    ]


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
        for command in _create_branch_commands(branch, base):
            await self._run_checked(command, action=f"create {branch} off {base}")
        await self._run_checked(
            _push_branch_command(branch), action=f"push {branch} to origin"
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
                _push_branch_command(into), action=f"push {into} after merge"
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
        the round's merged surface to review. The branch tips are fetched from origin first
        so the diff reads the pushed work, not a stale local ref. An empty branch list
        yields an empty diff (nothing merged, nothing to review).
        """
        sections: list[str] = []
        for branch in merged_branches:
            await self._run_checked(
                ["git", "fetch", "origin", branch], action=f"fetch {branch} for diff"
            )
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


class GitOpsError(RuntimeError):
    """A ``git`` command failed for a reason other than a recoverable merge conflict.

    Distinct from :class:`MergeConflict`: a conflict is handed to the resolver, but a
    hard error (unknown ref, not a repository, checkout failure) means the integration
    branch could not be advanced at all, so it propagates rather than masquerading as a
    conflict the resolver could fix.
    """


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


# --- real Agent-SDK conflict resolver --------------------------------------------
#
# The production :class:`ConflictResolver` re-runs the failed merge inside the same
# disposable container ``build_prd`` already cloned into (the merge was aborted before
# :class:`MergeConflict` surfaced, so the conflict must be recreated), reads the
# conflicted blobs, drives Claude over the Messages API to emit a full resolution for
# each one, writes them back, and stages + commits the merge. Two collaborators are
# injected — the already-present :class:`~retinue.container.Container` (the only git/FS
# side effect) and an HTTP transport for the one Anthropic call — so the bug-prone pure
# parts (auth-header routing, request payload assembly, response parsing, and reading
# the conflicted-path list out of git) are unit-tested without a live container, Docker,
# gh, Claude, or network. The retried merge in ``build_prd`` is the real gate, so a
# resolver that produces an incomplete or wrong resolution still escalates rather than
# merging a broken tree.
#
# SDK conventions match the slicer/reviewer: Opus 4.8 is the default model; an OAuth
# subscription token (``sk-ant-oat...``) rides ``Authorization: Bearer`` with the
# ``oauth-2025-04-20`` beta header, any other credential is a raw API key on
# ``x-api-key``; ``anthropic-version`` is always sent. The model must return only the
# JSON object matching the schema — one resolved full-file body per conflicted path.

_ANTHROPIC_VERSION = "2023-06-01"
_RESOLVE_OAUTH_BETA = "oauth-2025-04-20"
_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_RESOLVE_MAX_TOKENS = 32_000

# The conflict resolver's model and effort tier come from the
# :data:`~retinue.roles.Role.RESOLVER` registry entry (Opus 4.8 at the ``xhigh`` tier by
# default — the same tier the slicer uses). Drawing both from the registry keeps the tier
# from silently drifting between the two Opus call sites and lets a repo's ``models``
# override swap the model at the wiring site.

# Re-runs the merge (no auto-commit) so the working tree carries the conflict markers
# the resolver reads; ``--no-commit`` keeps it stopped at the conflict even when git
# could otherwise auto-resolve, so the agent always sees the full picture.
_RESOLVE_MERGE_COMMAND = ["git", "merge", "--no-ff", "--no-commit", "--no-edit"]
# Lists every path with a merge conflict (``U`` in either status column).
_CONFLICTED_PATHS_COMMAND = ["git", "diff", "--name-only", "--diff-filter=U"]

# Strict JSON schema the resolver must emit: one entry per conflicted path carrying the
# complete resolved file body. ``resolved`` lists exactly the paths it fixed; the caller
# stages those and lets the retried merge verify nothing was left conflicted.
_RESOLVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "resolved": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["resolved"],
    "additionalProperties": False,
}

# The resolver's brief. Frozen (no per-request interpolation) so the request prefix is
# cacheable across conflicts; the conflicted blobs ride in the user message.
_RESOLVE_SYSTEM = (
    "You resolve git merge conflicts. Each file below contains conflict markers "
    "(<<<<<<<, =======, >>>>>>>). For every file, return its complete resolved "
    "content with all markers removed, integrating both sides faithfully so the "
    "result is correct and compiles — never drop one side's changes wholesale. "
    "Return only the JSON object matching the schema; emit one 'resolved' entry per "
    "input file, each with the file's full post-resolution body. No prose."
)


@dataclass(frozen=True)
class AnthropicResponse:
    """The slice of an Anthropic Messages API response the resolver reads."""

    status_code: int
    body: dict[str, Any]


class AnthropicTransport(Protocol):
    """Async HTTP POST seam (httpx-style). The network edge of the resolver.

    A production implementation wraps an httpx client; tests inject a fake returning a
    canned :class:`AnthropicResponse`. Kept narrow — one POST — so the resolution flow
    is exercisable without network.
    """

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any]
    ) -> AnthropicResponse:
        """POST ``json`` to ``url`` with ``headers`` and return the response."""
        ...


class ConflictResolutionError(RuntimeError):
    """The resolver could not obtain a usable resolution (bad API status / payload).

    Distinct from a *claimed-but-wrong* resolution, which is caught downstream by the
    retried merge: this is a hard failure to even produce a candidate (non-200 API
    response, unparseable body), so the conflict escalates rather than silently merging.
    """


@dataclass(frozen=True)
class _ConflictedFile:
    """One conflicted path and its working-tree blob (with conflict markers)."""

    path: str
    content: str


def _write_file_command(path: str, content: str) -> list[str]:
    """Argv that writes ``content`` to ``path`` inside the container, byte-exact.

    ``run_command`` execs the argv directly (no shell, no stdin), so arbitrary file
    bodies can't be passed as a here-doc or piped. The content is base64-encoded and
    decoded in-container via positional args (``$1``/``$2``) — never interpolated into
    the command string — so conflict markers, quotes, and newlines survive untouched and
    nothing in the file body is interpreted as shell syntax.
    """
    blob = base64.b64encode(content.encode()).decode()
    script = 'printf %s "$1" | base64 -d > "$2"'
    return ["sh", "-c", script, "sh", blob, path]


def _conflicted_paths(stdout: str) -> list[str]:
    """Parse ``git diff --name-only --diff-filter=U`` output into a path list.

    One path per non-blank line; surrounding whitespace is trimmed. An empty result
    means git reported no conflicted paths, which the caller treats as nothing to do.
    """
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def _resolve_headers(credential: str) -> dict[str, str]:
    """Build the Messages API request headers, routing the credential to its scheme.

    An OAuth subscription token (``sk-ant-oat...``) is sent as ``Authorization: Bearer``
    with the OAuth beta header; any other value is treated as a raw API key on
    ``x-api-key``. ``anthropic-version`` is always sent.
    """
    headers = {
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    if credential.startswith("sk-ant-oat"):
        headers["authorization"] = f"Bearer {credential}"
        headers["anthropic-beta"] = _RESOLVE_OAUTH_BETA
    else:
        headers["x-api-key"] = credential
    return headers


def _resolve_payload(
    files: list[_ConflictedFile], *, source: str, into: str, model: str
) -> dict[str, Any]:
    """Assemble the Messages API request body for one conflict resolution.

    The frozen system brief leads; the user message carries the merge context plus each
    conflicted file's full marked-up body, fenced by path so the model can address each
    one. The strict schema forces a per-file resolved body back. The request carries the
    resolver's registry effort tier (``xhigh``) via ``output_config.effort`` (Opus 4.8
    removed the extended-thinking ``budget_tokens`` mechanism, which now returns HTTP 400).
    """
    blocks = "\n\n".join(
        f"### {file.path}\n```\n{file.content}\n```" for file in files
    )
    user = (
        f"Merging branch '{source}' into '{into}' hit conflicts in "
        f"{len(files)} file(s). Resolve each and return the full resolved body:\n\n"
        f"{blocks}"
    )
    return {
        "model": model,
        "max_tokens": _RESOLVE_MAX_TOKENS,
        "output_config": {"effort": resolve_effort(Role.RESOLVER)},
        "system": _RESOLVE_SYSTEM,
        "messages": [{"role": "user", "content": user}],
        "response_format": {"type": "json_schema", "json_schema": _RESOLVE_SCHEMA},
    }


def _parse_resolution(body: dict[str, Any]) -> dict[str, str]:
    """Parse a Messages API response body into a ``{path: resolved_content}`` map.

    Reads the concatenated ``text`` content blocks, loads them as the schema JSON, and
    maps each entry's path to its resolved body. A response missing text, carrying
    malformed JSON, or a non-list ``resolved`` raises :class:`ConflictResolutionError`
    rather than silently resolving nothing (which would let the retried merge merge an
    unresolved tree only if git could auto-resolve — never on a real conflict).
    """
    text = "".join(
        block.get("text", "")
        for block in body.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )
    if not text.strip():
        raise ConflictResolutionError("resolver response carried no text content")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConflictResolutionError(f"resolver emitted invalid JSON: {exc}") from exc
    raw = parsed.get("resolved")
    if not isinstance(raw, list):
        raise ConflictResolutionError("resolver JSON missing a 'resolved' list")
    return {str(item["path"]): str(item["content"]) for item in raw}


@dataclass(frozen=True)
class AgentSdkConflictResolver:
    """Real :class:`ConflictResolver`: resolve a merge conflict via the Agent SDK.

    Satisfies the resolver protocol ``(*, source, into) -> ConflictResolution`` via
    :meth:`__call__`, so it drops in where the fake resolver sits in tests and at the
    wiring site. It recreates the aborted merge in the injected container, reads the
    conflicted blobs, calls Claude once to resolve them, writes each resolution back,
    and stages + commits the merge — then returns ``RESOLVED``. With no conflicts to
    find it returns ``UNRESOLVED`` (nothing to fix); a hard API/parse failure raises
    :class:`ConflictResolutionError`. The retried merge in ``build_prd`` is the real
    gate, so an incomplete resolution still escalates rather than merging a broken tree.

    Attributes:
        container: The cloned-repo container the merge/resolution runs in.
        transport: The injected Anthropic HTTP POST seam.
        credential: The Anthropic credential (OAuth subscription token or API key).
        model: The resolving model id; defaults to the
            :data:`~retinue.roles.Role.RESOLVER` registry entry (Opus 4.8), which a
            repo's ``models`` override can replace at the wiring site.
    """

    container: Container
    transport: AnthropicTransport
    credential: str
    model: str = field(default_factory=lambda: resolve_model(Role.RESOLVER))

    async def __call__(self, *, source: str, into: str) -> ConflictResolution:
        """Resolve the conflict merging ``source`` into ``into``; stage and commit it."""
        files = await self._recreate_and_read(source)
        if not files:
            logger.warning("No conflicted paths found resolving %s into %s", source, into)
            return ConflictResolution.UNRESOLVED
        resolutions = await self._ask_claude(files, source=source, into=into)
        await self._apply(files, resolutions)
        logger.info("Resolved %d conflicted file(s) merging %s into %s", len(files), source, into)
        return ConflictResolution.RESOLVED

    async def _recreate_and_read(self, source: str) -> list[_ConflictedFile]:
        """Re-run the aborted merge to surface the conflict, then read each blob.

        The merge is expected to fail (that is the conflict being resolved), so its exit
        is not checked; the conflicted-path listing is the source of truth. Each path's
        working-tree content (carrying the conflict markers) is read back for the agent.
        """
        await self.container.run_command([*_RESOLVE_MERGE_COMMAND, f"origin/{source}"])
        listing = await self.container.run_command(_CONFLICTED_PATHS_COMMAND)
        files: list[_ConflictedFile] = []
        for path in _conflicted_paths(listing.stdout):
            blob = await self.container.run_command(["cat", path])
            files.append(_ConflictedFile(path=path, content=blob.stdout))
        return files

    async def _ask_claude(
        self, files: list[_ConflictedFile], *, source: str, into: str
    ) -> dict[str, str]:
        """Call the Messages API once and parse the per-path resolution map."""
        response = await self.transport.post(
            _ANTHROPIC_MESSAGES_URL,
            headers=_resolve_headers(self.credential),
            json=_resolve_payload(files, source=source, into=into, model=self.model),
        )
        if response.status_code != 200:
            raise ConflictResolutionError(
                f"Anthropic Messages API returned {response.status_code}"
            )
        return _parse_resolution(response.body)

    async def _apply(
        self, files: list[_ConflictedFile], resolutions: dict[str, str]
    ) -> None:
        """Write each resolved body back, stage it, then commit the merge.

        Only the paths the resolver actually returned are written and staged; any
        conflicted path it omitted is left unresolved, so the retried merge catches the
        gap and the slice escalates rather than committing a partial resolution.
        """
        for file in files:
            content = resolutions.get(file.path)
            if content is None:
                logger.warning("Resolver omitted conflicted path %s", file.path)
                continue
            await self.container.run_command(_write_file_command(file.path, content))
            await self.container.run_command(["git", "add", file.path])
        await self.container.run_command(
            ["git", *_GIT_IDENTITY_FLAGS, "commit", "--no-edit"]
        )


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
    container (:func:`_build_slice_in_container`): clone, check out a fresh ``issue-<N>``
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

    passed = await _build_slice_in_container(
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

    await _merge_green_slice(slice_, branch, config=config, git=git)
    return BuildResult(outcome=BuildOutcome.MERGED, integration_branch=branch)


async def _build_slice_in_container(
    slice_: Slice,
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
) -> bool:
    """Run one slice's full build in a single disposable container; return green/red.

    Owns the whole per-slice lifecycle, destroying the container on every path:

    1. parse the done-check and resolve the config's secrets (a missing one escalates on
       the report sink and propagates *before* any container starts),
    2. start the container with the secrets, the git committer identity, and the agent's
       credential all in its env (the env is fixed at ``start``),
    3. clone the repo and check out a fresh ``issue-<N>`` branch off ``base`` — the
       integration branch ``retinue/prd-<n>`` (already created off staging at build start),
       so a later-round slice builds on its already-merged sibling work,
    4. exec the implementer (``claude``) inside the container to build and commit the slice,
    5. run the done-check over the real changes and post the outcome,
    6. push ``issue-<N>`` to ``origin`` only when the done-check is green (so the merge seam
       can fetch it; a red slice pushes nothing).

    Returns:
        True only when the done-check passed (and the branch was pushed); False on red.
    """
    commands = parse_done_check(claude_md)
    env = await resolve_secrets_or_escalate(
        slice_.repo_full_name, slice_.issue_number, config, resolve_secret, report
    )
    auth_env = implementer.auth_env()
    start_env = {**env, **_GIT_COMMITTER_ENV, **auth_env}
    # The exact secret values injected into the container, scrubbed from a failing
    # done-check's report (repo-declared secrets plus the auth credential).
    secret_values = [*env.values(), *auth_env.values()]
    token = await auth.installation_token(slice_.repo_full_name)
    container = await runtime.start(image=image, env=start_env)
    try:
        await _clone_and_branch(
            container, token.clone_url, branch=slice_.branch, base=base
        )
        await implementer.implement(slice_, container=container)
        passed, detail = await run_done_check_commands(
            container, commands, secret_values=secret_values
        )
        if passed:
            await _push_branch(container, slice_.branch)
        await report(
            DoneCheckReport(
                repo_full_name=slice_.repo_full_name,
                issue_number=slice_.issue_number,
                passed=passed,
                escalated=False,
                detail=detail,
            )
        )
        logger.info(
            "Slice %s done-check %s", slice_.branch, "passed" if passed else "failed"
        )
        return passed
    finally:
        # Guaranteed teardown: the disposable container is destroyed on every path,
        # including when clone, implement, the done-check, the push, or report raises.
        await container.destroy()


async def _clone_and_branch(
    container: Container, clone_url: str, *, branch: str, base: str
) -> None:
    """Clone the repo into ``container`` and check out a fresh ``branch`` off ``base``."""
    clone = await container.run_command(_clone_command(clone_url))
    if not clone.ok:
        raise GitOpsError(f"clone failed (exit {clone.exit_code}): {clone.stderr}")
    for command in _create_branch_commands(branch, base):
        result = await container.run_command(command)
        if not result.ok:
            raise GitOpsError(
                f"failed to create slice branch {branch} off {base} "
                f"(exit {result.exit_code}): {result.stderr}"
            )


async def _push_branch(container: Container, branch: str) -> None:
    """Push ``branch`` to ``origin`` from inside ``container``; raise on failure."""
    result = await container.run_command(_push_branch_command(branch))
    if not result.ok:
        raise GitOpsError(
            f"failed to push {branch} (exit {result.exit_code}): {result.stderr}"
        )


async def _merge_green_slice(
    slice_: Slice, branch: str, *, config: RepoConfig, git: GitOps
) -> None:
    """Ensure the integration branch and merge a green slice onto it."""
    await git.ensure_integration_branch(branch=branch, base=config.staging_branch)
    await git.merge(source=slice_.branch, into=branch)
    logger.info("Merged %s into %s after green done-check", slice_.branch, branch)


# --- full-PRD driver (issue #7) --------------------------------------------------


class OrchestratorBusyError(Exception):
    """A second orchestrator run was attempted while one is already in flight.

    The single-run guarantee: :func:`build_prd` runs inside an injected lock that
    rejects a concurrent holder. Catching this (rather than blocking) makes the
    "at most one run at a time" contract observable to the caller.
    """

    def __init__(self) -> None:
        super().__init__("an orchestrator run is already in flight")


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
    """

    integration_branch: str
    merged_issues: list[int]
    blocked_issues: list[int]
    escalated_issues: list[int]
    skipped_issues: list[int]


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
        lock: The single-run lock; entering it raises :class:`OrchestratorBusyError`
            when a run is already in flight.
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
        OrchestratorBusyError: A run is already in flight (from the injected lock).
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
                config=config,
                git=git,
                resolve_conflict=resolve_conflict,
                state=state,
            )
            # After the round merges, the internal reviewer reviews the round's merged
            # diff and files review-fix follow-ups; each comes back as a slice that
            # enters a later round's ready set and builds in the same run.
            if review_round is not None and merged_this_round:
                fixes = await review_round.review(merged_issues=merged_this_round)
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
) -> list[tuple[PrdSlice, bool]]:
    """Fan out the ready slices, bounded by ``max_parallel``; return (slice, passed).

    Each slice's build runs in its own disposable container (clone → branch off ``base``
    → implement → done-check → push-on-green), concurrently across the round but capped at
    ``config.max_parallel`` live builds. ``base`` is the integration branch, so a later
    round branches off the sibling work an earlier round already merged onto it. Only the
    build phase runs here; merges are serialized afterwards in dependency order, against
    the branches the green builds pushed.
    """
    semaphore = asyncio.Semaphore(config.max_parallel or len(ready) or 1)

    async def build_one(slice_: PrdSlice) -> tuple[PrdSlice, bool]:
        async with semaphore:
            passed = await _build_slice_in_container(
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
            return slice_, passed

    return await asyncio.gather(*(build_one(s) for s in ready))


async def _merge_round(
    built: list[tuple[PrdSlice, bool]],
    branch: str,
    *,
    config: RepoConfig,
    git: GitOps,
    resolve_conflict: ConflictResolver | None,
    state: _PrdState,
) -> list[int]:
    """Merge a round's green slices in dependency order; return the issues merged.

    Merges are serialized (a shared integration branch) and ordered by a Kahn
    topological sort over the round's ``blocked_by`` graph, so an in-round blocker
    always merges before its dependent; issue number only breaks ties between
    mutually independent slices, keeping the order deterministic. A red slice is
    recorded blocked; a conflict is resolved under the done-check or escalated. The
    returned list is *this* round's merged issue numbers in merge order — the surface
    the per-round reviewer reviews.
    """
    passed_by_number = {slice_.issue_number: passed for slice_, passed in built}
    merged_this_round: list[int] = []
    for slice_ in _topo_merge_order([slice_ for slice_, _ in built]):
        if not passed_by_number[slice_.issue_number]:
            state.record(slice_, state.blocked)
            continue
        merged = await _merge_or_escalate(
            slice_, branch, config=config, git=git, resolve_conflict=resolve_conflict
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
    config: RepoConfig,
    git: GitOps,
    resolve_conflict: ConflictResolver | None,
) -> bool:
    """Merge a green slice; on a conflict, resolve-and-retry once or escalate.

    Returns True when the slice merged, False when it escalated (an unresolvable
    conflict, no resolver, or a retry that still conflicts).
    """
    try:
        await _merge_green_slice(slice_, branch, config=config, git=git)
        return True
    except MergeConflict as conflict:
        if resolve_conflict is None:
            logger.warning("Escalating %s: merge conflict, no resolver", slice_.branch)
            return False
        resolution = await resolve_conflict(source=conflict.source, into=conflict.into)
        if resolution is ConflictResolution.UNRESOLVED:
            logger.warning("Escalating %s: conflict unresolved", slice_.branch)
            return False
        return await _retry_merge_after_resolution(slice_, branch, config=config, git=git)


async def _retry_merge_after_resolution(
    slice_: PrdSlice, branch: str, *, config: RepoConfig, git: GitOps
) -> bool:
    """Retry a merge after a claimed resolution; escalate if it still conflicts."""
    try:
        await _merge_green_slice(slice_, branch, config=config, git=git)
        return True
    except MergeConflict:
        logger.warning("Escalating %s: still conflicts after resolution", slice_.branch)
        return False
