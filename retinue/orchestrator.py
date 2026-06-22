"""Orchestrator: build one ready slice, or build a full PRD in dependency order.

The single-slice primitive (issue #6) is :func:`build_slice`. For one ready slice the
orchestrator:

1. **spawn** — runs one implementer subagent (the Agent SDK seam) in an isolated git
   worktree inside the disposable container; it implements TDD-first and commits to
   an ``issue-<N>`` branch,
2. **done-check** — runs the repo's done-check via :func:`retinue.done_check.run_done_check`
   (auth -> clone -> inject -> run -> report -> teardown), which yields a pass/fail,
3. **merge** — only on a green done-check, ensures the integration branch
   ``retinue/prd-<n>`` exists (created off the config's ``staging_branch`` when absent)
   and merges ``issue-<N>`` into it. A red done-check **blocks** the merge: no red
   slice is ever merged.

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
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from retinue.container import Container, ContainerRuntime, RunResult
from retinue.done_check import (
    DEFAULT_IMAGE,
    ReportSink,
    SecretResolver,
    run_done_check,
)
from retinue.github_app import InstallationAuth
from retinue.repo_config import RepoConfig

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

    A production implementation spawns a Claude Agent-SDK subagent in an isolated git
    worktree inside the disposable container; the subagent implements TDD-first and
    commits to the slice's ``issue-<N>`` branch. Tests inject a fake that records the
    request without any real spawn. The contract is the commit on ``slice.branch``;
    the orchestrator does not read a return value, it gates on the done-check that
    follows.
    """

    async def implement(self, slice_: Slice) -> None:
        """Build ``slice_``, committing the work to its ``issue-<N>`` branch."""
        ...


# --- real Agent-SDK implementer (production adapter behind the Implementer seam) ---
#
# The production :class:`Implementer` spawns a Claude Agent-SDK subagent (the headless
# ``claude`` CLI the SDK drives) inside the already-cloned repo and lets it implement the
# slice TDD-first, edit files, and commit to the ``issue-<N>`` branch. The one side effect
# is the spawn, so it is taken behind the injected ``query`` seam (default:
# ``claude_agent_sdk.query``) exactly as the slicer/reviewer/resolver inject their model
# boundary — tests script the message stream without a live model, the SDK, Docker, gh, or
# network. The bug-prone pure parts — prompt assembly, option construction (cwd, model, and
# the api_key-vs-subscription env auth routing), and reading the terminal result — are
# factored into the free functions/methods below so they are unit-tested in isolation.
#
# Auth mirrors :class:`retinue.config.Settings`: ``auth_mode="api_key"`` threads the
# credential to the spawned CLI as ``ANTHROPIC_API_KEY``; ``auth_mode="subscription"``
# threads it as ``CLAUDE_CODE_OAUTH_TOKEN`` (the subscription OAuth env var). The contract
# is the commit on ``slice.branch``; the orchestrator gates on the done-check that follows,
# so a run the subagent finishes "successfully" but that fails to satisfy the repo is still
# caught downstream. A spawn that the SDK reports as errored raises :class:`ImplementError`.

_IMPLEMENT_MODEL = "claude-opus-4-8"
# The cloned-repo directory the done-check clones into and the implementer commits within;
# matches :data:`retinue.container.WORKSPACE_DIR` so the subagent runs over the same tree.
_IMPLEMENT_WORKSPACE = "/workspace"

# The implementer's brief. Frozen (no per-slice interpolation) so the spawned subagent's
# system prefix is stable; the slice specifics ride in the per-slice prompt.
_IMPLEMENT_SYSTEM = (
    "You are an autonomous implementer. Build the requested GitHub issue inside the "
    "repository you are running in, test-driven: write or update a failing test first, "
    "then write code until it passes and the repo's own checks are green. Make the "
    "smallest change that satisfies the issue; do not refactor unrelated code. When the "
    "work is complete and committed to the issue's branch, stop."
)


def _implement_prompt(slice_: Slice) -> str:
    """Assemble the per-slice prompt: which issue to build, on which branch.

    Names the target repo, the issue number to implement, and the ``issue-<N>`` branch the
    work must be committed to, so the spawned subagent commits where the orchestrator's
    merge seam expects to find it.
    """
    return (
        f"Implement issue #{slice_.issue_number} of {slice_.repo_full_name}. "
        f"Commit your work to the '{slice_.branch}' branch (create it off the current "
        "checkout if it does not exist). Implement test-driven and ensure the repo's "
        "checks pass before committing."
    )


def _implement_env(credential: str, auth_mode: str) -> dict[str, str]:
    """Build the env the spawned CLI authenticates with, routing the credential by mode.

    ``api_key`` mode threads the credential as ``ANTHROPIC_API_KEY``; ``subscription`` mode
    threads it as ``CLAUDE_CODE_OAUTH_TOKEN`` (the Claude subscription OAuth env var the
    headless CLI reads). Only the credential env var is set here — the SDK merges it onto
    the parent environment when it spawns the CLI.
    """
    if auth_mode == "subscription":
        return {"CLAUDE_CODE_OAUTH_TOKEN": credential}
    return {"ANTHROPIC_API_KEY": credential}


class ImplementError(RuntimeError):
    """The Agent-SDK implementer spawn ended in an error rather than a clean build.

    Distinct from a *clean-but-insufficient* build, which the orchestrator catches via the
    done-check that follows: this is the SDK reporting the run itself failed (a
    ``ResultMessage`` with ``is_error``), so the slice surfaces the failure rather than
    proceeding to a done-check over a half-built tree.
    """


# The injected spawn seam: ``claude_agent_sdk.query(prompt=..., options=...)`` returns an
# async iterator of SDK messages. Kept narrow so the implement flow is exercisable without
# the SDK, a live model, or a real clone; production binds the real ``query``.
QuerySeam = Callable[..., AsyncIterator[Any]]
# Builds the SDK ``ClaudeAgentOptions`` from the assembled kwargs. Injected alongside the
# ``query`` seam so the option-kwargs assembly is tested without the SDK installed; the
# default lazily constructs the real ``ClaudeAgentOptions``.
OptionsFactory = Callable[..., Any]


def _implement_option_kwargs(
    *, credential: str, auth_mode: str, model: str, workspace: str
) -> dict[str, Any]:
    """Assemble the ``ClaudeAgentOptions`` kwargs for one spawn — the pure, testable core.

    Runs the subagent in the cloned-repo ``workspace`` (so edits and the commit land in the
    tree the done-check clones), pins the implementing ``model``, leads with the frozen
    implementer brief, accepts edits non-interactively (no human approves the autonomous
    run), and threads the credential to the spawned CLI via the env var its ``auth_mode``
    selects.
    """
    return {
        "cwd": workspace,
        "model": model,
        "system_prompt": _IMPLEMENT_SYSTEM,
        "permission_mode": "acceptEdits",
        "env": _implement_env(credential, auth_mode),
    }


@dataclass(frozen=True)
class AgentSdkImplementer:
    """Real :class:`Implementer`: build a slice by spawning a Claude Agent-SDK subagent.

    Satisfies the implementer protocol ``implement(slice_) -> None`` so it drops in where
    the fake implementer sits in tests and at the wiring site. It spawns the headless
    ``claude`` CLI (via the injected ``query`` seam) inside the cloned repo, hands it the
    per-slice prompt, and lets it implement TDD-first and commit to ``slice.branch``. The
    orchestrator gates on the done-check that follows, so the contract here is only that the
    spawn ran and committed; a run the SDK reports as errored raises :class:`ImplementError`.

    Attributes:
        credential: The Anthropic credential (API key or subscription OAuth token).
        auth_mode: ``"api_key"`` (credential rides ``ANTHROPIC_API_KEY``) or
            ``"subscription"`` (credential rides ``CLAUDE_CODE_OAUTH_TOKEN``).
        query: The Agent-SDK spawn seam; defaults to ``claude_agent_sdk.query``.
        options_factory: Builds the SDK options from the assembled kwargs; defaults to the
            real ``ClaudeAgentOptions``. Injected so the spawn runs without the SDK in tests.
        model: The implementing model id; defaults to Opus 4.8.
        workspace: The cloned-repo directory the subagent runs in; the SDK's ``cwd``.
    """

    credential: str
    auth_mode: str = "api_key"
    query: QuerySeam | None = None
    options_factory: OptionsFactory | None = None
    model: str = _IMPLEMENT_MODEL
    workspace: str = _IMPLEMENT_WORKSPACE

    async def implement(self, slice_: Slice) -> None:
        """Spawn the subagent to build ``slice_``; raise on an errored run."""
        query = self._resolve_query()
        options = self._build_options()
        result = None
        async for message in query(prompt=_implement_prompt(slice_), options=options):
            if _is_result_message(message):
                result = message
        if result is not None and getattr(result, "is_error", False):
            raise ImplementError(
                f"implementer spawn for {slice_.branch} ended in error: "
                f"{getattr(result, 'result', None)}"
            )
        logger.info("Implementer spawn for %s completed", slice_.branch)

    def _resolve_query(self) -> QuerySeam:
        """The injected spawn seam, or the real ``claude_agent_sdk.query`` lazily imported.

        The lazy import keeps the module (and the unit tests over the pure prompt/option
        helpers) import-clean without the Agent SDK installed; type-ignored because the SDK
        is an optional runtime dependency.
        """
        if self.query is not None:
            return self.query
        from claude_agent_sdk import query  # type: ignore[import-not-found,unused-ignore]

        return cast(QuerySeam, query)

    def _build_options(self) -> Any:
        """Build the SDK options object from the assembled kwargs via the options factory.

        The kwargs are assembled by the pure :func:`_implement_option_kwargs`; the factory
        (default: the real ``ClaudeAgentOptions``, lazily imported) turns them into the SDK
        object. Tests inject a passthrough factory so the kwargs are asserted without the SDK.
        """
        kwargs = _implement_option_kwargs(
            credential=self.credential,
            auth_mode=self.auth_mode,
            model=self.model,
            workspace=self.workspace,
        )
        factory = self.options_factory or self._default_options_factory()
        return factory(**kwargs)

    @staticmethod
    def _default_options_factory() -> OptionsFactory:
        """Lazily import the real ``ClaudeAgentOptions`` as the default options factory."""
        from claude_agent_sdk import (  # type: ignore[import-not-found,unused-ignore]
            ClaudeAgentOptions,
        )

        return cast(OptionsFactory, ClaudeAgentOptions)


def _is_result_message(message: Any) -> bool:
    """Whether an SDK message is the terminal ``ResultMessage`` carrying ``is_error``.

    Matched structurally (the message exposes ``is_error``) rather than by importing the
    SDK type, so the result-reading path is unit-testable with a plain stand-in and the
    module imports without the SDK present.
    """
    return hasattr(message, "is_error")


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

# Substrings git prints to stdout/stderr when a merge stops on a content conflict, as
# opposed to a hard error (unknown ref, not a repo, …). Matched case-insensitively.
_CONFLICT_MARKERS = (
    "conflict",
    "automatic merge failed",
    "fix conflicts and then commit",
)


def _branch_exists_command(branch: str) -> list[str]:
    """Argv that exits 0 iff local ``branch`` already exists in the workspace."""
    return ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"]


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

    The integration-branch merges happen inside the same disposable container the
    done-check cloned the repo into, so this seam has no external dependency of its own
    beyond the injected :class:`~retinue.container.Container`. A merge that stops on a
    content conflict is aborted (to leave the workspace clean for the resolver) and
    surfaced as :class:`MergeConflict`; any other ``git`` failure is a hard
    :class:`GitOpsError`, never silently swallowed.
    """

    def __init__(self, container: Container) -> None:
        self._container = container

    async def ensure_integration_branch(self, *, branch: str, base: str) -> None:
        """Ensure ``branch`` exists locally, creating it off ``origin/<base>`` if absent."""
        exists = await self._container.run_command(_branch_exists_command(branch))
        if exists.ok:
            logger.info("Integration branch %s already exists", branch)
            return
        for command in _create_branch_commands(branch, base):
            await self._run_checked(command, action=f"create {branch} off {base}")
        logger.info("Created integration branch %s off %s", branch, base)

    async def merge(self, *, source: str, into: str) -> None:
        """Merge ``source`` into ``into``; raise :class:`MergeConflict` on a conflict.

        Runs checkout + fetch + merge. A merge that stops on a content conflict is
        aborted to leave the workspace clean, then raised as :class:`MergeConflict`; any
        other non-zero ``git`` exit is a :class:`GitOpsError`.
        """
        commands = _merge_commands(source, into)
        for command in commands[:-1]:
            await self._run_checked(command, action=f"prepare merge of {source}")
        result = await self._container.run_command(commands[-1])
        if result.ok:
            logger.info("Merged %s into %s", source, into)
            return
        if _is_merge_conflict(result):
            await self._container.run_command(_ABORT_MERGE_COMMAND)
            raise MergeConflict(source=source, into=into)
        raise GitOpsError(
            f"git merge of {source} into {into} failed "
            f"(exit {result.exit_code}): {result.stderr}"
        )

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

_RESOLVE_MODEL = "claude-opus-4-8"
_ANTHROPIC_VERSION = "2023-06-01"
_RESOLVE_OAUTH_BETA = "oauth-2025-04-20"
_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_RESOLVE_MAX_TOKENS = 32_000

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
    one. The strict schema forces a per-file resolved body back.
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
        model: The resolving model id; defaults to Opus 4.8.
    """

    container: Container
    transport: AnthropicTransport
    credential: str
    model: str = _RESOLVE_MODEL

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
    """Build one ready slice: spawn the implementer, gate on the done-check, merge.

    The implementer builds and commits the slice to its ``issue-<N>`` branch, then the
    repo's done-check runs in a fresh disposable container. The done-check result gates
    the merge: a green check merges ``issue-<N>`` into the integration branch
    ``retinue/prd-<n>`` (created off ``config.staging_branch`` if absent), while a red
    check blocks the merge so no failing slice is ever integrated.

    Args:
        slice_: The ready slice to build (repo, issue number, PRD number).
        config: The accepted repo config; its ``staging_branch`` is the base for a
            new integration branch and its ``secrets`` are injected into the check.
        claude_md: The repo's ``CLAUDE.md`` text, carrying the done-check command.
        implementer: Spawns the implementer subagent (the Agent SDK seam).
        git: Integration-branch git operations (the merge seam).
        auth: Mints the installation token used to clone (the auth seam).
        runtime: Spawns the disposable container the done-check runs in (Docker seam).
        resolve_secret: Resolves the config's declared secret names/refs to values.
        report: Sink the done-check outcome is posted to (commit status / comment).
        image: Container image the done-check runs in.

    Returns:
        A :class:`BuildResult`: ``MERGED`` when the green slice was merged, or
        ``BLOCKED`` when a red done-check stopped it.

    Raises:
        Propagates whatever ``run_done_check`` raises (e.g. a missing secret), and any
        merge error the git seam raises on a conflict.
    """
    branch = integration_branch(slice_.prd_number)

    await implementer.implement(slice_)
    passed = await _run_slice_done_check(
        slice_,
        config,
        claude_md,
        auth=auth,
        runtime=runtime,
        resolve_secret=resolve_secret,
        report=report,
        image=image,
    )

    if not passed:
        # A red slice is never merged: leave the integration branch untouched.
        logger.info(
            "Blocking merge of %s into %s: done-check failed",
            slice_.branch,
            branch,
        )
        return BuildResult(outcome=BuildOutcome.BLOCKED, integration_branch=branch)

    await _merge_green_slice(slice_, branch, config=config, git=git)
    return BuildResult(outcome=BuildOutcome.MERGED, integration_branch=branch)


async def _run_slice_done_check(
    slice_: Slice,
    config: RepoConfig,
    claude_md: str,
    *,
    auth: InstallationAuth,
    runtime: ContainerRuntime,
    resolve_secret: SecretResolver,
    report: ReportSink,
    image: str,
) -> bool:
    """Run the repo's done-check for ``slice_``; return True only when it is green."""
    check = await run_done_check(
        slice_.repo_full_name,
        config,
        claude_md,
        auth=auth,
        runtime=runtime,
        resolve_secret=resolve_secret,
        report=report,
        image=image,
    )
    return check.passed


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
    image: str = DEFAULT_IMAGE,
) -> PrdBuildResult:
    """Build a full PRD: ready set -> parallel fan-out -> topological merge -> loop.

    Runs under ``lock`` so at most one orchestrator run executes at a time. Each round
    picks the ready set (every ``blocked_by`` merged this run or already merged/closed
    before it), fans the slices out to implementers bounded by ``config.max_parallel``,
    then merges the green branches in dependency order under the done-check — resolving
    a conflict or escalating. Rounds repeat until no ready slice remains.

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
        image: Container image the done-check runs in.

    Returns:
        A :class:`PrdBuildResult` partitioning the slices into
        merged/blocked/escalated/skipped — a subtree pruned by a failed upstream
        slice lands in ``skipped``, so every slice is accounted for.

    Raises:
        OrchestratorBusyError: A run is already in flight (from the injected lock).
    """
    branch = integration_branch(slices[0].prd_number) if slices else integration_branch(0)
    async with lock:
        state = _PrdState(slices)
        while True:
            ready = state.ready_set()
            if not ready:
                break
            built = await _build_round(
                ready,
                config,
                claude_md,
                implementer=implementer,
                auth=auth,
                runtime=runtime,
                resolve_secret=resolve_secret,
                report=report,
                image=image,
            )
            await _merge_round(
                built,
                branch,
                config=config,
                git=git,
                resolve_conflict=resolve_conflict,
                state=state,
            )
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
    implementer: Implementer,
    auth: InstallationAuth,
    runtime: ContainerRuntime,
    resolve_secret: SecretResolver,
    report: ReportSink,
    image: str,
) -> list[tuple[PrdSlice, bool]]:
    """Fan out the ready slices, bounded by ``max_parallel``; return (slice, passed).

    Each slice's implementer runs then its done-check runs, concurrently across the
    round but capped at ``config.max_parallel`` live builds. Only the done-check phase
    runs here; merges are serialized afterwards in dependency order.
    """
    semaphore = asyncio.Semaphore(config.max_parallel or len(ready) or 1)

    async def build_one(slice_: PrdSlice) -> tuple[PrdSlice, bool]:
        async with semaphore:
            await implementer.implement(slice_)
            passed = await _run_slice_done_check(
                slice_,
                config,
                claude_md,
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
) -> None:
    """Merge a round's green slices in dependency order, recording each outcome.

    Merges are serialized (a shared integration branch) and ordered by issue number,
    which respects the ``blocked_by`` graph since a slice is only ready once its
    blockers merged. A red slice is recorded blocked; a conflict is resolved under the
    done-check or escalated.
    """
    for slice_, passed in sorted(built, key=lambda item: item[0].issue_number):
        if not passed:
            state.record(slice_, state.blocked)
            continue
        merged = await _merge_or_escalate(
            slice_, branch, config=config, git=git, resolve_conflict=resolve_conflict
        )
        state.record(slice_, state.merged if merged else state.escalated)


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
