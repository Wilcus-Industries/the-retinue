"""Tests for the single-slice orchestrator (issue #6).

The flow is spawn-implementer -> done-check -> merge-or-block. Every collaborator is
faked: a fake implementer records that it was asked to build the slice, the done-check
runs against the faked container/auth seams reused from the done-check tests, and a fake
git-ops records branch creation and merges. No Agent SDK, no Docker, no gh, no network.

A green done-check merges ``issue-<N>`` into the integration branch ``retinue/prd-<n>``
(created off ``staging`` if absent); a red done-check blocks the merge.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Mapping

import pytest

from retinue.classifier import ClassifyInput
from retinue.container import Container, RunResult
from retinue.done_check import DoneCheckReport, MissingSecretError
from retinue.orchestrator import (
    _DEFAULT_IMPLEMENT_MAX_TURNS,
    _IMPLEMENT_SYSTEM,
    _RESOLVE_SYSTEM,
    AgentSdkConflictResolver,
    AnthropicResponse,
    BuildOutcome,
    BuildResult,
    ConflictResolution,
    ConflictResolutionError,
    ContainerGitOps,
    ContainerImplementer,
    GitOpsError,
    Implementer,
    ImplementError,
    MergeConflict,
    Slice,
    _claude_argv,
    _claude_result_is_error,
    _conflicted_paths,
    _ConflictedFile,
    _implement_env,
    _implement_prompt,
    _parse_resolution,
    _resolve_headers,
    _resolve_payload,
    _write_file_command,
    build_slice,
    integration_branch,
)
from retinue.repo_config import (
    ModelEffort,
    RepoConfig,
    RoutingConfig,
    RoutingLevel,
    SecretsConfig,
)
from retinue.roles import CLAUDE_CODE_IDENTITY, ROLE_REGISTRY, Role, resolve_effort, resolve_model
from retinue.slicer import _EFFORT_XHIGH
from tests.test_done_check import (
    CLAUDE_MD,
    FakeAuth,
    FakeRuntime,
    _resolver,
    _sink,
)


class FakeImplementer:
    """Records the slice it was asked to build and marks the build container.

    Appends an ``implement`` marker to the container log so the per-slice lifecycle
    order (clone -> checkout -> implement -> done-check -> push) is assertable, and
    returns an empty ``auth_env`` (a fake needs no real Anthropic credential).
    """

    def __init__(self) -> None:
        self.built: list[Slice] = []

    async def implement(
        self, slice_: Slice, *, container: Container, plan_path: str | None = None
    ) -> None:
        self.built.append(slice_)
        await container.run_command(["implement", slice_.branch])

    def auth_env(self) -> dict[str, str]:
        return {}


class FakeGitOps:
    """In-memory git: records branch creation and merges, scripts existence/conflicts.

    ``existing`` is the set of branches that already exist on the remote. ``conflicts``
    is the set of source branches whose merge should raise (a conflict the orchestrator
    surfaces). ``log`` records each ensure/merge event in order.
    """

    def __init__(
        self,
        existing: set[str] | None = None,
        conflicts: set[str] | None = None,
        timeline: list[str] | None = None,
    ) -> None:
        self.existing = set(existing or set())
        self._conflicts = set(conflicts or set())
        self.log: list[str] = []
        self.merges: list[tuple[str, str]] = []
        # Optional shared event list, written to by both this seam and the runtime, so a
        # test can assert ordering *across* the git and container seams.
        self._timeline = timeline

    async def ensure_integration_branch(self, *, branch: str, base: str) -> None:
        if branch in self.existing:
            self.log.append(f"exists:{branch}")
            return
        event = f"create:{branch}<-{base}"
        self.log.append(event)
        if self._timeline is not None:
            self._timeline.append(event)
        self.existing.add(branch)

    async def merge(self, *, source: str, into: str) -> None:
        self.log.append(f"merge:{source}->{into}")
        if source in self._conflicts:
            raise MergeConflictError(source, into)
        self.merges.append((source, into))


class MergeConflictError(MergeConflict):
    """A merge could not complete because of a conflict (a typed ``MergeConflict``)."""


def _slice(issue_number: int = 7, prd_number: int = 1) -> Slice:
    return Slice(
        repo_full_name="owner/repo",
        issue_number=issue_number,
        prd_number=prd_number,
    )


async def _build(
    *,
    runtime: FakeRuntime,
    git: FakeGitOps,
    implementer: Implementer | None = None,
    config: RepoConfig | None = None,
    captured: list[DoneCheckReport] | None = None,
    slice_: Slice | None = None,
) -> BuildResult:
    return await build_slice(
        slice_ or _slice(),
        config or RepoConfig(),
        CLAUDE_MD,
        implementer=implementer or FakeImplementer(),
        git=git,
        auth=FakeAuth(),
        runtime=runtime,
        resolve_secret=_resolver({}),
        report=_sink(captured if captured is not None else []),
    )


# --- branch naming ---------------------------------------------------------------


def test_integration_branch_name() -> None:
    """The integration branch for PRD <n> is ``retinue/prd-<n>``."""
    assert integration_branch(1) == "retinue/prd-1"
    assert integration_branch(42) == "retinue/prd-42"


# --- happy path: green done-check merges -----------------------------------------


@pytest.mark.asyncio
async def test_green_done_check_merges_into_integration_branch() -> None:
    """A ready slice with a green done-check is merged into retinue/prd-<n>."""
    implementer = FakeImplementer()
    runtime = FakeRuntime()
    git = FakeGitOps()

    result = await _build(runtime=runtime, git=git, implementer=implementer)

    assert result.outcome is BuildOutcome.MERGED
    assert result.merged is True
    # The implementer was asked to build exactly this slice.
    assert implementer.built == [_slice()]
    # The merge happened from issue-7 into the integration branch.
    assert git.merges == [("issue-7", "retinue/prd-1")]


@pytest.mark.asyncio
async def test_integration_branch_created_off_staging_when_absent() -> None:
    """When retinue/prd-<n> is absent it is created off the config's staging branch."""
    git = FakeGitOps()
    config = RepoConfig(staging_branch="staging")

    await _build(runtime=FakeRuntime(), git=git, config=config)

    assert "create:retinue/prd-1<-staging" in git.log
    # Create precedes the merge.
    create_index = git.log.index("create:retinue/prd-1<-staging")
    merge_index = git.log.index("merge:issue-7->retinue/prd-1")
    assert create_index < merge_index


@pytest.mark.asyncio
async def test_integration_branch_reused_when_present() -> None:
    """An existing integration branch is reused, not recreated off staging."""
    git = FakeGitOps(existing={"retinue/prd-1"})

    await _build(runtime=FakeRuntime(), git=git)

    assert "exists:retinue/prd-1" in git.log
    assert not any(event.startswith("create:") for event in git.log)


@pytest.mark.asyncio
async def test_custom_staging_branch_is_the_merge_base() -> None:
    """A repo config's non-default staging branch is the base for the new branch."""
    git = FakeGitOps()
    config = RepoConfig(staging_branch="integration")

    await _build(runtime=FakeRuntime(), git=git, config=config)

    assert "create:retinue/prd-1<-integration" in git.log


# --- red done-check blocks the merge ---------------------------------------------


@pytest.mark.asyncio
async def test_red_done_check_blocks_the_merge() -> None:
    """A failing done-check blocks the merge: no merge runs, outcome is BLOCKED."""
    runtime = FakeRuntime(results={"uv": RunResult(exit_code=1, stderr="boom")})
    git = FakeGitOps()

    result = await _build(runtime=runtime, git=git)

    assert result.outcome is BuildOutcome.BLOCKED
    assert result.merged is False
    # No red slice is merged. The integration branch is still ensured up front (the
    # implementer needs it as a base), but nothing is ever merged onto it.
    assert git.merges == []
    assert not any(event.startswith("merge:") for event in git.log)


@pytest.mark.asyncio
async def test_red_done_check_still_built_and_reported() -> None:
    """A red slice is still built and the failing done-check is still reported."""
    runtime = FakeRuntime(results={"uv": RunResult(exit_code=1, stderr="boom")})
    implementer = FakeImplementer()
    captured: list[DoneCheckReport] = []

    await _build(
        runtime=runtime, git=FakeGitOps(), implementer=implementer, captured=captured
    )

    assert implementer.built == [_slice()]
    assert len(captured) == 1
    assert captured[0].passed is False


# --- per-slice container lifecycle: clone -> branch -> implement -> check -> push --


@pytest.mark.asyncio
async def test_slice_builds_in_one_container_in_order() -> None:
    """One container handles clone -> checkout issue-<N> -> implement -> done-check -> push."""
    runtime = FakeRuntime()
    implementer = FakeImplementer()

    result = await _build(runtime=runtime, git=FakeGitOps(), implementer=implementer)

    assert result.outcome is BuildOutcome.MERGED
    # Exactly one container built the whole slice and was destroyed at the end.
    assert runtime.container is not None and runtime.container.destroyed
    log = runtime.log
    clone = next(i for i, e in enumerate(log) if "git clone" in e)
    checkout = next(i for i, e in enumerate(log) if "checkout -B issue-7" in e)
    implement = next(i for i, e in enumerate(log) if e.startswith("run:implement"))
    done_check = next(i for i, e in enumerate(log) if "uv run pytest" in e)
    push = next(i for i, e in enumerate(log) if "git push origin issue-7" in e)
    assert clone < checkout < implement < done_check < push


@pytest.mark.asyncio
async def test_issue_branch_roots_on_the_integration_branch_not_staging() -> None:
    """The implementer's issue-<N> branch is rooted on retinue/prd-<n>, not staging.

    PRD: one integration branch per PRD off staging; implementers branch off *it* so a
    later-round slice builds on already-merged sibling work. The container checks out
    issue-<N> off ``origin/retinue/prd-<n>``, never off ``origin/staging``.
    """
    runtime = FakeRuntime()
    config = RepoConfig(staging_branch="staging")

    await _build(runtime=runtime, git=FakeGitOps(), config=config)

    log = runtime.log
    assert any("git checkout -B issue-7 origin/retinue/prd-1" in e for e in log)
    assert not any("checkout -B issue-7 origin/staging" in e for e in log)


@pytest.mark.asyncio
async def test_integration_branch_exists_before_the_implementer_runs() -> None:
    """retinue/prd-<n> is created off staging before any build container starts.

    The integration branch must exist (on origin) before the implementer's container
    fetches it as its branch base, so the create must precede the container start.
    """
    timeline: list[str] = []
    git = FakeGitOps(timeline=timeline)
    runtime = FakeRuntime(timeline=timeline)
    config = RepoConfig(staging_branch="staging")

    await _build(runtime=runtime, git=git, config=config)

    assert timeline[0] == "create:retinue/prd-1<-staging"
    assert any(e.startswith("start:") for e in timeline)
    create_index = timeline.index("create:retinue/prd-1<-staging")
    start_index = next(i for i, e in enumerate(timeline) if e.startswith("start:"))
    assert create_index < start_index


@pytest.mark.asyncio
async def test_red_slice_pushes_nothing() -> None:
    """A red done-check leaves the issue branch unpushed; the container is torn down."""
    runtime = FakeRuntime(results={"uv": RunResult(exit_code=1, stderr="boom")})

    result = await _build(runtime=runtime, git=FakeGitOps())

    assert result.outcome is BuildOutcome.BLOCKED
    assert not any("git push" in e for e in runtime.log)
    assert runtime.container is not None and runtime.container.destroyed


# --- hollow implement: zero commits fails the slice -------------------------------


@pytest.mark.asyncio
async def test_implement_landing_no_commits_raises_instead_of_vacuous_green() -> None:
    """An implementer run that lands zero commits fails the slice before the done-check.

    The hollow-implement failure: the agent no-ops, exits 0, and the done-check passes
    vacuously over the untouched tree — merging an empty branch. Counting commits since
    ``origin/<base>`` right after the implement catches it: a ``0`` count raises
    ``ImplementError`` (routing into the errored/escalate lane), pushes nothing, and
    still tears the container down.
    """
    runtime = FakeRuntime(
        results={"git rev-list": RunResult(exit_code=0, stdout="0\n")}
    )

    with pytest.raises(ImplementError, match="landed no commits"):
        await _build(runtime=runtime, git=FakeGitOps())

    assert not any("git push" in e for e in runtime.log)
    assert runtime.container is not None and runtime.container.destroyed


@pytest.mark.asyncio
async def test_failed_commit_count_probe_raises_not_passes() -> None:
    """A rev-list probe that itself fails (empty stdout, bad exit) raises, not passes.

    An unreadable count must not be read as "commits exist" — that would re-open the
    vacuous-green hole whenever the probe breaks.
    """
    runtime = FakeRuntime(
        results={"git rev-list": RunResult(exit_code=128, stderr="fatal: bad rev")}
    )

    with pytest.raises(ImplementError):
        await _build(runtime=runtime, git=FakeGitOps())


@pytest.mark.asyncio
async def test_commit_count_probe_runs_between_implement_and_done_check() -> None:
    """The guard probes commits after the implement and before the done-check runs."""
    runtime = FakeRuntime()

    await _build(runtime=runtime, git=FakeGitOps())

    implement_at = runtime.log.index("run:implement issue-7")
    probe_at = next(i for i, e in enumerate(runtime.log) if "git rev-list" in e)
    check_at = next(i for i, e in enumerate(runtime.log) if "uv run pytest" in e)
    assert implement_at < probe_at < check_at


@pytest.mark.asyncio
async def test_secrets_and_git_identity_injected_into_container_env() -> None:
    """Resolved secrets, the git committer identity, and the agent creds ride the env."""
    runtime = FakeRuntime()
    config = RepoConfig(
        secrets=SecretsConfig(values={"OPENAI_API_KEY": "${{ secrets.OPENAI_API_KEY }}"})
    )

    class CredImplementer(FakeImplementer):
        def auth_env(self) -> dict[str, str]:
            return {"ANTHROPIC_API_KEY": "sk-real"}

    await build_slice(
        _slice(),
        config,
        CLAUDE_MD,
        implementer=CredImplementer(),
        git=FakeGitOps(),
        auth=FakeAuth(),
        runtime=runtime,
        resolve_secret=_resolver({"OPENAI_API_KEY": "sk-secret"}),
        report=_sink([]),
    )

    env = runtime.started_env
    assert env is not None
    assert env["OPENAI_API_KEY"] == "sk-secret"
    assert env["ANTHROPIC_API_KEY"] == "sk-real"  # the implementer's auth_env
    assert env["GIT_AUTHOR_NAME"] and env["GIT_COMMITTER_EMAIL"]


@pytest.mark.asyncio
async def test_missing_secret_escalates_before_any_container_starts() -> None:
    """A missing required secret escalates on the report sink and starts no container."""
    runtime = FakeRuntime()
    captured: list[DoneCheckReport] = []
    config = RepoConfig(secrets=SecretsConfig(values={"OPENAI_API_KEY": "${{ secrets.X }}"}))

    with pytest.raises(MissingSecretError):
        await build_slice(
            _slice(),
            config,
            CLAUDE_MD,
            implementer=FakeImplementer(),
            git=FakeGitOps(),
            auth=FakeAuth(),
            runtime=runtime,
            resolve_secret=_resolver({}),
            report=_sink(captured),
        )

    assert runtime.log == []  # no container ever started
    assert len(captured) == 1 and captured[0].escalated


@pytest.mark.asyncio
async def test_container_torn_down_when_clone_fails() -> None:
    """A clone failure raises GitOpsError but the container is still destroyed."""
    runtime = FakeRuntime(results={"git": RunResult(exit_code=128, stderr="no auth")})

    with pytest.raises(GitOpsError):
        await _build(runtime=runtime, git=FakeGitOps())

    assert runtime.container is not None and runtime.container.destroyed


# --- merge conflict surfaces -----------------------------------------------------


@pytest.mark.asyncio
async def test_merge_conflict_propagates() -> None:
    """A merge conflict on a green slice propagates rather than silently passing."""
    git = FakeGitOps(conflicts={"issue-7"})

    with pytest.raises(MergeConflictError):
        await _build(runtime=FakeRuntime(), git=git)


# --- real container-backed GitOps adapter ----------------------------------------
#
# Exercises the production ContainerGitOps over a scripted in-memory container: argv
# assembly, the branch-exists shortcut, and classifying a failed merge as a conflict
# (aborted + raised) vs. a hard git error. No live container/Docker/network.


class ScriptedContainer:
    """In-memory :class:`Container` returning canned results keyed by argv substring.

    ``results`` maps a marker (a substring of the joined argv, matched in insertion
    order) to the :class:`RunResult` to return; an unmatched command returns exit 0.
    Every command is recorded in ``commands`` so argv assembly can be asserted.
    """

    def __init__(self, results: dict[str, RunResult] | None = None) -> None:
        self._results = results or {}
        self.commands: list[list[str]] = []
        # Per-exec env overrides, keyed by the first argv token, so a test can assert
        # what env a command was exec'd with (e.g. the implementer's IS_SANDBOX).
        self.command_env: dict[str, Mapping[str, str]] = {}

    async def run_command(
        self, command: list[str], *, env: Mapping[str, str] | None = None
    ) -> RunResult:
        self.commands.append(command)
        if env is not None:
            self.command_env[command[0]] = env
        joined = " ".join(command)
        for marker, result in self._results.items():
            if marker in joined:
                return result
        return RunResult(exit_code=0)

    async def destroy(self) -> None:  # pragma: no cover - unused by GitOps
        pass


def _joined(container: ScriptedContainer) -> list[str]:
    return [" ".join(cmd) for cmd in container.commands]


@pytest.mark.asyncio
async def test_ensure_branch_reuses_existing_on_origin_without_creating() -> None:
    """When the branch already exists on origin, no fetch/checkout/push is issued."""
    container = ScriptedContainer()  # ls-remote returns exit 0 -> branch on origin
    git = ContainerGitOps(container)

    await git.ensure_integration_branch(branch="retinue/prd-1", base="main")

    cmds = _joined(container)
    assert cmds == ["git ls-remote --exit-code origin refs/heads/retinue/prd-1"]


@pytest.mark.asyncio
async def test_ensure_branch_creates_off_origin_base_and_pushes_when_absent() -> None:
    """An absent branch is fetched, checked out (-B) off ``origin/<base>``, and pushed.

    The integration branch must exist on origin before implementers branch off it, so a
    freshly created branch is pushed to origin within ensure (not only at merge time).
    """
    container = ScriptedContainer(
        {"ls-remote": RunResult(exit_code=2)}  # branch missing on origin
    )
    git = ContainerGitOps(container)

    await git.ensure_integration_branch(branch="retinue/prd-1", base="staging")

    cmds = _joined(container)
    assert "git fetch origin staging" in cmds
    assert "git checkout -B retinue/prd-1 origin/staging" in cmds
    assert "git push origin retinue/prd-1" in cmds


@pytest.mark.asyncio
async def test_ensure_branch_raises_gitops_error_on_checkout_failure() -> None:
    """A hard git failure while creating the branch is a GitOpsError, not silent."""
    container = ScriptedContainer(
        {
            "ls-remote": RunResult(exit_code=2),
            "checkout": RunResult(exit_code=128, stderr="fatal: bad object"),
        }
    )
    git = ContainerGitOps(container)

    with pytest.raises(GitOpsError):
        await git.ensure_integration_branch(branch="retinue/prd-1", base="main")


@pytest.mark.asyncio
async def test_merge_assembles_checkout_fetch_merge_argv() -> None:
    """A clean merge checks out the target, fetches the source, then merges its tip."""
    container = ScriptedContainer()  # all exit 0 -> clean merge
    git = ContainerGitOps(container)

    await git.merge(source="issue-7", into="retinue/prd-1")

    cmds = _joined(container)
    assert cmds[0] == "git checkout retinue/prd-1"
    assert cmds[1] == "git fetch origin issue-7"
    # The merge is a no-ff, no-edit commit under a fixed committer identity.
    merge_cmd = cmds[2]
    assert "merge --no-ff --no-edit origin/issue-7" in merge_cmd
    assert "user.name=" in merge_cmd
    assert "user.email=" in merge_cmd
    # The integration branch is pushed so the staging PR has a real remote head.
    assert cmds[3] == "git push origin retinue/prd-1"


@pytest.mark.asyncio
async def test_merge_conflict_is_aborted_and_raised() -> None:
    """A content conflict aborts the merge (clean workspace) and raises MergeConflict."""
    container = ScriptedContainer(
        {
            "merge --no-ff": RunResult(
                exit_code=1,
                stdout="Auto-merging x\nCONFLICT (content): Merge conflict in x",
                stderr="Automatic merge failed; fix conflicts and then commit",
            )
        }
    )
    git = ContainerGitOps(container)

    with pytest.raises(MergeConflict) as excinfo:
        await git.merge(source="issue-7", into="retinue/prd-1")

    assert excinfo.value.source == "issue-7"
    assert excinfo.value.into == "retinue/prd-1"
    assert "git merge --abort" in _joined(container)


@pytest.mark.asyncio
async def test_merge_hard_error_is_gitops_error_not_conflict() -> None:
    """A non-conflict merge failure (e.g. unknown ref) is a GitOpsError, not a conflict."""
    container = ScriptedContainer(
        {
            "merge --no-ff": RunResult(
                exit_code=128,
                stderr="merge: origin/issue-7 - not something we can merge",
            )
        }
    )
    git = ContainerGitOps(container)

    with pytest.raises(GitOpsError):
        await git.merge(source="issue-7", into="retinue/prd-1")
    # A hard error must not leave a phantom --abort claiming a conflict was handled.
    assert "git merge --abort" not in _joined(container)


@pytest.mark.asyncio
async def test_round_diff_diffs_each_branch_without_refetching() -> None:
    """The round diff diffs each merged branch (3-dot) over the base, fetching nothing.

    Feeds the internal reviewer the round's merged surface: each ``issue-<N>`` branch's
    contribution since it diverged from the integration branch, concatenated. Only merged
    branches are diffed and :meth:`merge` already fetched each one's tip into this same
    container, so ``round_diff`` must not re-fetch — a redundant ``git fetch`` per branch.
    """
    container = ScriptedContainer(
        {"diff retinue/prd-1...origin/issue-7": RunResult(exit_code=0, stdout="DIFF-7")}
    )
    git = ContainerGitOps(container)

    diff = await git.round_diff(
        merged_branches=["issue-7", "issue-8"], base="retinue/prd-1"
    )

    cmds = _joined(container)
    assert "git diff retinue/prd-1...origin/issue-7" in cmds
    assert "git diff retinue/prd-1...origin/issue-8" in cmds
    # The merge already fetched each branch tip; round_diff issues no fetch of its own.
    assert not any(c.startswith("git fetch") for c in cmds)
    # The matched branch's diff body is carried through into the concatenated result.
    assert "DIFF-7" in diff


@pytest.mark.asyncio
async def test_round_diff_empty_when_no_branches_merged() -> None:
    """No merged branches means an empty diff and no git commands issued."""
    container = ScriptedContainer()
    git = ContainerGitOps(container)

    diff = await git.round_diff(merged_branches=[], base="retinue/prd-1")

    assert diff == ""
    assert container.commands == []


# --- real Agent-SDK conflict resolver --------------------------------------------
#
# Exercises the production AgentSdkConflictResolver's pure/parseable parts (auth header,
# request payload, response parsing, conflicted-path + write-file argv) and its
# container/transport collaboration over an in-memory container + fake transport. No
# live container/Docker/Claude/network.


class FakeAnthropicTransport:
    """Records the POST and returns a canned :class:`AnthropicResponse`."""

    def __init__(self, response: AnthropicResponse) -> None:
        self._response = response
        self.calls: list[tuple[str, dict[str, str], dict[str, object]]] = []

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, object]
    ) -> AnthropicResponse:
        self.calls.append((url, headers, json))
        return self._response


def _text_response(payload: dict[str, object], *, status: int = 200) -> AnthropicResponse:
    """An Anthropic response whose single text block carries ``payload`` as JSON."""
    return AnthropicResponse(
        status_code=status,
        body={"content": [{"type": "text", "text": json.dumps(payload)}]},
    )


def test_resolve_headers_routes_oauth_token_to_bearer() -> None:
    """An OAuth subscription token rides Authorization: Bearer + the OAuth beta header."""
    headers = _resolve_headers("sk-ant-oat01-secret")

    assert headers["authorization"] == "Bearer sk-ant-oat01-secret"
    assert headers["anthropic-beta"] == "oauth-2025-04-20"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "x-api-key" not in headers


def test_resolve_headers_routes_api_key_to_x_api_key() -> None:
    """A raw API key rides x-api-key with no OAuth beta header."""
    headers = _resolve_headers("sk-ant-api03-secret")

    assert headers["x-api-key"] == "sk-ant-api03-secret"
    assert "authorization" not in headers
    assert "anthropic-beta" not in headers


def test_resolve_payload_carries_each_conflicted_blob_and_schema() -> None:
    """The request payload fences each conflicted file and forces the strict schema.

    The schema must ride ``output_config.format`` (the canonical Messages API shape);
    the OpenAI-style top-level ``response_format`` is not a Claude API parameter and
    400s on the live wire.
    """
    files = [
        _ConflictedFile(path="a.py", content="<<<<<<< ours\nx\n=======\ny\n>>>>>>> theirs"),
        _ConflictedFile(path="b.py", content="conflict-b"),
    ]
    payload = _resolve_payload(
        files,
        source="issue-7",
        into="retinue/prd-1",
        model="m",
        effort=_EFFORT_XHIGH,
        is_oauth=False,
    )

    assert payload["model"] == "m"
    fmt = payload["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["required"] == ["resolved"]
    assert "response_format" not in payload
    user = payload["messages"][0]["content"]
    # Both paths and both blobs ride in the user message, fenced by path.
    assert "### a.py" in user and "### b.py" in user
    assert "<<<<<<< ours" in user and "conflict-b" in user
    assert "issue-7" in user and "retinue/prd-1" in user


def test_resolve_payload_carries_xhigh_effort() -> None:
    """The conflict resolver's default (registry) effort tier threads through unchanged.

    Opus 4.8 removed the ``thinking`` budget mechanism (400 on the live Messages API);
    ``output_config.effort`` is the current effort control. Passing the registry's
    ``xhigh`` tier explicitly proves the value round-trips onto the payload.
    """
    files = [_ConflictedFile(path="a.py", content="conflict-a")]

    payload = _resolve_payload(
        files,
        source="issue-7",
        into="retinue/prd-1",
        model="m",
        effort=_EFFORT_XHIGH,
        is_oauth=False,
    )

    assert payload["output_config"]["effort"] == _EFFORT_XHIGH
    assert _EFFORT_XHIGH == "xhigh"
    assert "thinking" not in payload


def test_resolve_payload_carries_the_given_effort_tier() -> None:
    """The resolver's effort tier is not hardcoded — a caller-supplied tier threads through."""
    files = [_ConflictedFile(path="a.py", content="conflict-a")]

    payload = _resolve_payload(
        files,
        source="issue-7",
        into="retinue/prd-1",
        model="m",
        effort="low",
        is_oauth=False,
    )

    assert payload["output_config"]["effort"] == "low"


def test_resolve_payload_oauth_leads_system_with_claude_code_identity() -> None:
    """With an OAuth credential the system field leads with the identity block.

    A subscription OAuth token reaches the premium resolving model over the raw Messages
    API only when the first system block is the Claude Code identity string; the
    resolver's own brief follows it as the second block.
    """
    files = [_ConflictedFile(path="a.py", content="conflict-a")]

    payload = _resolve_payload(
        files,
        source="issue-7",
        into="retinue/prd-1",
        model="m",
        effort=_EFFORT_XHIGH,
        is_oauth=True,
    )

    assert payload["system"] == [
        {"type": "text", "text": CLAUDE_CODE_IDENTITY},
        {"type": "text", "text": _RESOLVE_SYSTEM},
    ]


def test_resolve_payload_api_key_keeps_plain_string_system() -> None:
    """With an API-key credential the system field stays the unchanged plain brief."""
    files = [_ConflictedFile(path="a.py", content="conflict-a")]

    payload = _resolve_payload(
        files,
        source="issue-7",
        into="retinue/prd-1",
        model="m",
        effort=_EFFORT_XHIGH,
        is_oauth=False,
    )

    assert payload["system"] == _RESOLVE_SYSTEM
    assert isinstance(payload["system"], str)


def test_parse_resolution_maps_paths_to_resolved_bodies() -> None:
    """A well-formed response yields a {path: resolved_content} map."""
    response_body = {
        "content": [
            {
                "type": "text",
                "text": '{"resolved": ['
                '{"path": "a.py", "content": "merged-a"},'
                '{"path": "b.py", "content": "merged-b"}]}',
            }
        ]
    }

    resolutions = _parse_resolution(response_body)

    assert resolutions == {"a.py": "merged-a", "b.py": "merged-b"}


def test_parse_resolution_raises_on_missing_text() -> None:
    """A response with no text block escalates rather than resolving nothing."""
    with pytest.raises(ConflictResolutionError):
        _parse_resolution({"content": []})


def test_parse_resolution_raises_on_invalid_json() -> None:
    """A non-JSON text body is a hard resolver error, not a silent empty resolution."""
    with pytest.raises(ConflictResolutionError):
        _parse_resolution({"content": [{"type": "text", "text": "not json"}]})


def test_parse_resolution_raises_on_non_list_resolved() -> None:
    """A payload missing the 'resolved' list fails loudly."""
    with pytest.raises(ConflictResolutionError):
        _parse_resolution({"content": [{"type": "text", "text": '{"resolved": {}}'}]})


def test_conflicted_paths_parses_one_path_per_nonblank_line() -> None:
    """``git diff --name-only --diff-filter=U`` output splits into trimmed paths."""
    assert _conflicted_paths("a.py\n  b/c.py  \n\n") == ["a.py", "b/c.py"]
    assert _conflicted_paths("") == []


def test_write_file_command_round_trips_arbitrary_content() -> None:
    """The write-file argv carries content base64-encoded as a positional arg, byte-exact."""
    content = '<<<<<<< ours\n"x" & $y\n=======\nz\n>>>>>>> theirs\n'
    argv = _write_file_command("dir/f.py", content)

    # No shell interpolation: the body is a base64 positional arg, the path another.
    assert argv[0] == "sh" and argv[-1] == "dir/f.py"
    blob = argv[-2]
    assert base64.b64decode(blob).decode() == content


@pytest.mark.asyncio
async def test_resolver_recreates_merge_resolves_and_commits() -> None:
    """A conflict is recreated, resolved via Claude, written back, staged, and committed."""
    container = ScriptedContainer(
        {
            "diff --name-only": RunResult(exit_code=1, stdout="a.py\n"),
            "cat a.py": RunResult(
                exit_code=0, stdout="<<<<<<< ours\nx\n=======\ny\n>>>>>>> theirs"
            ),
        }
    )
    transport = FakeAnthropicTransport(
        _text_response({"resolved": [{"path": "a.py", "content": "merged"}]})
    )
    resolver = AgentSdkConflictResolver(
        container=container, transport=transport, credential="sk-ant-api03-k"
    )

    result = await resolver(source="issue-7", into="retinue/prd-1")

    assert result is ConflictResolution.RESOLVED
    cmds = _joined(container)
    # The merge is recreated no-commit, the resolved blob is written + staged, committed.
    assert any("merge --no-ff --no-commit --no-edit origin/issue-7" in c for c in cmds)
    assert "git add a.py" in cmds
    assert any(c.startswith("git ") and "commit --no-edit" in c for c in cmds)
    # The resolved body was written byte-exact via the base64 round-trip.
    write = next(cmd for cmd in container.commands if cmd[0] == "sh")
    assert base64.b64decode(write[-2]).decode() == "merged"


@pytest.mark.asyncio
async def test_resolver_unresolved_when_no_conflicted_paths() -> None:
    """No conflicted paths means nothing to fix: UNRESOLVED, no Claude call, no commit."""
    container = ScriptedContainer({"diff --name-only": RunResult(exit_code=0, stdout="")})
    transport = FakeAnthropicTransport(_text_response({"resolved": []}))
    resolver = AgentSdkConflictResolver(
        container=container, transport=transport, credential="k"
    )

    result = await resolver(source="issue-7", into="retinue/prd-1")

    assert result is ConflictResolution.UNRESOLVED
    assert transport.calls == []
    assert "git commit --no-edit" not in " ".join(_joined(container))


@pytest.mark.asyncio
async def test_resolver_escalates_on_non_200_status() -> None:
    """A non-200 Messages API response is a hard error: escalate, don't commit junk."""
    container = ScriptedContainer(
        {
            "diff --name-only": RunResult(exit_code=1, stdout="a.py\n"),
            "cat a.py": RunResult(exit_code=0, stdout="conflict"),
        }
    )
    transport = FakeAnthropicTransport(_text_response({"resolved": []}, status=500))
    resolver = AgentSdkConflictResolver(
        container=container, transport=transport, credential="k"
    )

    with pytest.raises(ConflictResolutionError):
        await resolver(source="issue-7", into="retinue/prd-1")
    # No commit and no staging happened on a failed resolution. ``commit`` is a discrete
    # argv element here, so matching the token avoids the ``--no-commit`` recreate flag.
    assert not any("commit" in cmd for cmd in container.commands)
    assert not any("add" in cmd for cmd in container.commands)


@pytest.mark.asyncio
async def test_conflict_resolver_reads_model_and_effort_from_a_default_level_override() -> None:
    """The resolver's model + effort resolve via the routing table's ``default:`` level.

    ``AgentSdkConflictResolver`` has no production wiring site (``resolve_conflict`` is
    always ``None`` in ``build_prd`` today), so this drives the mechanism directly: a
    ``RoutingConfig`` whose ``default`` level overrides :data:`Role.RESOLVER` with a
    custom model *and* effort, resolved via :func:`resolve_model`/:func:`resolve_effort`
    (the same helpers a future wiring site would call), and asserts both land on the
    request the resolver sends.
    """
    config = RepoConfig(
        routing=RoutingConfig(
            default="standard",
            levels={
                "standard": RoutingLevel(
                    description="Ordinary work.",
                    roles={
                        Role.RESOLVER.value: ModelEffort(
                            model="resolver-custom", effort="low"
                        )
                    },
                )
            },
        )
    )
    container = ScriptedContainer(
        {
            "diff --name-only": RunResult(exit_code=1, stdout="a.py\n"),
            "cat a.py": RunResult(exit_code=0, stdout="conflict"),
        }
    )
    transport = FakeAnthropicTransport(
        _text_response({"resolved": [{"path": "a.py", "content": "merged"}]})
    )
    resolver = AgentSdkConflictResolver(
        container=container,
        transport=transport,
        credential="k",
        model=resolve_model(Role.RESOLVER, config),
        effort=resolve_effort(Role.RESOLVER, config),
    )

    result = await resolver(source="issue-7", into="retinue/prd-1")

    assert result is ConflictResolution.RESOLVED
    _, _, payload = transport.calls[0]
    assert payload["model"] == "resolver-custom"
    output_config = payload["output_config"]
    assert isinstance(output_config, dict)
    assert output_config["effort"] == "low"


# --- real container-exec implementer ---------------------------------------------
#
# Exercises the production ContainerImplementer's pure parts (prompt assembly, env-routed
# auth, claude argv, json-result error detection) and its in-container collaboration over
# a scripted container. No live model, CLI, Docker, gh, or clone.


def test_implement_prompt_names_issue_repo_and_branch() -> None:
    """The per-slice prompt names the issue, repo, and the issue-<N> commit branch."""
    prompt = _implement_prompt(_slice(issue_number=7, prd_number=1))

    assert "#7" in prompt
    assert "owner/repo" in prompt
    assert "issue-7" in prompt


def test_implement_prompt_injects_no_plan_file_for_the_prd_lane() -> None:
    """With no plan_path (the PRD lane), the prompt carries no plan-file instruction.

    The plan file is an ad-hoc-lane concern (#37); ``build_slice``/``build_prd`` must be
    unaffected, so the default-path prompt names no ``.retinue/plan.md`` and no read-plan
    preamble.
    """
    prompt = _implement_prompt(_slice(issue_number=7, prd_number=1))

    assert ".retinue/plan.md" not in prompt
    assert not prompt.startswith("Read the implementation plan")


def test_implement_prompt_appends_issue_facts_as_the_spec() -> None:
    """With facts, the prompt carries the issue title/body as the authoritative spec.

    The build container has no gh and no GitHub token in its env, so the agent cannot
    read the issue itself; without the title/body baked into the prompt it no-ops (the
    hollow-implement failure). The section must also say the container cannot reach
    GitHub, so the agent works from the baked text instead of hunting for the issue.
    """
    facts = ClassifyInput(
        title="Add retry cap", body="The worker must retry twice.", labels=[]
    )

    prompt = _implement_prompt(_slice(issue_number=7, prd_number=1), facts=facts)

    assert "Add retry cap" in prompt
    assert "The worker must retry twice." in prompt
    assert "cannot reach GitHub" in prompt


def test_implement_prompt_without_facts_carries_no_issue_section() -> None:
    """With no facts (seam unset), the prompt is the bare pre-existing shape."""
    prompt = _implement_prompt(_slice(issue_number=7, prd_number=1))

    assert "Issue title:" not in prompt
    assert "Issue body:" not in prompt


def test_implement_env_api_key_mode_uses_anthropic_api_key() -> None:
    """api_key mode threads the credential to the CLI as ANTHROPIC_API_KEY."""
    env = _implement_env("sk-ant-api03-secret", "api_key")

    assert env == {"ANTHROPIC_API_KEY": "sk-ant-api03-secret"}
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env


def test_implement_env_subscription_mode_uses_oauth_token() -> None:
    """subscription mode threads the credential as CLAUDE_CODE_OAUTH_TOKEN."""
    env = _implement_env("sk-ant-oat01-secret", "subscription")

    assert env == {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-secret"}
    assert "ANTHROPIC_API_KEY" not in env


def test_claude_argv_assembles_headless_invocation() -> None:
    """The argv runs the headless CLI: print mode, model, bypassPermissions, json output.

    ``acceptEdits`` only auto-accepts *file edits*; Bash calls — ``git commit``, the
    repo's checks — stay blocked pending an approval a headless ``-p`` run can never
    give, so the agent edits for its whole run and exits 0 with zero commits (the
    second hollow-implement cause, verified live on slice #72). The container is
    disposable and isolated, so the run must bypass permissions entirely.
    """
    argv = _claude_argv(prompt="do it", model="m", max_turns=80)

    assert argv[0] == "claude"
    assert argv[1:3] == ["-p", "do it"]
    assert "--model" in argv and "m" in argv
    assert "--permission-mode" in argv and "bypassPermissions" in argv
    assert "acceptEdits" not in argv
    assert "--output-format" in argv and "json" in argv


def test_claude_argv_caps_the_agent_loop_with_max_turns() -> None:
    """The argv pins ``--max-turns`` so an uncapped implement loop cannot run unbounded.

    Without this cap, a thrashing run (e.g. a doc task that re-runs the full check suite
    each turn) is bounded only by the arq job_timeout, which kills the whole job — including
    the done-check — rather than the agent stopping and letting the done-check report.
    """
    argv = _claude_argv(prompt="do it", model="m", max_turns=42)

    assert argv[argv.index("--max-turns") + 1] == "42"


def test_implement_system_prompt_excuses_doc_only_tasks_from_tdd() -> None:
    """The system brief no longer mandates a failing test for untestable doc/config work.

    The hard TDD mandate made a documentation-only issue (no behavior to test) thrash —
    hunting for a test that cannot exist. The brief now keeps TDD the default for testable
    changes but excuses doc/config-only changes from inventing one.
    """
    assert "documentation" in _IMPLEMENT_SYSTEM
    assert "config" in _IMPLEMENT_SYSTEM
    assert "nothing to test" in _IMPLEMENT_SYSTEM


def test_implement_prompt_excuses_doc_only_tasks_from_tdd() -> None:
    """The per-slice prompt likewise excuses a doc/config-only change from a test."""
    prompt = _implement_prompt(_slice(issue_number=7, prd_number=1))

    assert "documentation" in prompt
    assert "no test" in prompt


def test_claude_result_is_error_reads_json_flag() -> None:
    """Only a json result flagging ``is_error`` is an error; non-json/empty is not."""
    assert _claude_result_is_error('{"is_error": true}') is True
    assert _claude_result_is_error('{"is_error": false}') is False
    assert _claude_result_is_error("not json") is False
    assert _claude_result_is_error("") is False


def test_claude_result_is_error_warns_on_unparseable_stdout(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unparseable/empty CLI stdout is not an error but is logged, not swallowed silently.

    ``--output-format json`` was requested, so a non-json or empty result is unexpected:
    the exit code stays authoritative (returns not-an-error), but the anomaly is surfaced
    as a warning rather than passing silently.
    """
    with caplog.at_level(logging.WARNING, logger="retinue.orchestrator"):
        assert _claude_result_is_error("not json") is False
        assert _claude_result_is_error("") is False

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert all("not parseable JSON" in r.getMessage() for r in warnings)


@pytest.mark.asyncio
async def test_container_implementer_execs_claude_with_prompt_and_model() -> None:
    """A clean run execs ``claude`` in the container with the per-slice prompt + model."""
    container = ScriptedContainer()
    implementer = ContainerImplementer(credential="sk-ant-api03-k")

    await implementer.implement(_slice(issue_number=7), container=container)

    cmd = next(c for c in container.commands if c and c[0] == "claude")
    prompt = cmd[cmd.index("-p") + 1]
    assert "issue-7" in prompt
    # Production wires ContainerImplementer with no model override, so the default
    # constant is the source of truth. Per the PRD, implementers default to Sonnet.
    assert "claude-sonnet-4-6" in cmd


@pytest.mark.asyncio
async def test_container_implementer_bakes_fetched_issue_facts_into_prompt() -> None:
    """With an issue-facts source, the exec'd prompt carries the fetched title/body.

    The fetch runs on the worker (which has gh + the installation token) before the
    container exec, so the containerized agent — which cannot reach GitHub — receives
    the issue content it is asked to build.
    """
    calls: list[tuple[str, int]] = []

    async def facts(repo_full_name: str, issue_number: int) -> ClassifyInput:
        calls.append((repo_full_name, issue_number))
        return ClassifyInput(title="Add retry cap", body="Retry twice.", labels=[])

    container = ScriptedContainer()
    implementer = ContainerImplementer(credential="k", issue_facts=facts)

    await implementer.implement(_slice(issue_number=7), container=container)

    assert calls == [("owner/repo", 7)]
    cmd = next(c for c in container.commands if c and c[0] == "claude")
    prompt = cmd[cmd.index("-p") + 1]
    assert "Add retry cap" in prompt
    assert "Retry twice." in prompt


@pytest.mark.asyncio
async def test_container_implementer_without_facts_source_keeps_bare_prompt() -> None:
    """With no issue-facts source (the default), the prompt is unchanged."""
    container = ScriptedContainer()

    await ContainerImplementer(credential="k").implement(_slice(), container=container)

    cmd = next(c for c in container.commands if c and c[0] == "claude")
    assert "Issue title:" not in cmd[cmd.index("-p") + 1]


@pytest.mark.asyncio
async def test_container_implementer_threads_max_turns_into_the_argv() -> None:
    """The implementer caps its exec at its ``max_turns`` (overridable; sane default)."""
    container = ScriptedContainer()
    await ContainerImplementer(credential="k", max_turns=7).implement(
        _slice(), container=container
    )

    cmd = next(c for c in container.commands if c and c[0] == "claude")
    assert cmd[cmd.index("--max-turns") + 1] == "7"
    # A bare production-wired implementer still carries a positive cap.
    assert ContainerImplementer(credential="k").max_turns == _DEFAULT_IMPLEMENT_MAX_TURNS
    assert _DEFAULT_IMPLEMENT_MAX_TURNS > 0


def test_container_implementer_defaults_to_sonnet() -> None:
    """The production-wired implementer (no model override) defaults to Sonnet.

    pipeline.py constructs ``ContainerImplementer`` with only credential + auth_mode, so
    the registry's ``IMPLEMENTER`` entry is the live model. Per the PRD, implementers
    default to Sonnet (slicer/reviewer/resolver stay on Opus).
    """
    assert ROLE_REGISTRY[Role.IMPLEMENTER].model == "claude-sonnet-4-6"
    assert ContainerImplementer(credential="k").model == "claude-sonnet-4-6"


def test_container_implementer_auth_env_routes_by_mode() -> None:
    """auth_env routes the credential to the env var the auth mode selects."""
    assert ContainerImplementer(credential="k").auth_env() == {"ANTHROPIC_API_KEY": "k"}
    assert ContainerImplementer(
        credential="t", auth_mode="subscription"
    ).auth_env() == {"CLAUDE_CODE_OAUTH_TOKEN": "t"}


@pytest.mark.asyncio
async def test_container_implementer_raises_on_nonzero_exit() -> None:
    """A non-zero ``claude`` exit surfaces as ImplementError, not a clean build."""
    container = ScriptedContainer({"claude": RunResult(exit_code=1, stderr="boom")})

    with pytest.raises(ImplementError):
        await ContainerImplementer(credential="k").implement(_slice(), container=container)


@pytest.mark.asyncio
async def test_container_implementer_raises_on_is_error_json_result() -> None:
    """A clean exit whose json result flags is_error still raises ImplementError."""
    container = ScriptedContainer(
        {"claude": RunResult(exit_code=0, stdout='{"is_error": true}')}
    )

    with pytest.raises(ImplementError):
        await ContainerImplementer(credential="k").implement(_slice(), container=container)


@pytest.mark.asyncio
async def test_container_implementer_execs_claude_with_is_sandbox_env() -> None:
    """The claude exec carries IS_SANDBOX=1 so bypassPermissions runs as root.

    The runner container execs as root, and the CLI refuses
    ``--dangerously-skip-permissions`` (bypassPermissions) under root unless
    ``IS_SANDBOX=1`` marks the environment as a disposable sandbox — which this
    container is. Without it every implement exits 1 instantly (seen live: the whole
    retry budget burned in two seconds).
    """
    container = ScriptedContainer()

    await ContainerImplementer(credential="k").implement(_slice(), container=container)

    assert container.command_env["claude"]["IS_SANDBOX"] == "1"


@pytest.mark.asyncio
async def test_container_implementer_logs_run_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A completed run logs the CLI result summary (turns + result snippet).

    The CLI's stdout is consumed here and the container is destroyed after the build,
    so without this line a clean-but-wrong run (e.g. the agent explaining why it could
    not commit) leaves zero forensic trace — exactly what made the hollow-implement
    failures blind.
    """
    container = ScriptedContainer(
        {
            "claude": RunResult(
                exit_code=0,
                stdout='{"is_error": false, "num_turns": 12, "result": "Committed the endpoint."}',
            )
        }
    )

    with caplog.at_level(logging.INFO, logger="retinue.orchestrator"):
        await ContainerImplementer(credential="k").implement(
            _slice(), container=container
        )

    message = "\n".join(r.getMessage() for r in caplog.records)
    assert "12 turns" in message
    assert "Committed the endpoint." in message


@pytest.mark.asyncio
async def test_container_implementer_clean_run_does_not_raise() -> None:
    """A zero exit with a non-error json result completes cleanly."""
    container = ScriptedContainer(
        {"claude": RunResult(exit_code=0, stdout='{"is_error": false}')}
    )

    await ContainerImplementer(credential="k").implement(_slice(), container=container)


def test_container_implementer_satisfies_implementer_protocol() -> None:
    """ContainerImplementer is usable wherever the Implementer protocol is expected."""
    implementer: Implementer = ContainerImplementer(credential="k")

    assert hasattr(implementer, "implement") and hasattr(implementer, "auth_env")
