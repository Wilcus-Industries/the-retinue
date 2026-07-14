"""Tests for the convergence handoff and the merge reap (issue #12).

Two flows, both with every gh/push/network touch faked or injected:

1. **convergence handoff** — when heimdall converges on the staging PR, the retinue
   fires a "test & merge" notification (a push + a PR comment) and NEVER merges. There
   is no merge seam to call; the test asserts the announcement landed and that the
   handoff signature is the loopback ``Handoff`` shape so it wires in directly.
2. **merge reap** — on a ``pull_request`` closed+merged signal (the human merged the
   PR), the retinue closes the PR's slice issues, then reaps the PRD: it closes the PRD
   IFF every non-``hitl`` child issue is closed. An open ``hitl`` child does not block
   the reap; an open non-``hitl`` child does.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable

import pytest

from retinue.handoff import (
    ChildIssue,
    Handoff,
    MergedPullRequest,
    ReapOutcome,
    announce_handoff,
    reap_merged_pr,
)
from retinue.loopback import Handoff as LoopbackHandoff
from retinue.notify import CommentRequest, LabelRequest, Notifier, PushRequest


class _RecordingSinks:
    """Captures notifier sink calls so a test can assert the announcement fired."""

    def __init__(self) -> None:
        self.pushes: list[PushRequest] = []
        self.comments: list[CommentRequest] = []
        self.labels: list[LabelRequest] = []

    async def push(self, request: PushRequest) -> None:
        self.pushes.append(request)

    async def comment(self, request: CommentRequest) -> None:
        self.comments.append(request)

    async def label(self, request: LabelRequest) -> None:
        self.labels.append(request)


def _notifier(sinks: _RecordingSinks) -> Notifier:
    return Notifier(push=sinks.push, comment=sinks.comment, label=sinks.label)


class _RecordingGh:
    """A fake gh seam: records issue closes and scripts a PRD's children."""

    def __init__(self, children: dict[int, list[ChildIssue]] | None = None) -> None:
        self.closed: list[tuple[str, int]] = []
        self._children = children or {}

    async def close_issue(self, *, repo_full_name: str, issue_number: int) -> None:
        self.closed.append((repo_full_name, issue_number))

    async def children_of(self, *, repo_full_name: str, prd_number: int) -> list[ChildIssue]:
        return self._children.get(prd_number, [])


class _RaisingGh(_RecordingGh):
    """A gh seam whose ``close_issue`` raises for a chosen set of issue numbers.

    Models a transient per-issue ``gh`` failure so a test can assert one failed close does
    not abort the others and is not credited to the reap gate.
    """

    def __init__(
        self,
        *,
        fail_on: set[int],
        children: dict[int, list[ChildIssue]] | None = None,
    ) -> None:
        super().__init__(children)
        self._fail_on = set(fail_on)

    async def close_issue(self, *, repo_full_name: str, issue_number: int) -> None:
        if issue_number in self._fail_on:
            raise RuntimeError(f"boom closing #{issue_number}")
        await super().close_issue(
            repo_full_name=repo_full_name, issue_number=issue_number
        )


# --- convergence handoff: fire "test & merge" (push + PR comment), never merge ----


@pytest.mark.asyncio
async def test_convergence_fires_test_and_merge_push_and_comment() -> None:
    """The handoff fires a push + a PR comment telling a human to test & merge."""
    sinks = _RecordingSinks()

    await announce_handoff(
        repo_full_name="owner/repo",
        pr_number=42,
        notifier=_notifier(sinks),
    )

    # A push went out (the out-of-band heads-up).
    assert len(sinks.pushes) == 1
    # A PR comment landed on the PR (the durable in-repo record).
    assert len(sinks.comments) == 1
    assert sinks.comments[0].issue_number == 42
    assert sinks.comments[0].repo_full_name == "owner/repo"
    # The announcement is a "test & merge" handoff, naming the PR.
    body = sinks.comments[0].body.lower()
    assert "test" in body and "merge" in body
    assert "#42" in sinks.comments[0].body


@pytest.mark.asyncio
async def test_convergence_never_merges() -> None:
    """The handoff has no merge seam: the retinue never merges on convergence."""
    sinks = _RecordingSinks()

    await announce_handoff(
        repo_full_name="owner/repo",
        pr_number=42,
        notifier=_notifier(sinks),
    )

    # There is no merge collaborator at all on the handoff signature.
    params = inspect.signature(announce_handoff).parameters
    assert not any("merge" in name.lower() for name in params)


def test_announce_handoff_matches_loopback_handoff_seam() -> None:
    """``announce_handoff`` is callable as the loopback ``Handoff`` seam."""
    # The loopback seam is ``Callable[..., Awaitable[None]]``; the converge path calls
    # it with ``repo_full_name=`` and ``pr_number=``. announce_handoff must accept both
    # so it can be partially applied (notifier bound) into the loopback seam.
    def handoff(*, repo_full_name: str, pr_number: int) -> Awaitable[None]:
        return announce_handoff(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            notifier=_notifier(_RecordingSinks()),
        )

    seam: LoopbackHandoff = handoff
    params = inspect.signature(announce_handoff).parameters
    assert "repo_full_name" in params
    assert "pr_number" in params
    assert callable(seam)


# --- merge reap: a merged PR closes slice issues, then reaps the PRD --------------


@pytest.mark.asyncio
async def test_merged_pr_closes_slice_issues_then_reaps_prd_when_all_closed() -> None:
    """A merged PR closes its slices, then closes the PRD when all non-hitl kids closed."""
    # Both children are closed non-hitl issues -> PRD is reaped.
    gh = _RecordingGh(
        children={
            1: [
                ChildIssue(number=10, closed=True, hitl=False),
                ChildIssue(number=11, closed=True, hitl=False),
            ]
        }
    )
    merged = MergedPullRequest(
        repo_full_name="owner/repo",
        pr_number=42,
        prd_number=1,
        slice_issues=[10, 11],
    )

    result = await reap_merged_pr(merged, gh=gh)

    # The PR's slice issues were closed first.
    assert ("owner/repo", 10) in gh.closed
    assert ("owner/repo", 11) in gh.closed
    # Then the PRD itself was reaped (closed).
    assert ("owner/repo", 1) in gh.closed
    assert result.outcome is ReapOutcome.REAPED
    assert result.prd_closed is True


@pytest.mark.asyncio
async def test_open_non_hitl_child_blocks_the_reap() -> None:
    """An open non-hitl child leaves the PRD open: the reap does not fire."""
    gh = _RecordingGh(
        children={
            1: [
                ChildIssue(number=10, closed=True, hitl=False),
                ChildIssue(number=99, closed=False, hitl=False),
            ]
        }
    )
    merged = MergedPullRequest(
        repo_full_name="owner/repo",
        pr_number=42,
        prd_number=1,
        slice_issues=[10],
    )

    result = await reap_merged_pr(merged, gh=gh)

    # The slice issue was closed, but the PRD was NOT closed.
    assert ("owner/repo", 10) in gh.closed
    assert ("owner/repo", 1) not in gh.closed
    assert result.outcome is ReapOutcome.KEPT_OPEN
    assert result.prd_closed is False


@pytest.mark.asyncio
async def test_open_hitl_child_does_not_block_the_reap() -> None:
    """An open hitl child is excluded from the reap gate: the PRD still closes."""
    gh = _RecordingGh(
        children={
            1: [
                ChildIssue(number=10, closed=True, hitl=False),
                ChildIssue(number=50, closed=False, hitl=True),
            ]
        }
    )
    merged = MergedPullRequest(
        repo_full_name="owner/repo",
        pr_number=42,
        prd_number=1,
        slice_issues=[10],
    )

    result = await reap_merged_pr(merged, gh=gh)

    # The open child is hitl, so it does not block: the PRD is reaped.
    assert ("owner/repo", 1) in gh.closed
    assert result.outcome is ReapOutcome.REAPED
    assert result.prd_closed is True


@pytest.mark.asyncio
async def test_reap_closes_slice_issues_before_checking_children() -> None:
    """Slice issues close first; the just-closed slice can complete the reap gate."""
    # The only non-hitl child is the slice issue itself, still reported open by the
    # children query. Closing it first must let the reap fire.
    gh = _RecordingGh(
        children={
            1: [ChildIssue(number=10, closed=False, hitl=False)]
        }
    )
    merged = MergedPullRequest(
        repo_full_name="owner/repo",
        pr_number=42,
        prd_number=1,
        slice_issues=[10],
    )

    result = await reap_merged_pr(merged, gh=gh)

    assert ("owner/repo", 10) in gh.closed
    assert ("owner/repo", 1) in gh.closed
    assert result.outcome is ReapOutcome.REAPED


@pytest.mark.asyncio
async def test_a_failed_slice_close_does_not_abort_the_other_closes() -> None:
    """One slice close raising must not stop the others — closes run concurrently.

    The serial-await form aborted the whole reap on the first failing close; closing the
    slices with ``asyncio.gather`` means a transient failure on #10 still lets #11 close.
    """
    gh = _RaisingGh(
        fail_on={10},
        children={1: [ChildIssue(number=11, closed=True, hitl=False)]},
    )
    merged = MergedPullRequest(
        repo_full_name="owner/repo",
        pr_number=42,
        prd_number=1,
        slice_issues=[10, 11],
    )

    result = await reap_merged_pr(merged, gh=gh)

    # #10 failed to close, but #11 was still closed despite #10 raising first-in-order.
    assert ("owner/repo", 10) not in gh.closed
    assert ("owner/repo", 11) in gh.closed
    # The successfully-closed slice is the only one reported closed.
    assert result.closed_slice_issues == [11]


@pytest.mark.asyncio
async def test_a_failed_slice_close_is_not_credited_to_the_reap_gate() -> None:
    """A slice whose close FAILED is not treated as closed by the reap gate.

    The only non-hitl child is #10, still reported open by the children query, and its
    close fails — so the gate must not credit it as just-closed and the PRD stays open.
    """
    gh = _RaisingGh(
        fail_on={10},
        children={1: [ChildIssue(number=10, closed=False, hitl=False)]},
    )
    merged = MergedPullRequest(
        repo_full_name="owner/repo",
        pr_number=42,
        prd_number=1,
        slice_issues=[10],
    )

    result = await reap_merged_pr(merged, gh=gh)

    assert result.outcome is ReapOutcome.KEPT_OPEN
    assert ("owner/repo", 1) not in gh.closed
    assert result.prd_closed is False


@pytest.mark.asyncio
async def test_reap_with_no_children_closes_prd() -> None:
    """A PRD whose children are all merged-and-closed reaps with an empty gate."""
    gh = _RecordingGh(children={1: []})
    merged = MergedPullRequest(
        repo_full_name="owner/repo",
        pr_number=42,
        prd_number=1,
        slice_issues=[],
    )

    result = await reap_merged_pr(merged, gh=gh)

    assert ("owner/repo", 1) in gh.closed
    assert result.outcome is ReapOutcome.REAPED


def test_handoff_module_exposes_no_merge_seam() -> None:
    """The reap gh seam protocol has no merge method: the retinue never merges."""
    methods = {name for name in dir(Handoff) if not name.startswith("_")}
    assert not any("merge" in name.lower() for name in methods)
    assert "close_issue" in methods
    assert "children_of" in methods


# --- production gh-cli HandoffGh: pure command-assembly + payload parsing ----------

from collections.abc import Mapping, Sequence  # noqa: E402

from retinue.handoff import (  # noqa: E402
    HandoffGh,
    _auth_env,
    _children_query_argv,
    _close_issue_argv,
    _parse_children,
)
from retinue.slicer import HITL_LABEL  # noqa: E402


class _RecordingRunner:
    """A fake gh subprocess runner: records argv + env, replays scripted stdout."""

    def __init__(self, stdout: str = "") -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self._stdout = stdout

    async def __call__(
        self, argv: Sequence[str], env: Mapping[str, str]
    ) -> str:
        self.calls.append((list(argv), dict(env)))
        return self._stdout


def test_auth_env_injects_token_via_gh_token() -> None:
    """A supplied token is threaded into the gh env as GH_TOKEN."""
    env = _auth_env("ghs_secret")
    assert env["GH_TOKEN"] == "ghs_secret"


def test_auth_env_without_token_omits_gh_token() -> None:
    """No token leaves GH_TOKEN unset so the host's own gh auth is used."""
    env = _auth_env(None)
    assert "GH_TOKEN" not in env


def test_auth_env_does_not_leak_the_parent_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gh env carries only GH_TOKEN + what gh needs — never the worker's other secrets.

    The old ``dict(os.environ)`` form handed the gh child the worker's ENTIRE environment
    (Anthropic credentials and the like). The minimal env forwards only ``PATH`` (to locate
    ``gh``) and ``HOME`` (its config / host auth), plus the injected ``GH_TOKEN``.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/home/worker")

    env = _auth_env("ghs_tok")

    assert env == {"PATH": "/usr/bin", "HOME": "/home/worker", "GH_TOKEN": "ghs_tok"}
    assert "ANTHROPIC_API_KEY" not in env


def test_close_issue_argv_targets_repo_and_issue() -> None:
    """close_issue assembles `gh issue close <n> --repo owner/repo`."""
    argv = _close_issue_argv(repo_full_name="owner/repo", issue_number=10)
    assert argv == ["issue", "close", "10", "--repo", "owner/repo"]


def test_children_query_argv_searches_part_of_marker_all_states() -> None:
    """The child query searches the `Part of #<prd>` body marker over all states."""
    argv = _children_query_argv(repo_full_name="owner/repo", prd_number=1)
    assert argv[:4] == ["issue", "list", "--repo", "owner/repo"]
    assert "--state" in argv and argv[argv.index("--state") + 1] == "all"
    search = argv[argv.index("--search") + 1]
    assert "Part of #1" in search
    fields = argv[argv.index("--json") + 1]
    assert "number" in fields and "state" in fields and "labels" in fields


def test_parse_children_maps_state_and_hitl_label() -> None:
    """gh JSON parses state->closed and the hitl label->hitl."""
    stdout = json.dumps(
        [
            {"number": 10, "state": "CLOSED", "labels": [{"name": "ready-for-agent"}]},
            {"number": 50, "state": "OPEN", "labels": [{"name": HITL_LABEL}]},
        ]
    )
    children = _parse_children(stdout)
    assert children[0] == ChildIssue(number=10, closed=True, hitl=False)
    assert children[1] == ChildIssue(number=50, closed=False, hitl=True)


def test_parse_children_handles_lowercase_state_and_empty_stdout() -> None:
    """State is case-insensitive; empty stdout yields no children."""
    assert _parse_children("") == []
    assert _parse_children("   \n") == []
    children = _parse_children(json.dumps([{"number": 1, "state": "open", "labels": []}]))
    assert children == [ChildIssue(number=1, closed=False, hitl=False)]


@pytest.mark.asyncio
async def test_handoff_gh_close_issue_runs_assembled_argv() -> None:
    """HandoffGh.close_issue drives the runner with the close argv and token env."""
    runner = _RecordingRunner()
    gh = HandoffGh(token="ghs_tok", runner=runner)

    await gh.close_issue(repo_full_name="owner/repo", issue_number=7)

    assert len(runner.calls) == 1
    argv, env = runner.calls[0]
    assert argv == ["issue", "close", "7", "--repo", "owner/repo"]
    assert env["GH_TOKEN"] == "ghs_tok"


@pytest.mark.asyncio
async def test_handoff_gh_children_of_runs_query_and_parses() -> None:
    """HandoffGh.children_of runs the query argv and parses the JSON stdout."""
    stdout = json.dumps(
        [{"number": 11, "state": "CLOSED", "labels": [{"name": HITL_LABEL}]}]
    )
    runner = _RecordingRunner(stdout=stdout)
    gh = HandoffGh(runner=runner)

    children = await gh.children_of(repo_full_name="owner/repo", prd_number=3)

    argv, _ = runner.calls[0]
    assert argv[:2] == ["issue", "list"]
    assert children == [ChildIssue(number=11, closed=True, hitl=True)]


@pytest.mark.asyncio
async def test_handoff_gh_satisfies_handoff_protocol_in_reap() -> None:
    """HandoffGh drives reap_merged_pr end to end through the fake runner."""
    stdout = json.dumps([{"number": 10, "state": "CLOSED", "labels": []}])
    runner = _RecordingRunner(stdout=stdout)
    gh: Handoff = HandoffGh(runner=runner)
    merged = MergedPullRequest(
        repo_full_name="owner/repo", pr_number=42, prd_number=1, slice_issues=[10]
    )

    result = await reap_merged_pr(merged, gh=gh)

    assert result.outcome is ReapOutcome.REAPED
    # The slice close and the PRD close both ran as `gh issue close` argv.
    close_calls = [c for c in runner.calls if c[0][:2] == ["issue", "close"]]
    assert ["issue", "close", "10", "--repo", "owner/repo"] in [c[0] for c in close_calls]
    assert ["issue", "close", "1", "--repo", "owner/repo"] in [c[0] for c in close_calls]
