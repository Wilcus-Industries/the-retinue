"""Tests for the ``issue-<N>`` -> target-branch PR opener (issue #10).

Once a build pushes a green ``issue-<N>`` branch, the orchestrator opens exactly one
PR ``issue-<N>`` -> ``config.require_target_branch()`` — but only after one precheck
passes: the target branch must exist. A missing one escalates (notify + label) and
opens no PR.

When the precheck passes, the head branch is brought up to date with the target branch
and exactly one PR is opened. Every gh-touching collaborator — the target-branch check,
the bring-up-to-date, and the open-PR action — is an injected seam faked here, so no
real gh and no network are touched. Escalations reuse the ``Notifier`` fan-out (push +
comment + label).
"""

from __future__ import annotations

import pytest

from retinue.gh import GhCommandError, GhResult
from retinue.notify import (
    CommentRequest,
    LabelRequest,
    Notifier,
    PushRequest,
)
from retinue.pr_opener import (
    GhCliPrOps,
    OpenPrRequest,
    PrOpenOutcome,
    PrOpenResult,
    PullRequest,
    open_staging_pr,
)
from retinue.repo_config import RepoConfig


class _RecordingSinks:
    """Captures each notify sink call so a test can assert an escalation fired."""

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


class FakePrOps:
    """In-memory gh seams: target-branch check, sync, and open-PR.

    ``staging_exists`` scripts the one precheck. ``synced`` records each
    bring-up-to-date call; ``opened`` records every PR opened so a test can assert
    exactly one PR was opened (or none).
    """

    def __init__(
        self,
        *,
        staging_exists: bool = True,
        existing_pr: PullRequest | None = None,
    ) -> None:
        self._staging_exists = staging_exists
        self._existing_pr = existing_pr
        self.synced: list[tuple[str, str]] = []
        self.opened: list[OpenPrRequest] = []
        self.existing_queries: list[tuple[str, str]] = []

    async def staging_exists(self, *, repo_full_name: str, branch: str) -> bool:
        return self._staging_exists

    async def existing_open_pr(
        self, *, repo_full_name: str, head: str, base: str
    ) -> PullRequest | None:
        self.existing_queries.append((head, base))
        return self._existing_pr

    async def bring_up_to_date(
        self, *, repo_full_name: str, branch: str, base: str
    ) -> None:
        self.synced.append((branch, base))

    async def open_pr(self, request: OpenPrRequest) -> PullRequest:
        self.opened.append(request)
        return PullRequest(number=101, url=f"https://gh/{request.head}->{request.base}")


def _notifier(sinks: _RecordingSinks) -> Notifier:
    return Notifier(push=sinks.push, comment=sinks.comment, label=sinks.label)


def _config(target_branch: str = "staging") -> RepoConfig:
    return RepoConfig(target_branch=target_branch)


async def _open(
    *,
    ops: FakePrOps,
    sinks: _RecordingSinks | None = None,
    config: RepoConfig | None = None,
    prd_number: int = 1,
    repo_full_name: str = "owner/repo",
    issue_number: int = 1,
    head: str | None = None,
) -> PrOpenResult:
    return await open_staging_pr(
        repo_full_name=repo_full_name,
        prd_number=prd_number,
        prd_issue_number=issue_number,
        config=config or _config(),
        ops=ops,
        notifier=_notifier(sinks or _RecordingSinks()),
        head=head or f"retinue/prd-{prd_number}",
    )


# --- happy path: target branch exists -> exactly one PR ---------------------------


@pytest.mark.asyncio
async def test_built_prd_opens_exactly_one_pr_into_staging() -> None:
    """A fully built PRD opens exactly one PR retinue/prd-<n> -> staging."""
    ops = FakePrOps()

    result = await _open(ops=ops, prd_number=7)

    assert result.outcome is PrOpenOutcome.OPENED
    assert len(ops.opened) == 1
    request = ops.opened[0]
    assert request.head == "retinue/prd-7"
    assert request.base == "staging"
    assert result.pull_request is not None
    assert result.pull_request.number == 101


@pytest.mark.asyncio
async def test_adhoc_head_opens_issue_branch_straight_into_staging() -> None:
    """An explicit ``head`` (the ad-hoc lane's ``issue-<N>``) opens straight into staging.

    The ad-hoc lane has no integration branch: it passes its ``issue-<N>`` branch as the
    head, so the head is synced and opened with no ``retinue/prd-<n>`` branch involved,
    while the precheck still runs.
    """
    ops = FakePrOps()

    result = await _open(ops=ops, prd_number=31, head="issue-31")

    assert result.outcome is PrOpenOutcome.OPENED
    assert ops.synced == [("issue-31", "staging")]  # the issue branch, not retinue/prd-31
    assert ops.opened[0].head == "issue-31"
    assert ops.opened[0].base == "staging"


@pytest.mark.asyncio
async def test_existing_open_pr_short_circuits_to_it() -> None:
    """An already-open head -> staging PR is returned instead of opening a second.

    Webhook redelivery, an arq retry, or a startup-sweep double-resume must never
    stack duplicate PRs onto staging — the opener is idempotent, as the reconcile
    contract assumes.
    """
    existing = PullRequest(number=88, url="https://gh/existing")
    ops = FakePrOps(existing_pr=existing)

    result = await _open(ops=ops, prd_number=7)

    assert result.outcome is PrOpenOutcome.OPENED
    assert result.pull_request == existing
    # No second PR was opened, and the branch wasn't re-synced under the open PR.
    assert ops.opened == []
    assert ops.synced == []
    assert ops.existing_queries == [("retinue/prd-7", "staging")]


@pytest.mark.asyncio
async def test_integration_branch_brought_up_to_date_before_pr() -> None:
    """The integration branch is synced with staging before the PR is opened."""
    ops = FakePrOps()

    await _open(ops=ops, prd_number=3)

    assert ops.synced == [("retinue/prd-3", "staging")]
    assert len(ops.opened) == 1


@pytest.mark.asyncio
async def test_custom_target_branch_is_the_pr_base() -> None:
    """A repo's non-default target branch is the sync base and the PR base."""
    ops = FakePrOps()
    config = _config(target_branch="release")

    result = await _open(ops=ops, config=config, prd_number=2)

    assert result.outcome is PrOpenOutcome.OPENED
    assert ops.synced == [("retinue/prd-2", "release")]
    assert ops.opened[0].base == "release"


@pytest.mark.asyncio
async def test_happy_path_does_not_escalate() -> None:
    """A clean open touches no escalation sinks."""
    ops = FakePrOps()
    sinks = _RecordingSinks()

    await _open(ops=ops, sinks=sinks)

    assert sinks.comments == []
    assert sinks.labels == []


# --- precheck: missing target branch escalates on its own path --------------------


@pytest.mark.asyncio
async def test_missing_target_branch_escalates_and_opens_no_pr() -> None:
    """A missing target branch escalates and opens no PR."""
    ops = FakePrOps(staging_exists=False)
    sinks = _RecordingSinks()

    result = await _open(ops=ops, sinks=sinks)

    assert result.outcome is PrOpenOutcome.STAGING_MISSING
    assert result.pull_request is None
    assert ops.opened == []
    assert ops.synced == []
    assert len(sinks.comments) == 1
    assert len(sinks.labels) == 1


@pytest.mark.asyncio
async def test_missing_target_branch_names_the_branch_in_the_escalation() -> None:
    """The escalation names the missing branch so the human knows what to fix."""
    ops = FakePrOps(staging_exists=False)
    sinks = _RecordingSinks()
    config = _config(target_branch="release")

    await _open(ops=ops, sinks=sinks, config=config, issue_number=42)

    assert "release" in sinks.comments[0].body
    assert "does not exist" in sinks.comments[0].body
    label = sinks.labels[0]
    assert label.label == "hitl"
    assert label.issue_number == 42


# --- production gh-cli PrOps: command assembly + payload parsing -------------------
#
# These exercise GhCliPrOps's pure/parseable surface against a recording fake gh
# runner that returns canned output. No live gh, no network: the fake captures every
# (args, env) so we can assert how each command is assembled and how its stdout is
# parsed back into a PR handle / boolean.


class _FakeGhRunner:
    """Records each gh invocation and replays a scripted GhResult per call.

    ``results`` is consumed in order; when it runs dry a default ok/empty result is
    returned so single-call methods need no scripting.
    """

    def __init__(self, results: list[GhResult] | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self._results = list(results or [])

    async def run(self, args: list[str], *, env: dict[str, str]) -> GhResult:
        self.calls.append((args, env))
        if self._results:
            return self._results.pop(0)
        return GhResult(exit_code=0, stdout="")


def _ops(runner: _FakeGhRunner, *, token: str = "ghs_secret") -> GhCliPrOps:
    return GhCliPrOps(runner, token=token)


@pytest.mark.asyncio
async def test_open_pr_assembles_gh_pr_create_with_title_body_base() -> None:
    """open_pr builds `gh pr create` with the repo, base, head, title, and body.

    ``gh pr create`` has no ``--json`` flag (json output belongs to list/view) —
    passing it exits 1 with "unknown flag", which broke the first live PR open —
    so the argv must carry no ``--json`` and the handle is parsed from the PR URL
    that create prints to stdout.
    """
    runner = _FakeGhRunner(
        [GhResult(exit_code=0, stdout="https://github.com/owner/repo/pull/99\n")]
    )
    request = OpenPrRequest(
        repo_full_name="owner/repo",
        head="retinue/prd-5",
        base="staging",
        title="land it",
        body="the body",
    )

    pr = await _ops(runner).open_pr(request)

    args, _ = runner.calls[0]
    assert args[:2] == ["pr", "create"]
    assert "--repo" in args and args[args.index("--repo") + 1] == "owner/repo"
    assert args[args.index("--base") + 1] == "staging"
    assert args[args.index("--head") + 1] == "retinue/prd-5"
    assert args[args.index("--title") + 1] == "land it"
    assert args[args.index("--body") + 1] == "the body"
    assert "--json" not in args
    # The returned handle is parsed straight from the URL create prints.
    assert pr == PullRequest(number=99, url="https://github.com/owner/repo/pull/99")


@pytest.mark.asyncio
async def test_open_pr_authenticates_with_gh_token_bearer() -> None:
    """Every gh call carries the token in GH_TOKEN so gh sends the bearer header."""
    runner = _FakeGhRunner(
        [GhResult(exit_code=0, stdout="https://github.com/o/r/pull/1\n")]
    )
    request = OpenPrRequest(repo_full_name="o/r", head="h", base="b", title="t", body="x")

    await _ops(runner, token="ghs_abc").open_pr(request)

    _, env = runner.calls[0]
    assert env == {"GH_TOKEN": "ghs_abc"}


@pytest.mark.asyncio
async def test_open_pr_rejects_output_with_no_pr_url() -> None:
    """Create output carrying no PR URL fails loudly, not a bogus PR handle."""
    runner = _FakeGhRunner([GhResult(exit_code=0, stdout="something went sideways")])
    request = OpenPrRequest(repo_full_name="o/r", head="h", base="b", title="t", body="x")

    with pytest.raises(ValueError):
        await _ops(runner).open_pr(request)


@pytest.mark.asyncio
async def test_open_pr_parses_the_url_amid_other_output_lines() -> None:
    """The PR URL is found even when create prints extra lines around it."""
    runner = _FakeGhRunner(
        [
            GhResult(
                exit_code=0,
                stdout=(
                    "Warning: 1 uncommitted change\n"
                    "https://github.com/o/r/pull/123\n"
                ),
            )
        ]
    )
    request = OpenPrRequest(repo_full_name="o/r", head="h", base="b", title="t", body="x")

    pr = await _ops(runner).open_pr(request)

    assert pr == PullRequest(number=123, url="https://github.com/o/r/pull/123")


@pytest.mark.asyncio
async def test_failed_gh_invocation_raises_with_stderr() -> None:
    """A non-zero gh exit surfaces as GhCommandError carrying its stderr."""
    runner = _FakeGhRunner([GhResult(exit_code=1, stdout="", stderr="boom: not allowed")])
    request = OpenPrRequest(repo_full_name="o/r", head="h", base="b", title="t", body="x")

    with pytest.raises(GhCommandError) as excinfo:
        await _ops(runner).open_pr(request)
    assert "boom: not allowed" in str(excinfo.value)


@pytest.mark.asyncio
async def test_staging_exists_maps_404_to_false() -> None:
    """A non-zero gh exit (the 404 for a missing branch) reads as not existing."""
    runner = _FakeGhRunner([GhResult(exit_code=1, stderr="HTTP 404")])

    exists = await _ops(runner).staging_exists(
        repo_full_name="owner/repo", branch="staging"
    )

    assert exists is False
    args, _ = runner.calls[0]
    assert args == ["api", "repos/owner/repo/branches/staging"]


@pytest.mark.asyncio
async def test_staging_exists_true_on_ok() -> None:
    """A 200 (ok gh exit) for the branch lookup reads as existing."""
    runner = _FakeGhRunner([GhResult(exit_code=0, stdout='{"name": "staging"}')])

    assert (
        await _ops(runner).staging_exists(
            repo_full_name="owner/repo", branch="staging"
        )
        is True
    )


@pytest.mark.asyncio
async def test_bring_up_to_date_posts_a_merge() -> None:
    """bring_up_to_date POSTs base into branch via the repo merges API."""
    runner = _FakeGhRunner([GhResult(exit_code=0, stdout="{}")])

    await _ops(runner).bring_up_to_date(
        repo_full_name="owner/repo", branch="retinue/prd-5", base="staging"
    )

    args, _ = runner.calls[0]
    assert args[:2] == ["api", "--method"]
    assert "repos/owner/repo/merges" in args
    assert "base=retinue/prd-5" in args
    assert "head=staging" in args


@pytest.mark.asyncio
async def test_bring_up_to_date_targets_repo_explicitly_not_cwd() -> None:
    """The merges call names the repo in the path, never a cwd-resolved placeholder.

    The worker runs in the retinue source, not a clone of the target repo, so any
    ``{owner}/{repo}`` placeholder would resolve against the wrong (or no) git remote.
    The repo must be threaded into the API path explicitly.
    """
    runner = _FakeGhRunner([GhResult(exit_code=0, stdout="{}")])

    await _ops(runner).bring_up_to_date(
        repo_full_name="acme/widgets", branch="retinue/prd-9", base="staging"
    )

    args, _ = runner.calls[0]
    assert "repos/acme/widgets/merges" in args
    # No gh placeholder anywhere in the argv — nothing resolves from process cwd.
    assert not any("{owner}" in arg or "{repo}" in arg for arg in args)
