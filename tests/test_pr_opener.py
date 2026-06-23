"""Tests for the staging PR opener with its heimdall precheck (issue #10).

Once a PRD's ready set drains, the orchestrator opens exactly one PR
``retinue/prd-<n>`` -> ``staging`` — but only after two prechecks pass:

1. **heimdall installed** — the repo must have the heimdall check installed; a repo
   without it escalates (notify + label) and opens no PR,
2. **staging exists** — the target ``staging`` branch must exist; a missing one
   escalates on its own path and opens no PR.

When both pass, the integration branch is brought up to date with ``staging`` and
exactly one PR is opened. Every gh-touching collaborator — the heimdall precheck, the
staging-branch check, the bring-up-to-date, and the open-PR action — is an injected
seam faked here, so no real gh and no network are touched. Escalations reuse the
``Notifier`` fan-out (push + comment + label).
"""

from __future__ import annotations

import pytest

from retinue.notify import (
    CommentRequest,
    LabelRequest,
    Notifier,
    PushRequest,
)
from retinue.pr_opener import (
    GhCliPrOps,
    GhCommandError,
    GhResult,
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
    """In-memory gh seams: heimdall precheck, staging check, sync, and open-PR.

    ``heimdall_installed`` and ``staging_exists`` script the two prechecks.
    ``synced`` records each bring-up-to-date call; ``opened`` records every PR opened
    so a test can assert exactly one PR was opened (or none).
    """

    def __init__(
        self,
        *,
        heimdall_installed: bool = True,
        staging_exists: bool = True,
    ) -> None:
        self._heimdall_installed = heimdall_installed
        self._staging_exists = staging_exists
        self.synced: list[tuple[str, str]] = []
        self.opened: list[OpenPrRequest] = []

    async def heimdall_installed(self, repo_full_name: str) -> bool:
        return self._heimdall_installed

    async def staging_exists(self, *, repo_full_name: str, branch: str) -> bool:
        return self._staging_exists

    async def bring_up_to_date(
        self, *, repo_full_name: str, branch: str, base: str
    ) -> None:
        self.synced.append((branch, base))

    async def open_pr(self, request: OpenPrRequest) -> PullRequest:
        self.opened.append(request)
        return PullRequest(number=101, url=f"https://gh/{request.head}->{request.base}")


def _notifier(sinks: _RecordingSinks) -> Notifier:
    return Notifier(push=sinks.push, comment=sinks.comment, label=sinks.label)


async def _open(
    *,
    ops: FakePrOps,
    sinks: _RecordingSinks | None = None,
    config: RepoConfig | None = None,
    prd_number: int = 1,
    repo_full_name: str = "owner/repo",
    issue_number: int = 1,
) -> PrOpenResult:
    return await open_staging_pr(
        repo_full_name=repo_full_name,
        prd_number=prd_number,
        prd_issue_number=issue_number,
        config=config or RepoConfig(),
        ops=ops,
        notifier=_notifier(sinks or _RecordingSinks()),
    )


# --- happy path: heimdall installed + staging exists -> exactly one PR ------------


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
async def test_integration_branch_brought_up_to_date_before_pr() -> None:
    """The integration branch is synced with staging before the PR is opened."""
    ops = FakePrOps()

    await _open(ops=ops, prd_number=3)

    assert ops.synced == [("retinue/prd-3", "staging")]
    assert len(ops.opened) == 1


@pytest.mark.asyncio
async def test_custom_staging_branch_is_the_pr_base() -> None:
    """A repo's non-default staging branch is the sync base and the PR base."""
    ops = FakePrOps()
    config = RepoConfig(staging_branch="release")

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


# --- precheck: heimdall not installed escalates, opens no PR ----------------------


@pytest.mark.asyncio
async def test_missing_heimdall_escalates_and_opens_no_pr() -> None:
    """A repo without heimdall installed escalates (notify + label), opens no PR."""
    ops = FakePrOps(heimdall_installed=False)
    sinks = _RecordingSinks()

    result = await _open(ops=ops, sinks=sinks)

    assert result.outcome is PrOpenOutcome.HEIMDALL_MISSING
    assert result.pull_request is None
    # No PR, and the branch is never even brought up to date.
    assert ops.opened == []
    assert ops.synced == []
    # The escalation landed on the comment + label sinks.
    assert len(sinks.comments) == 1
    assert len(sinks.labels) == 1


@pytest.mark.asyncio
async def test_missing_heimdall_labels_hitl() -> None:
    """The heimdall escalation applies the hitl label so a human can pick it up."""
    ops = FakePrOps(heimdall_installed=False)
    sinks = _RecordingSinks()

    await _open(ops=ops, sinks=sinks, issue_number=42)

    label = sinks.labels[0]
    assert label.label == "hitl"
    assert label.issue_number == 42
    assert "heimdall" in sinks.comments[0].body.lower()


# --- precheck: missing staging branch escalates on its own path -------------------


@pytest.mark.asyncio
async def test_missing_staging_escalates_and_opens_no_pr() -> None:
    """A missing staging branch escalates on its own path and opens no PR."""
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
async def test_missing_staging_names_the_branch_in_the_escalation() -> None:
    """The staging escalation names the missing branch so the human knows what to fix."""
    ops = FakePrOps(staging_exists=False)
    sinks = _RecordingSinks()
    config = RepoConfig(staging_branch="release")

    await _open(ops=ops, sinks=sinks, config=config)

    assert "release" in sinks.comments[0].body
    assert sinks.labels[0].label == "hitl"


@pytest.mark.asyncio
async def test_heimdall_checked_before_staging() -> None:
    """A repo missing both heimdall and staging escalates on the heimdall path first."""
    ops = FakePrOps(heimdall_installed=False, staging_exists=False)
    sinks = _RecordingSinks()

    result = await _open(ops=ops, sinks=sinks)

    assert result.outcome is PrOpenOutcome.HEIMDALL_MISSING
    assert "heimdall" in sinks.comments[0].body.lower()


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
    """open_pr builds `gh pr create` with the repo, base, head, title, and body."""
    runner = _FakeGhRunner(
        [GhResult(exit_code=0, stdout='{"number": 99, "url": "https://gh/pr/99"}')]
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
    assert args[args.index("--json") + 1] == "number,url"
    # The returned handle is parsed straight from gh's JSON payload.
    assert pr == PullRequest(number=99, url="https://gh/pr/99")


@pytest.mark.asyncio
async def test_open_pr_authenticates_with_gh_token_bearer() -> None:
    """Every gh call carries the token in GH_TOKEN so gh sends the bearer header."""
    runner = _FakeGhRunner([GhResult(exit_code=0, stdout='{"number": 1, "url": "u"}')])
    request = OpenPrRequest(repo_full_name="o/r", head="h", base="b", title="t", body="x")

    await _ops(runner, token="ghs_abc").open_pr(request)

    _, env = runner.calls[0]
    assert env == {"GH_TOKEN": "ghs_abc"}


@pytest.mark.asyncio
async def test_open_pr_rejects_non_json_output() -> None:
    """A non-JSON gh response fails loudly rather than yielding a bogus PR handle."""
    runner = _FakeGhRunner([GhResult(exit_code=0, stdout="not json")])
    request = OpenPrRequest(repo_full_name="o/r", head="h", base="b", title="t", body="x")

    with pytest.raises(ValueError):
        await _ops(runner).open_pr(request)


@pytest.mark.asyncio
async def test_open_pr_rejects_payload_missing_fields() -> None:
    """A JSON object lacking number/url is rejected, not silently coerced."""
    runner = _FakeGhRunner([GhResult(exit_code=0, stdout='{"number": 7}')])
    request = OpenPrRequest(repo_full_name="o/r", head="h", base="b", title="t", body="x")

    with pytest.raises(ValueError):
        await _ops(runner).open_pr(request)


@pytest.mark.asyncio
async def test_failed_gh_invocation_raises_with_stderr() -> None:
    """A non-zero gh exit surfaces as GhCommandError carrying its stderr."""
    runner = _FakeGhRunner([GhResult(exit_code=1, stdout="", stderr="boom: not allowed")])
    request = OpenPrRequest(repo_full_name="o/r", head="h", base="b", title="t", body="x")

    with pytest.raises(GhCommandError) as excinfo:
        await _ops(runner).open_pr(request)
    assert "boom: not allowed" in str(excinfo.value)


@pytest.mark.asyncio
async def test_heimdall_installed_true_when_check_in_rulesets() -> None:
    """heimdall_installed parses the ruleset contexts and membership-tests the name."""
    runner = _FakeGhRunner([GhResult(exit_code=0, stdout='["ci", "heimdall", "lint"]')])

    assert await _ops(runner).heimdall_installed("owner/repo") is True
    args, _ = runner.calls[0]
    assert args[0] == "api"
    assert args[1] == "repos/owner/repo/rulesets"


@pytest.mark.asyncio
async def test_heimdall_installed_false_when_absent() -> None:
    """A ruleset list without the heimdall context reads as not installed."""
    runner = _FakeGhRunner([GhResult(exit_code=0, stdout='["ci", "lint"]')])

    assert await _ops(runner).heimdall_installed("owner/repo") is False


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
