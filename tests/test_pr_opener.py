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
        existing_pr: PullRequest | None = None,
    ) -> None:
        self._heimdall_installed = heimdall_installed
        self._staging_exists = staging_exists
        self._existing_pr = existing_pr
        self.synced: list[tuple[str, str]] = []
        self.opened: list[OpenPrRequest] = []
        self.existing_queries: list[tuple[str, str]] = []

    async def heimdall_installed(self, repo_full_name: str) -> bool:
        return self._heimdall_installed

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
        config=config or RepoConfig(),
        ops=ops,
        notifier=_notifier(sinks or _RecordingSinks()),
        head=head,
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
async def test_adhoc_head_opens_issue_branch_straight_into_staging() -> None:
    """An explicit ``head`` (the ad-hoc lane's ``issue-<N>``) opens straight into staging.

    The ad-hoc lane has no integration branch: it passes its ``issue-<N>`` branch as the
    head, so the head is synced and opened with no ``retinue/prd-<n>`` branch involved,
    while every shared precheck still runs.
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
async def test_heimdall_installed_reads_each_rulesets_detail() -> None:
    """heimdall_installed lists ruleset ids, then reads each ruleset's *detail* for the check.

    The repo-rulesets *list* endpoint omits each ruleset's ``rules``, so the required-check
    contexts must be read per-ruleset from the detail endpoint — querying ``.rules`` on the
    list response always yields nothing (the prior bug that left no PR ever opened).
    """
    runner = _FakeGhRunner(
        [
            GhResult(exit_code=0, stdout="[7, 9]"),  # list: ruleset ids
            GhResult(exit_code=0, stdout='["ci", "heimdall"]'),  # detail of 7: match
        ]
    )

    assert await _ops(runner).heimdall_installed("owner/repo") is True
    list_args, _ = runner.calls[0]
    assert list_args[0] == "api"
    assert list_args[1] == "repos/owner/repo/rulesets"
    assert list_args[list_args.index("--jq") + 1] == "[.[].id]"
    detail_args, _ = runner.calls[1]
    assert detail_args[1] == "repos/owner/repo/rulesets/7"


@pytest.mark.asyncio
async def test_heimdall_installed_scans_every_ruleset_until_found() -> None:
    """The check may live in a later ruleset; each ruleset's detail is scanned until matched."""
    runner = _FakeGhRunner(
        [
            GhResult(exit_code=0, stdout="[7, 9]"),
            GhResult(exit_code=0, stdout='["ci"]'),  # detail 7: no heimdall
            GhResult(exit_code=0, stdout='["heimdall"]'),  # detail 9: match
        ]
    )

    assert await _ops(runner).heimdall_installed("owner/repo") is True
    assert [c[0][1] for c in runner.calls] == [
        "repos/owner/repo/rulesets",
        "repos/owner/repo/rulesets/7",
        "repos/owner/repo/rulesets/9",
    ]


@pytest.mark.asyncio
async def test_heimdall_installed_false_when_no_ruleset_requires_it() -> None:
    """Every ruleset read, none requiring the heimdall check, reads as not installed."""
    runner = _FakeGhRunner(
        [
            GhResult(exit_code=0, stdout="[7]"),
            GhResult(exit_code=0, stdout='["ci", "lint"]'),
        ]
    )

    assert await _ops(runner).heimdall_installed("owner/repo") is False


@pytest.mark.asyncio
async def test_heimdall_installed_false_and_no_detail_call_when_no_rulesets() -> None:
    """No rulesets at all reads as not installed, with no per-ruleset detail call made."""
    runner = _FakeGhRunner([GhResult(exit_code=0, stdout="[]")])

    assert await _ops(runner).heimdall_installed("owner/repo") is False
    assert len(runner.calls) == 1  # only the list call


@pytest.mark.asyncio
async def test_heimdall_installed_false_when_rulesets_feature_unavailable() -> None:
    """A 403 from the rulesets list reads as not installed, not a crash.

    GitHub returns HTTP 403 ("Upgrade to GitHub Pro or make this repository public
    to enable this feature.") for a private repo on a free plan — the rulesets
    feature does not exist there, so no ruleset can require the heimdall check.
    That is the HEIMDALL_MISSING escalation path, not a gh failure to raise on;
    raising here crashed the PRD resume sweep after a green merged round.
    """
    runner = _FakeGhRunner(
        [
            GhResult(
                exit_code=1,
                stdout="",
                stderr=(
                    "gh: Upgrade to GitHub Pro or make this repository public to "
                    "enable this feature. (HTTP 403)"
                ),
            )
        ]
    )

    assert await _ops(runner).heimdall_installed("owner/repo") is False
    assert len(runner.calls) == 1


@pytest.mark.asyncio
async def test_heimdall_installed_still_raises_on_other_gh_failures() -> None:
    """A non-403 gh failure (auth, network) still raises rather than reading False."""
    runner = _FakeGhRunner(
        [GhResult(exit_code=1, stdout="", stderr="gh: Bad credentials (HTTP 401)")]
    )

    with pytest.raises(GhCommandError):
        await _ops(runner).heimdall_installed("owner/repo")


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
