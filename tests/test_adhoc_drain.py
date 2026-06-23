"""Tests for the ad-hoc drain (issue #32).

The ad-hoc drain lists every open ``ready-for-agent`` non-PRD issue via the gh seam,
ranks them by ``priority:<severity>`` (no-priority lowest), and drives the ad-hoc
build+PR primitive for each up to the concurrency cap (``max_parallel``):

1. **list** — pull the repo's open ``ready-for-agent`` issues (number, labels, body),
2. **filter** — keep only the ad-hoc lane (consuming :func:`retinue.lane.classify`):
   drop any PRD-labeled issue and any issue carrying a ``Part of #<prd>`` link, since
   those route to the orchestrator lane,
3. **rank** — order by ``priority:<severity>`` with no-priority lowest,
4. **drive** — materialize each ranked issue into an :class:`AdhocIssue` through
   :meth:`AdhocIssue.from_fetched_issue` (fed the fetched body, so the ``Chain-depth:``
   lineage marker is read back and the #39/#40 review-fix chain bound stays live) and
   run the injected ad-hoc build callable, bounded by ``max_parallel``.

Every collaborator — the gh issue query and the downstream build — is injected and
faked, so the whole drain runs with no real ``gh``, no Docker, and no network.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence

import pytest

from retinue.adhoc_build import AdhocIssue, render_chain_depth
from retinue.adhoc_drain import (
    GhCli,
    ReadyIssue,
    run_adhoc_drain,
)
from retinue.loopback import Severity
from retinue.repo_config import RepoConfig


def _ready(
    number: int, *, labels: list[str] | None = None, body: str = ""
) -> ReadyIssue:
    """A ``ready-for-agent`` issue as the gh seam reports it (number, labels, body)."""
    return ReadyIssue(
        number=number,
        labels=["ready-for-agent", *(labels or [])],
        body=body,
    )


class FakeAdhocGh:
    """In-memory ready-for-agent query: returns the scripted issues."""

    def __init__(self, issues: list[ReadyIssue]) -> None:
        self._issues = issues
        self.calls: list[str] = []

    async def list_ready(self, *, repo_full_name: str) -> list[ReadyIssue]:
        self.calls.append(repo_full_name)
        return list(self._issues)


class RecordingAdhocBuild:
    """Records each AdhocIssue handed to the downstream build (the mocked build+PR)."""

    def __init__(self, *, gate: asyncio.Event | None = None) -> None:
        self.built: list[AdhocIssue] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self._gate = gate

    async def __call__(self, issue: AdhocIssue, *, repo_full_name: str) -> None:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self._gate is not None:
                await self._gate.wait()
            self.built.append(issue)
        finally:
            self.in_flight -= 1


# --- listing + filtering: only open ready-for-agent non-PRD issues ----------------


@pytest.mark.asyncio
async def test_drain_drives_the_build_for_each_ready_adhoc_issue() -> None:
    """AC1/AC3: the drain drives the ad-hoc build primitive for each ready issue."""
    gh = FakeAdhocGh([_ready(7), _ready(9)])
    build = RecordingAdhocBuild()

    await run_adhoc_drain(repo_full_name="owner/repo", gh=gh, build=build, config=RepoConfig())

    assert {issue.issue_number for issue in build.built} == {7, 9}
    assert gh.calls == ["owner/repo"]


@pytest.mark.asyncio
async def test_prd_labeled_issues_are_excluded() -> None:
    """AC4: a PRD-labeled (``prd``) issue is not an ad-hoc issue, so it is dropped."""
    gh = FakeAdhocGh([_ready(7), _ready(8, labels=["prd"])])
    build = RecordingAdhocBuild()

    await run_adhoc_drain(repo_full_name="owner/repo", gh=gh, build=build, config=RepoConfig())

    assert [issue.issue_number for issue in build.built] == [7]


@pytest.mark.asyncio
async def test_part_of_prd_issues_are_excluded() -> None:
    """AC1/AC4: a ``Part of #<prd>`` issue routes to the orchestrator lane, not ad-hoc."""
    gh = FakeAdhocGh(
        [_ready(7), _ready(8, body="Implements the thing.\n\nPart of #42")]
    )
    build = RecordingAdhocBuild()

    await run_adhoc_drain(repo_full_name="owner/repo", gh=gh, build=build, config=RepoConfig())

    assert [issue.issue_number for issue in build.built] == [7]


# --- ranking: priority:<sev>, no-priority lowest ----------------------------------


@pytest.mark.asyncio
async def test_issues_are_ranked_by_priority_no_priority_lowest() -> None:
    """AC2: issues are built highest-priority first; a no-priority issue ranks lowest."""
    gh = FakeAdhocGh(
        [
            _ready(1),  # no priority -> lowest
            _ready(2, labels=["priority:high"]),
            _ready(3, labels=["priority:critical"]),
            _ready(4, labels=["priority:low"]),
        ]
    )
    build = RecordingAdhocBuild()

    await run_adhoc_drain(repo_full_name="owner/repo", gh=gh, build=build, config=RepoConfig())

    assert [issue.issue_number for issue in build.built] == [3, 2, 4, 1]


@pytest.mark.asyncio
async def test_an_unknown_priority_label_ranks_lowest() -> None:
    """A stray ``priority:*`` value is treated as no priority (lowest), never raises."""
    gh = FakeAdhocGh(
        [_ready(1, labels=["priority:bogus"]), _ready(2, labels=["priority:high"])]
    )
    build = RecordingAdhocBuild()

    await run_adhoc_drain(repo_full_name="owner/repo", gh=gh, build=build, config=RepoConfig())

    assert [issue.issue_number for issue in build.built] == [2, 1]


# --- concurrency: bounded by max_parallel -----------------------------------------


@pytest.mark.asyncio
async def test_the_drain_is_bounded_by_max_parallel() -> None:
    """AC3: at most ``max_parallel`` ad-hoc builds run concurrently."""
    gate = asyncio.Event()
    gh = FakeAdhocGh([_ready(n) for n in range(10)])
    build = RecordingAdhocBuild(gate=gate)

    config = RepoConfig(max_parallel=3)
    drain = asyncio.create_task(
        run_adhoc_drain(repo_full_name="owner/repo", gh=gh, build=build, config=config)
    )
    # Let the scheduler fill the semaphore before releasing the gate.
    for _ in range(50):
        await asyncio.sleep(0)
    gate.set()
    await drain

    assert build.max_in_flight == 3
    assert len(build.built) == 10


@pytest.mark.asyncio
async def test_an_unset_max_parallel_builds_all_visible_issues() -> None:
    """An unset ``max_parallel`` does not block: every ready issue is still built."""
    gh = FakeAdhocGh([_ready(n) for n in range(5)])
    build = RecordingAdhocBuild()

    await run_adhoc_drain(
        repo_full_name="owner/repo", gh=gh, build=build, config=RepoConfig()
    )

    assert len(build.built) == 5


@pytest.mark.asyncio
async def test_an_empty_ready_set_drives_no_build() -> None:
    """An empty ready set drives no build and touches no downstream."""
    gh = FakeAdhocGh([])
    build = RecordingAdhocBuild()

    await run_adhoc_drain(
        repo_full_name="owner/repo", gh=gh, build=build, config=RepoConfig()
    )

    assert build.built == []


# --- chain-depth: built through from_fetched_issue (the #39/#40 bound stays live) --


@pytest.mark.asyncio
async def test_each_issue_is_built_through_from_fetched_issue() -> None:
    """AC5: a fetched body carrying ``Chain-depth: <n>`` yields ``chain_depth == n``.

    The drain MUST materialize each issue via
    :meth:`AdhocIssue.from_fetched_issue` fed the fetched body — not the bare
    constructor — so the lineage marker is read back and the #39/#40 review-fix chain
    bound stays live.
    """
    body = f"a review-fix to apply.\n\n{render_chain_depth(2)}"
    gh = FakeAdhocGh([_ready(503, body=body)])
    build = RecordingAdhocBuild()

    await run_adhoc_drain(
        repo_full_name="owner/repo", gh=gh, build=build, config=RepoConfig()
    )

    assert build.built == [
        AdhocIssue(repo_full_name="owner/repo", issue_number=503, chain_depth=2)
    ]


@pytest.mark.asyncio
async def test_a_marker_less_body_builds_a_chain_origin() -> None:
    """A ready issue with no ``Chain-depth:`` marker is a chain origin (depth 0)."""
    gh = FakeAdhocGh([_ready(29, body="a hand-filed nit, no marker")])
    build = RecordingAdhocBuild()

    await run_adhoc_drain(
        repo_full_name="owner/repo", gh=gh, build=build, config=RepoConfig()
    )

    assert build.built == [AdhocIssue(repo_full_name="owner/repo", issue_number=29)]


# --- real GhCli: command assembly, auth env, payload parsing ----------------------


class CapturingGhRunner:
    """Records the argv + env it was called with and returns a canned stdout payload."""

    def __init__(self, stdout: bytes = b"[]") -> None:
        self._stdout = stdout
        self.argv: Sequence[str] | None = None
        self.env: Mapping[str, str] | None = None

    async def __call__(self, argv: Sequence[str], env: Mapping[str, str]) -> bytes:
        self.argv = argv
        self.env = env
        return self._stdout


@pytest.mark.asyncio
async def test_ghcli_assembles_the_ready_list_command() -> None:
    """GhCli runs ``gh issue list`` scoped to the repo's open ``ready-for-agent`` issues."""
    runner = CapturingGhRunner()
    gh = GhCli(token="t0ken", runner=runner, list_limit=50)

    await gh.list_ready(repo_full_name="owner/repo")

    argv = list(runner.argv or [])
    assert argv[:3] == ["gh", "issue", "list"]
    assert "--repo" in argv and argv[argv.index("--repo") + 1] == "owner/repo"
    assert "--label" in argv and argv[argv.index("--label") + 1] == "ready-for-agent"
    assert "--state" in argv and argv[argv.index("--state") + 1] == "open"
    assert "--limit" in argv and argv[argv.index("--limit") + 1] == "50"
    # The body must be surfaced so the drain can feed from_fetched_issue.
    assert argv[argv.index("--json") + 1] == "number,labels,body"


@pytest.mark.asyncio
async def test_ghcli_puts_the_token_in_the_env_not_the_argv() -> None:
    """The token authenticates via GH_TOKEN in the child env, never on the command line."""
    runner = CapturingGhRunner()
    gh = GhCli(token="s3cret", runner=runner)

    await gh.list_ready(repo_full_name="owner/repo")

    assert (runner.env or {}).get("GH_TOKEN") == "s3cret"
    assert "s3cret" not in list(runner.argv or [])


@pytest.mark.asyncio
async def test_ghcli_omits_the_auth_env_when_no_token() -> None:
    """With no token GhCli leaves the auth env empty, deferring to gh's ambient auth."""
    runner = CapturingGhRunner()
    gh = GhCli(token=None, runner=runner)

    await gh.list_ready(repo_full_name="owner/repo")

    assert "GH_TOKEN" not in (runner.env or {})


@pytest.mark.asyncio
async def test_ghcli_parses_the_gh_json_payload() -> None:
    """GhCli parses gh's JSON listing into ReadyIssue objects with labels + body."""
    payload = json.dumps(
        [
            {
                "number": 7,
                "body": f"a fix.\n\n{render_chain_depth(1)}",
                "labels": [{"name": "ready-for-agent"}, {"name": "priority:high"}],
            },
            {
                "number": 9,
                "body": "",
                "labels": [{"name": "ready-for-agent"}],
            },
        ]
    ).encode()
    gh = GhCli(runner=CapturingGhRunner(stdout=payload))

    issues = await gh.list_ready(repo_full_name="owner/repo")

    assert [issue.number for issue in issues] == [7, 9]
    assert issues[0].labels == ["ready-for-agent", "priority:high"]
    assert issues[0].body == f"a fix.\n\n{render_chain_depth(1)}"
    assert issues[0].severity() is Severity.HIGH
    assert issues[1].severity() is None


@pytest.mark.asyncio
async def test_ghcli_rejects_a_non_array_payload() -> None:
    """A payload that is not a JSON array raises rather than silently dropping issues."""
    gh = GhCli(runner=CapturingGhRunner(stdout=b'{"number": 7}'))

    with pytest.raises(ValueError):
        await gh.list_ready(repo_full_name="owner/repo")
