"""Shared in-memory fakes and helpers reused across the test suite.

These are plain importable classes and helper functions (not pytest fixtures) that
several test modules lean on. They live here — rather than in a de-facto fixture module
like ``tests/test_done_check.py`` — so tests import shared fakes from one place instead
of reaching into sibling test modules. Test-specific fakes stay local to their module.

This module imports only from the ``retinue`` package and the stdlib; it must never
import from ``tests.test_*`` (that would risk an import cycle).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from retinue.adhoc_drain import FlightSnapshot, FlightState, ReadyIssue
from retinue.budget import AuthMode, BudgetGovernor, BudgetLedger
from retinue.container import Container, RunResult
from retinue.container_build import Slice
from retinue.done_check import DoneCheckReport, ReportSink, SecretResolver
from retinue.github_app import InstallationToken
from retinue.handoff import ChildIssue
from retinue.issues import CreatedIssue, IssueDraft
from retinue.pr_opener import OpenPrRequest, PullRequest
from retinue.run_ledger import RunLedgerStore

CLAUDE_MD = """# CLAUDE.md

## Definition of done

```
uv run pytest
uv run ruff check .
```
"""


class FakeAuth:
    """Mints a canned installation token and records that auth was called."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def installation_token(self, repo_full_name: str) -> InstallationToken:
        self.calls.append(repo_full_name)
        return InstallationToken(
            token="ghs_faketoken",
            clone_url=f"https://x-access-token:ghs_faketoken@github.com/{repo_full_name}.git",
        )


class FakeContainer:
    """In-memory container that scripts per-command results and records teardown.

    ``results`` maps the first argv token (e.g. "git", "uv") — or the first two, for a
    subcommand-precise script (e.g. "git rev-list") that must not also hit clone/push —
    to the :class:`RunResult` to return; the two-token key wins. An unscripted command
    returns success, except ``git rev-list`` which returns a count of ``1`` (the
    orchestrator's landed-no-commits guard; the fake models an implementer that
    committed, so green-path tests stay green by default). ``log`` appends each event
    so a test can assert command order and that destroy ran.
    """

    def __init__(self, log: list[str], results: dict[str, RunResult]) -> None:
        self._log = log
        self._results = results
        self.destroyed = False
        # Per-command exec env overrides, keyed by the first argv token, so a test can
        # assert the done-check blanks the Anthropic credential before running pytest.
        self.command_env: dict[str, Mapping[str, str]] = {}

    async def run_command(
        self, command: list[str], *, env: Mapping[str, str] | None = None
    ) -> RunResult:
        self._log.append("run:" + " ".join(command))
        if env is not None:
            self.command_env[command[0]] = env
        for key in (" ".join(command[:2]), command[0]):
            if key in self._results:
                return self._results[key]
        if command[:2] == ["git", "rev-list"]:
            return RunResult(exit_code=0, stdout="1\n")
        return RunResult(exit_code=0)

    async def destroy(self) -> None:
        self.destroyed = True
        self._log.append("destroy")


class FakeRuntime:
    """Spawns one :class:`FakeContainer`, recording the start event and injected env."""

    def __init__(
        self,
        results: dict[str, RunResult] | None = None,
        timeline: list[str] | None = None,
    ) -> None:
        self.log: list[str] = []
        self.started_env: dict[str, str] | None = None
        self.container: FakeContainer | None = None
        self._results = results or {}
        # Optional shared event list, written to by both this runtime and the git seam,
        # so a test can assert ordering *across* the container and git seams.
        self._timeline = timeline

    async def start(self, *, image: str, env: dict[str, str]) -> Container:
        self.log.append(f"start:{image}")
        if self._timeline is not None:
            self._timeline.append(f"start:{image}")
        self.started_env = env
        self.container = FakeContainer(self.log, self._results)
        return self.container


def _resolver(known: dict[str, str]) -> SecretResolver:
    async def resolve(name: str) -> str | None:
        return known.get(name)

    return resolve


def _sink(captured: list[DoneCheckReport]) -> ReportSink:
    async def report(result: DoneCheckReport) -> None:
        captured.append(result)

    return report


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


CLOCK_DEFAULT = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


class FakeClock:
    """A deterministic, advanceable time source tests read ``now()`` from.

    Constructable bare (defaults to :data:`CLOCK_DEFAULT`) or with a specific instant,
    and advanceable via :meth:`advance` — one clock satisfying the budget, cron, and
    heartbeat call sites. Tests that anchor ancillary timestamps (e.g. issue
    ``created_at``) relative to the clock should reference :data:`CLOCK_DEFAULT`.
    """

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start if start is not None else CLOCK_DEFAULT

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta


class OrchestratorBusyError(Exception):
    """Raised when a second orchestrator run tries to acquire a held run-lock."""


class OneAtATimeLock:
    """An async lock that records contention; refuses a second concurrent holder.

    Models the single-orchestrator-run guarantee: ``__aenter__`` raises
    :class:`OrchestratorBusyError` if the lock is already held rather than blocking,
    so a second run is rejected, not silently serialized.
    """

    def __init__(self) -> None:
        self.held = False
        self.acquisitions = 0

    async def __aenter__(self) -> OneAtATimeLock:
        if self.held:
            raise OrchestratorBusyError()
        self.held = True
        self.acquisitions += 1
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.held = False


class FakeAdhocGh:
    """In-memory ready-for-agent query + whole-repo flight-state truth (the fast path).

    ``flight_snapshot`` answers the dedup + stranded-recovery question for the whole repo in
    one shot (the query the drain prefers): an issue in ``in_flight_numbers`` has a branch
    *and* an open PR (a build under way or landed -> :attr:`FlightState.IN_FLIGHT`, skip); an
    issue in ``stranded_numbers`` has a pushed ``issue-<N>`` branch but *no* open PR (a green
    build whose PR never opened -> :attr:`FlightState.STRANDED`, open its PR without
    rebuilding); every other issue is :attr:`FlightState.ABSENT` (build it). ``flight_state``
    is retained so this fake still satisfies :class:`AdhocGh`, but the drain classifies from
    the snapshot, so ``flight_state`` is not exercised on the fast path.
    """

    def __init__(
        self,
        issues: list[ReadyIssue],
        *,
        in_flight_numbers: set[int] | None = None,
        stranded_numbers: set[int] | None = None,
        closed_numbers: set[int] | None = None,
        native_blockers: dict[int, list[int]] | None = None,
    ) -> None:
        self._issues = issues
        self._in_flight = in_flight_numbers or set()
        self._stranded = stranded_numbers or set()
        self._closed = closed_numbers or set()
        self._native_blockers = native_blockers or {}
        self.calls: list[str] = []
        self.list_labels: list[str] = []
        self.snapshot_calls: list[str] = []
        self.flight_state_calls: list[int] = []

    async def list_ready(
        self, *, repo_full_name: str, label: str
    ) -> list[ReadyIssue]:
        self.calls.append(repo_full_name)
        self.list_labels.append(label)
        return list(self._issues)

    async def native_blockers(
        self, *, repo_full_name: str, issue_number: int
    ) -> list[int]:
        return list(self._native_blockers.get(issue_number, []))

    async def is_closed(self, *, repo_full_name: str, issue_number: int) -> bool:
        return issue_number in self._closed

    async def flight_snapshot(self, *, repo_full_name: str) -> FlightSnapshot:
        self.snapshot_calls.append(repo_full_name)
        return FlightSnapshot(
            open_pr_heads=frozenset(f"issue-{n}" for n in self._in_flight),
            issue_branches=frozenset(
                f"issue-{n}" for n in self._in_flight | self._stranded
            ),
        )

    async def flight_state(
        self, *, repo_full_name: str, issue_number: int
    ) -> FlightState:
        self.flight_state_calls.append(issue_number)
        if issue_number in self._in_flight:
            return FlightState.IN_FLIGHT
        if issue_number in self._stranded:
            return FlightState.STRANDED
        return FlightState.ABSENT


class _FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 6, 22, tzinfo=UTC)


@dataclass
class _RecordingNotifier:
    notes: list[object] = field(default_factory=list)

    async def notify(self, notification: object) -> None:
        self.notes.append(notification)


@dataclass
class _FakePrOps:
    staging: bool = True
    opened: list[OpenPrRequest] = field(default_factory=list)

    async def staging_exists(self, *, repo_full_name: str, branch: str) -> bool:
        return self.staging

    async def existing_open_pr(
        self, *, repo_full_name: str, head: str, base: str
    ) -> PullRequest | None:
        return None

    async def bring_up_to_date(
        self, *, repo_full_name: str, branch: str, base: str
    ) -> None:
        return None

    async def open_pr(self, request: OpenPrRequest) -> PullRequest:
        self.opened.append(request)
        return PullRequest(number=99, url="https://github.com/owner/repo/pull/99")


@dataclass
class _FakeReapGh:
    children: list[ChildIssue] = field(default_factory=list)
    closed: list[int] = field(default_factory=list)

    async def close_issue(self, *, repo_full_name: str, issue_number: int) -> None:
        self.closed.append(issue_number)

    async def children_of(
        self, *, repo_full_name: str, prd_number: int
    ) -> list[ChildIssue]:
        return self.children


def _run_ledger(tmp_path: Path) -> RunLedgerStore:
    return RunLedgerStore(tmp_path / "run-ledger.sqlite3")


def _governor(tmp_path: Path, *, weekly: float = 1_000_000.0) -> BudgetGovernor:
    ledger = BudgetLedger(
        tmp_path / "budget.sqlite3",
        clock=_FixedClock(),
        auth_mode=AuthMode.API_KEY,
        weekly_budget=weekly,
    )
    return BudgetGovernor(ledger)


async def _created(draft: IssueDraft) -> CreatedIssue:
    return CreatedIssue(issue_number=1000)


async def _noop_rebuild(request: object) -> None:
    return None


def _settings(tmp_path: Path, **extra: object) -> object:
    from retinue.config import Settings

    base = dict(
        webhook_secret="s",
        api_service_token="svc-tok",
        dedupe_db_path=str(tmp_path / "dedupe.sqlite3"),
        budget_db_path=str(tmp_path / "budget.sqlite3"),
        weekly_budget=1000.0,
    )
    base.update(extra)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type, call-arg]


@dataclass
class _RecordingAdhocPipeline:
    """A pipeline recording every ``process_adhoc_pr`` call the bound build makes.

    Only the two collaborators :func:`bind_adhoc_build` touches are modeled: the
    ``process_adhoc_pr`` step (recorded here) and the ``create_issue`` the advisory review
    pass is wired through (a harmless recording stub, never invoked by these tests since
    the build is faked).
    """

    pr_calls: list[tuple[object, object]] = field(default_factory=list)
    pr_result: object | None = None
    # The shared budget governor the per-issue classify hop meters on; ``None`` is fine
    # for the table-less chain tests (they never classify) and for tests that fake the hop.
    governor: object | None = None

    async def process_adhoc_pr(self, issue: object, build: object) -> object | None:
        self.pr_calls.append((issue, build))
        return self.pr_result

    @staticmethod
    async def create_issue(draft: IssueDraft) -> CreatedIssue:
        return CreatedIssue(issue_number=1000)


def _fake_build_adhoc_issue(
    captured: dict[str, object], result: object
) -> object:
    """A drop-in ``build_adhoc_issue`` capturing its call and returning ``result``.

    Replaces the real build (which spawns a container + execs ``claude``) so the bound
    build's chain — build then ``process_adhoc_pr(issue, result)`` — is exercised with no
    Docker, gh, model, or network.
    """

    async def fake(
        issue: object,
        config: object,
        claude_md: object,
        **kwargs: object,
    ) -> object:
        captured["issue"] = issue
        captured["config"] = config
        captured["claude_md"] = claude_md
        captured["kwargs"] = kwargs
        return result

    return fake
