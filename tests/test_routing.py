"""Tests for retinue.routing: per-issue implementer-model routing in the PRD build lane.

Two layers are exercised, all offline (no gh, no network, no Docker):

* **Unit** — the ``gh issue view`` argv/parse helpers and :class:`GhCliIssueFacts`, plus
  the four :class:`PerIssueImplementerRouter` paths (classify success, pre-existing-label
  short-circuit, classification failure) against recording label/comment sinks, a fake
  classifier, and a real :class:`BudgetGovernor` on a temp ledger.
* **Integration** — :func:`retinue.wiring.bind_build_prd` driven with a real router over
  the container-exec :class:`ContainerImplementer` and the in-memory runtime/git/auth
  fakes, so the recording container runtime records the actual ``claude --model <...>``
  exec each slice launched with. This layer also proves a loopback fix-issue (a
  ``ready-for-agent`` + ``Part of #<prd>`` slice) classifies and routes through the same
  seam, and — over the ad-hoc :func:`retinue.adhoc_build.build_adhoc_issue` harness — that
  the ad-hoc planner/implementer/reviewer all launch at the issue's resolved level.

The shared :func:`retinue.routing.resolve_issue_level` hop and the ad-hoc
:func:`retinue.pipeline._resolve_adhoc_level` best-effort resolver are unit-tested directly
against the same recording fakes.

Reuses ``FakeRuntime``/``FakeAuth``/``CLAUDE_MD``/``_resolver``/``_sink`` from
``tests/test_done_check.py``, ``FakeGitOps`` from ``tests/test_orchestrator.py``, and
``FakeClock`` from ``tests/test_budget.py``.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from retinue.adhoc_build import (
    AdhocBuildResult,
    AdhocIssue,
    ContainerAdhocReviewer,
    ContainerPlanner,
    build_adhoc_issue,
)
from retinue.budget import AuthMode, BudgetGovernor, BudgetLedger
from retinue.classifier import ClassifyInput, ClassifyResult, ClaudeIssueClassifier
from retinue.container_build import Slice
from retinue.messages_api import HttpResponse
from retinue.notify import (
    CommentRequest,
    CommentSink,
    LabelRequest,
    LabelSink,
    Notifier,
    PushRequest,
)
from retinue.orchestrator import ContainerImplementer, PrdSlice
from retinue.pipeline import _CLASSIFIER_ESTIMATED_AMOUNT, _resolve_adhoc_level
from retinue.repo_config import RepoConfig, RoutingConfig, RoutingLevel
from retinue.reviewer import AgentSdkReviewGenerator
from retinue.roles import Role, resolve_effort, resolve_model
from retinue.routing import (
    _FAILURE_COMMENT,
    GhCliIssueFacts,
    PerIssueImplementer,
    PerIssueImplementerRouter,
    _issue_facts_argv,
    _parse_issue_facts,
    resolve_issue_level,
)
from retinue.wiring import BoundBuildResult, bind_build_prd
from tests.test_budget import FakeClock
from tests.test_done_check import CLAUDE_MD, FakeAuth, FakeRuntime, _resolver, _sink
from tests.test_orchestrator import FakeGitOps

pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnhandledThreadExceptionWarning"
)


# --- shared fixtures / fakes -----------------------------------------------------


def _routing() -> RoutingConfig:
    """A two-level table whose implementer models differ per level; default standard."""
    return RoutingConfig(
        default="standard",
        levels={
            "trivial": RoutingLevel(
                description="tiny one-file typo or doc fix",
                roles={"implementer": {"model": "implementer-trivial"}},  # type: ignore[dict-item]
            ),
            "standard": RoutingLevel(
                description="a normal multi-file feature slice",
                roles={"implementer": {"model": "implementer-standard"}},  # type: ignore[dict-item]
            ),
        },
    )


def _config() -> RepoConfig:
    return RepoConfig(staging_branch="staging", max_parallel=1, routing=_routing())


class _RecordingLabels:
    """Records every label applied (models gh's add-only ``--add-label``)."""

    def __init__(self) -> None:
        self.calls: list[LabelRequest] = []

    async def __call__(self, request: LabelRequest) -> None:
        self.calls.append(request)


class _RecordingComments:
    """Records every comment posted."""

    def __init__(self) -> None:
        self.calls: list[CommentRequest] = []

    async def __call__(self, request: CommentRequest) -> None:
        self.calls.append(request)


class _RecordingClassifier:
    """Canned classifier returning a scripted result; records each call."""

    def __init__(self, result: ClassifyResult) -> None:
        self._result = result
        self.calls: list[ClassifyInput] = []

    async def __call__(self, issue: ClassifyInput) -> ClassifyResult:
        self.calls.append(issue)
        return self._result


class _UnusedClassifier:
    """Fails the test if invoked — for the pre-existing-label short-circuit path."""

    async def __call__(self, issue: ClassifyInput) -> ClassifyResult:
        raise AssertionError("classifier must not be called")


class _FactsFor:
    """A fake :data:`IssueFactsSource` returning per-issue-number canned facts."""

    def __init__(self, facts: dict[int, ClassifyInput]) -> None:
        self._facts = facts
        self.calls: list[tuple[str, int]] = []

    async def __call__(
        self, repo_full_name: str, issue_number: int
    ) -> ClassifyInput:
        self.calls.append((repo_full_name, issue_number))
        return self._facts[issue_number]


class _RaisingFacts:
    """A fake :data:`IssueFactsSource` that raises — models a gh flake / bad JSON."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __call__(
        self, repo_full_name: str, issue_number: int
    ) -> ClassifyInput:
        raise self._exc


class _RaisingComments:
    """A comment sink that raises — models a failed failure-comment post."""

    async def __call__(self, request: CommentRequest) -> None:
        raise RuntimeError("comment post failed")


def _governor(tmp_path: Path, *, weekly: float = 1000.0) -> BudgetGovernor:
    return BudgetGovernor(
        BudgetLedger(
            tmp_path / "budget.sqlite3",
            clock=FakeClock(),
            auth_mode=AuthMode.API_KEY,
            weekly_budget=weekly,
        )
    )


def _slice(issue_number: int = 1) -> Slice:
    return Slice(repo_full_name="owner/repo", issue_number=issue_number, prd_number=7)


# --- gh issue-facts helpers ------------------------------------------------------


def test_issue_facts_argv_requests_title_body_labels() -> None:
    """The argv fetches exactly title/body/labels for the one issue in the repo."""
    assert _issue_facts_argv("owner/repo", 42) == [
        "issue",
        "view",
        "42",
        "--repo",
        "owner/repo",
        "--json",
        "title,body,labels",
    ]


def test_parse_issue_facts_maps_title_body_and_label_names() -> None:
    """The parser reads title/body and flattens ``labels[].name`` into a ClassifyInput."""
    stdout = json.dumps(
        {
            "title": "Add a widget",
            "body": "Wire it in.",
            "labels": [{"name": "ready-for-agent"}, {"name": "level:trivial"}],
        }
    )

    facts = _parse_issue_facts(stdout)

    assert facts == ClassifyInput(
        title="Add a widget",
        body="Wire it in.",
        labels=["ready-for-agent", "level:trivial"],
        prd_body=None,
    )


def test_parse_issue_facts_tolerates_missing_keys() -> None:
    """A payload lacking keys yields empty defaults rather than raising KeyError."""
    facts = _parse_issue_facts("{}")

    assert facts == ClassifyInput(title="", body="", labels=[], prd_body=None)


@pytest.mark.asyncio
async def test_gh_cli_issue_facts_runs_argv_and_parses() -> None:
    """GhCliIssueFacts runs the view argv through the runner and parses its JSON."""
    seen: list[list[str]] = []

    async def runner(argv: list[str]) -> str:
        seen.append(argv)
        return json.dumps(
            {"title": "T", "body": "B", "labels": [{"name": "feature"}]}
        )

    facts = await GhCliIssueFacts(runner)("owner/repo", 7)

    assert seen == [_issue_facts_argv("owner/repo", 7)]
    assert facts == ClassifyInput(title="T", body="B", labels=["feature"])


# --- PerIssueImplementerRouter: the four routing paths ---------------------------


def _router(
    tmp_path: Path,
    *,
    classify: object,
    labels: _RecordingLabels,
    comments: _RecordingComments,
    facts: _FactsFor,
    governor: BudgetGovernor,
) -> PerIssueImplementerRouter:
    return PerIssueImplementerRouter(
        base_implementer=ContainerImplementer(credential="k"),
        config=_config(),
        classify=classify,  # type: ignore[arg-type]
        label_sink=labels,
        comment_sink=comments,
        issue_facts=facts,
        governor=governor,
        classifier_charge=_CLASSIFIER_ESTIMATED_AMOUNT,
    )


@pytest.mark.asyncio
async def test_router_classify_success_routes_model_labels_and_charges(
    tmp_path: Path,
) -> None:
    """No pre-existing label -> classify -> the level's model, a level label, one charge."""
    labels, comments = _RecordingLabels(), _RecordingComments()
    governor = _governor(tmp_path)
    facts = _FactsFor({1: ClassifyInput(title="typo", body="b", labels=[])})
    router = _router(
        tmp_path,
        classify=_RecordingClassifier(ClassifyResult(level="trivial")),
        labels=labels,
        comments=comments,
        facts=facts,
        governor=governor,
    )

    implementer = await router(_slice(1))

    assert isinstance(implementer, ContainerImplementer)
    assert implementer.model == "implementer-trivial"
    assert [r.label for r in labels.calls] == ["level:trivial"]
    assert comments.calls == []
    assert await governor._ledger.trailing_24h_spend() == pytest.approx(
        _CLASSIFIER_ESTIMATED_AMOUNT
    )


@pytest.mark.asyncio
async def test_router_logs_the_routed_level_and_model(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Each routed slice leaves a log line naming its level and implementer model.

    The routed model is otherwise invisible in production — the implementer runs
    silently in its container — so the log line is the live-verification surface
    for per-issue routing (PRD #58's dogfood gate reads it).
    """
    facts = _FactsFor({1: ClassifyInput(title="typo", body="b", labels=[])})
    router = _router(
        tmp_path,
        classify=_RecordingClassifier(ClassifyResult(level="trivial")),
        labels=_RecordingLabels(),
        comments=_RecordingComments(),
        facts=facts,
        governor=_governor(tmp_path),
    )

    with caplog.at_level("INFO", logger="retinue.routing"):
        await router(_slice(1))

    assert "level 'trivial'" in caplog.text
    assert "implementer-trivial" in caplog.text


@pytest.mark.asyncio
async def test_router_preexisting_label_short_circuits_without_charge(
    tmp_path: Path,
) -> None:
    """A known ``level:`` label skips the classifier and records no charge."""
    labels, comments = _RecordingLabels(), _RecordingComments()
    governor = _governor(tmp_path)
    facts = _FactsFor(
        {1: ClassifyInput(title="x", body="b", labels=["level:standard"])}
    )
    router = _router(
        tmp_path,
        classify=_UnusedClassifier(),
        labels=labels,
        comments=comments,
        facts=facts,
        governor=governor,
    )

    implementer = await router(_slice(1))

    assert implementer.model == "implementer-standard"  # type: ignore[attr-defined]
    assert await governor._ledger.trailing_24h_spend() == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_router_classification_failure_defaults_and_comments(
    tmp_path: Path,
) -> None:
    """Classifier returns no level -> default level's model + an explanatory comment."""
    labels, comments = _RecordingLabels(), _RecordingComments()
    governor = _governor(tmp_path)
    facts = _FactsFor({1: ClassifyInput(title="x", body="b", labels=[])})
    router = _router(
        tmp_path,
        classify=_RecordingClassifier(ClassifyResult(level=None)),
        labels=labels,
        comments=comments,
        facts=facts,
        governor=governor,
    )

    implementer = await router(_slice(1))

    # The default level ("standard") builds the slice.
    assert implementer.model == "implementer-standard"  # type: ignore[attr-defined]
    # One comment, naming the applied default level.
    assert len(comments.calls) == 1
    assert comments.calls[0].body == _FAILURE_COMMENT.format(level="standard")
    assert comments.calls[0].repo_full_name == "owner/repo"
    assert comments.calls[0].issue_number == 1
    # The classifier ran, so its charge is metered even on failure.
    assert await governor._ledger.trailing_24h_spend() == pytest.approx(
        _CLASSIFIER_ESTIMATED_AMOUNT
    )


@pytest.mark.asyncio
async def test_router_facts_failure_falls_back_to_base_implementer(
    tmp_path: Path,
) -> None:
    """A resolution failure (gh flake / bad JSON) falls back to the base implementer.

    The facts fetch raising must not propagate — that would escalate the slice with
    zero triage retries. Instead the router logs and returns the injected base
    implementer unchanged, mirroring resolve_level's best-effort label contract.
    """
    labels, comments = _RecordingLabels(), _RecordingComments()
    governor = _governor(tmp_path)
    base = ContainerImplementer(credential="k", model="base-model")
    router = PerIssueImplementerRouter(
        base_implementer=base,
        config=_config(),
        classify=_UnusedClassifier(),
        label_sink=labels,
        comment_sink=comments,
        issue_facts=_RaisingFacts(RuntimeError("gh view exited non-zero")),
        governor=governor,
        classifier_charge=_CLASSIFIER_ESTIMATED_AMOUNT,
    )

    implementer = await router(_slice(1))

    # The exact injected base implementer is returned — no model swap, no charge.
    assert implementer is base
    assert comments.calls == []
    assert await governor._ledger.trailing_24h_spend() == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_router_misuse_without_a_routing_table_fails_loudly(
    tmp_path: Path,
) -> None:
    """Constructing the router for a table-less repo is misuse and must propagate.

    The best-effort fallback exists for runtime flakes (gh, JSON, comment posts) — not
    for programming errors. ``resolve_issue_level`` asserts the table exists; that
    :class:`AssertionError` must escape ``__call__``'s guard rather than be silently
    downgraded to a base-implementer fallback, or the misuse would never surface.
    """
    router = PerIssueImplementerRouter(
        base_implementer=ContainerImplementer(credential="k", model="base-model"),
        config=RepoConfig(staging_branch="staging"),  # no routing table
        classify=_UnusedClassifier(),
        label_sink=_RecordingLabels(),
        comment_sink=_RecordingComments(),
        issue_facts=_FactsFor({1: ClassifyInput(title="x", body="b", labels=[])}),
        governor=_governor(tmp_path),
        classifier_charge=_CLASSIFIER_ESTIMATED_AMOUNT,
    )

    with pytest.raises(AssertionError):
        await router(_slice(1))


@pytest.mark.asyncio
async def test_router_failure_comment_post_failure_falls_back_to_base(
    tmp_path: Path,
) -> None:
    """A failed failure-comment post falls back to the base implementer, not an escalation."""
    labels = _RecordingLabels()
    governor = _governor(tmp_path)
    facts = _FactsFor({1: ClassifyInput(title="x", body="b", labels=[])})
    base = ContainerImplementer(credential="k", model="base-model")
    router = PerIssueImplementerRouter(
        base_implementer=base,
        config=_config(),
        classify=_RecordingClassifier(ClassifyResult(level=None)),
        label_sink=labels,
        comment_sink=_RaisingComments(),
        issue_facts=facts,
        governor=governor,
        classifier_charge=_CLASSIFIER_ESTIMATED_AMOUNT,
    )

    implementer = await router(_slice(1))

    assert implementer is base


# --- integration: bind_build_prd driving a real router --------------------------


class _TitleRoutingTransport:
    """A fake Messages-API transport routing on the issue title in the prompt.

    :meth:`ClaudeIssueClassifier._build_prompt` embeds ``Issue title: <title>`` in the
    user message; this transport reads that content and picks ``trivial`` when the title
    carries the ``PICKTRIVIAL`` sentinel (else ``standard``), so two slices classify to two
    different levels. The sentinel — not the bare word "trivial", which also appears in the
    prompt's level list — is what routes. ``non_200`` returns a failing response on every
    call (to exercise the classification-failure path).
    """

    def __init__(self, *, non_200: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self._non_200 = non_200

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, object]
    ) -> HttpResponse:
        self.calls.append(json)
        if self._non_200:
            return HttpResponse(status_code=503, body={})
        messages = json["messages"]
        assert isinstance(messages, list)
        content = messages[0]["content"]
        assert isinstance(content, str)
        level = "trivial" if "PICKTRIVIAL" in content else "standard"
        import json as _json

        return HttpResponse(
            status_code=200,
            body={"content": [{"type": "text", "text": _json.dumps({"level": level})}]},
        )


async def _noop_push(request: PushRequest) -> None:
    return None


async def _noop_comment(request: CommentRequest) -> None:
    return None


async def _noop_label(request: LabelRequest) -> None:
    return None


def _quiet_notifier() -> Notifier:
    return Notifier(push=_noop_push, comment=_noop_comment, label=_noop_label)


async def _no_create(draft: object) -> object:
    raise AssertionError("create_issue must not be called on a clean build")


def _prd_slice(issue_number: int) -> PrdSlice:
    return PrdSlice(
        repo_full_name="owner/repo", issue_number=issue_number, prd_number=7
    )


def _build_router(
    *,
    config: RepoConfig,
    transport: _TitleRoutingTransport,
    facts: _FactsFor,
    comments: CommentSink,
    governor: BudgetGovernor,
    labels: LabelSink = _noop_label,
) -> PerIssueImplementerRouter:
    assert config.routing is not None
    return PerIssueImplementerRouter(
        base_implementer=ContainerImplementer(
            credential="sk-ant-api03-k",
            model=resolve_model(Role.IMPLEMENTER, config),
        ),
        config=config,
        classify=ClaudeIssueClassifier(
            credential="sk-ant-api03-k",
            transport=transport,
            routing=config.routing,
        ),
        label_sink=labels,
        comment_sink=comments,
        issue_facts=facts,
        governor=governor,
        classifier_charge=_CLASSIFIER_ESTIMATED_AMOUNT,
    )


def _bind(
    *,
    tmp_path: Path,
    governor: BudgetGovernor,
    runtime: FakeRuntime,
    resolve_implementer: PerIssueImplementer | None,
    base: ContainerImplementer,
    estimated_amount: float = 1.0,
) -> Callable[..., Awaitable[BoundBuildResult]]:
    return bind_build_prd(
        implementer=base,
        governor=governor,
        notifier=_quiet_notifier(),
        create_issue=_no_create,  # type: ignore[arg-type]
        retry_store_path=tmp_path / "retries.sqlite3",
        estimated_amount=estimated_amount,
        git=FakeGitOps(),
        auth=FakeAuth(),
        runtime=runtime,
        resolve_secret=_resolver({}),
        report=_sink([]),
        resolve_implementer=resolve_implementer,
    )


def _launched_models(runtime: FakeRuntime) -> list[str]:
    """The ``--model`` value of every ``claude`` exec the runtime recorded, in order."""
    models: list[str] = []
    for event in runtime.log:
        if not event.startswith("run:claude"):
            continue
        parts = event.split()
        idx = parts.index("--model")
        models.append(parts[idx + 1])
    return models


@pytest.mark.asyncio
async def test_two_slices_launch_on_their_levels_models(tmp_path: Path) -> None:
    """AC-1: two slices classifying to different levels each exec their level's model."""
    config = _config()
    governor = _governor(tmp_path)
    runtime = FakeRuntime()
    transport = _TitleRoutingTransport()
    facts = _FactsFor(
        {
            1: ClassifyInput(title="PICKTRIVIAL doc fix", body="b", labels=[]),
            2: ClassifyInput(title="a normal feature", body="b", labels=[]),
        }
    )
    base = ContainerImplementer(
        credential="sk-ant-api03-k", model=resolve_model(Role.IMPLEMENTER, config)
    )
    router = _build_router(
        config=config,
        transport=transport,
        facts=facts,
        comments=_RecordingComments(),
        governor=governor,
    )
    build_prd = _bind(
        tmp_path=tmp_path,
        governor=governor,
        runtime=runtime,
        resolve_implementer=router,
        base=base,
    )

    result = await build_prd(
        repo_full_name="owner/repo",
        prd_number=7,
        slices=[_prd_slice(1), _prd_slice(2)],
        config=config,
        claude_md=CLAUDE_MD,
    )

    assert result.prd_build is not None
    assert result.prd_build.merged_issues == [1, 2]
    models = _launched_models(runtime)
    assert "implementer-trivial" in models
    assert "implementer-standard" in models


@pytest.mark.asyncio
async def test_classification_failure_builds_default_and_comments(
    tmp_path: Path,
) -> None:
    """AC-2: a classifier failure builds at the default level and posts an explanation."""
    config = _config()
    governor = _governor(tmp_path)
    runtime = FakeRuntime()
    transport = _TitleRoutingTransport(non_200=True)  # both attempts fail
    facts = _FactsFor({1: ClassifyInput(title="x", body="b", labels=[])})
    comments = _RecordingComments()
    base = ContainerImplementer(
        credential="sk-ant-api03-k", model=resolve_model(Role.IMPLEMENTER, config)
    )
    router = _build_router(
        config=config,
        transport=transport,
        facts=facts,
        comments=comments,
        governor=governor,
    )
    build_prd = _bind(
        tmp_path=tmp_path,
        governor=governor,
        runtime=runtime,
        resolve_implementer=router,
        base=base,
    )

    result = await build_prd(
        repo_full_name="owner/repo",
        prd_number=7,
        slices=[_prd_slice(1)],
        config=config,
        claude_md=CLAUDE_MD,
    )

    assert result.prd_build is not None
    assert result.prd_build.merged_issues == [1]
    assert _launched_models(runtime) == ["implementer-standard"]
    assert len(comments.calls) == 1
    assert comments.calls[0].body == _FAILURE_COMMENT.format(level="standard")


@pytest.mark.asyncio
async def test_classifier_charge_lands_and_gate_estimate_unchanged(
    tmp_path: Path,
) -> None:
    """AC-3: one classifying slice charges the gate estimate plus one classifier charge."""
    config = _config()
    governor = _governor(tmp_path)
    runtime = FakeRuntime()
    transport = _TitleRoutingTransport()
    facts = _FactsFor({1: ClassifyInput(title="PICKTRIVIAL fix", body="b", labels=[])})
    base = ContainerImplementer(
        credential="sk-ant-api03-k", model=resolve_model(Role.IMPLEMENTER, config)
    )
    router = _build_router(
        config=config,
        transport=transport,
        facts=facts,
        comments=_RecordingComments(),
        governor=governor,
    )
    build_prd = _bind(
        tmp_path=tmp_path,
        governor=governor,
        runtime=runtime,
        resolve_implementer=router,
        base=base,
        estimated_amount=1.0,
    )

    await build_prd(
        repo_full_name="owner/repo",
        prd_number=7,
        slices=[_prd_slice(1)],
        config=config,
        claude_md=CLAUDE_MD,
    )

    assert await governor._ledger.trailing_24h_spend() == pytest.approx(
        1.0 + _CLASSIFIER_ESTIMATED_AMOUNT
    )


@pytest.mark.asyncio
async def test_table_less_repo_uses_registry_default_and_no_classifier(
    tmp_path: Path,
) -> None:
    """AC-4: with no routing table the build uses the plain default and never classifies."""
    config = RepoConfig(staging_branch="staging", max_parallel=1)  # no routing
    governor = _governor(tmp_path)
    runtime = FakeRuntime()
    transport = _TitleRoutingTransport()
    facts = _FactsFor({1: ClassifyInput(title="x", body="b", labels=[])})
    base = ContainerImplementer(
        credential="sk-ant-api03-k", model=resolve_model(Role.IMPLEMENTER, config)
    )
    build_prd = _bind(
        tmp_path=tmp_path,
        governor=governor,
        runtime=runtime,
        resolve_implementer=None,  # production passes None for a table-less repo
        base=base,
    )

    result = await build_prd(
        repo_full_name="owner/repo",
        prd_number=7,
        slices=[_prd_slice(1)],
        config=config,
        claude_md=CLAUDE_MD,
    )

    assert result.prd_build is not None
    assert result.prd_build.merged_issues == [1]
    assert _launched_models(runtime) == [resolve_model(Role.IMPLEMENTER, RepoConfig())]
    assert _launched_models(runtime) == ["claude-sonnet-4-6"]
    # The classifier transport and issue-facts fetch were never touched.
    assert transport.calls == []
    assert facts.calls == []


# --- resolve_issue_level: the shared classify-one-issue hop -----------------------


@pytest.mark.asyncio
async def test_resolve_issue_level_classify_success_labels_and_charges(
    tmp_path: Path,
) -> None:
    """No pre-existing label -> classify -> the level name, a level label, one charge."""
    labels, comments = _RecordingLabels(), _RecordingComments()
    governor = _governor(tmp_path)
    facts = _FactsFor({1: ClassifyInput(title="typo", body="b", labels=[])})

    level = await resolve_issue_level(
        "owner/repo",
        1,
        _config(),
        classify=_RecordingClassifier(ClassifyResult(level="trivial")),
        label_sink=labels,
        comment_sink=comments,
        issue_facts=facts,
        governor=governor,
        classifier_charge=_CLASSIFIER_ESTIMATED_AMOUNT,
    )

    assert level == "trivial"
    assert [r.label for r in labels.calls] == ["level:trivial"]
    assert comments.calls == []
    assert await governor._ledger.trailing_24h_spend() == pytest.approx(
        _CLASSIFIER_ESTIMATED_AMOUNT
    )


@pytest.mark.asyncio
async def test_resolve_issue_level_preexisting_label_short_circuits(
    tmp_path: Path,
) -> None:
    """A known ``level:`` label skips the classifier: no call, no meter, no label write."""
    labels, comments = _RecordingLabels(), _RecordingComments()
    governor = _governor(tmp_path)
    facts = _FactsFor(
        {1: ClassifyInput(title="x", body="b", labels=["level:standard"])}
    )

    level = await resolve_issue_level(
        "owner/repo",
        1,
        _config(),
        classify=_UnusedClassifier(),
        label_sink=labels,
        comment_sink=comments,
        issue_facts=facts,
        governor=governor,
        classifier_charge=_CLASSIFIER_ESTIMATED_AMOUNT,
    )

    assert level == "standard"
    assert labels.calls == []
    assert await governor._ledger.trailing_24h_spend() == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_resolve_issue_level_failure_defaults_and_comments(
    tmp_path: Path,
) -> None:
    """A classifier failure returns the default level, comments, and still meters once."""
    labels, comments = _RecordingLabels(), _RecordingComments()
    governor = _governor(tmp_path)
    facts = _FactsFor({1: ClassifyInput(title="x", body="b", labels=[])})

    level = await resolve_issue_level(
        "owner/repo",
        1,
        _config(),
        classify=_RecordingClassifier(ClassifyResult(level=None)),
        label_sink=labels,
        comment_sink=comments,
        issue_facts=facts,
        governor=governor,
        classifier_charge=_CLASSIFIER_ESTIMATED_AMOUNT,
    )

    assert level == "standard"
    assert labels.calls == []
    assert len(comments.calls) == 1
    assert comments.calls[0].body == _FAILURE_COMMENT.format(level="standard")
    assert await governor._ledger.trailing_24h_spend() == pytest.approx(
        _CLASSIFIER_ESTIMATED_AMOUNT
    )


# --- loopback fix-issues route through the identical seam -------------------------


def _fix_issue_facts(issue_number: int, *, title: str) -> ClassifyInput:
    """Facts shaped like a loopback fix-issue: ``ready-for-agent`` + ``Part of #<prd>``.

    A loopback fix-issue (``retinue.loopback._file_fix_issue``) is an ordinary
    orchestrator-lane slice — no ``level:`` label — so it must classify and route through
    the same :class:`PerIssueImplementerRouter` seam as any PRD slice.
    """
    return ClassifyInput(
        title=title,
        body="A heimdall blocking finding to fix.\n\nPart of #7",
        labels=["ready-for-agent"],
    )


@pytest.mark.asyncio
async def test_loopback_fix_issue_classifies_labels_and_routes(tmp_path: Path) -> None:
    """A fix-issue-shaped slice classifies once, gets its label, meters, and routes."""
    config = _config()
    governor = _governor(tmp_path)
    runtime = FakeRuntime()
    transport = _TitleRoutingTransport()
    labels = _RecordingLabels()
    facts = _FactsFor(
        {1: _fix_issue_facts(1, title="Heimdall fix: PICKTRIVIAL correct the typo")}
    )
    base = ContainerImplementer(
        credential="sk-ant-api03-k", model=resolve_model(Role.IMPLEMENTER, config)
    )
    router = _build_router(
        config=config,
        transport=transport,
        facts=facts,
        comments=_RecordingComments(),
        governor=governor,
        labels=labels,
    )
    build_prd = _bind(
        tmp_path=tmp_path,
        governor=governor,
        runtime=runtime,
        resolve_implementer=router,
        base=base,
    )

    result = await build_prd(
        repo_full_name="owner/repo",
        prd_number=7,
        slices=[_prd_slice(1)],
        config=config,
        claude_md=CLAUDE_MD,
    )

    assert result.prd_build is not None
    assert result.prd_build.merged_issues == [1]
    # The classifier ran exactly once for the one fix-issue.
    assert len(transport.calls) == 1
    # Its resolved level label was applied.
    assert [r.label for r in labels.calls] == ["level:trivial"]
    # It launched on the resolved level's implementer model.
    assert _launched_models(runtime) == ["implementer-trivial"]
    # The classifier charge was metered on top of the build gate estimate.
    assert await governor._ledger.trailing_24h_spend() == pytest.approx(
        1.0 + _CLASSIFIER_ESTIMATED_AMOUNT
    )


@pytest.mark.asyncio
async def test_two_loopback_fix_issues_each_classified_individually(
    tmp_path: Path,
) -> None:
    """Two fix-issues classifying to different levels each exec their own level's model."""
    config = _config()
    governor = _governor(tmp_path)
    runtime = FakeRuntime()
    transport = _TitleRoutingTransport()
    labels = _RecordingLabels()
    facts = _FactsFor(
        {
            1: _fix_issue_facts(1, title="Heimdall fix: PICKTRIVIAL a one-line typo"),
            2: _fix_issue_facts(2, title="Heimdall fix: a substantive refactor"),
        }
    )
    base = ContainerImplementer(
        credential="sk-ant-api03-k", model=resolve_model(Role.IMPLEMENTER, config)
    )
    router = _build_router(
        config=config,
        transport=transport,
        facts=facts,
        comments=_RecordingComments(),
        governor=governor,
        labels=labels,
    )
    build_prd = _bind(
        tmp_path=tmp_path,
        governor=governor,
        runtime=runtime,
        resolve_implementer=router,
        base=base,
    )

    result = await build_prd(
        repo_full_name="owner/repo",
        prd_number=7,
        slices=[_prd_slice(1), _prd_slice(2)],
        config=config,
        claude_md=CLAUDE_MD,
    )

    assert result.prd_build is not None
    assert result.prd_build.merged_issues == [1, 2]
    # Each fix-issue was classified on its own facts.
    assert len(transport.calls) == 2
    models = _launched_models(runtime)
    assert "implementer-trivial" in models
    assert "implementer-standard" in models
    assert sorted(r.label for r in labels.calls) == ["level:standard", "level:trivial"]


# --- ad-hoc lane: _resolve_adhoc_level (unit) ------------------------------------


def _adhoc_issue(issue_number: int = 1) -> AdhocIssue:
    return AdhocIssue(repo_full_name="owner/repo", issue_number=issue_number)


@pytest.mark.asyncio
async def test_resolve_adhoc_level_table_less_returns_none_without_classifying(
    tmp_path: Path,
) -> None:
    """A table-less repo resolves ``None`` and never fetches facts or classifies."""
    governor = _governor(tmp_path)
    facts = _FactsFor({1: ClassifyInput(title="x", body="b", labels=[])})

    level = await _resolve_adhoc_level(
        _adhoc_issue(),
        RepoConfig(staging_branch="staging"),  # no routing table
        classify=_UnusedClassifier(),  # type: ignore[arg-type]
        label_sink=_RecordingLabels(),  # type: ignore[arg-type]
        comment_sink=_RecordingComments(),  # type: ignore[arg-type]
        issue_facts=facts,  # type: ignore[arg-type]
        governor=governor,
    )

    assert level is None
    assert facts.calls == []
    assert await governor._ledger.trailing_24h_spend() == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_resolve_adhoc_level_classify_success_labels_and_charges(
    tmp_path: Path,
) -> None:
    """A routing repo classifies once, applies the label, and meters one charge."""
    labels, comments = _RecordingLabels(), _RecordingComments()
    governor = _governor(tmp_path)
    facts = _FactsFor({1: ClassifyInput(title="typo", body="b", labels=[])})

    level = await _resolve_adhoc_level(
        _adhoc_issue(),
        _config(),
        classify=_RecordingClassifier(ClassifyResult(level="trivial")),  # type: ignore[arg-type]
        label_sink=labels,  # type: ignore[arg-type]
        comment_sink=comments,  # type: ignore[arg-type]
        issue_facts=facts,  # type: ignore[arg-type]
        governor=governor,
    )

    assert level == "trivial"
    assert [r.label for r in labels.calls] == ["level:trivial"]
    assert comments.calls == []
    assert await governor._ledger.trailing_24h_spend() == pytest.approx(
        _CLASSIFIER_ESTIMATED_AMOUNT
    )


@pytest.mark.asyncio
async def test_resolve_adhoc_level_preexisting_label_skips_classifier(
    tmp_path: Path,
) -> None:
    """A known ``level:`` label routes by that label with no classifier call or charge."""
    labels = _RecordingLabels()
    governor = _governor(tmp_path)
    facts = _FactsFor(
        {1: ClassifyInput(title="x", body="b", labels=["level:standard"])}
    )

    level = await _resolve_adhoc_level(
        _adhoc_issue(),
        _config(),
        classify=_UnusedClassifier(),  # type: ignore[arg-type]
        label_sink=labels,  # type: ignore[arg-type]
        comment_sink=_RecordingComments(),  # type: ignore[arg-type]
        issue_facts=facts,  # type: ignore[arg-type]
        governor=governor,
    )

    assert level == "standard"
    assert labels.calls == []
    assert await governor._ledger.trailing_24h_spend() == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_resolve_adhoc_level_failure_defaults_and_comments(
    tmp_path: Path,
) -> None:
    """A classifier failure builds at the default level and posts the failure comment."""
    comments = _RecordingComments()
    governor = _governor(tmp_path)
    facts = _FactsFor({1: ClassifyInput(title="x", body="b", labels=[])})

    level = await _resolve_adhoc_level(
        _adhoc_issue(),
        _config(),
        classify=_RecordingClassifier(ClassifyResult(level=None)),  # type: ignore[arg-type]
        label_sink=_RecordingLabels(),  # type: ignore[arg-type]
        comment_sink=comments,  # type: ignore[arg-type]
        issue_facts=facts,  # type: ignore[arg-type]
        governor=governor,
    )

    assert level == "standard"
    assert len(comments.calls) == 1
    assert comments.calls[0].body == _FAILURE_COMMENT.format(level="standard")


@pytest.mark.asyncio
async def test_resolve_adhoc_level_facts_flake_falls_back_without_propagating(
    tmp_path: Path,
) -> None:
    """A gh/facts flake falls back to the table default and never propagates out."""
    governor = _governor(tmp_path)

    level = await _resolve_adhoc_level(
        _adhoc_issue(),
        _config(),
        classify=_UnusedClassifier(),  # type: ignore[arg-type]
        label_sink=_RecordingLabels(),  # type: ignore[arg-type]
        comment_sink=_RecordingComments(),  # type: ignore[arg-type]
        issue_facts=_RaisingFacts(RuntimeError("gh view exited non-zero")),  # type: ignore[arg-type]
        governor=governor,
    )

    assert level == "standard"  # config.routing.default


# --- ad-hoc lane: all three roles launch at the resolved level (integration) ------


def _adhoc_routing() -> RoutingConfig:
    """A table whose ``trivial`` level overrides planner, implementer, and reviewer.

    Each override names a model distinct from that role's registry default so an exec/model
    assertion is meaningful; the reviewer also overrides its effort tier.
    """
    return RoutingConfig(
        default="standard",
        levels={
            "trivial": RoutingLevel(
                description="a tiny one-file fix",
                roles={
                    "planner": {"model": "planner-trivial"},  # type: ignore[dict-item]
                    "implementer": {"model": "implementer-trivial"},  # type: ignore[dict-item]
                    "reviewer": {"model": "reviewer-trivial", "effort": "low"},  # type: ignore[dict-item]
                },
            ),
            "standard": RoutingLevel(
                description="a normal feature",
                roles={"implementer": {"model": "implementer-standard"}},  # type: ignore[dict-item]
            ),
        },
    )


@pytest.mark.asyncio
async def test_adhoc_roles_all_launch_at_the_resolved_level(tmp_path: Path) -> None:
    """AC-1: the ad-hoc planner + implementer exec the level's models; the reviewer too.

    Mirrors the reworked ``bind_adhoc_build`` closure: construct the three per-issue roles
    at the resolved level and drive ``build_adhoc_issue`` over a fake runtime. The planner
    and implementer exec ``claude --model <level's model>`` in-container; the reviewer's
    generator carries the level's model and effort (it reviews over HTTP, not in-container).
    """
    config = RepoConfig(staging_branch="staging", routing=_adhoc_routing())
    level = "trivial"
    runtime = FakeRuntime()

    planner = ContainerPlanner(
        credential="sk-ant-api03-k",
        auth_mode="api_key",
        model=resolve_model(Role.PLANNER, config, level=level),
    )
    implementer = ContainerImplementer(
        credential="sk-ant-api03-k",
        auth_mode="api_key",
        model=resolve_model(Role.IMPLEMENTER, config, level=level),
    )
    review_generate = AgentSdkReviewGenerator(
        credential="sk-ant-api03-k",
        transport=_TitleRoutingTransport(),
        model=resolve_model(Role.REVIEWER, config, level=level),
        effort=resolve_effort(Role.REVIEWER, config, level=level),
    )
    reviewer = ContainerAdhocReviewer(
        repo_full_name="owner/repo",
        config=config,
        generate=review_generate,
        create_issue=_no_create,  # type: ignore[arg-type]
    )

    result = await build_adhoc_issue(
        _adhoc_issue(29),
        config,
        CLAUDE_MD,
        planner=planner,
        implementer=implementer,
        auth=FakeAuth(),
        runtime=runtime,
        resolve_secret=_resolver({}),
        report=_sink([]),
        reviewer=None,  # the reviewer's routing is asserted on the generator directly
    )

    assert result.passed is True
    assert _launched_models(runtime) == ["planner-trivial", "implementer-trivial"]
    assert review_generate.model == "reviewer-trivial"
    assert reviewer.generate is review_generate
    assert review_generate.effort == resolve_effort(Role.REVIEWER, config, level=level)
    assert review_generate.effort == "low"

@pytest.mark.asyncio
async def test_bind_adhoc_build_routes_through_the_real_classify_hop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock-debt #70: the ad-hoc closure's routing runs the real classify chain.

    :func:`bind_adhoc_build`'s per-issue closure is driven through the **real**
    ``_resolve_adhoc_level`` -> :func:`resolve_issue_level` ->
    :class:`ClaudeIssueClassifier` chain — nothing on the classify path is mocked. Only
    the boundary seams are faked: the Messages-API transport (:class:`_TitleRoutingTransport`
    routes on the issue title), the gh subprocess runner (returns the ``issue view`` JSON
    :class:`GhCliIssueFacts` parses for real), and the gh label/comment sinks.

    The issue classifies to ``trivial`` — a **non-default** level whose planner/
    implementer/reviewer models are distinct from both the registry defaults and the
    ``standard`` default level's map — so a closure that dropped the resolved level
    (passing ``level=None``, which resolves via the default level) would construct
    ``implementer-standard`` and the registry planner/reviewer models and fail the
    asserts below. The classifier's charge is asserted on the **pipeline's own**
    governor's ledger, pinning the ``governor=pipeline.governor`` kwarg the closure
    passes.
    """
    import retinue.pipeline as pipeline_mod
    from retinue.pipeline import bind_adhoc_build
    from tests.test_pipeline import (
        _fake_build_adhoc_issue,
        _RecordingAdhocPipeline,
        _settings,
    )

    transport = _TitleRoutingTransport()
    labels, comments = _RecordingLabels(), _RecordingComments()

    async def gh_runner(argv: list[str]) -> str:
        # The ``gh issue view --json title,body,labels`` stdout GhCliIssueFacts parses;
        # PICKTRIVIAL routes the transport to the non-default ``trivial`` level.
        assert argv[:2] == ["issue", "view"]
        return json.dumps({"title": "PICKTRIVIAL doc fix", "body": "b", "labels": []})

    monkeypatch.setattr(pipeline_mod, "HttpxTransport", lambda: transport)
    monkeypatch.setattr(pipeline_mod, "GhLabelSink", lambda token: labels)
    monkeypatch.setattr(pipeline_mod, "GhCommentSink", lambda token: comments)
    monkeypatch.setattr(pipeline_mod, "ReconcileGhRunner", lambda token: gh_runner)
    captured: dict[str, object] = {}
    green = AdhocBuildResult(branch="issue-31", passed=True)
    monkeypatch.setattr(
        pipeline_mod, "build_adhoc_issue", _fake_build_adhoc_issue(captured, green)
    )

    config = RepoConfig(staging_branch="staging", routing=_adhoc_routing())
    governor = _governor(tmp_path)
    pipeline = _RecordingAdhocPipeline(governor=governor)
    settings = _settings(tmp_path, anthropic_credential="sk-ant-api03-k")
    build = bind_adhoc_build(
        settings,  # type: ignore[arg-type]
        FakeAuth(),
        pipeline=pipeline,  # type: ignore[arg-type]
        repo_full_name="owner/repo",
        token="ghs_x",
        config=config,
        claude_md=CLAUDE_MD,
    )

    issue = _adhoc_issue(31)
    await build(issue, repo_full_name="owner/repo")

    # The classifier really ran: exactly one Messages-API POST over the fake transport,
    # and its verdict was persisted as the additive ``level:trivial`` label.
    assert len(transport.calls) == 1
    assert [request.label for request in labels.calls] == ["level:trivial"]
    assert comments.calls == []  # classification succeeded — no failure comment
    # Every role was constructed at the resolved non-default level's model — the
    # ``level=None`` default-level resolution would yield none of these three ids.
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["planner"].model == "planner-trivial"
    assert kwargs["implementer"].model == "implementer-trivial"
    assert kwargs["reviewer"].generate.model == "reviewer-trivial"
    assert kwargs["reviewer"].generate.effort == "low"
    # The classifier charge landed on the pipeline's own governor's ledger.
    assert await governor._ledger.trailing_24h_spend() == pytest.approx(
        _CLASSIFIER_ESTIMATED_AMOUNT
    )
    # The green result still chained into process_adhoc_pr.
    assert pipeline.pr_calls == [(issue, green)]
