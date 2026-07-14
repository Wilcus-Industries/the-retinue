"""Tests for retinue.routing: per-issue implementer-model routing in the PRD build lane.

Two layers are exercised, all offline (no gh, no network, no Docker):

* **Unit** — the ``gh issue view`` argv/parse helpers and :class:`GhCliIssueFacts`, plus
  the four :class:`PerIssueImplementerRouter` paths (classify success, pre-existing-label
  short-circuit, classification failure) against recording label/comment sinks, a fake
  classifier, and a real :class:`BudgetGovernor` on a temp ledger.
* **Integration** — :func:`retinue.wiring.bind_build_prd` driven with a real router over
  the container-exec :class:`ContainerImplementer` and the in-memory runtime/git/auth
  fakes, so the recording container runtime records the actual ``claude --model <...>``
  exec each slice launched with.

Reuses ``FakeRuntime``/``FakeAuth``/``CLAUDE_MD``/``_resolver``/``_sink`` from
``tests/test_done_check.py``, ``FakeGitOps`` from ``tests/test_orchestrator.py``, and
``FakeClock`` from ``tests/test_budget.py``.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from retinue.budget import AuthMode, BudgetGovernor, BudgetLedger
from retinue.classifier import ClassifyInput, ClassifyResult, ClaudeIssueClassifier
from retinue.notify import (
    CommentRequest,
    CommentSink,
    LabelRequest,
    Notifier,
    PushRequest,
)
from retinue.orchestrator import ContainerImplementer, PrdSlice, Slice
from retinue.pipeline import _CLASSIFIER_ESTIMATED_AMOUNT
from retinue.repo_config import RepoConfig, RoutingConfig, RoutingLevel
from retinue.reviewer import HttpResponse
from retinue.roles import Role, resolve_model
from retinue.routing import (
    _FAILURE_COMMENT,
    GhCliIssueFacts,
    PerIssueImplementer,
    PerIssueImplementerRouter,
    _issue_facts_argv,
    _parse_issue_facts,
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
        label_sink=_noop_label,
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
