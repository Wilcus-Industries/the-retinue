"""Tests for the headless PRD slicer.

The slicer turns a PRD body into tracer-bullet vertical-slice issues. The two
external seams are faked: the Agent-SDK slice generator (``generate``) and the gh
issue creator (``create_issue``). A well-formed PRD yields labeled, dependency-
ordered slices; a thin or malformed PRD escalates through the notifier and creates
no slices. No real network, Agent SDK, or gh is touched.
"""

from __future__ import annotations

import json

import pytest

from retinue.notify import (
    CommentRequest,
    LabelRequest,
    Notifier,
    PushRequest,
)
from retinue.roles import CLAUDE_CODE_IDENTITY
from retinue.slicer import (
    _EFFORT_XHIGH,
    _SLICE_SYSTEM,
    ClaudeSliceGenerator,
    CreatedIssue,
    GhCliIssueCreator,
    GhCommandError,
    GhResult,
    IssueDraft,
    SliceOutcome,
    SlicePlan,
    _blocked_by_numbers,
    _issue_create_args,
    _parse_issue_number,
    slice_prd,
)

PRD_NUMBER = 1
REPO = "owner/repo"

# A well-formed PRD: two ordinary slices plus one genuinely human-only slice, with
# a Blocked-by graph the slicer must resolve to created issue numbers.
WELL_FORMED_PRD = """\
## Testing Decisions

Use pytest with mocked sinks; no real network in tests.

## Slices

1. Transport spine — webhook + queue.
2. Worker gate — depends on the transport spine.
3. Provision the production Pushover account and API token.
"""

THIN_PRD = "TODO: write this later."


class _Recorder:
    """Captures notify sink calls and gh issue creations for assertions."""

    def __init__(self) -> None:
        self.pushes: list[PushRequest] = []
        self.comments: list[CommentRequest] = []
        self.labels: list[LabelRequest] = []
        self.created: list[IssueDraft] = []
        self._next_number = 100

    async def push(self, request: PushRequest) -> None:
        self.pushes.append(request)

    async def comment(self, request: CommentRequest) -> None:
        self.comments.append(request)

    async def label(self, request: LabelRequest) -> None:
        self.labels.append(request)

    async def create_issue(self, draft: IssueDraft) -> CreatedIssue:
        self._next_number += 1
        self.created.append(draft)
        return CreatedIssue(issue_number=self._next_number)

    def notifier(self) -> Notifier:
        return Notifier(push=self.push, comment=self.comment, label=self.label)


def _well_formed_plan() -> SlicePlan:
    """The fake Agent-SDK output for the well-formed PRD.

    The slicer's own logic — labels, Part-of, blocked-by resolution, escalation —
    is what is under test; the generator that produces this plan is a faked seam.
    """
    return SlicePlan(
        slices=[
            IssueDraft(title="Transport spine", body="webhook + queue", blocked_by=[]),
            IssueDraft(title="Worker gate", body="gate the PRD", blocked_by=[1]),
            IssueDraft(
                title="Provision Pushover account",
                body="create the prod account + token",
                blocked_by=[],
                hitl=True,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_well_formed_prd_creates_labeled_slices() -> None:
    """Each slice issue carries ready-for-agent + prd-slice + Part of #<prd>; gh is mocked."""
    rec = _Recorder()

    async def generate(prd_body: str) -> SlicePlan:
        return _well_formed_plan()

    result = await slice_prd(
        repo_full_name=REPO,
        prd_number=PRD_NUMBER,
        prd_body=WELL_FORMED_PRD,
        generate=generate,
        create_issue=rec.create_issue,
        notifier=rec.notifier(),
    )

    assert result.outcome is SliceOutcome.SLICED
    assert len(rec.created) == 3
    for draft in rec.created:
        assert "ready-for-agent" in draft.labels
        assert "prd-slice" in draft.labels
        assert f"Part of #{PRD_NUMBER}" in draft.body


@pytest.mark.asyncio
async def test_blocked_by_graph_resolves_to_created_issue_numbers() -> None:
    """A slice's intra-PRD dependency is rewritten to the real created issue number."""
    rec = _Recorder()

    async def generate(prd_body: str) -> SlicePlan:
        return _well_formed_plan()

    result = await slice_prd(
        repo_full_name=REPO,
        prd_number=PRD_NUMBER,
        prd_body=WELL_FORMED_PRD,
        generate=generate,
        create_issue=rec.create_issue,
        notifier=rec.notifier(),
    )

    assert result.outcome is SliceOutcome.SLICED
    # First created slice gets number 101; the second slice was "blocked_by=[1]".
    spine_number = result.created_numbers[0]
    gate_draft = rec.created[1]
    assert f"## Blocked by\n#{spine_number}" in gate_draft.body


@pytest.mark.asyncio
async def test_human_only_slice_is_hitl_others_are_not() -> None:
    """Only the genuinely human-only slice carries the hitl label."""
    rec = _Recorder()

    async def generate(prd_body: str) -> SlicePlan:
        return _well_formed_plan()

    await slice_prd(
        repo_full_name=REPO,
        prd_number=PRD_NUMBER,
        prd_body=WELL_FORMED_PRD,
        generate=generate,
        create_issue=rec.create_issue,
        notifier=rec.notifier(),
    )

    by_title = {draft.title: draft for draft in rec.created}
    assert "hitl" in by_title["Provision Pushover account"].labels
    assert "hitl" not in by_title["Transport spine"].labels
    assert "hitl" not in by_title["Worker gate"].labels


@pytest.mark.asyncio
async def test_thin_prd_escalates_and_creates_no_slices() -> None:
    """A thin/malformed PRD escalates (push + comment + label) and slices nothing."""
    rec = _Recorder()

    async def generate(prd_body: str) -> SlicePlan:
        raise AssertionError("generate must not be called for a thin PRD")

    result = await slice_prd(
        repo_full_name=REPO,
        prd_number=PRD_NUMBER,
        prd_body=THIN_PRD,
        generate=generate,
        create_issue=rec.create_issue,
        notifier=rec.notifier(),
    )

    assert result.outcome is SliceOutcome.ESCALATED
    assert rec.created == []
    assert len(rec.pushes) == 1
    assert len(rec.comments) == 1
    assert len(rec.labels) == 1
    assert rec.labels[0].label == "hitl"


@pytest.mark.asyncio
async def test_empty_generated_plan_escalates() -> None:
    """A well-formed PRD whose generator yields zero slices escalates, not silently."""
    rec = _Recorder()

    async def generate(prd_body: str) -> SlicePlan:
        return SlicePlan(slices=[])

    result = await slice_prd(
        repo_full_name=REPO,
        prd_number=PRD_NUMBER,
        prd_body=WELL_FORMED_PRD,
        generate=generate,
        create_issue=rec.create_issue,
        notifier=rec.notifier(),
    )

    assert result.outcome is SliceOutcome.ESCALATED
    assert rec.created == []
    assert len(rec.comments) == 1


# --- Real Agent-SDK generator: pure/parseable parts (no network, SDK, or gh) ---


def test_api_key_mode_builds_no_extra_headers_and_uses_x_api_key() -> None:
    """api_key mode passes the key as api_key= and adds no OAuth beta header."""
    gen = ClaudeSliceGenerator(token="sk-ant-123", auth_mode="api_key")

    assert gen._build_auth_headers() == {}
    assert gen._client_kwargs() == {"api_key": "sk-ant-123"}


def test_subscription_mode_uses_bearer_token_and_oauth_beta_header() -> None:
    """subscription mode sends the OAuth token as auth_token= with the oauth beta header."""
    gen = ClaudeSliceGenerator(token="oauth-tok", auth_mode="subscription")

    assert gen._build_auth_headers() == {"anthropic-beta": "oauth-2025-04-20"}
    assert gen._client_kwargs() == {
        "auth_token": "oauth-tok",
        "default_headers": {"anthropic-beta": "oauth-2025-04-20"},
    }


def test_request_kwargs_carry_model_prd_body_and_strict_schema() -> None:
    """The assembled request pins the model, the PRD body, and a strict JSON schema."""
    gen = ClaudeSliceGenerator(token="sk-ant-123", model="claude-opus-4-8")

    kwargs = gen._build_request_kwargs("Slice this PRD into vertical slices.")

    assert kwargs["model"] == "claude-opus-4-8"
    user_content = kwargs["messages"][0]["content"]
    assert kwargs["messages"][0]["role"] == "user"
    assert "Slice this PRD into vertical slices." in user_content
    fmt = kwargs["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["properties"]["slices"]["type"] == "array"
    assert "response_format" not in kwargs
    assert kwargs["max_tokens"] > 0


def test_subscription_mode_request_system_leads_with_claude_code_identity() -> None:
    """In subscription (OAuth) mode the system field leads with the identity block.

    A subscription OAuth token reaches the premium slicing model over the raw Messages
    API only when the first system block is the Claude Code identity string; the slicer's
    own brief follows it as the second block.
    """
    gen = ClaudeSliceGenerator(token="oauth-tok", auth_mode="subscription")

    system = gen._build_request_kwargs("body")["system"]

    assert system == [
        {"type": "text", "text": CLAUDE_CODE_IDENTITY},
        {"type": "text", "text": _SLICE_SYSTEM},
    ]


def test_api_key_mode_request_system_stays_the_plain_role_prompt() -> None:
    """In api_key mode the system field stays the unchanged plain role brief."""
    gen = ClaudeSliceGenerator(token="sk-ant-123", auth_mode="api_key")

    system = gen._build_request_kwargs("body")["system"]

    assert system == _SLICE_SYSTEM
    assert isinstance(system, str)


def test_request_injects_prd_testing_decisions_as_authoritative_seam() -> None:
    """The PRD's ## Testing Decisions section is injected as the labeled testing seam.

    The testing seam is read from the PRD (drift fix #6): the slicer must extract the
    ``## Testing Decisions`` section and hand it to the model as an explicit, labeled
    block so slices inherit the PRD's testing decisions rather than inventing their own.
    """
    gen = ClaudeSliceGenerator(token="sk-ant-123")

    kwargs = gen._build_request_kwargs(WELL_FORMED_PRD)

    user_content = kwargs["messages"][0]["content"]
    assert "TESTING SEAM" in user_content
    assert "Use pytest with mocked sinks; no real network in tests." in user_content
    # The whole PRD body still reaches the model alongside the labeled seam.
    assert "Transport spine — webhook + queue." in user_content


def test_request_omits_testing_seam_block_when_prd_has_no_testing_decisions() -> None:
    """A PRD without a ## Testing Decisions section carries no labeled seam block."""
    gen = ClaudeSliceGenerator(token="sk-ant-123")

    kwargs = gen._build_request_kwargs("## Slices\n\n1. Just one slice, no testing section.")

    user_content = kwargs["messages"][0]["content"]
    assert "TESTING SEAM" not in user_content
    assert "Just one slice, no testing section." in user_content


_TRAILING_SECTION_PRD = "## Testing Decisions\nlast section, no trailing heading."


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (WELL_FORMED_PRD, "Use pytest with mocked sinks; no real network in tests."),
        (_TRAILING_SECTION_PRD, "last section, no trailing heading."),
        ("no heading at all", None),
        ("## Other\n\nstuff", None),
    ],
)
def test_extract_testing_decisions_pulls_only_the_section(
    body: str, expected: str | None
) -> None:
    """The extractor returns the Testing Decisions section text, or None when absent."""
    from retinue.slicer import _extract_testing_decisions

    assert _extract_testing_decisions(body) == expected


def test_request_kwargs_carry_xhigh_effort() -> None:
    """The slicer (Opus 4.8) runs at the xhigh effort tier via output_config.effort.

    Opus 4.8 removed ``thinking={"type": "enabled", "budget_tokens": N}`` (400 on the
    live Messages API); ``output_config.effort`` is the current effort control. The
    effort tier must ride the *same* ``output_config`` dict that already carries the
    JSON-schema ``format`` so the slicer sends one output_config, not two.
    """
    gen = ClaudeSliceGenerator(token="sk-ant-123")

    kwargs = gen._build_request_kwargs("Slice this PRD into vertical slices.")

    assert kwargs["output_config"]["effort"] == _EFFORT_XHIGH
    assert _EFFORT_XHIGH == "xhigh"
    # Effort lives alongside the schema format, not as a separate thinking budget.
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in kwargs


def test_parse_plan_maps_payload_to_ordered_drafts() -> None:
    """A well-formed payload parses into ordered drafts preserving blocked_by + hitl."""
    payload = json.dumps(
        {
            "slices": [
                {"title": "Spine", "body": "webhook + queue", "blocked_by": [], "hitl": False},
                {"title": "Gate", "body": "gate the PRD", "blocked_by": [1], "hitl": False},
                {"title": "Provision", "body": "prod account", "blocked_by": [], "hitl": True},
            ]
        }
    )

    plan = ClaudeSliceGenerator._parse_plan(payload)

    assert [d.title for d in plan.slices] == ["Spine", "Gate", "Provision"]
    assert plan.slices[1].blocked_by == [1]
    assert plan.slices[2].hitl is True


def test_parse_plan_defaults_optional_fields() -> None:
    """blocked_by and hitl default when absent; title and body are kept."""
    plan = ClaudeSliceGenerator._parse_plan(json.dumps({"slices": [{"title": "T", "body": "B"}]}))

    assert plan.slices[0].blocked_by == []
    assert plan.slices[0].hitl is False


def test_parse_plan_empty_slices_is_a_valid_empty_plan() -> None:
    """An empty slices array parses to an empty plan (the caller escalates it)."""
    plan = ClaudeSliceGenerator._parse_plan(json.dumps({"slices": []}))

    assert plan.slices == []


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        json.dumps({"slices": "nope"}),
        json.dumps({"wrong": []}),
        json.dumps({"slices": [{"title": "no body"}]}),
    ],
)
def test_parse_plan_rejects_malformed_payloads(payload: str) -> None:
    """A non-object / missing-array / missing-field payload raises rather than filing junk."""
    with pytest.raises(ValueError):
        ClaudeSliceGenerator._parse_plan(payload)


# --- Real gh-cli IssueCreator: pure/parseable parts (no network, gh, or subprocess) ---


class _FakeGhRunner:
    """Records each gh invocation and returns a canned result for the create call."""

    def __init__(self, result: GhResult) -> None:
        self._result = result
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    async def run(self, args: list[str], *, env: dict[str, str]) -> GhResult:
        self.calls.append((args, env))
        return self._result


def test_issue_create_args_carry_repo_title_body_labels_and_blocked_by() -> None:
    """The argv pins the repo/title/body and rides one flag per label and blocked-by ref."""
    draft = IssueDraft(
        title="Worker gate",
        body="gate the PRD\n\nPart of #1\n\n## Blocked by\n#101",
        labels=["ready-for-agent", "hitl"],
    )

    args = _issue_create_args(draft, "owner/repo", [101])

    assert args[:4] == ["issue", "create", "--repo", "owner/repo"]
    assert args[args.index("--title") + 1] == "Worker gate"
    assert args[args.index("--body") + 1] == draft.body
    # One --label per label, one --blocked-by per resolved dependency.
    assert [args[i + 1] for i, a in enumerate(args) if a == "--label"] == [
        "ready-for-agent",
        "hitl",
    ]
    assert [args[i + 1] for i, a in enumerate(args) if a == "--blocked-by"] == ["101"]


def test_blocked_by_numbers_reads_back_resolved_refs_from_finalized_body() -> None:
    """The resolved #<n> refs are recovered from the rendered ## Blocked by block."""
    body = "do the thing\n\nPart of #1\n\n## Blocked by\n#101\n#102"

    assert _blocked_by_numbers(body) == [101, 102]


def test_blocked_by_numbers_is_empty_without_a_blocked_by_block() -> None:
    """A draft with no ## Blocked by section yields no native blocked-by links."""
    assert _blocked_by_numbers("do the thing\n\nPart of #1") == []


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("https://github.com/owner/repo/issues/123\n", 123),
        ("https://github.com/owner/repo/issues/7", 7),
        ("https://github.com/owner/repo/issues/42/", 42),
    ],
)
def test_parse_issue_number_pulls_the_trailing_number_from_the_url(
    stdout: str, expected: int
) -> None:
    """gh issue create prints the issue URL; the number is its trailing path segment."""
    assert _parse_issue_number(stdout) == expected


def test_parse_issue_number_rejects_output_with_no_number() -> None:
    """A URL-less / number-less output raises rather than yielding a bogus number."""
    with pytest.raises(ValueError):
        _parse_issue_number("not a url")


def test_auth_uses_gh_token_env_no_leading_gh_in_argv() -> None:
    """The runner is handed GH_TOKEN in env and an argv that omits the leading 'gh'."""
    from retinue.slicer import _auth_env

    assert _auth_env("tok-123") == {"GH_TOKEN": "tok-123"}


@pytest.mark.asyncio
async def test_gh_cli_issue_creator_files_and_returns_parsed_number() -> None:
    """The real creator dispatches one gh create call (GH_TOKEN env) and parses the number."""
    runner = _FakeGhRunner(
        GhResult(exit_code=0, stdout="https://github.com/owner/repo/issues/207\n")
    )
    creator = GhCliIssueCreator(runner, token="tok-123", repo_full_name="owner/repo")
    draft = IssueDraft(
        title="Spine",
        body="webhook + queue\n\nPart of #1",
        labels=["ready-for-agent"],
    )

    created = await creator(draft)

    assert created == CreatedIssue(issue_number=207)
    assert len(runner.calls) == 1
    args, env = runner.calls[0]
    assert args[0] == "issue"
    assert env == {"GH_TOKEN": "tok-123"}


@pytest.mark.asyncio
async def test_gh_cli_issue_creator_raises_on_nonzero_exit() -> None:
    """A non-zero gh exit raises GhCommandError rather than returning a bogus issue."""
    runner = _FakeGhRunner(GhResult(exit_code=1, stderr="not authenticated"))
    creator = GhCliIssueCreator(runner, token="tok-123", repo_full_name="owner/repo")

    with pytest.raises(GhCommandError):
        await creator(IssueDraft(title="T", body="B", labels=["ready-for-agent"]))
