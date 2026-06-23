"""Tests for the internal reviewer (issue #9).

After a round's merge, the reviewer reads the round's merged diff and merged issue
numbers, runs the injected Agent-SDK review seam, and for each genuine finding files
a ``review-fix`` follow-up issue (``ready-for-agent`` + ``Part of #<prd>``) and wires
it into the ``## Blocked by`` of the relevant dependent open issues so the fix builds
before the work layered on it. The reviewer never edits code.

The three side-effecting seams are faked: the review generator (Agent SDK), the gh
issue creator (reused from the slicer), and the gh issue-body editor (the Blocked-by
wiring). A clean diff files nothing. A filed review-fix issue is also fed back into
``build_prd`` to prove it is picked up and built in a subsequent round. No real Agent
SDK, gh, or network is touched.
"""

from __future__ import annotations

import pytest

from retinue.orchestrator import PrdBuildResult, PrdSlice, build_prd
from retinue.repo_config import RepoConfig
from retinue.reviewer import (
    AgentSdkReviewGenerator,
    EditBlockedByRequest,
    GhCliBlockedByEditor,
    GhCommandError,
    GhResult,
    HttpResponse,
    ReviewFinding,
    ReviewGenerationError,
    ReviewInput,
    ReviewPlan,
    ReviewResult,
    add_blocked_by,
    review_round,
)
from retinue.slicer import _EFFORT_MAX, CreatedIssue, IssueDraft
from tests.test_done_check import CLAUDE_MD, FakeAuth, FakeRuntime, _resolver, _sink
from tests.test_orchestrator import FakeGitOps, FakeImplementer
from tests.test_prd_build import OneAtATimeLock

PRD_NUMBER = 1
REPO = "owner/repo"

# A merged round: issues #2 and #3 were merged; #3 was built on top of #2's work.
MERGED_ISSUES = [2, 3]
PLANTED_DEFECT_DIFF = """\
diff --git a/retinue/widget.py b/retinue/widget.py
+def total(items):
+    return sum(items) + 1  # off-by-one planted defect
"""
CLEAN_DIFF = """\
diff --git a/retinue/widget.py b/retinue/widget.py
+def total(items):
+    return sum(items)
"""


class _Recorder:
    """Captures filed review-fix issues and Blocked-by edits for assertions."""

    def __init__(self) -> None:
        self.created: list[IssueDraft] = []
        self.edits: list[EditBlockedByRequest] = []
        self._next_number = 200

    async def create_issue(self, draft: IssueDraft) -> CreatedIssue:
        self._next_number += 1
        self.created.append(draft)
        return CreatedIssue(issue_number=self._next_number)

    async def edit_blocked_by(self, request: EditBlockedByRequest) -> None:
        self.edits.append(request)


def _input(diff: str) -> ReviewInput:
    return ReviewInput(
        repo_full_name=REPO,
        prd_number=PRD_NUMBER,
        merged_issues=list(MERGED_ISSUES),
        diff=diff,
    )


@pytest.mark.asyncio
async def test_planted_defect_files_review_fix_with_labels_and_wiring() -> None:
    """A finding files a review-fix issue (correct labels) wired into a dependent."""
    rec = _Recorder()

    async def generate(review_input: ReviewInput) -> ReviewPlan:
        # The reviewer flagged the off-by-one in #2; #3 depends on it, so the fix
        # must block #3 (build the fix before the work layered on the defect).
        return ReviewPlan(
            findings=[
                ReviewFinding(
                    title="Fix off-by-one in total()",
                    body="total() adds a stray +1.",
                    blocks_issues=[3],
                )
            ]
        )

    result = await review_round(
        _input(PLANTED_DEFECT_DIFF),
        generate=generate,
        create_issue=rec.create_issue,
        edit_blocked_by=rec.edit_blocked_by,
    )

    assert isinstance(result, ReviewResult)
    assert len(rec.created) == 1
    draft = rec.created[0]
    assert "review-fix" in draft.labels
    assert "ready-for-agent" in draft.labels
    assert f"Part of #{PRD_NUMBER}" in draft.body
    # The new review-fix issue (#201) is wired into dependent #3's Blocked by.
    new_number = result.filed_issues[0]
    assert new_number == 201
    assert rec.edits == [
        EditBlockedByRequest(
            repo_full_name=REPO, issue_number=3, add_blocker=new_number
        )
    ]


@pytest.mark.asyncio
async def test_clean_diff_files_nothing() -> None:
    """A clean review yields no findings, so no issue is filed and nothing is wired."""
    rec = _Recorder()

    async def generate(review_input: ReviewInput) -> ReviewPlan:
        return ReviewPlan(findings=[])

    result = await review_round(
        _input(CLEAN_DIFF),
        generate=generate,
        create_issue=rec.create_issue,
        edit_blocked_by=rec.edit_blocked_by,
    )

    assert result.filed_issues == []
    assert rec.created == []
    assert rec.edits == []


@pytest.mark.asyncio
async def test_finding_with_no_dependents_files_issue_without_wiring() -> None:
    """A finding that blocks nothing still files a review-fix issue, no Blocked-by edit."""
    rec = _Recorder()

    async def generate(review_input: ReviewInput) -> ReviewPlan:
        return ReviewPlan(
            findings=[
                ReviewFinding(title="Stale doc", body="README mentions old flag.")
            ]
        )

    result = await review_round(
        _input(PLANTED_DEFECT_DIFF),
        generate=generate,
        create_issue=rec.create_issue,
        edit_blocked_by=rec.edit_blocked_by,
    )

    assert len(result.filed_issues) == 1
    assert rec.edits == []


@pytest.mark.asyncio
async def test_review_fix_issue_is_built_in_a_subsequent_round() -> None:
    """The filed review-fix issue is picked up and built by a later build_prd round."""
    rec = _Recorder()

    async def generate(review_input: ReviewInput) -> ReviewPlan:
        return ReviewPlan(
            findings=[
                ReviewFinding(
                    title="Fix off-by-one in total()",
                    body="total() adds a stray +1.",
                    blocks_issues=[3],
                )
            ]
        )

    review = await review_round(
        _input(PLANTED_DEFECT_DIFF),
        generate=generate,
        create_issue=rec.create_issue,
        edit_blocked_by=rec.edit_blocked_by,
    )

    # A subsequent orchestrator round picks up the filed review-fix issue as a slice.
    fix_number = review.filed_issues[0]
    git = FakeGitOps()
    result: PrdBuildResult = await build_prd(
        [PrdSlice(repo_full_name=REPO, issue_number=fix_number, prd_number=PRD_NUMBER)],
        RepoConfig(),
        CLAUDE_MD,
        implementer=FakeImplementer(),
        git=git,
        auth=FakeAuth(),
        runtime=FakeRuntime(),
        resolve_secret=_resolver({}),
        report=_sink([]),
        lock=OneAtATimeLock(),
    )

    assert result.merged_issues == [fix_number]
    assert (f"issue-{fix_number}", "retinue/prd-1") in git.merges


# --- Real Agent-SDK ReviewGenerator: pure/parseable parts, no network ---


class _FakeTransport:
    """Records the one POST and returns a canned response. No network."""

    def __init__(self, response: HttpResponse) -> None:
        self._response = response
        self.calls: list[tuple[str, dict[str, str], dict[str, object]]] = []

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, object]
    ) -> HttpResponse:
        self.calls.append((url, headers, json))
        return self._response


def _text_response(payload: object, *, status_code: int = 200) -> HttpResponse:
    """A Messages API response whose single text block is ``payload`` as JSON."""
    import json as _json

    return HttpResponse(
        status_code=status_code,
        body={"content": [{"type": "text", "text": _json.dumps(payload)}]},
    )


def test_headers_oauth_token_uses_bearer_and_beta() -> None:
    """An OAuth subscription token rides Authorization: Bearer + the oauth beta."""
    gen = AgentSdkReviewGenerator(
        credential="sk-ant-oat-abc",
        transport=_FakeTransport(_text_response({"findings": []})),
    )
    headers = gen._headers()

    assert headers["authorization"] == "Bearer sk-ant-oat-abc"
    assert headers["anthropic-beta"] == "oauth-2025-04-20"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "x-api-key" not in headers


def test_headers_api_key_uses_x_api_key() -> None:
    """A raw API key rides x-api-key, with no bearer/oauth-beta header."""
    gen = AgentSdkReviewGenerator(
        credential="sk-ant-api-xyz",
        transport=_FakeTransport(_text_response({"findings": []})),
    )
    headers = gen._headers()

    assert headers["x-api-key"] == "sk-ant-api-xyz"
    assert "authorization" not in headers
    assert "anthropic-beta" not in headers


def test_payload_carries_model_schema_diff_and_merged_issues() -> None:
    """The request body assembles model, schema, and the diff + merged issues."""
    gen = AgentSdkReviewGenerator(
        credential="k", transport=_FakeTransport(_text_response({"findings": []}))
    )
    payload = gen._payload(_input(PLANTED_DEFECT_DIFF))

    assert payload["model"] == "claude-opus-4-8"
    assert payload["response_format"]["type"] == "json_schema"
    user = payload["messages"][0]["content"]
    assert "#2, #3" in user
    assert "off-by-one planted defect" in user
    assert f"PRD #{PRD_NUMBER}" in user


def test_payload_carries_max_effort() -> None:
    """The internal reviewer (Opus 4.8) runs at the max effort tier via output_config.

    Opus 4.8 removed the ``thinking`` budget mechanism (400 on the live Messages API);
    ``output_config.effort`` is the current effort control, and ``max`` is the highest-
    rigor tier the PRD pins the internal reviewer to.
    """
    gen = AgentSdkReviewGenerator(
        credential="k", transport=_FakeTransport(_text_response({"findings": []}))
    )

    payload = gen._payload(_input(PLANTED_DEFECT_DIFF))

    assert payload["output_config"]["effort"] == _EFFORT_MAX
    assert _EFFORT_MAX == "max"
    assert "thinking" not in payload


@pytest.mark.asyncio
async def test_real_generator_parses_findings_from_response() -> None:
    """A response with findings parses into a ReviewPlan of ReviewFindings."""
    transport = _FakeTransport(
        _text_response(
            {
                "findings": [
                    {
                        "title": "Fix off-by-one in total()",
                        "body": "total() adds a stray +1.",
                        "blocks_issues": [3],
                    }
                ]
            }
        )
    )
    gen = AgentSdkReviewGenerator(credential="sk-ant-oat-1", transport=transport)

    plan = await gen(_input(PLANTED_DEFECT_DIFF))

    assert plan == ReviewPlan(
        findings=[
            ReviewFinding(
                title="Fix off-by-one in total()",
                body="total() adds a stray +1.",
                blocks_issues=[3],
            )
        ]
    )
    # It POSTed exactly once to the Messages endpoint.
    assert transport.calls[0][0] == "https://api.anthropic.com/v1/messages"


@pytest.mark.asyncio
async def test_real_generator_clean_review_parses_empty_plan() -> None:
    """An empty findings list parses into a clean ReviewPlan."""
    gen = AgentSdkReviewGenerator(
        credential="k", transport=_FakeTransport(_text_response({"findings": []}))
    )

    plan = await gen(_input(CLEAN_DIFF))

    assert plan == ReviewPlan(findings=[])


@pytest.mark.asyncio
async def test_real_generator_raises_on_non_200() -> None:
    """A non-200 from the API raises rather than filing a phantom clean review."""
    gen = AgentSdkReviewGenerator(
        credential="k",
        transport=_FakeTransport(_text_response({"findings": []}, status_code=429)),
    )

    with pytest.raises(ReviewGenerationError):
        await gen(_input(CLEAN_DIFF))


@pytest.mark.asyncio
async def test_real_generator_raises_on_malformed_json() -> None:
    """A text block that is not valid JSON raises ReviewGenerationError."""
    bad = HttpResponse(
        status_code=200, body={"content": [{"type": "text", "text": "not json {"}]}
    )
    gen = AgentSdkReviewGenerator(credential="k", transport=_FakeTransport(bad))

    with pytest.raises(ReviewGenerationError):
        await gen(_input(CLEAN_DIFF))


@pytest.mark.asyncio
async def test_real_generator_raises_when_findings_missing() -> None:
    """JSON without a 'findings' list raises rather than silently producing none."""
    gen = AgentSdkReviewGenerator(
        credential="k", transport=_FakeTransport(_text_response({"oops": True}))
    )

    with pytest.raises(ReviewGenerationError):
        await gen(_input(CLEAN_DIFF))


# --- Real gh-cli BlockedByEditor: pure/parseable parts, no gh/network ---


class _FakeGhRunner:
    """Records each gh invocation and returns scripted results. No subprocess."""

    def __init__(self, results: list[GhResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    async def run(self, args: list[str], *, env: dict[str, str]) -> GhResult:
        self.calls.append((list(args), dict(env)))
        return self._results.pop(0)


def _body_result(body: str) -> GhResult:
    """A ``gh issue view --json body`` result carrying ``body``."""
    import json as _json

    return GhResult(exit_code=0, stdout=_json.dumps({"body": body}))


def test_add_blocked_by_appends_block_when_absent() -> None:
    """A body with no Blocked-by block grows one appended at the end."""
    body = "Some finding.\n\nPart of #1"
    assert add_blocked_by(body, 201) == "Some finding.\n\nPart of #1\n\n## Blocked by\n#201"


def test_add_blocked_by_appends_to_existing_block() -> None:
    """An existing Blocked-by block gains the new reference on its own line."""
    body = "Body.\n\n## Blocked by\n#5"
    assert add_blocked_by(body, 201) == "Body.\n\n## Blocked by\n#5\n#201"


def test_add_blocked_by_is_idempotent() -> None:
    """A blocker already listed leaves the body unchanged."""
    body = "Body.\n\n## Blocked by\n#5\n#201"
    assert add_blocked_by(body, 201) == body


@pytest.mark.asyncio
async def test_real_editor_reads_then_writes_with_added_blocker() -> None:
    """The editor views the body, then edits it back with the new Blocked-by ref."""
    runner = _FakeGhRunner([_body_result("Dependent body.\n\nPart of #1"), GhResult(0)])
    editor = GhCliBlockedByEditor(runner=runner, token="ght_abc")

    await editor(EditBlockedByRequest(repo_full_name=REPO, issue_number=3, add_blocker=201))

    view_args, view_env = runner.calls[0]
    assert view_args == ["issue", "view", "3", "--repo", REPO, "--json", "body"]
    assert view_env == {"GH_TOKEN": "ght_abc"}
    edit_args, _ = runner.calls[1]
    assert edit_args[:5] == ["issue", "edit", "3", "--repo", REPO]
    assert edit_args[5] == "--body"
    assert "## Blocked by\n#201" in edit_args[6]


@pytest.mark.asyncio
async def test_real_editor_skips_edit_when_already_blocked() -> None:
    """An already-present blocker reads the body but writes nothing back."""
    runner = _FakeGhRunner([_body_result("Body.\n\n## Blocked by\n#201")])
    editor = GhCliBlockedByEditor(runner=runner, token="ght_abc")

    await editor(EditBlockedByRequest(repo_full_name=REPO, issue_number=3, add_blocker=201))

    assert len(runner.calls) == 1  # view only, no edit


@pytest.mark.asyncio
async def test_real_editor_raises_on_gh_failure() -> None:
    """A non-zero gh exit raises GhCommandError rather than silently dropping the wire."""
    runner = _FakeGhRunner([GhResult(exit_code=1, stderr="not found")])
    editor = GhCliBlockedByEditor(runner=runner, token="ght_abc")

    with pytest.raises(GhCommandError):
        await editor(
            EditBlockedByRequest(repo_full_name=REPO, issue_number=3, add_blocker=201)
        )


def test_real_editor_raises_on_malformed_view_output() -> None:
    """A view payload missing the body field fails loudly rather than clobbering it."""
    from retinue.reviewer import _parse_issue_body

    with pytest.raises(ValueError):
        _parse_issue_body('{"oops": true}')
