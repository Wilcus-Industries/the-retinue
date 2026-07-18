"""Tests for the internal reviewer (issue #9, reshaped for the #80 review gate).

The reviewer is the headless Agent-SDK seam the ad-hoc build's review gate runs over one
freshly-built ``issue-<N>`` diff. It reads the diff, runs the injected review generator,
and returns a :class:`ReviewPlan` of :class:`ReviewFinding`, each carrying a
:class:`~retinue.vocab.Severity`. The reviewer itself files, wires, and edits nothing —
the gate partitions and acts on the findings.

The one side-effecting seam — the HTTP transport — is faked, so the pure/parseable parts
(header build, payload assembly, response parsing) are exercised with no network.
"""

from __future__ import annotations

import pytest

from retinue.messages_api import HttpResponse
from retinue.reviewer import (
    _REVIEW_SYSTEM,
    AgentSdkReviewGenerator,
    ReviewFinding,
    ReviewGenerationError,
    ReviewInput,
    ReviewPlan,
)
from retinue.roles import CLAUDE_CODE_IDENTITY, EFFORT_MAX
from retinue.vocab import Severity

ISSUE_NUMBER = 17
REPO = "owner/repo"

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


def _input(diff: str) -> ReviewInput:
    return ReviewInput(repo_full_name=REPO, issue_number=ISSUE_NUMBER, diff=diff)


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


def test_payload_carries_model_schema_and_issue_diff() -> None:
    """The request body assembles model, severity schema, and the issue's diff.

    The schema must ride ``output_config.format`` (the canonical Messages API shape);
    the OpenAI-style top-level ``response_format`` is not a Claude API parameter and
    400s on the live wire — the reviewer-in-subscription-mode bug.
    """
    gen = AgentSdkReviewGenerator(
        credential="k", transport=_FakeTransport(_text_response({"findings": []}))
    )
    payload = gen._payload(_input(PLANTED_DEFECT_DIFF))

    assert payload["model"] == "claude-opus-4-8"
    fmt = payload["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["required"] == ["findings"]
    severity_prop = fmt["schema"]["properties"]["findings"]["items"]["properties"][
        "severity"
    ]
    assert severity_prop["enum"] == ["low", "medium", "high", "critical"]
    assert "response_format" not in payload
    user = payload["messages"][0]["content"]
    assert f"#{ISSUE_NUMBER}" in user
    assert "off-by-one planted defect" in user


def test_payload_keeps_a_small_diff_whole() -> None:
    """A diff under the cap rides the user message verbatim, no truncation note."""
    gen = AgentSdkReviewGenerator(
        credential="k", transport=_FakeTransport(_text_response({"findings": []}))
    )

    user = gen._payload(_input(PLANTED_DEFECT_DIFF))["messages"][0]["content"]

    assert PLANTED_DEFECT_DIFF.strip() in user
    assert "truncated" not in user


def test_payload_caps_an_oversized_diff_with_a_note() -> None:
    """A huge issue diff is clamped before interpolation so the request stays bounded.

    An unbounded diff would blow the request body (and the reviewer's context) on a big
    change; the payload keeps the head of the diff up to the cap and appends an explicit
    truncation note so the model knows the diff is partial.
    """
    gen = AgentSdkReviewGenerator(
        credential="k", transport=_FakeTransport(_text_response({"findings": []}))
    )
    big_diff = "\n".join(f"+line {n}: {'x' * 60}" for n in range(1000))
    assert len(big_diff) > 8_000

    user = gen._payload(_input(big_diff))["messages"][0]["content"]

    assert "+line 0:" in user  # head kept
    assert big_diff not in user  # tail dropped
    assert "truncated" in user
    assert len(user) < 10_000


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

    assert payload["output_config"]["effort"] == EFFORT_MAX
    assert EFFORT_MAX == "max"
    assert "thinking" not in payload


def test_payload_carries_an_explicit_effort_override() -> None:
    """An explicit ``effort=`` at construction overrides the registry default tier.

    A repo's routing table can replace the reviewer's effort tier at the wiring site
    (:mod:`retinue.pipeline`); this proves the instance's resolved tier reaches the
    request rather than always sending the registry's ``max`` default.
    """
    gen = AgentSdkReviewGenerator(
        credential="k",
        transport=_FakeTransport(_text_response({"findings": []})),
        effort="low",
    )

    payload = gen._payload(_input(PLANTED_DEFECT_DIFF))

    assert payload["output_config"]["effort"] == "low"
    assert "thinking" not in payload


def test_payload_oauth_credential_leads_system_with_claude_code_identity() -> None:
    """An OAuth credential makes the system field lead with the identity block.

    A subscription OAuth token reaches the premium reviewing model over the raw Messages
    API only when the first system block is the Claude Code identity string; the
    reviewer's own brief follows it as the second block.
    """
    gen = AgentSdkReviewGenerator(
        credential="sk-ant-oat-abc",
        transport=_FakeTransport(_text_response({"findings": []})),
    )

    payload = gen._payload(_input(PLANTED_DEFECT_DIFF))

    assert payload["system"] == [
        {"type": "text", "text": CLAUDE_CODE_IDENTITY},
        {"type": "text", "text": _REVIEW_SYSTEM},
    ]


def test_payload_api_key_credential_keeps_plain_string_system() -> None:
    """A raw API-key credential keeps the system field as the unchanged plain brief."""
    gen = AgentSdkReviewGenerator(
        credential="sk-ant-api-xyz",
        transport=_FakeTransport(_text_response({"findings": []})),
    )

    payload = gen._payload(_input(PLANTED_DEFECT_DIFF))

    assert payload["system"] == _REVIEW_SYSTEM
    assert isinstance(payload["system"], str)


@pytest.mark.asyncio
async def test_real_generator_parses_findings_with_severity() -> None:
    """A response with findings parses into a ReviewPlan of severity-carrying findings."""
    transport = _FakeTransport(
        _text_response(
            {
                "findings": [
                    {
                        "title": "Fix off-by-one in total()",
                        "body": "total() adds a stray +1.",
                        "severity": "high",
                    },
                    {
                        "title": "Stale doc",
                        "body": "README mentions an old flag.",
                        "severity": "low",
                    },
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
                severity=Severity.HIGH,
            ),
            ReviewFinding(
                title="Stale doc",
                body="README mentions an old flag.",
                severity=Severity.LOW,
            ),
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


@pytest.mark.asyncio
async def test_real_generator_raises_on_unknown_severity() -> None:
    """A severity outside the vocabulary is a contract breach, surfaced loudly."""
    gen = AgentSdkReviewGenerator(
        credential="k",
        transport=_FakeTransport(
            _text_response(
                {
                    "findings": [
                        {"title": "t", "body": "b", "severity": "catastrophic"}
                    ]
                }
            )
        ),
    )

    with pytest.raises(ReviewGenerationError):
        await gen(_input(PLANTED_DEFECT_DIFF))
