"""Tests for the issue classifier adapter.

:class:`~retinue.classifier.ClaudeIssueClassifier` routes one issue to a level of a
repo's routing table via the Messages API. The pure parts — headers, payload, schema,
prompt, and response parsing — are exercised offline with a fake transport; the retry
contract is exercised by returning non-200 / non-conforming responses. No network, Agent
SDK, or gh is touched. Mirrors ``tests/test_reviewer.py``'s pure-parts style.
"""

from __future__ import annotations

import json

import pytest

from retinue.classifier import (
    ClassifyInput,
    ClassifyResult,
    ClaudeIssueClassifier,
)
from retinue.repo_config import ModelEffort, RoutingConfig, RoutingLevel
from retinue.reviewer import HttpResponse
from retinue.roles import CLAUDE_CODE_IDENTITY, EFFORT_LOW, ROLE_REGISTRY, Role


class _FakeTransport:
    """Records every POST and returns canned responses in order. No network.

    A single response is repeated; a list is consumed one per call so a retry can see a
    different response than the first attempt.
    """

    def __init__(self, response: HttpResponse | list[HttpResponse]) -> None:
        self._responses = response if isinstance(response, list) else None
        self._single = None if isinstance(response, list) else response
        self.calls: list[tuple[str, dict[str, str], dict[str, object]]] = []

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, object]
    ) -> HttpResponse:
        self.calls.append((url, headers, json))
        if self._responses is not None:
            return self._responses[len(self.calls) - 1]
        assert self._single is not None
        return self._single


def _text_response(payload: object, *, status_code: int = 200) -> HttpResponse:
    """A Messages API response whose single text block is ``payload`` as JSON."""
    return HttpResponse(
        status_code=status_code,
        body={"content": [{"type": "text", "text": json.dumps(payload)}]},
    )


def _routing(*, classifier: ModelEffort | None = None) -> RoutingConfig:
    """A two-level routing table (``trivial`` / ``standard``), default ``standard``."""
    return RoutingConfig(
        default="standard",
        classifier=classifier,
        levels={
            "trivial": RoutingLevel(description="tiny one-file typo or doc fix"),
            "standard": RoutingLevel(description="a normal multi-file feature slice"),
        },
    )


def _issue(*, prd_body: str | None = None) -> ClassifyInput:
    return ClassifyInput(
        title="Add a widget",
        body="Wire the widget into the dashboard.",
        labels=["ready-for-agent", "feature"],
        prd_body=prd_body,
    )


def _classifier(
    transport: _FakeTransport,
    *,
    credential: str = "k",
    routing: RoutingConfig | None = None,
) -> ClaudeIssueClassifier:
    return ClaudeIssueClassifier(
        credential=credential,
        transport=transport,
        routing=routing or _routing(),
    )


def test_schema_constrains_level_to_the_table_names() -> None:
    """The output schema's ``level`` enum is exactly the routing table's level names."""
    gen = _classifier(_FakeTransport(_text_response({"level": "trivial"})))
    payload = gen._payload(_issue())

    schema = payload["output_config"]["format"]["schema"]
    assert set(schema["properties"]["level"]["enum"]) == {"trivial", "standard"}
    assert schema["required"] == ["level"]
    assert schema["additionalProperties"] is False
    # The canonical Claude structured-output shape, never the OpenAI-style parameter.
    assert "response_format" not in payload
    assert payload["output_config"]["format"]["type"] == "json_schema"


def test_prompt_lists_every_level_and_the_issue_fields() -> None:
    """The user message carries each level's name + description and the issue fields."""
    gen = _classifier(_FakeTransport(_text_response({"level": "trivial"})))
    prompt = gen._payload(_issue())["messages"][0]["content"]

    assert "trivial" in prompt and "tiny one-file typo or doc fix" in prompt
    assert "standard" in prompt and "a normal multi-file feature slice" in prompt
    assert "Add a widget" in prompt
    assert "Wire the widget into the dashboard." in prompt
    assert "ready-for-agent" in prompt and "feature" in prompt


def test_prompt_includes_the_prd_body_only_when_present() -> None:
    """The parent PRD body appears in the prompt exactly when ``prd_body`` is set."""
    gen = _classifier(_FakeTransport(_text_response({"level": "trivial"})))

    without = gen._payload(_issue())["messages"][0]["content"]
    assert "Parent PRD body" not in without

    with_prd = gen._payload(_issue(prd_body="the umbrella PRD text"))
    prompt = with_prd["messages"][0]["content"]
    assert "Parent PRD body" in prompt
    assert "the umbrella PRD text" in prompt


def test_oauth_credential_leads_system_with_the_identity_block() -> None:
    """An OAuth token makes the system field lead with the Claude Code identity block."""
    gen = _classifier(
        _FakeTransport(_text_response({"level": "trivial"})),
        credential="sk-ant-oat-abc",
    )
    system = gen._payload(_issue())["system"]

    assert system[0] == {"type": "text", "text": CLAUDE_CODE_IDENTITY}


def test_api_key_credential_keeps_the_plain_system_string() -> None:
    """An API key leaves the system field as the plain classifier brief string."""
    gen = _classifier(
        _FakeTransport(_text_response({"level": "trivial"})),
        credential="sk-key",
    )
    system = gen._payload(_issue())["system"]

    assert isinstance(system, str)
    assert "route one GitHub issue" in system


def test_headers_oauth_uses_bearer_and_beta() -> None:
    """An OAuth token rides Authorization: Bearer + the oauth beta, no x-api-key."""
    gen = _classifier(
        _FakeTransport(_text_response({"level": "trivial"})),
        credential="sk-ant-oat-abc",
    )
    headers = gen._headers()

    assert headers["authorization"] == "Bearer sk-ant-oat-abc"
    assert "anthropic-beta" in headers
    assert "x-api-key" not in headers


def test_headers_api_key_uses_x_api_key() -> None:
    """A raw API key rides x-api-key, with no bearer or beta header."""
    gen = _classifier(
        _FakeTransport(_text_response({"level": "trivial"})),
        credential="sk-key",
    )
    headers = gen._headers()

    assert headers["x-api-key"] == "sk-key"
    assert "authorization" not in headers
    assert "anthropic-beta" not in headers


def test_default_model_and_effort_come_from_the_registry() -> None:
    """With no ``classifier:`` override, model and effort are the registry defaults."""
    gen = _classifier(_FakeTransport(_text_response({"level": "trivial"})))
    payload = gen._payload(_issue())

    assert payload["model"] == ROLE_REGISTRY[Role.CLASSIFIER].model == "claude-haiku-4-5"
    assert payload["output_config"]["effort"] == EFFORT_LOW


def test_classifier_override_steers_model_and_effort() -> None:
    """A routing ``classifier:`` override replaces both model and output effort."""
    routing = _routing(
        classifier=ModelEffort(model="claude-haiku-4-5-override", effort="max")
    )
    gen = _classifier(
        _FakeTransport(_text_response({"level": "trivial"})), routing=routing
    )
    payload = gen._payload(_issue())

    assert payload["model"] == "claude-haiku-4-5-override"
    assert payload["output_config"]["effort"] == "max"


@pytest.mark.asyncio
async def test_success_returns_the_chosen_level_in_one_call() -> None:
    """A valid response yields the chosen level with exactly one POST."""
    transport = _FakeTransport(_text_response({"level": "trivial"}))
    gen = _classifier(transport)

    result = await gen(_issue())

    assert result == ClassifyResult(level="trivial")
    assert result.failed is False
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_non_200_retries_once_then_fails() -> None:
    """A non-200 status retries exactly once, then returns ``level=None``."""
    transport = _FakeTransport(
        [
            _text_response({"level": "trivial"}, status_code=429),
            _text_response({"level": "trivial"}, status_code=429),
        ]
    )
    gen = _classifier(transport)

    result = await gen(_issue())

    assert result.level is None
    assert result.failed is True
    assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_non_conforming_output_retries_once_then_fails() -> None:
    """A level outside the table retries once, then returns ``level=None``."""
    transport = _FakeTransport(
        [
            _text_response({"level": "does-not-exist"}),
            _text_response({"level": "does-not-exist"}),
        ]
    )
    gen = _classifier(transport)

    result = await gen(_issue())

    assert result.level is None
    assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_unhashable_level_retries_once_then_fails() -> None:
    """A degenerate non-string ``level`` (e.g. a list) fails without raising.

    ``level not in self.routing.levels`` would raise ``TypeError`` on an unhashable
    value; the type check must turn it into a retried :class:`ClassificationError`.
    """
    transport = _FakeTransport(
        [
            _text_response({"level": ["trivial"]}),
            _text_response({"level": ["trivial"]}),
        ]
    )
    gen = _classifier(transport)

    result = await gen(_issue())

    assert result == ClassifyResult(level=None)
    assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_non_list_content_retries_once_then_fails() -> None:
    """A 200 body whose ``content`` is not a list fails without raising."""
    transport = _FakeTransport(
        [
            HttpResponse(status_code=200, body={"content": 42}),
            HttpResponse(status_code=200, body={"content": 42}),
        ]
    )
    gen = _classifier(transport)

    result = await gen(_issue())

    assert result == ClassifyResult(level=None)
    assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_non_string_text_block_retries_once_then_fails() -> None:
    """A 200 body with a non-string ``text`` value fails without raising."""
    transport = _FakeTransport(
        [
            HttpResponse(status_code=200, body={"content": [{"type": "text", "text": 42}]}),
            HttpResponse(status_code=200, body={"content": [{"type": "text", "text": 42}]}),
        ]
    )
    gen = _classifier(transport)

    result = await gen(_issue())

    assert result == ClassifyResult(level=None)
    assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_retry_succeeds_after_a_first_failure() -> None:
    """A first-attempt non-200 then a valid response yields the level in two calls."""
    transport = _FakeTransport(
        [
            _text_response({"level": "standard"}, status_code=429),
            _text_response({"level": "standard"}),
        ]
    )
    gen = _classifier(transport)

    result = await gen(_issue())

    assert result == ClassifyResult(level="standard")
    assert len(transport.calls) == 2
