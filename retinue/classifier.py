"""Issue classifier: route one issue to a level of a repo's routing table.

:class:`ClaudeIssueClassifier` is the Messages-API adapter for the
:data:`~retinue.roles.Role.CLASSIFIER` role. Given one issue (title, body, labels, and
optionally its parent PRD body) and a validated :class:`~retinue.repo_config.RoutingConfig`,
it asks a Haiku-class model to pick the single level whose description best fits the
work, constraining the answer with a JSON schema whose ``level`` is an ``enum`` of
exactly the table's level names. It reuses the shared Messages-API wire helpers in
:mod:`retinue.roles` (:func:`~retinue.roles.oauth_system`,
:func:`~retinue.roles.structured_output_config`) and the HTTP seam
(:class:`~retinue.reviewer.HttpTransport` / :class:`~retinue.reviewer.HttpResponse`) the
reviewer already defines, so no new transport protocol is introduced.

Classification is best-effort: a non-200 status or non-conforming output triggers exactly
one retry, and a second failure returns :class:`ClassifyResult` with ``level=None`` rather
than raising. A caller falls back to the routing table's ``default`` level on such a
failure (that fallback, and the production wiring that constructs this adapter, live in a
later PRD #58 slice — this module delivers the registry role and the adapter only).

A repo's routing table may carry a top-level ``classifier:`` override
(:class:`~retinue.repo_config.ModelEffort`); when present it replaces the registry model
and, if it names one, the effort tier for the classifier's own request.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from retinue.repo_config import RoutingConfig
from retinue.reviewer import HttpResponse, HttpTransport
from retinue.roles import (
    Role,
    oauth_system,
    resolve_effort,
    resolve_model,
    structured_output_config,
)

logger = logging.getLogger(__name__)

# Local copies of the Messages-API wire constants, mirroring the edit-isolation pattern
# the other adapters use (reviewer, slicer, resolver each hold their own). The visible
# output is a single level name, so the token ceiling is small.
_ANTHROPIC_VERSION = "2023-06-01"
_OAUTH_BETA = "oauth-2025-04-20"
_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_MAX_TOKENS = 1_024

# The classifier's frozen brief. It is appended after the Claude Code identity block in
# OAuth mode (see :func:`~retinue.roles.oauth_system`).
_CLASSIFY_SYSTEM = (
    "You route one GitHub issue to exactly one level of a repo's routing table. "
    "Choose the single level whose description best fits the issue's work. Return only "
    "the JSON object matching the schema — the chosen level name and nothing else. "
    "No prose."
)


class ClassificationError(RuntimeError):
    """A classification attempt failed: a non-200 status or unusable output.

    Raised internally per attempt and caught by :meth:`ClaudeIssueClassifier.__call__`,
    which retries once and then returns a failure :class:`ClassifyResult` rather than
    propagating. Mirrors :class:`~retinue.reviewer.ReviewGenerationError`.
    """


@dataclass(frozen=True)
class ClassifyInput:
    """One issue to classify against a routing table.

    Attributes:
        title: The issue title.
        body: The issue body (Markdown).
        labels: The issue's label names; ``(none)`` is shown when empty.
        prd_body: The parent PRD's body when this issue is a PRD slice; ``None`` for a
            standalone issue. Included in the prompt only when set.
    """

    title: str
    body: str
    labels: list[str]
    prd_body: str | None = None


@dataclass(frozen=True)
class ClassifyResult:
    """The outcome of classifying one issue.

    A failure is returned, never raised: after the retry is exhausted ``level`` is
    ``None`` and :attr:`failed` is true, and the caller falls back to the routing
    table's ``default`` level (that fallback is a later slice's concern).

    Attributes:
        level: The chosen level name on success; ``None`` when both attempts failed.
    """

    level: str | None

    @property
    def failed(self) -> bool:
        """True when classification produced no level (the caller should fall back)."""
        return self.level is None


def _classifier_model_effort(routing: RoutingConfig) -> tuple[str, str]:
    """Resolve the classifier request's ``(model, effort)`` for a routing table.

    A routing table's top-level ``classifier:`` override replaces the registry model,
    and — when it names one — the registry effort tier; otherwise both come from the
    :data:`~retinue.roles.Role.CLASSIFIER` registry entry.
    """
    override = routing.classifier
    model = override.model if override is not None else resolve_model(Role.CLASSIFIER)
    effort = (
        override.effort
        if override is not None and override.effort is not None
        else resolve_effort(Role.CLASSIFIER)
    )
    return model, effort


@dataclass(frozen=True)
class ClaudeIssueClassifier:
    """Classify one issue to a routing level via the Messages API.

    Callable as ``(ClassifyInput) -> Awaitable[ClassifyResult]``. Holds the credential,
    the injected HTTP transport, and the repo's routing table; everything testable
    offline — :meth:`_headers`, :meth:`_payload`, :meth:`_schema`, :meth:`_build_prompt`,
    :meth:`_parse` — is a pure method. Mirrors
    :class:`~retinue.reviewer.AgentSdkReviewGenerator`.

    Attributes:
        credential: The Anthropic credential. An OAuth subscription token
            (``sk-ant-oat...``) rides ``Authorization: Bearer`` with the OAuth beta
            header and leads the system field with the Claude Code identity block; any
            other value is a raw API key on ``x-api-key``.
        transport: The injected HTTP POST seam.
        routing: The repo's validated routing table — its level names bound the schema
            enum and its descriptions build the prompt; a ``classifier:`` override on it
            steers the request's model and effort.
    """

    credential: str
    transport: HttpTransport
    routing: RoutingConfig

    async def __call__(self, issue: ClassifyInput) -> ClassifyResult:
        """Classify ``issue``, retrying once, and never raise.

        Builds the request once, then attempts at most twice (initial + one retry). A
        :class:`ClassificationError` or leaked :class:`httpx.HTTPError` on either attempt
        is logged and retried; after both fail the result carries ``level=None``.
        """
        headers, payload = self._headers(), self._payload(issue)
        for attempt in (1, 2):
            try:
                return ClassifyResult(level=await self._request(headers, payload))
            except (ClassificationError, httpx.HTTPError) as exc:
                logger.warning(
                    "Classifier attempt %d/2 failed for %r: %s",
                    attempt,
                    issue.title,
                    exc,
                )
        return ClassifyResult(level=None)

    async def _request(
        self, headers: dict[str, str], payload: dict[str, Any]
    ) -> str:
        """POST one classification request and return the chosen level name."""
        response: HttpResponse = await self.transport.post(
            _MESSAGES_URL, headers=headers, json=payload
        )
        if response.status_code != 200:
            raise ClassificationError(
                f"Anthropic Messages API returned {response.status_code}: "
                f"{json.dumps(response.body)[:500]}"
            )
        return self._parse(response.body)

    def _headers(self) -> dict[str, str]:
        """Build the request headers, routing the credential to its auth scheme."""
        headers = {
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        if self.credential.startswith("sk-ant-oat"):
            headers["authorization"] = f"Bearer {self.credential}"
            headers["anthropic-beta"] = _OAUTH_BETA
        else:
            headers["x-api-key"] = self.credential
        return headers

    def _payload(self, issue: ClassifyInput) -> dict[str, Any]:
        """Assemble the Messages API request body for one classification.

        The model and effort ride the routing table's ``classifier:`` override when set,
        else the registry defaults. The lone ``output_config`` carries that effort and
        the enum-constrained schema via the shared
        :func:`~retinue.roles.structured_output_config` helper — the canonical Claude
        structured-output shape (the OpenAI-style top-level ``response_format`` 400s).
        """
        model, effort = _classifier_model_effort(self.routing)
        return {
            "model": model,
            "max_tokens": _MAX_TOKENS,
            "output_config": structured_output_config(
                Role.CLASSIFIER, self._schema(), model=model, effort=effort
            ),
            "system": oauth_system(
                _CLASSIFY_SYSTEM, is_oauth=self.credential.startswith("sk-ant-oat")
            ),
            "messages": [{"role": "user", "content": self._build_prompt(issue)}],
        }

    def _schema(self) -> dict[str, Any]:
        """The strict JSON schema: a required ``level`` bound to the table's names."""
        return {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": list(self.routing.levels)}
            },
            "required": ["level"],
            "additionalProperties": False,
        }

    def _build_prompt(self, issue: ClassifyInput) -> str:
        """Build the user message: the levels, then the issue (and PRD body if any)."""
        levels = "\n".join(
            f"- {name}: {level.description}"
            for name, level in self.routing.levels.items()
        )
        labels = ", ".join(issue.labels) or "(none)"
        parts = [
            "Routing levels (name: description):",
            levels,
            "",
            f"Issue title: {issue.title}",
            f"Issue labels: {labels}",
            "Issue body:",
            issue.body,
        ]
        if issue.prd_body is not None:
            parts += ["", "Parent PRD body:", issue.prd_body]
        parts += [
            "",
            "Return only the JSON object matching the schema — the chosen level name.",
        ]
        return "\n".join(parts)

    def _parse(self, body: dict[str, Any]) -> str:
        """Parse a Messages API response into the chosen level name.

        Concatenates the ``text`` content blocks, loads them as the schema JSON, and
        reads ``level``. Empty text, malformed JSON, a non-object payload, or a ``level``
        that is not a table name raises :class:`ClassificationError` — enforcing the enum
        a second time so non-conforming output triggers the retry. Every guard tolerates a
        degenerate 200 body (non-list ``content``, unhashable ``level``) so no shape of a
        200 response can raise anything but :class:`ClassificationError`.
        """
        content = body.get("content")
        blocks = content if isinstance(content, list) else []
        text = "".join(
            block["text"]
            for block in blocks
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
        if not text.strip():
            raise ClassificationError("Messages API response carried no text content")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ClassificationError(f"Classifier emitted invalid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise ClassificationError("Classifier JSON is not an object")
        level = parsed.get("level")
        if not isinstance(level, str) or level not in self.routing.levels:
            raise ClassificationError(
                f"Classifier chose {level!r}, not one of {sorted(self.routing.levels)}"
            )
        return level
