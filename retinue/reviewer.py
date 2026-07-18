"""Internal reviewer: the in-session pre-PR review of one issue's diff.

The reviewer is the headless Agent-SDK seam the ad-hoc build's review gate runs after a
green push (see :func:`retinue.adhoc_build.build_adhoc_issue`). It reads one freshly-built
``issue-<N>`` branch's diff over the target base, runs the Anthropic Messages API headless
to review it, and returns a :class:`ReviewPlan`: an ordered list of :class:`ReviewFinding`,
each carrying a :class:`~retinue.vocab.Severity`. The build's gate then partitions those
findings — blocking (severity at or above the threshold, default
:attr:`~retinue.vocab.Severity.HIGH`) versus backlog — and acts on them; the reviewer
itself never files, wires, or edits anything.

The HTTP call is the only side effect, taken behind the injected :class:`HttpTransport`
protocol: production wires a concrete httpx-style client, tests inject a fake. The pure
parts — auth header build, request payload assembly, and response parsing — are exercised
without network.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from retinue.messages_api import (
    MESSAGES_URL,
    HttpTransport,
    extract_json_object,
    request_headers,
)
from retinue.roles import (
    Role,
    is_oauth_credential,
    oauth_system,
    resolve_effort,
    resolve_model,
    structured_output_config,
)
from retinue.vocab import Severity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReviewInput:
    """What the reviewer reviews: one built issue's diff.

    Attributes:
        repo_full_name: e.g. "owner/repo"; carried through to the payload framing.
        issue_number: The issue whose ``issue-<N>`` diff is under review.
        diff: The issue branch's contribution over the target base — the review surface
            for correctness bugs and stale docs.
    """

    repo_full_name: str
    issue_number: int
    diff: str


@dataclass
class ReviewFinding:
    """One thing the reviewer flagged in the issue's diff.

    Attributes:
        title: Follow-up issue title (used when the gate files a backlog nit).
        body: The finding's what/why.
        severity: How severe the finding is; the build's gate blocks the PR when this is
            at or above the configured threshold (default
            :attr:`~retinue.vocab.Severity.HIGH`) and files the rest as backlog nits
            tagged with the matching ``priority:<severity>`` label.
    """

    title: str
    body: str
    severity: Severity


@dataclass(frozen=True)
class ReviewPlan:
    """The headless reviewer's output: the findings the gate partitions and acts on."""

    findings: list[ReviewFinding]


# The injected review seam: reviews one issue's diff and returns its findings. Async and
# faked in tests; production wires :class:`AgentSdkReviewGenerator`.
ReviewGenerator = Callable[[ReviewInput], Awaitable[ReviewPlan]]


# ---------------------------------------------------------------------------
# Real Agent-SDK reviewer (production adapter behind the ReviewGenerator seam).
#
# Drives the Anthropic Messages API headless to review one issue's diff and emit
# structured, severity-carrying findings. The HTTP call is the only side effect, so it is
# taken behind the injected :class:`HttpTransport` protocol: production wires a concrete
# httpx-style client, tests inject a fake. The pure parts — auth header build, request
# payload assembly, and response parsing — are exercised without network.
#
# SDK conventions match the slicer: the model and effort tier come from the
# :data:`~retinue.roles.Role.REVIEWER` registry entry (Opus 4.8 at the ``max`` tier by
# default); a subscription OAuth token goes on ``Authorization: Bearer`` with the
# ``oauth-2025-04-20`` beta header, while a raw API key goes on ``x-api-key``;
# ``anthropic-version`` is always sent. The model must return only the JSON object
# matching the schema.
# ---------------------------------------------------------------------------

_MAX_TOKENS = 16_000

# Cap on the issue diff interpolated into the reviewer's user message, mirroring the
# done-check's failure-detail clamp. An unbounded diff would blow the request body (and the
# reviewer's context) on a big change; the head is kept — a diff's file headers and hunks
# read front-to-back — with an explicit note so the model knows the review surface is
# partial.
_DIFF_MAX_CHARS = 8_000

# Strict JSON schema the headless reviewer must emit: an ordered list of findings, each
# with a severity drawn from the :class:`~retinue.vocab.Severity` vocabulary. The gate
# partitions on severity — blocking at or above the threshold, backlog below.
_SEVERITY_NAMES = [severity.name.lower() for severity in Severity]
_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "severity": {"type": "string", "enum": _SEVERITY_NAMES},
                },
                "required": ["title", "body", "severity"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}

# The headless reviewer's brief. Frozen (no per-request interpolation) so the request
# prefix is cacheable across issues; the diff rides the user message.
_REVIEW_SYSTEM = (
    "You review one issue's pull-request diff, before the PR opens, for genuine "
    "defects: correctness bugs introduced by the diff and documentation the diff made "
    "stale. Report only real, actionable findings — never style nits or speculation; a "
    "clean diff yields an empty 'findings' list. Assign each finding a 'severity' of "
    "'critical', 'high', 'medium', or 'low' by its impact: a defect that breaks "
    "correctness or ships broken behaviour is 'high' or 'critical', while a minor or "
    "cosmetic concern is 'medium' or 'low'. Return only the JSON object matching the "
    "schema; no prose."
)


def _clip_diff(diff: str) -> str:
    """Clamp ``diff`` to :data:`_DIFF_MAX_CHARS`, noting the truncation explicitly.

    Keeps the head — a diff's file headers and hunks read front-to-back — and appends
    a note carrying the original size so the model knows the review surface is
    partial rather than silently reviewing a diff that just stops.
    """
    if len(diff) <= _DIFF_MAX_CHARS:
        return diff
    return (
        f"{diff[:_DIFF_MAX_CHARS]}\n"
        f"[diff truncated: first {_DIFF_MAX_CHARS} of {len(diff)} characters shown]"
    )


@dataclass(frozen=True)
class AgentSdkReviewGenerator:
    """Real :data:`ReviewGenerator`: review one issue's diff via the Messages API.

    Satisfies the ``generate`` protocol ``(ReviewInput) -> Awaitable[ReviewPlan]``
    by calling itself. Holds the credential and the HTTP transport; everything
    that can be tested offline — :meth:`_headers`, :meth:`_payload`,
    :meth:`_parse` — is a pure method.

    Attributes:
        credential: The Anthropic credential. An OAuth subscription token
            (``sk-ant-oat...``) is sent as ``Authorization: Bearer`` with the
            OAuth beta header; any other value is treated as a raw API key on
            ``x-api-key``.
        transport: The injected HTTP POST seam.
        model: The reviewing model id; defaults to the
            :data:`~retinue.roles.Role.REVIEWER` registry entry (Opus 4.8), which a
            repo's routing level can replace at the wiring site.
        effort: The review request's reasoning-effort tier; defaults to the
            registry entry's tier, which a repo's routing level can replace at
            the wiring site.
    """

    credential: str
    transport: HttpTransport
    model: str = field(default_factory=lambda: resolve_model(Role.REVIEWER))
    effort: str = field(default_factory=lambda: resolve_effort(Role.REVIEWER))

    async def __call__(self, review_input: ReviewInput) -> ReviewPlan:
        """Review ``review_input``'s diff and return the parsed :class:`ReviewPlan`."""
        response = await self.transport.post(
            MESSAGES_URL,
            headers=self._headers(),
            json=self._payload(review_input),
        )
        if response.status_code != 200:
            raise ReviewGenerationError(
                f"Anthropic Messages API returned {response.status_code}"
            )
        return self._parse(response.body)

    def _headers(self) -> dict[str, str]:
        """Build the request headers, routing the credential to its auth scheme."""
        return request_headers(self.credential)

    def _payload(self, review_input: ReviewInput) -> dict[str, Any]:
        """Assemble the Messages API request body for one issue's review.

        The internal reviewer is the highest-rigor Opus role; the shared
        :func:`~retinue.roles.structured_output_config` helper carries the instance's
        resolved effort tier (``max`` by default) and the findings JSON schema on one
        ``output_config`` dict — the canonical Messages API structured-output shape
        (the OpenAI-style top-level ``response_format`` is not a Claude API parameter
        and 400s). The issue diff is clamped to :data:`_DIFF_MAX_CHARS` before
        interpolation so a big change cannot blow the request body.
        """
        user = (
            f"Pre-PR review of issue #{review_input.issue_number} in "
            f"{review_input.repo_full_name}.\n"
            "Review the following diff and emit findings as JSON matching the "
            "schema:\n\n"
            f"{_clip_diff(review_input.diff)}"
        )
        return {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "output_config": structured_output_config(
                Role.REVIEWER, _REVIEW_SCHEMA, model=self.model, effort=self.effort
            ),
            "system": oauth_system(
                _REVIEW_SYSTEM, is_oauth=is_oauth_credential(self.credential)
            ),
            "messages": [{"role": "user", "content": user}],
        }

    def _parse(self, body: dict[str, Any]) -> ReviewPlan:
        """Parse a Messages API response body into a :class:`ReviewPlan`.

        Reads the concatenated ``text`` content blocks, loads them as the schema
        JSON, and builds one :class:`ReviewFinding` per entry, mapping each entry's
        ``severity`` name back onto the :class:`~retinue.vocab.Severity` enum. A
        response missing text, or carrying malformed JSON or a non-list ``findings``,
        raises :class:`ReviewGenerationError` rather than silently reviewing nothing.
        """
        parsed = extract_json_object(body, who="Reviewer", error=ReviewGenerationError)
        raw_findings = parsed.get("findings")
        if not isinstance(raw_findings, list):
            raise ReviewGenerationError("Reviewer JSON missing a 'findings' list")

        findings = [
            ReviewFinding(
                title=str(item["title"]),
                body=str(item["body"]),
                severity=_parse_severity(item["severity"]),
            )
            for item in raw_findings
        ]
        return ReviewPlan(findings=findings)


def _parse_severity(raw: Any) -> Severity:
    """Map a schema ``severity`` name (e.g. ``"high"``) onto the :class:`Severity` enum.

    The schema constrains the value to one of the four names, so an unknown value is a
    contract breach the reviewer surfaces loudly rather than defaulting a severity.
    """
    try:
        return Severity[str(raw).upper()]
    except KeyError as exc:
        raise ReviewGenerationError(f"Reviewer emitted unknown severity {raw!r}") from exc


class ReviewGenerationError(RuntimeError):
    """The headless reviewer failed to produce a usable :class:`ReviewPlan`."""
