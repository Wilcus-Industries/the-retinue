"""Internal reviewer: file review-fix follow-ups after a round's merge.

:func:`review_round` is the entry point. After a PRD round merges (see
:func:`retinue.orchestrator.build_prd`), the reviewer takes that round's merged diff
and merged issue numbers, runs the headless Agent-SDK review seam (``generate``,
injected) over them, and for each genuine finding:

1. files a follow-up issue via the slicer's ``create_issue`` seam, reusing the
   ``ready-for-agent`` + ``Part of #<prd>`` shape and adding a ``review-fix`` label
   so the agent loop routes it as a correctness/stale-doc fix, and
2. wires that new issue into the ``## Blocked by`` of each dependent open issue it
   flags (the ``edit_blocked_by`` seam, a ``gh issue edit``), so the fix builds in a
   later round *before* the work layered on top of the defect.

The reviewer **never edits code** — it only files and wires issues. All three
side-effecting seams (the Agent-SDK reviewer, the gh issue creator reused from the
slicer, and the gh issue-body editor) are injected, so the flow is unit-testable
without the Agent SDK, gh, or network.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from retinue.slicer import READY_LABEL, CreatedIssue, IssueCreator, IssueDraft

logger = logging.getLogger(__name__)

REVIEW_FIX_LABEL = "review-fix"


@dataclass(frozen=True)
class ReviewInput:
    """What the reviewer reviews: one merged round's diff and its merged issues.

    Attributes:
        repo_full_name: e.g. "owner/repo"; targets the issue creation and edits.
        prd_number: The parent PRD; review-fix issues link back via ``Part of #``.
        merged_issues: Issue numbers merged in the round, in merge order. These are
            the issues whose work the review-fix may need to block.
        diff: The round's merged diff — the review surface for correctness and stale
            docs.
    """

    repo_full_name: str
    prd_number: int
    merged_issues: list[int]
    diff: str


@dataclass
class ReviewFinding:
    """One thing the reviewer flagged in the round's diff.

    Attributes:
        title: Follow-up issue title.
        body: The finding's what/why — the review-fix issue body, enriched in place
            with the ``Part of`` footer before creation.
        blocks_issues: Issue numbers (from ``merged_issues``) whose work is layered on
            this defect; the filed review-fix issue is wired into each one's
            ``## Blocked by`` so the fix lands first. Empty for a standalone fix (e.g.
            a stale doc) that nothing depends on.
    """

    title: str
    body: str
    blocks_issues: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ReviewPlan:
    """The headless reviewer's output: the findings to file as review-fix issues."""

    findings: list[ReviewFinding]


@dataclass(frozen=True)
class EditBlockedByRequest:
    """Payload handed to the issue-body editor: add one Blocked-by reference.

    A ``gh issue edit`` that appends ``add_blocker`` to ``issue_number``'s
    ``## Blocked by`` block, so the dependent builds only after the fix merges.
    """

    repo_full_name: str
    issue_number: int
    add_blocker: int


@dataclass(frozen=True)
class ReviewResult:
    """Result of reviewing one merged round.

    Attributes:
        filed_issues: Issue numbers of the review-fix follow-ups filed, in finding
            order (empty when the review was clean).
    """

    filed_issues: list[int] = field(default_factory=list)


# Injected seams. ``generate`` runs the headless Agent-SDK reviewer over the round's
# diff; ``create_issue`` (reused from the slicer) files one issue via gh;
# ``edit_blocked_by`` wires the new issue into a dependent's ``## Blocked by``. All are
# async and faked in tests — no Agent SDK, gh, or network.
ReviewGenerator = Callable[[ReviewInput], Awaitable[ReviewPlan]]
BlockedByEditor = Callable[[EditBlockedByRequest], Awaitable[None]]


# ---------------------------------------------------------------------------
# Real Agent-SDK reviewer (production adapter behind the ReviewGenerator seam).
#
# Drives the Anthropic Messages API headless to review a round's merged diff and
# emit structured findings. The HTTP call is the only side effect, so it is taken
# behind the injected :class:`HttpTransport` protocol: production wires a concrete
# httpx-style client, tests inject a fake. The pure parts — auth header build,
# request payload assembly, and response parsing — are exercised without network.
#
# SDK conventions match the slicer: Opus 4.8 is the default model; a subscription
# OAuth token goes on ``Authorization: Bearer`` with the ``oauth-2025-04-20`` beta
# header, while a raw API key goes on ``x-api-key``; ``anthropic-version`` is
# always sent. The model must return only the JSON object matching the schema.
# ---------------------------------------------------------------------------

_REVIEW_MODEL = "claude-opus-4-8"
_ANTHROPIC_VERSION = "2023-06-01"
_OAUTH_BETA = "oauth-2025-04-20"
_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_MAX_TOKENS = 16_000

# Strict JSON schema the headless reviewer must emit: an ordered list of findings,
# each with the dependent issue numbers (from the round) whose work is layered on
# the defect. ``blocks_issues`` is empty for a standalone fix (e.g. a stale doc).
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
                    "blocks_issues": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["title", "body", "blocks_issues"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}

# The headless reviewer's brief. Frozen (no per-request interpolation) so the
# request prefix is cacheable across rounds; the diff and issue list ride in the
# user message.
_REVIEW_SYSTEM = (
    "You review a merged pull-request diff for genuine defects: correctness bugs "
    "introduced by the diff and documentation the diff made stale. Report only "
    "real, actionable findings — never style nits or speculation; a clean diff "
    "yields an empty 'findings' list. For each finding, list in 'blocks_issues' "
    "the issue numbers (drawn only from the round's merged issues) whose work is "
    "layered on the defect, so the fix builds first; leave it empty for a "
    "standalone fix nothing depends on. Return only the JSON object matching the "
    "schema; no prose."
)


@dataclass(frozen=True)
class HttpResponse:
    """The slice of an HTTP response the reviewer reads: status and JSON body."""

    status_code: int
    body: dict[str, Any]


class HttpTransport(Protocol):
    """Async HTTP POST seam (httpx-style). The network edge of the reviewer.

    A production implementation wraps an httpx client; tests inject a fake that
    returns a canned :class:`HttpResponse`. Kept narrow — one POST — so the real
    review flow is exercisable without network.
    """

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any]
    ) -> HttpResponse:
        """POST ``json`` to ``url`` with ``headers`` and return the response."""
        ...


@dataclass(frozen=True)
class AgentSdkReviewGenerator:
    """Real :data:`ReviewGenerator`: review a round's diff via the Messages API.

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
        model: The reviewing model id; defaults to Opus 4.8.
    """

    credential: str
    transport: HttpTransport
    model: str = _REVIEW_MODEL

    async def __call__(self, review_input: ReviewInput) -> ReviewPlan:
        """Review ``review_input``'s diff and return the parsed :class:`ReviewPlan`."""
        response = await self.transport.post(
            _MESSAGES_URL,
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

    def _payload(self, review_input: ReviewInput) -> dict[str, Any]:
        """Assemble the Messages API request body for one round's review."""
        merged = ", ".join(f"#{n}" for n in review_input.merged_issues) or "(none)"
        user = (
            f"Merged round of PRD #{review_input.prd_number} in "
            f"{review_input.repo_full_name}.\n"
            f"Merged issues, in merge order: {merged}.\n"
            "Review the following merged diff and emit findings as JSON matching "
            "the schema:\n\n"
            f"{review_input.diff}"
        )
        return {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "system": _REVIEW_SYSTEM,
            "messages": [{"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": _REVIEW_SCHEMA},
        }

    def _parse(self, body: dict[str, Any]) -> ReviewPlan:
        """Parse a Messages API response body into a :class:`ReviewPlan`.

        Reads the concatenated ``text`` content blocks, loads them as the schema
        JSON, and builds one :class:`ReviewFinding` per entry. A response missing
        text, or carrying malformed JSON or a non-list ``findings``, raises
        :class:`ReviewGenerationError` rather than silently filing nothing.
        """
        text = "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text.strip():
            raise ReviewGenerationError("Messages API response carried no text content")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ReviewGenerationError(f"Reviewer emitted invalid JSON: {exc}") from exc

        raw_findings = parsed.get("findings")
        if not isinstance(raw_findings, list):
            raise ReviewGenerationError("Reviewer JSON missing a 'findings' list")

        findings = [
            ReviewFinding(
                title=str(item["title"]),
                body=str(item["body"]),
                blocks_issues=[int(n) for n in item.get("blocks_issues", [])],
            )
            for item in raw_findings
        ]
        return ReviewPlan(findings=findings)


class ReviewGenerationError(RuntimeError):
    """The headless reviewer failed to produce a usable :class:`ReviewPlan`."""


async def review_round(
    review_input: ReviewInput,
    *,
    generate: ReviewGenerator,
    create_issue: IssueCreator,
    edit_blocked_by: BlockedByEditor,
) -> ReviewResult:
    """Review a merged round; file and wire a review-fix issue per finding.

    Runs ``generate`` over the round's merged diff + issue numbers, then for every
    finding files a ``review-fix`` + ``ready-for-agent`` + ``Part of #<prd>`` issue and
    wires it into the ``## Blocked by`` of each dependent open issue it flags. A clean
    review (no findings) files nothing. The reviewer never edits code.

    Args:
        review_input: The round's diff, merged issue numbers, repo, and PRD number.
        generate: Async headless reviewer (Agent SDK seam) producing a ReviewPlan.
        create_issue: Async issue creator (gh seam) filing one review-fix issue;
            reused from the slicer so the labeling/Part-of shape is shared.
        edit_blocked_by: Async issue-body editor (gh seam) appending a Blocked-by ref
            to a dependent issue.

    Returns:
        A :class:`ReviewResult` with the filed review-fix issue numbers in finding
        order — empty when the review was clean.
    """
    plan = await generate(review_input)
    if not plan.findings:
        logger.info(
            "Review of round (PRD #%d, %s) found nothing to fix",
            review_input.prd_number,
            review_input.repo_full_name,
        )
        return ReviewResult()

    filed: list[int] = []
    for finding in plan.findings:
        created = await _file_review_fix(finding, review_input, create_issue)
        await _wire_blocked_by(created.issue_number, finding, review_input, edit_blocked_by)
        filed.append(created.issue_number)
    return ReviewResult(filed_issues=filed)


async def _file_review_fix(
    finding: ReviewFinding,
    review_input: ReviewInput,
    create_issue: IssueCreator,
) -> CreatedIssue:
    """File one finding as a labeled, PRD-linked review-fix issue via the gh seam."""
    draft = IssueDraft(
        title=finding.title,
        body=f"{finding.body.rstrip()}\n\nPart of #{review_input.prd_number}",
        labels=[READY_LABEL, REVIEW_FIX_LABEL],
    )
    return await create_issue(draft)


async def _wire_blocked_by(
    fix_number: int,
    finding: ReviewFinding,
    review_input: ReviewInput,
    edit_blocked_by: BlockedByEditor,
) -> None:
    """Add ``fix_number`` to each flagged dependent's ``## Blocked by`` block.

    Only issues that were merged in this round can be wired — a finding that names an
    issue outside the round is dropped with a warning rather than editing an unrelated
    issue, since the reviewer must not touch work it did not just review.
    """
    for dependent in finding.blocks_issues:
        if dependent not in review_input.merged_issues:
            logger.warning(
                "Dropping review-fix wiring: #%d is not in the reviewed round %s",
                dependent,
                review_input.merged_issues,
            )
            continue
        await edit_blocked_by(
            EditBlockedByRequest(
                repo_full_name=review_input.repo_full_name,
                issue_number=dependent,
                add_blocker=fix_number,
            )
        )
