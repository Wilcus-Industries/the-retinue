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

from retinue.roles import (
    Role,
    oauth_system,
    resolve_effort,
    resolve_model,
    structured_output_config,
)
from retinue.slicer import (
    READY_LABEL,
    CreatedIssue,
    IssueCreator,
    IssueDraft,
)

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
# SDK conventions match the slicer: the model and effort tier come from the
# :data:`~retinue.roles.Role.REVIEWER` registry entry (Opus 4.8 at the ``max`` tier by
# default); a subscription OAuth token goes on ``Authorization: Bearer`` with the
# ``oauth-2025-04-20`` beta header, while a raw API key goes on ``x-api-key``;
# ``anthropic-version`` is always sent. The model must return only the JSON object
# matching the schema.
# ---------------------------------------------------------------------------

_ANTHROPIC_VERSION = "2023-06-01"
_OAUTH_BETA = "oauth-2025-04-20"
_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_MAX_TOKENS = 16_000

# Cap on the round diff interpolated into the reviewer's user message, mirroring the
# done-check's failure-detail clamp. An unbounded merged diff would blow the request
# body (and the reviewer's context) on a big round; the head is kept — a diff's file
# headers and hunks read front-to-back — with an explicit note so the model knows the
# review surface is partial.
_DIFF_MAX_CHARS = 8_000

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
        """Assemble the Messages API request body for one round's review.

        The internal reviewer is the highest-rigor Opus role; the shared
        :func:`~retinue.roles.structured_output_config` helper carries the instance's
        resolved effort tier (``max`` by default) and the findings JSON schema on one
        ``output_config`` dict — the canonical Messages API structured-output shape
        (the OpenAI-style top-level ``response_format`` is not a Claude API parameter
        and 400s). The round diff is clamped to :data:`_DIFF_MAX_CHARS` before
        interpolation so a big round cannot blow the request body.
        """
        merged = ", ".join(f"#{n}" for n in review_input.merged_issues) or "(none)"
        user = (
            f"Merged round of PRD #{review_input.prd_number} in "
            f"{review_input.repo_full_name}.\n"
            f"Merged issues, in merge order: {merged}.\n"
            "Review the following merged diff and emit findings as JSON matching "
            "the schema:\n\n"
            f"{_clip_diff(review_input.diff)}"
        )
        return {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "output_config": structured_output_config(
                Role.REVIEWER, _REVIEW_SCHEMA, effort=self.effort
            ),
            "system": oauth_system(
                _REVIEW_SYSTEM, is_oauth=self.credential.startswith("sk-ant-oat")
            ),
            "messages": [{"role": "user", "content": user}],
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


# ---------------------------------------------------------------------------
# Real gh-cli BlockedByEditor (production adapter behind the BlockedByEditor seam).
#
# Wires a filed review-fix issue into a dependent's ``## Blocked by`` block by
# editing the dependent's issue body via ``gh``. The flow is read-modify-write:
# read the dependent's current body (``gh issue view --json body``), add the
# ``#<fix>`` reference to its ``## Blocked by`` block (matching the slicer's block
# shape), then write it back (``gh issue edit --body``). The process spawn is the
# only side effect, taken behind the injected :class:`GhRunner` protocol so the
# pure parts — auth-env build, command assembly, body parsing, block rendering —
# are exercised without a live ``gh`` or network.
#
# gh is authenticated via ``GH_TOKEN`` in the environment (gh sends it as
# ``Authorization: Bearer`` on every REST/GraphQL call), so the adapter never
# assembles an auth header itself — it injects the token and lets gh own the wire.
# ---------------------------------------------------------------------------

_BLOCKED_BY_HEADING = "## Blocked by"


@dataclass(frozen=True)
class GhResult:
    """Captured result of a single ``gh`` invocation.

    Attributes:
        exit_code: ``gh``'s process exit status; ``0`` means success.
        stdout: Captured standard output (the issue body payload to parse).
        stderr: Captured standard error (surfaced in the error on failure).
    """

    exit_code: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        """True when ``gh`` exited successfully (exit code 0)."""
        return self.exit_code == 0


class GhRunner(Protocol):
    """Runs a single ``gh`` command. The process-spawn seam under the editor.

    A production implementation spawns ``gh`` as a subprocess with ``env`` merged
    into its environment (so ``GH_TOKEN`` authenticates the call) and returns the
    captured :class:`GhResult`; tests inject a fake that records each ``(args, env)``
    and returns a canned result. ``args`` never includes the leading ``"gh"`` — the
    runner owns the executable name.
    """

    async def run(self, args: list[str], *, env: dict[str, str]) -> GhResult:
        """Run ``gh <args>`` with ``env`` in the environment and capture the result."""
        ...


class GhCommandError(RuntimeError):
    """A ``gh`` invocation exited non-zero. Carries the args and stderr for debugging."""

    def __init__(self, command: list[str], result: GhResult) -> None:
        self.command = command
        self.result = result
        super().__init__(
            f"gh {' '.join(command)} exited {result.exit_code}: {result.stderr.strip()}"
        )


def _gh_auth_env(token: str) -> dict[str, str]:
    """Build the env that authenticates ``gh``: a ``GH_TOKEN`` bearer for the API.

    ``gh`` reads ``GH_TOKEN`` and sends it as ``Authorization: Bearer <token>`` on
    every REST/GraphQL call, so the adapter injects the token here and lets ``gh``
    own the wire format rather than assembling a header itself.
    """
    return {"GH_TOKEN": token}


def _issue_view_args(repo_full_name: str, issue_number: int) -> list[str]:
    """Assemble the ``gh issue view`` argv reading the dependent's body as JSON."""
    return [
        "issue",
        "view",
        str(issue_number),
        "--repo",
        repo_full_name,
        "--json",
        "body",
    ]


def _issue_edit_args(repo_full_name: str, issue_number: int, body: str) -> list[str]:
    """Assemble the ``gh issue edit`` argv writing ``body`` back to the dependent."""
    return [
        "issue",
        "edit",
        str(issue_number),
        "--repo",
        repo_full_name,
        "--body",
        body,
    ]


def _parse_issue_body(stdout: str) -> str:
    """Parse ``gh issue view --json body`` output into the issue body string.

    ``gh`` emits ``{"body": "<str>"}``. Raises :class:`ValueError` when the payload
    is not JSON or is missing the ``body`` field, so a malformed response fails
    loudly rather than clobbering the dependent's body with junk.
    """
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"gh issue view returned non-JSON output: {stdout!r}") from exc
    if not isinstance(payload, dict) or "body" not in payload:
        raise ValueError(f"gh issue view output missing body: {stdout!r}")
    return str(payload["body"])


def add_blocked_by(body: str, blocker: int) -> str:
    """Return ``body`` with ``#<blocker>`` added to its ``## Blocked by`` block.

    Matches the slicer's block shape: a ``## Blocked by`` heading followed by one
    ``#N`` reference per line. The edit is idempotent — a blocker already listed
    yields the body unchanged — and a body without the block grows one appended at
    the end, so a dependent the slicer never gave a block still gets wired.
    """
    existing, prefix = _split_blocked_by(body)
    if blocker in existing:
        return body
    refs = "\n".join(f"#{n}" for n in [*existing, blocker])
    block = f"{_BLOCKED_BY_HEADING}\n{refs}"
    return f"{prefix}\n\n{block}" if prefix else block


def _split_blocked_by(body: str) -> tuple[list[int], str]:
    """Split ``body`` into its existing blocker numbers and the text before the block.

    Returns ``(blockers, prefix)`` where ``prefix`` is everything ahead of the
    ``## Blocked by`` heading (the whole body, right-stripped, when there is no
    block). Only well-formed ``#<int>`` lines inside the block are read as blockers;
    a non-reference line ends the block so trailing prose is preserved in ``prefix``.
    """
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == _BLOCKED_BY_HEADING:
            blockers: list[int] = []
            for ref in lines[index + 1 :]:
                stripped = ref.strip()
                if stripped.startswith("#") and stripped[1:].isdigit():
                    blockers.append(int(stripped[1:]))
                elif stripped:
                    break
            prefix = "\n".join(lines[:index]).rstrip()
            return blockers, prefix
    return [], body.rstrip()


@dataclass(frozen=True)
class GhCliBlockedByEditor:
    """Real :data:`BlockedByEditor`: wire a review-fix into a dependent via ``gh``.

    Satisfies the ``edit`` protocol ``(EditBlockedByRequest) -> Awaitable[None]`` by
    calling itself: it reads the dependent issue's body, adds the ``#<fix>``
    reference to its ``## Blocked by`` block (idempotently), and writes the body
    back. The injected :class:`GhRunner` is the only side-effecting seam, so command
    assembly and body rendering are unit-testable without a live ``gh`` or network.

    Attributes:
        runner: The process-spawn seam that runs each ``gh`` command.
        token: The token ``gh`` authenticates with (sent as ``GH_TOKEN``).
    """

    runner: GhRunner
    token: str

    async def __call__(self, request: EditBlockedByRequest) -> None:
        """Add ``request.add_blocker`` to the dependent's ``## Blocked by`` block."""
        body = await self._read_body(request.repo_full_name, request.issue_number)
        updated = add_blocked_by(body, request.add_blocker)
        if updated == body:
            logger.info(
                "#%d already blocked by #%d in %s; no edit",
                request.issue_number,
                request.add_blocker,
                request.repo_full_name,
            )
            return
        await self._gh(
            _issue_edit_args(request.repo_full_name, request.issue_number, updated)
        )

    async def _read_body(self, repo_full_name: str, issue_number: int) -> str:
        """Read the dependent issue's current body via ``gh issue view``."""
        result = await self._gh(_issue_view_args(repo_full_name, issue_number))
        return _parse_issue_body(result.stdout)

    async def _gh(self, args: list[str]) -> GhResult:
        """Run one ``gh`` command authenticated with the token, raising on failure."""
        result = await self.runner.run(args, env=_gh_auth_env(self.token))
        if not result.ok:
            raise GhCommandError(args, result)
        return result


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
