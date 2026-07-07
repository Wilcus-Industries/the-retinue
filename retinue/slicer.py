"""Headless PRD slicer: turn a PRD body into tracer-bullet vertical slices.

:func:`slice_prd` is the entry point. It checks the PRD is substantive, runs the
headless slice generator (the Agent SDK, injected as the ``generate`` seam) over
the body, then creates one GitHub issue per slice via the ``create_issue`` seam
(``gh``, injected). Every slice issue is labeled ``ready-for-agent`` + ``prd-slice``
and carries ``Part of #<prd>``; a genuinely human-only slice also gets ``hitl``. Intra-PRD
``blocked_by`` references are resolved to the real created issue numbers, so the
``## Blocked by`` graph is resolvable in dependency order.

A thin or malformed PRD — too little to slice, or a generator that yields no
slices — is **not** invented around: it escalates through the shared
:class:`retinue.notify.Notifier` (push + comment + label) and creates no issues.

Both side-effecting seams (the Agent-SDK generator and the gh issue creator) are
injected so the slicer is unit-testable without network, Agent SDK, or gh.
"""

from __future__ import annotations

import enum
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from retinue.notify import Notification, Notifier
from retinue.roles import (
    EFFORT_MAX,
    EFFORT_XHIGH,
    Role,
    oauth_system,
    resolve_effort,
    resolve_model,
)

logger = logging.getLogger(__name__)

READY_LABEL = "ready-for-agent"
HITL_LABEL = "hitl"
# Provenance marker: a slice the slicer filed off a PRD. Stamped alongside
# ``ready-for-agent`` so a PRD slice is distinguishable from ad-hoc ``ready-for-agent``
# work at the label layer (the lane router reads the ``Part of #<prd>`` link, not this
# label, but downstream tooling and humans can tell a slice apart from pickup). See
# :mod:`retinue.lane`.
PRD_SLICE_LABEL = "prd-slice"

# The OAuth beta header a subscription token requires. See the claude-api skill: OAuth
# tokens go on Authorization: Bearer with the oauth beta header. The slicing model and
# effort tier are owned by :mod:`retinue.roles` (the :data:`Role.SLICER` registry entry),
# resolved at construction/request time rather than pinned to a local constant.
_OAUTH_BETA = "oauth-2025-04-20"
_MAX_TOKENS = 16_000

# Re-exported from :mod:`retinue.roles` so the conflict resolver and internal reviewer
# keep importing the shared effort tiers from the slicer; the registry is the source of
# truth, these names are aliases that preserve the existing import surface.
_EFFORT_XHIGH = EFFORT_XHIGH
_EFFORT_MAX = EFFORT_MAX

# Strict JSON schema the headless slicer must emit: an ordered list of vertical
# slices, each with its 1-based intra-PRD blocked_by indices and a hitl flag.
_SLICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "slices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "blocked_by": {"type": "array", "items": {"type": "integer"}},
                    "hitl": {"type": "boolean"},
                },
                "required": ["title", "body", "blocked_by", "hitl"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["slices"],
    "additionalProperties": False,
}

# The headless slicer's brief. Kept frozen (no per-request interpolation) so the
# request prefix is cacheable across PRDs.
_SLICE_SYSTEM = (
    "You slice a Product Requirements Doc into tracer-bullet vertical slices. "
    "Each slice cuts every layer and is demoable on its own. Emit them in "
    "dependency order — a slice may only depend on earlier ones. Reference an "
    "earlier slice via its 1-based index in 'blocked_by'. Set 'hitl' true only "
    "for a genuinely human-only slice (a secret, an external account, or a "
    "design call) the agent loop must not attempt. Return only the JSON object "
    "matching the schema; no prose."
)

# The PRD section the testing seam is read from, and the labeled block the slicer
# wraps it in so the model honors the PRD's testing decisions instead of inventing
# its own. The seam is read from the PRD (a locked design decision), so it is injected
# explicitly rather than left to ride opaquely inside the body.
_TESTING_DECISIONS_HEADING = "## Testing Decisions"
_TESTING_SEAM_LABEL = "TESTING SEAM — read from the PRD, honor this, do not invent:"

# A PRD body shorter than this (after stripping) is too thin to slice responsibly.
_MIN_PRD_BODY_CHARS = 40


@dataclass
class IssueDraft:
    """One vertical slice to file as a GitHub issue.

    The generator produces drafts with ``blocked_by`` holding **1-based indices
    into the same plan**; :func:`slice_prd` rewrites those to real issue numbers
    and renders them into the body before creation, and appends the labels +
    ``Part of`` line. ``labels`` is filled by the slicer, not the generator.

    Attributes:
        title: Issue title.
        body: Issue body (the slice's what/why/acceptance), enriched in place.
        blocked_by: 1-based indices of sibling slices this one depends on.
        hitl: True only for a genuinely human-only slice (secret / external
            account / design call) that the agent loop must not attempt.
        labels: Labels applied to the issue; populated by the slicer.
    """

    title: str
    body: str
    blocked_by: list[int] = field(default_factory=list)
    hitl: bool = False
    labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SlicePlan:
    """The headless generator's output: an ordered list of slice drafts.

    Order is dependency order — a slice may only depend on earlier ones — so the
    created issue numbers are known by the time a later slice references them.
    """

    slices: list[IssueDraft]


@dataclass(frozen=True)
class CreatedIssue:
    """The result of filing one slice issue."""

    issue_number: int


class SliceOutcome(enum.Enum):
    """Why the slicer sliced the PRD or escalated instead."""

    SLICED = "sliced"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class SliceResult:
    """Result of slicing one PRD.

    Attributes:
        outcome: Whether the PRD was sliced or escalated.
        created_numbers: Issue numbers created, in plan order (empty on escalate).
    """

    outcome: SliceOutcome
    created_numbers: list[int] = field(default_factory=list)


# Injected seams. ``generate`` runs the headless Agent-SDK slicer over the PRD
# body; ``create_issue`` files one issue via gh. Both are async and faked in tests.
SliceGenerator = Callable[[str], Awaitable[SlicePlan]]
IssueCreator = Callable[[IssueDraft], Awaitable[CreatedIssue]]


@dataclass(frozen=True)
class ClaudeSliceGenerator:
    """Real :data:`SliceGenerator`: slice a PRD with the Agent SDK (Anthropic API).

    An instance is callable as ``await generator(prd_body)`` — it satisfies the
    :data:`SliceGenerator` protocol via :meth:`generate`, so it drops straight in
    where the fake generator sits in tests and at the wiring site.

    The dollar/token metering split mirrors :class:`retinue.config.Settings`:
    ``auth_mode="api_key"`` authenticates with an ``ANTHROPIC_API_KEY`` (the
    ``x-api-key`` header the SDK builds from ``api_key``); ``auth_mode=
    "subscription"`` authenticates with a short-lived OAuth token on
    ``Authorization: Bearer`` plus the ``oauth-2025-04-20`` beta header.

    The Anthropic SDK is imported lazily inside :meth:`generate` so the module —
    and the unit tests over the pure header/request/parse helpers — import with
    no SDK or network present.

    Attributes:
        token: The API key (``api_key`` mode) or OAuth bearer token
            (``subscription`` mode).
        auth_mode: ``"api_key"`` or ``"subscription"``.
        model: The model the headless slicer runs on; defaults to the
            :data:`~retinue.roles.Role.SLICER` registry entry, which a repo's
            ``models`` override can replace at the wiring site.
    """

    token: str
    auth_mode: str = "api_key"
    model: str = field(default_factory=lambda: resolve_model(Role.SLICER))

    async def generate(self, prd_body: str) -> SlicePlan:
        """Run the headless slicer over ``prd_body`` and return its :class:`SlicePlan`.

        Streams the request (large ``max_tokens``) and parses the strict-schema
        JSON payload into ordered :class:`IssueDraft` slices. Raises on a response
        the slicer can't parse; an empty plan is a valid result the caller escalates.
        """
        # Lazy import keeps the module (and its unit tests) import-clean: the heavy
        # ``anthropic`` client is only pulled in when a real slice generation runs.
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(**self._client_kwargs())
        async with client.messages.stream(**self._build_request_kwargs(prd_body)) as stream:
            message = await stream.get_final_message()
        return self._parse_plan(_first_text(message.content))

    def _client_kwargs(self) -> dict[str, Any]:
        """Client constructor kwargs: the key/token and any OAuth default headers.

        In ``api_key`` mode the token rides ``api_key=`` (the SDK renders the
        ``x-api-key`` header). In ``subscription`` mode it rides ``auth_token=``
        (rendered as ``Authorization: Bearer``) and the OAuth beta header is added
        as a default header so it is sent on every request.
        """
        if self.auth_mode == "subscription":
            return {
                "auth_token": self.token,
                "default_headers": self._build_auth_headers(),
            }
        return {"api_key": self.token}

    def _build_auth_headers(self) -> dict[str, str]:
        """Extra headers required by the auth mode.

        ``subscription`` (OAuth) must carry ``anthropic-beta: oauth-2025-04-20``;
        ``api_key`` needs no extra header (the ``x-api-key`` header is built from
        the key by the SDK).
        """
        if self.auth_mode == "subscription":
            return {"anthropic-beta": _OAUTH_BETA}
        return {}

    def _build_request_kwargs(self, prd_body: str) -> dict[str, Any]:
        """Assemble the streaming-request kwargs for one PRD slice run.

        The slicer is the PRD-decision Opus role, so the request carries the "xhigh"
        reasoning-effort tier via ``output_config.effort`` (Opus 4.8 removed the thinking
        ``budget_tokens`` mechanism). The effort rides the same ``output_config`` dict as
        the JSON-schema ``format`` — one output_config carries both.

        The PRD's ``## Testing Decisions`` section is the authoritative testing seam, so
        it is extracted and prepended as an explicit, labeled block instructing the model
        to honor it; the full PRD body follows. A PRD without that section keeps the body
        verbatim (and the caller escalates a thin/malformed PRD before reaching here).
        """
        return {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "system": oauth_system(
                _SLICE_SYSTEM, is_oauth=self.auth_mode == "subscription"
            ),
            "output_config": {
                "effort": resolve_effort(Role.SLICER),
                "format": {"type": "json_schema", "schema": _SLICE_SCHEMA},
            },
            "messages": [{"role": "user", "content": _build_user_content(prd_body)}],
        }

    @staticmethod
    def _parse_plan(payload: str) -> SlicePlan:
        """Parse the strict-schema JSON ``payload`` into an ordered :class:`SlicePlan`.

        ``blocked_by`` and ``hitl`` are optional in a malformed-but-parseable
        payload and default to the empty list / ``False``; ``title`` and ``body``
        are required. Raises :class:`ValueError` if the payload isn't the expected
        object shape so the caller fails loudly rather than filing junk issues.
        """
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"slicer returned non-JSON payload: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("slices"), list):
            raise ValueError("slicer payload missing a 'slices' array")

        slices: list[IssueDraft] = []
        for raw in data["slices"]:
            if not isinstance(raw, dict) or "title" not in raw or "body" not in raw:
                raise ValueError(f"slice is missing title/body: {raw!r}")
            slices.append(
                IssueDraft(
                    title=str(raw["title"]),
                    body=str(raw["body"]),
                    blocked_by=[int(i) for i in raw.get("blocked_by", [])],
                    hitl=bool(raw.get("hitl", False)),
                )
            )
        return SlicePlan(slices=slices)


def _build_user_content(prd_body: str) -> str:
    """Compose the slice request's user message: the testing seam, then the PRD body.

    When the PRD carries a ``## Testing Decisions`` section, its text is prepended as an
    explicit, labeled block (see :data:`_TESTING_SEAM_LABEL`) so the model honors the
    PRD's testing decisions instead of inventing its own; the full body still follows so
    the slicer sees the whole PRD. A PRD without that section yields the body verbatim.
    """
    decisions = _extract_testing_decisions(prd_body)
    if decisions is None:
        return prd_body
    return f"{_TESTING_SEAM_LABEL}\n{decisions}\n\n{prd_body}"


def _extract_testing_decisions(prd_body: str) -> str | None:
    """Return the text of the PRD's ``## Testing Decisions`` section, or ``None``.

    The section runs from its heading to the next ``## `` heading (or end of body); the
    returned text is stripped. ``None`` means the PRD has no such section, so the caller
    leaves the body untouched rather than fabricating a testing seam.
    """
    lines = prd_body.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == _TESTING_DECISIONS_HEADING:
            section: list[str] = []
            for body_line in lines[index + 1 :]:
                if body_line.startswith("## "):
                    break
                section.append(body_line)
            text = "\n".join(section).strip()
            return text or None
    return None


def _first_text(content: list[Any]) -> str:
    """Return the first text block's text from a message's content blocks.

    ``output_config.format`` guarantees the JSON arrives as a single leading text
    block; an empty content array means the model produced nothing to parse.
    """
    for block in content:
        if getattr(block, "type", None) == "text":
            return str(block.text)
    raise ValueError("slicer response carried no text block")


async def slice_prd(
    *,
    repo_full_name: str,
    prd_number: int,
    prd_body: str,
    generate: SliceGenerator,
    create_issue: IssueCreator,
    notifier: Notifier,
) -> SliceResult:
    """Slice a PRD into labeled, dependency-ordered issues, or escalate.

    A substantive PRD is run through ``generate`` and each resulting slice is
    filed with ``ready-for-agent`` + ``Part of #<prd>`` (and ``hitl`` when the
    slice is human-only), with its ``## Blocked by`` graph resolved to real issue
    numbers. A thin PRD, or a generator that yields no slices, escalates through
    ``notifier`` and creates nothing.

    Args:
        repo_full_name: e.g. "owner/repo".
        prd_number: The PRD issue number; slices link back via ``Part of #``.
        prd_body: The PRD issue body to slice.
        generate: Async headless slicer (Agent SDK seam) producing a SlicePlan.
        create_issue: Async issue creator (gh seam) filing one slice issue.
        notifier: Shared notify primitive used to escalate a thin/malformed PRD.

    Returns:
        A :class:`SliceResult`: ``SLICED`` with the created issue numbers, or
        ``ESCALATED`` with an empty list.
    """
    if not _is_substantive(prd_body):
        logger.warning("PRD #%d in %s is too thin to slice; escalating", prd_number, repo_full_name)
        return await _escalate(
            repo_full_name,
            prd_number,
            "PRD is too thin or malformed to slice. A human needs to flesh it out.",
            notifier,
        )

    plan = await generate(prd_body)
    if not plan.slices:
        logger.warning("Slicer produced no slices for PRD #%d; escalating", prd_number)
        return await _escalate(
            repo_full_name,
            prd_number,
            "The slicer produced no vertical slices for this PRD. A human should review it.",
            notifier,
        )

    return await _create_slices(repo_full_name, prd_number, plan, create_issue)


def _is_substantive(prd_body: str) -> bool:
    """A PRD with real content to slice: non-trivial length and not a bare stub."""
    stripped = prd_body.strip()
    if len(stripped) < _MIN_PRD_BODY_CHARS:
        return False
    return not stripped.lower().startswith("todo")


async def _escalate(
    repo_full_name: str,
    prd_number: int,
    reason: str,
    notifier: Notifier,
) -> SliceResult:
    """Route a thin/malformed PRD through the notifier and create no slices."""
    await notifier.notify(
        Notification(
            repo_full_name=repo_full_name,
            issue_number=prd_number,
            title=f"Retinue can't slice PRD #{prd_number}",
            body=reason,
            label=HITL_LABEL,
        )
    )
    return SliceResult(outcome=SliceOutcome.ESCALATED)


async def _create_slices(
    repo_full_name: str,
    prd_number: int,
    plan: SlicePlan,
    create_issue: IssueCreator,
) -> SliceResult:
    """File each slice in order, resolving blocked-by to real issue numbers.

    Slices are created in plan order so a later slice's dependency on an earlier
    one resolves to an already-known issue number. Index ``i`` (1-based) in any
    ``blocked_by`` list maps to ``created_numbers[i - 1]``.
    """
    created_numbers: list[int] = []
    for draft in plan.slices:
        _finalize_draft(draft, prd_number, created_numbers)
        result = await create_issue(draft)
        created_numbers.append(result.issue_number)
    return SliceResult(outcome=SliceOutcome.SLICED, created_numbers=created_numbers)


def _finalize_draft(
    draft: IssueDraft,
    prd_number: int,
    created_numbers: list[int],
) -> None:
    """Apply labels and render the Part-of + Blocked-by footer onto ``draft``.

    Mutates ``draft`` in place: sets its labels (``ready-for-agent`` + ``prd-slice``
    always, ``hitl`` for a human-only slice) and rewrites the body to carry the
    ``Part of #<prd>`` line plus a resolved ``## Blocked by`` block. The ``prd-slice``
    provenance marker rides alongside the ``ready-for-agent`` build trigger so a slice is
    distinguishable from ad-hoc ``ready-for-agent`` work (see :mod:`retinue.lane`).
    """
    draft.labels = [READY_LABEL, PRD_SLICE_LABEL]
    if draft.hitl:
        draft.labels.append(HITL_LABEL)

    footer = [f"Part of #{prd_number}"]
    blocked_refs = _resolve_blocked_by(draft.blocked_by, created_numbers)
    if blocked_refs:
        footer.append("## Blocked by\n" + "\n".join(f"#{n}" for n in blocked_refs))
    draft.body = f"{draft.body.rstrip()}\n\n" + "\n\n".join(footer)


def _resolve_blocked_by(blocked_by: list[int], created_numbers: list[int]) -> list[int]:
    """Map 1-based plan indices to real created issue numbers.

    An index referencing a slice that has not been created yet (a forward or
    out-of-range reference) is dropped with a warning rather than rendered as a
    dangling ``#`` reference — the generator is expected to emit dependency order.
    """
    resolved: list[int] = []
    for index in blocked_by:
        if 1 <= index <= len(created_numbers):
            resolved.append(created_numbers[index - 1])
        else:
            logger.warning(
                "Dropping unresolvable blocked-by index %d (only %d slices created so far)",
                index,
                len(created_numbers),
            )
    return resolved


# --- production gh-cli IssueCreator ------------------------------------------------
#
# :func:`slice_prd` (and the review flows that reuse this seam) depend only on the
# :data:`IssueCreator` protocol. Production wires the concrete
# :class:`GhCliIssueCreator` below; tests inject a fake that records create calls and
# returns canned numbers. ``GhCliIssueCreator`` itself does not shell out — it assembles
# the ``gh issue create`` argv and parses gh's output, delegating the actual process
# spawn to an injected :class:`GhRunner`. That keeps every pure/parseable part (auth-env
# build, command assembly, URL parsing) testable with a recording fake runner, never a
# live ``gh``/network. The local :class:`GhRunner`/:class:`GhResult` mirror the gh-seam
# shape used in :mod:`retinue.pr_opener` / :mod:`retinue.reconcile`; each module keeps its
# own copy so the layers stay edit-isolated.


@dataclass(frozen=True)
class GhResult:
    """Captured result of a single ``gh`` invocation.

    Attributes:
        exit_code: ``gh``'s process exit status; ``0`` means success.
        stdout: Captured standard output (the issue URL ``GhCliIssueCreator`` parses).
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
    """Runs a single ``gh`` command. The process-spawn seam under :class:`GhCliIssueCreator`.

    A production implementation spawns ``gh`` as a subprocess with ``env`` merged into
    its environment (so ``GH_TOKEN`` authenticates the call) and returns the captured
    :class:`GhResult`; tests inject a fake that records each ``(args, env)`` and returns a
    canned result. ``args`` never includes the leading ``"gh"`` — the runner owns the
    executable name.
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


def _auth_env(token: str) -> dict[str, str]:
    """Build the env that authenticates ``gh``: a ``GH_TOKEN`` bearer for the API.

    ``gh`` reads ``GH_TOKEN`` and sends it as ``Authorization: Bearer <token>`` on every
    REST/GraphQL call, so the adapter never assembles a header itself — it injects the
    token here and lets ``gh`` own the wire format.
    """
    return {"GH_TOKEN": token}


def _issue_create_args(
    draft: IssueDraft, repo_full_name: str, blocked_by_numbers: list[int]
) -> list[str]:
    """Assemble the ``gh issue create`` argv for ``draft`` (no leading ``"gh"``).

    Each of the draft's labels rides its own ``--label`` flag, and every resolved
    ``Blocked by`` number rides its own ``--blocked-by`` flag so gh records the native
    dependency link in addition to the ``## Blocked by`` block already rendered into the
    body. The body is passed verbatim — the slicer finalized it before calling this seam.
    """
    args = [
        "issue",
        "create",
        "--repo",
        repo_full_name,
        "--title",
        draft.title,
        "--body",
        draft.body,
    ]
    for label in draft.labels:
        args += ["--label", label]
    for number in blocked_by_numbers:
        args += ["--blocked-by", str(number)]
    return args


def _parse_issue_number(stdout: str) -> int:
    """Parse the issue number from ``gh issue create``'s output.

    ``gh issue create`` prints the created issue's URL (e.g.
    ``https://github.com/owner/repo/issues/123``) to stdout. The number is the trailing
    path segment. Raises :class:`ValueError` when the output has no trailing integer, so a
    malformed response fails loudly rather than yielding a bogus issue number.
    """
    tail = stdout.strip().rstrip("/").rsplit("/", 1)[-1]
    try:
        return int(tail)
    except ValueError as exc:
        raise ValueError(f"gh issue create returned no issue number: {stdout!r}") from exc


def _blocked_by_numbers(body: str) -> list[int]:
    """Pull the resolved ``#<n>`` refs out of a finalized draft's ``## Blocked by`` block.

    :func:`_finalize_draft` renders dependencies as a trailing ``## Blocked by`` section of
    ``#<n>`` lines (real, already-created issue numbers). This reads them back so the gh
    create call can also carry native ``--blocked-by`` links. A draft with no such section
    yields an empty list.
    """
    _, _, block = body.partition("## Blocked by")
    if not block:
        return []
    numbers: list[int] = []
    for line in block.splitlines():
        ref = line.strip()
        if ref.startswith("#") and ref[1:].isdigit():
            numbers.append(int(ref[1:]))
    return numbers


class GhCliIssueCreator:
    """Production :data:`IssueCreator`: files one slice issue via ``gh issue create``.

    An instance is callable as ``await creator(draft)`` — it satisfies the
    :data:`IssueCreator` protocol via :meth:`__call__`, so it drops straight in where the
    fake issue creator sits in tests and at the wiring site (``slice_prd`` and the review
    flows that reuse this seam). It assembles the ``gh issue create`` argv (labels +
    native ``--blocked-by`` links read back from the finalized body) and dispatches it
    through the injected :class:`GhRunner`, authenticated with a ``GH_TOKEN`` bearer (see
    :func:`_auth_env`). The runner is the only side-effecting seam, which keeps command
    assembly and number parsing unit-testable with no live ``gh``/network.

    Args:
        runner: The process-spawn seam that runs each ``gh`` command.
        token: The installation/access token ``gh`` authenticates with.
        repo_full_name: The repo the slice issues are filed against, e.g. "owner/repo".
    """

    def __init__(self, runner: GhRunner, *, token: str, repo_full_name: str) -> None:
        self._runner = runner
        self._token = token
        self._repo_full_name = repo_full_name

    async def __call__(self, draft: IssueDraft) -> CreatedIssue:
        """File ``draft`` via ``gh issue create`` and return the parsed issue number."""
        args = _issue_create_args(
            draft, self._repo_full_name, _blocked_by_numbers(draft.body)
        )
        result = await self._runner.run(args, env=_auth_env(self._token))
        if not result.ok:
            raise GhCommandError(args, result)
        return CreatedIssue(issue_number=_parse_issue_number(result.stdout))
