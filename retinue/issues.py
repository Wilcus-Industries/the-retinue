"""Filing GitHub issues: the draft shape and the ``gh issue create`` seam.

The retinue files issues from several places — the in-session reviewer's backlog
findings, the ad-hoc review-fix follow-ups, the escalation flows. They all share one
draft shape (:class:`IssueDraft`), one result (:class:`CreatedIssue`), and one injected
seam (:data:`IssueCreator`), with :class:`GhCliIssueCreator` the production ``gh issue
create`` adapter. Kept in its own low-level module (importing only :mod:`retinue.gh`) so
every filer speaks one vocabulary without a dependency cycle.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from retinue.gh import GhCommandError, GhRunner, auth_env


@dataclass
class IssueDraft:
    """One issue to file.

    Attributes:
        title: Issue title.
        body: Issue body.
        labels: Labels applied to the issue.
        blocked_by: Native ``blocked_by`` issue numbers to link (rendered separately in
            the body when present); the ad-hoc filers leave this empty.
        hitl: Escalation flag retained for filers that mark a human-only issue.
    """

    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    blocked_by: list[int] = field(default_factory=list)
    hitl: bool = False


@dataclass(frozen=True)
class CreatedIssue:
    """The result of filing one issue."""

    issue_number: int


# The injected issue-filing seam: files one issue and returns its number. Async and faked
# in tests; production wires :class:`GhCliIssueCreator`.
IssueCreator = Callable[[IssueDraft], Awaitable[CreatedIssue]]


def _issue_create_args(
    draft: IssueDraft, repo_full_name: str, blocked_by_numbers: list[int]
) -> list[str]:
    """Assemble the ``gh issue create`` argv for ``draft`` (no leading ``"gh"``).

    Each of the draft's labels rides its own ``--label`` flag, and every resolved
    ``Blocked by`` number rides its own ``--blocked-by`` flag so gh records the native
    dependency link in addition to any ``## Blocked by`` block rendered into the body. The
    body is passed verbatim.
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
    """Pull the resolved ``#<n>`` refs out of a body's ``## Blocked by`` block.

    Reads back the ``## Blocked by`` section of ``#<n>`` lines so the gh create call can
    also carry native ``--blocked-by`` links. A body with no such section yields an empty
    list (the common case for a hand-filed or review-fix issue).
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
    """Production :data:`IssueCreator`: files one issue via ``gh issue create``.

    An instance is callable as ``await creator(draft)`` — it satisfies the
    :data:`IssueCreator` protocol via :meth:`__call__`, so it drops straight in where the
    fake issue creator sits in tests and at the wiring site. It assembles the ``gh issue
    create`` argv (labels + native ``--blocked-by`` links read back from the body) and
    dispatches it through the injected :class:`~retinue.gh.GhRunner`, authenticated with a
    ``GH_TOKEN`` bearer. The runner is the only side-effecting seam, which keeps command
    assembly and number parsing unit-testable with no live ``gh``/network.

    Args:
        runner: The process-spawn seam that runs each ``gh`` command.
        token: The installation/access token ``gh`` authenticates with.
        repo_full_name: The repo the issues are filed against, e.g. "owner/repo".
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
        result = await self._runner.run(args, env=auth_env(self._token))
        if not result.ok:
            raise GhCommandError(args, result)
        return CreatedIssue(issue_number=_parse_issue_number(result.stdout))
