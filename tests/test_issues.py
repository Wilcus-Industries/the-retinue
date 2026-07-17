"""Tests for the issue-filing seam relocated from the slicer.

The draft shape, the ``gh issue create`` argv assembly, the blocked-by read-back, the
issue-number parse, and the production :class:`GhCliIssueCreator` adapter live in
:mod:`retinue.issues`. The gh subprocess is faked, so no real network or ``gh`` is touched.
"""

from __future__ import annotations

import pytest

from retinue.gh import GhCommandError, GhResult
from retinue.issues import (
    CreatedIssue,
    GhCliIssueCreator,
    IssueDraft,
    _blocked_by_numbers,
    _issue_create_args,
    _parse_issue_number,
)


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
        await creator(IssueDraft(title="x", body="y"))
