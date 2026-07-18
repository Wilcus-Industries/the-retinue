"""Static invariant guard for the ad-hoc review-fix chain-depth bound (issue #82).

The chain-depth apparatus in :mod:`retinue.adhoc_build` bounds the review-fix chain
``#29 -> #501 -> #503 -> ...`` entirely through a ``Chain-depth:`` marker carried in the
issue body: each hop is a fresh GitHub number with no shared state, so the *only* thing
that keeps the bound live is that every production admission reads the marker back via
:meth:`AdhocIssue.from_fetched_issue`. The docstrings warn repeatedly that hand-building
``AdhocIssue(repo_full_name=..., issue_number=...)`` in a production path defaults
``chain_depth`` to 0 and *silently* makes the bound inert — but until now nothing stopped
a future edit from doing exactly that.

This is the missing safety net #82 flags: an AST walk over ``retinue/**.py`` that fails if
any production code constructs :class:`AdhocIssue` directly rather than through the
``from_fetched_issue`` classmethod. It mirrors how the repo pins other wiring invariants in
``tests/test_wiring.py``. The classmethod's own internal ``cls(...)`` call is not a
``Name(id="AdhocIssue")`` call, so it is not — and must not be — flagged.
"""

from __future__ import annotations

import ast
from pathlib import Path

import retinue

_RETINUE_DIR = Path(retinue.__file__).parent


def _production_sources() -> list[Path]:
    """Every ``.py`` file in the ``retinue`` production package."""
    return sorted(_RETINUE_DIR.rglob("*.py"))


def _bare_adhoc_issue_calls(source: str) -> list[int]:
    """Line numbers of any ``AdhocIssue(...)`` direct-construction call in ``source``.

    A direct construction is an :class:`ast.Call` whose callee is the bare name
    ``AdhocIssue``. The canonical ``AdhocIssue.from_fetched_issue(...)`` constructor is an
    attribute call (``func`` is an :class:`ast.Attribute`), and the classmethod's internal
    ``cls(...)`` names ``cls`` — neither is a bare ``AdhocIssue`` name, so both are ignored.
    """
    tree = ast.parse(source)
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AdhocIssue"
    ]


def test_production_never_hand_constructs_adhoc_issue() -> None:
    """AC (issue #82): production admits ad-hoc issues only via ``from_fetched_issue``.

    A bare ``AdhocIssue(...)`` anywhere under ``retinue/`` would default ``chain_depth`` to
    0 and quietly re-open the unbounded review-fix chain the marker apparatus exists to
    terminate. Tests are free to hand-build issues to script a specific depth; production
    must not — so this guard scopes to the ``retinue`` package only.
    """
    offenders = {
        path.relative_to(_RETINUE_DIR.parent).as_posix(): lines
        for path in _production_sources()
        if (lines := _bare_adhoc_issue_calls(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}, (
        "Production code must construct AdhocIssue via AdhocIssue.from_fetched_issue "
        "(which reads the Chain-depth: marker), not the bare constructor — a bare "
        f"AdhocIssue(...) defaults chain_depth=0 and makes the review-fix bound inert: "
        f"{offenders}"
    )


def test_guard_detects_a_bare_construction() -> None:
    """The guard's own detector fires on a bare call and ignores the canonical seam.

    Locks the guard against silently degrading into a no-op (e.g. if the AST shape it keys
    on ever drifts): a real ``AdhocIssue(...)`` must be flagged, while the
    ``from_fetched_issue`` attribute call and the classmethod's ``cls(...)`` must not.
    """
    bare = "AdhocIssue(repo_full_name='owner/repo', issue_number=29)"
    seam = "AdhocIssue.from_fetched_issue('owner/repo', 29, body)"
    cls_call = "cls(repo_full_name='owner/repo', issue_number=29, chain_depth=0)"

    assert _bare_adhoc_issue_calls(bare) == [1]
    assert _bare_adhoc_issue_calls(seam) == []
    assert _bare_adhoc_issue_calls(cls_call) == []
