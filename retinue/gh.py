"""The one gh-CLI seam: result shape, runner protocols, errors, auth env, subprocess runners.

Every module that shells out to ``gh`` shares this seam instead of keeping its own copy.
Adapters (slicer, PR-opener, loopback, reviewer, cron, done-check, notify, handoff,
reconcile) stay pure — they assemble argv and parse payloads — and dispatch the actual
process spawn through one of the runners here, injected so tests exercise the pure parts
with recording fakes and never a live ``gh``/network.

Three runner shapes coexist, each matching the contract its callers were built on:

- :class:`GhRunner` / :class:`SubprocessGhRunner` — ``run(args, *, env) -> GhResult``:
  captures the exit code instead of raising, for adapters that branch on failure
  (e.g. a 404 read as a False existence answer).
- :data:`GhBytesRunner` / :func:`run_gh_subprocess` — ``(argv, env) -> bytes`` where the
  argv *includes* the leading ``"gh"``; raises :class:`GhCliError` on a non-zero exit.
- :data:`GhTextRunner` / :func:`run_gh` — ``(args, env) -> str`` with the child
  environment passed *exactly* (nothing from the worker's environment leaks in unless the
  caller forwards it); raises :class:`GhCommandError` on a non-zero exit.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GhResult:
    """Captured result of a single ``gh`` invocation.

    Attributes:
        exit_code: ``gh``'s process exit status; ``0`` means success.
        stdout: Captured standard output (the payload the calling adapter parses).
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
    """Runs a single ``gh`` command. The process-spawn seam under the gh-cli adapters.

    A production implementation (:class:`SubprocessGhRunner`) spawns ``gh`` as a
    subprocess with ``env`` merged into its environment (so ``GH_TOKEN`` authenticates
    the call) and returns the captured :class:`GhResult`; tests inject a fake that
    records each ``(args, env)`` and returns a canned result. ``args`` never includes
    the leading ``"gh"`` — the runner owns the executable name.
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


class GhCliError(RuntimeError):
    """A raw-argv ``gh`` invocation (:func:`run_gh_subprocess`) failed (non-zero exit).

    Carries the argv and the captured stderr so the failure is debuggable rather than a
    bare ``CalledProcessError``. Kept distinct from :class:`GhCommandError` — the two
    error generations carry different attributes (``argv``/``returncode``/``stderr`` vs
    ``command``/``result``) and different message shapes, and their callers assert on
    both; merging them would change behavior for no dedup win.
    """

    def __init__(self, argv: Sequence[str], *, returncode: int, stderr: str) -> None:
        self.argv = list(argv)
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"gh exited {returncode} for {' '.join(argv)}: {stderr.strip()}")


def auth_env(token: str | None) -> dict[str, str]:
    """Build the env that authenticates ``gh``: a ``GH_TOKEN`` bearer for the API.

    ``gh`` reads ``GH_TOKEN`` and sends it as ``Authorization: Bearer <token>`` on every
    REST/GraphQL call, so an adapter never assembles a header itself — it injects the
    token here and lets ``gh`` own the wire format. The token rides the env, never the
    argv, so it cannot land in a process listing or a log of the command. With no token
    the mapping is empty, deferring to ``gh``'s ambient auth (e.g. a logged-in CLI).
    """
    return {"GH_TOKEN": token} if token else {}


async def _capture(
    argv: Sequence[str], env: Mapping[str, str]
) -> tuple[int, bytes, bytes]:
    """Spawn ``argv`` (argv list, no shell) with exactly ``env``; capture the outcome.

    The single subprocess spawn behind every production runner in this module. Returns
    ``(exit_code, stdout, stderr)`` without judging the exit code — each public wrapper
    applies its own error contract.
    """
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=dict(env),
    )
    stdout, stderr = await process.communicate()
    return process.returncode or 0, stdout, stderr


class SubprocessGhRunner:
    """Production :class:`GhRunner`: spawns ``gh`` and captures the result, never raising.

    Spawns ``gh <args>`` (argv list, no shell, so nothing is interpolated into a command
    line) with ``env`` merged over the ambient environment and returns the captured
    :class:`GhResult` — a non-zero exit is data for the caller to branch on, not an
    exception.
    """

    async def run(self, args: list[str], *, env: dict[str, str]) -> GhResult:
        """Run ``gh <args>`` with ``env`` in the environment and capture the result."""
        exit_code, stdout, stderr = await _capture(["gh", *args], {**os.environ, **env})
        return GhResult(
            exit_code=exit_code,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )


# An async runner for a raw ``gh`` argv (leading ``"gh"`` included): returns the
# command's stdout bytes, raising :class:`GhCliError` on a non-zero exit. The injectable
# seam type under the cron/done-check/ad-hoc adapters; production uses
# :func:`run_gh_subprocess`.
GhBytesRunner = Callable[[Sequence[str], Mapping[str, str]], Awaitable[bytes]]


async def run_gh_subprocess(argv: Sequence[str], env: Mapping[str, str]) -> bytes:
    """Spawn ``argv`` with ``env`` layered over the ambient env; return stdout bytes.

    The production :data:`GhBytesRunner`. ``argv`` carries the full command including
    the leading ``"gh"``. Uses :func:`asyncio.create_subprocess_exec` (no shell, so no
    argument is ever interpolated into a command line) and raises :class:`GhCliError`
    on a non-zero exit so the query fails loudly.
    """
    exit_code, stdout, stderr = await _capture(argv, {**os.environ, **env})
    if exit_code != 0:
        raise GhCliError(
            argv, returncode=exit_code, stderr=stderr.decode(errors="replace")
        )
    return stdout


# An async runner for a ``gh`` command (argv without the leading ``"gh"``): returns the
# command's stdout as text, raising on a non-zero exit. The injectable seam type under
# the handoff adapter; production uses :func:`run_gh`.
GhTextRunner = Callable[[Sequence[str], Mapping[str, str]], Awaitable[str]]


async def run_gh(argv: Sequence[str], env: Mapping[str, str]) -> str:
    """Run a real ``gh`` subprocess, returning stdout text; raise on a non-zero exit.

    The production :data:`GhTextRunner`. ``env`` is the child's environment *exactly* —
    nothing from the worker's environment (Anthropic credentials and the like) is
    forwarded unless the caller put it there — and a non-zero exit raises
    :class:`GhCommandError`.
    """
    exit_code, stdout, stderr = await _capture(["gh", *argv], env)
    if exit_code != 0:
        raise GhCommandError(
            list(argv),
            GhResult(
                exit_code=exit_code,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
            ),
        )
    return stdout.decode()


def parse_json_array(stdout: bytes) -> list[object]:
    """Parse a ``gh ... --json`` payload into its JSON array of entries.

    ``gh`` list commands emit a JSON array of objects; a payload that is not valid JSON
    or not an array raises :class:`ValueError` rather than silently yielding nothing.
    Shared by every gh lister (cron backlog, ad-hoc drain) so they parse the array
    envelope identically; each lister maps the entries into its own issue dataclass.
    """
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"gh issue list returned non-JSON output: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"gh issue list expected a JSON array, got {type(payload)}")
    return payload
