"""Tests for the shared gh-CLI seam (:mod:`retinue.gh`).

Covers the seam's own contract — the result shape, the auth env build, both error
types' messages, JSON-array payload parsing, and the production subprocess runners'
argv/env/error handling. The spawn itself is faked (a recording
``create_subprocess_exec`` stand-in), so no test runs a real ``gh`` or touches the
network; the per-module adapter tests keep exercising their pure command assembly
against injected fake runners.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from retinue.gh import (
    GhCliError,
    GhCommandError,
    GhResult,
    SubprocessGhRunner,
    auth_env,
    parse_json_array,
    run_gh,
    run_gh_subprocess,
)

# --- result shape ------------------------------------------------------------------


def test_gh_result_ok_only_on_zero_exit() -> None:
    """``ok`` is True exactly when gh exited 0."""
    assert GhResult(exit_code=0).ok
    assert not GhResult(exit_code=1, stderr="boom").ok


# --- auth env ----------------------------------------------------------------------


def test_auth_env_carries_only_the_gh_token_bearer() -> None:
    """The auth env injects the token as ``GH_TOKEN`` and nothing else."""
    assert auth_env("ghs_abc123") == {"GH_TOKEN": "ghs_abc123"}


def test_auth_env_empty_without_token() -> None:
    """With no token the env is empty, deferring to gh's ambient auth."""
    assert auth_env(None) == {}


# --- error messages ----------------------------------------------------------------


def test_gh_command_error_carries_command_exit_and_stderr() -> None:
    """The GhResult-shaped error names the command, exit code, and stderr."""
    error = GhCommandError(
        ["pr", "edit"], GhResult(exit_code=1, stderr="no such PR\n")
    )
    assert str(error) == "gh pr edit exited 1: no such PR"
    assert error.command == ["pr", "edit"]
    assert error.result.exit_code == 1


def test_gh_cli_error_carries_argv_exit_and_stderr() -> None:
    """The raw-argv error names the exit code, full argv, and stderr."""
    error = GhCliError(["gh", "issue", "list"], returncode=1, stderr="Not Found\n")
    assert str(error) == "gh exited 1 for gh issue list: Not Found"
    assert error.argv == ["gh", "issue", "list"]
    assert error.returncode == 1
    assert error.stderr == "Not Found\n"


# --- payload parsing ---------------------------------------------------------------


def test_parse_json_array_returns_the_entries() -> None:
    """A well-formed gh list payload parses to its entries."""
    assert parse_json_array(b'[{"number": 7}]') == [{"number": 7}]


def test_parse_json_array_rejects_non_json() -> None:
    """A non-JSON payload raises rather than silently yielding nothing."""
    with pytest.raises(ValueError, match="non-JSON"):
        parse_json_array(b"not json")


def test_parse_json_array_rejects_a_non_array_payload() -> None:
    """A JSON payload that is not an array raises rather than being coerced."""
    with pytest.raises(ValueError, match="expected a JSON array"):
        parse_json_array(b'{"number": 7}')


# --- production subprocess runners (faked spawn; no real gh) ------------------------


class _FakeProcess:
    """A canned ``create_subprocess_exec`` result: fixed exit code and output."""

    def __init__(self, returncode: int, stdout: bytes, stderr: bytes) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


def _install_fake_spawn(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> list[tuple[list[str], dict[str, str]]]:
    """Replace the module's subprocess spawn; return the recorded ``(argv, env)`` calls."""
    calls: list[tuple[list[str], dict[str, str]]] = []
    canned = (stdout, stderr)

    async def fake_spawn(
        *argv: str, stdout: object = None, stderr: object = None, env: object = None
    ) -> _FakeProcess:
        assert isinstance(env, dict)
        calls.append((list(argv), dict(env)))
        return _FakeProcess(returncode, *canned)

    # retinue.gh resolves ``asyncio.create_subprocess_exec`` at call time, so patching
    # the asyncio module attribute (restored by monkeypatch) intercepts the spawn.
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    return calls


@pytest.mark.asyncio
async def test_subprocess_gh_runner_prefixes_gh_and_merges_the_ambient_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GhResult runner owns the executable name and layers env over the ambient."""
    monkeypatch.setenv("AMBIENT_MARKER", "present")
    calls = _install_fake_spawn(monkeypatch, stdout=b"https://x/pull/9\n")

    result = await SubprocessGhRunner().run(
        ["pr", "create"], env={"GH_TOKEN": "tok"}
    )

    argv, env = calls[0]
    assert argv == ["gh", "pr", "create"]
    assert env["GH_TOKEN"] == "tok"
    assert env["AMBIENT_MARKER"] == "present"
    assert result == GhResult(exit_code=0, stdout="https://x/pull/9\n", stderr="")


@pytest.mark.asyncio
async def test_subprocess_gh_runner_captures_a_failure_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero exit is data in the GhResult, not an exception."""
    _install_fake_spawn(monkeypatch, returncode=1, stderr=b"HTTP 404")

    result = await SubprocessGhRunner().run(["api", "x"], env={})

    assert not result.ok
    assert result.stderr == "HTTP 404"


@pytest.mark.asyncio
async def test_run_gh_subprocess_takes_the_raw_argv_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bytes runner spawns exactly the given argv (leading ``gh`` included)."""
    calls = _install_fake_spawn(monkeypatch, stdout=b"[]")

    stdout = await run_gh_subprocess(["gh", "issue", "list"], {"GH_TOKEN": "t"})

    argv, env = calls[0]
    assert argv == ["gh", "issue", "list"]
    assert env["GH_TOKEN"] == "t"
    assert stdout == b"[]"


@pytest.mark.asyncio
async def test_run_gh_subprocess_raises_gh_cli_error_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero exit surfaces as GhCliError carrying the stderr."""
    _install_fake_spawn(monkeypatch, returncode=1, stderr=b"Not Found")

    with pytest.raises(GhCliError, match="Not Found"):
        await run_gh_subprocess(["gh", "api", "x"], {})


@pytest.mark.asyncio
async def test_run_gh_passes_the_child_env_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The text runner forwards nothing from the worker env the caller did not pass."""
    monkeypatch.setenv("AMBIENT_MARKER", "present")
    calls = _install_fake_spawn(monkeypatch, stdout=b"[]\n")

    stdout = await run_gh(["issue", "list"], {"PATH": os.environ.get("PATH", "")})

    argv, env = calls[0]
    assert argv == ["gh", "issue", "list"]
    assert "AMBIENT_MARKER" not in env
    assert stdout == "[]\n"


@pytest.mark.asyncio
async def test_run_gh_raises_gh_command_error_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero exit surfaces as GhCommandError with the command in the message."""
    _install_fake_spawn(monkeypatch, returncode=2, stderr=b"boom\n")

    with pytest.raises(GhCommandError, match="gh issue close exited 2: boom"):
        await run_gh(["issue", "close"], {})
