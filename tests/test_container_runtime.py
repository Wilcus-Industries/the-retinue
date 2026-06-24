"""Tests for the real Docker-backed ContainerRuntime's pure, parseable parts.

The live impl talks to the Docker Engine API over its Unix socket; that path needs a
daemon and is not exercised here. What *is* tested is everything pure: the registry
auth header it builds, the JSON payloads it assembles for container/exec create, and
how it parses Docker's multiplexed output stream and exec exit code. These are the
bug-prone seams, and they run with no Docker, no network.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from retinue.container import (
    DockerContainer,
    DockerError,
    DockerRuntime,
    RunResult,
    _create_container_payload,
    _demux_stream,
    _exec_create_payload,
    _parse_exec_exit_code,
    _registry_auth_header,
)

# --- registry auth header --------------------------------------------------------


def test_registry_auth_header_is_base64url_json() -> None:
    """X-Registry-Auth is base64url(JSON) of the credentials Docker expects."""
    header = _registry_auth_header("alice", "s3cret")
    decoded = json.loads(base64.urlsafe_b64decode(header))
    assert decoded == {"username": "alice", "password": "s3cret"}
    # base64url, so never the standard-alphabet '+' or '/' and no padding newlines.
    assert "\n" not in header


# --- container create payload ----------------------------------------------------


def test_create_container_payload_maps_env_and_keeps_alive() -> None:
    """Env becomes Docker's KEY=value list and the container idles so exec can attach."""
    payload = _create_container_payload(
        image="img:latest", env={"A": "1", "B": "two"}
    )
    assert payload["Image"] == "img:latest"
    assert sorted(payload["Env"]) == ["A=1", "B=two"]
    # A keep-alive entrypoint so the container stays up for run_command to exec into.
    assert payload["Cmd"]
    assert payload["WorkingDir"]


def test_create_container_payload_empty_env_yields_empty_list() -> None:
    """No secrets means an empty Env list, never a missing key or None."""
    payload = _create_container_payload(image="img:latest", env={})
    assert payload["Env"] == []


# --- exec create payload ---------------------------------------------------------


def test_exec_create_payload_attaches_both_streams() -> None:
    """Exec captures stdout and stderr and carries the command verbatim."""
    payload = _exec_create_payload(["uv", "run", "pytest"])
    assert payload["Cmd"] == ["uv", "run", "pytest"]
    assert payload["AttachStdout"] is True
    assert payload["AttachStderr"] is True


# --- multiplexed stream demux ----------------------------------------------------


def _frame(stream: int, data: bytes) -> bytes:
    # Docker frame header: [stream, 0,0,0, big-endian uint32 length] then payload.
    return bytes([stream, 0, 0, 0]) + len(data).to_bytes(4, "big") + data


def test_demux_stream_splits_stdout_and_stderr() -> None:
    """The 8-byte-framed stream is split back into stdout (1) and stderr (2)."""
    raw = _frame(1, b"hello ") + _frame(2, b"oops") + _frame(1, b"world")
    stdout, stderr = _demux_stream(raw)
    assert stdout == "hello world"
    assert stderr == "oops"


def test_demux_stream_empty_is_empty() -> None:
    """No frames yields empty stdout and stderr, not an error."""
    assert _demux_stream(b"") == ("", "")


def test_demux_stream_ignores_trailing_partial_frame() -> None:
    """A truncated trailing frame is dropped rather than crashing the parser."""
    raw = _frame(1, b"ok") + b"\x01\x00\x00\x00\x00\x00"  # header short of 8 bytes
    stdout, stderr = _demux_stream(raw)
    assert stdout == "ok"
    assert stderr == ""


# --- exec inspect exit code ------------------------------------------------------


def test_parse_exec_exit_code_reads_field() -> None:
    """The exec inspect payload's ExitCode is surfaced as the int exit code."""
    assert _parse_exec_exit_code({"ExitCode": 0, "Running": False}) == 0
    assert _parse_exec_exit_code({"ExitCode": 1, "Running": False}) == 1


def test_parse_exec_exit_code_running_treated_as_failure() -> None:
    """A null ExitCode (still running / unknown) is a non-zero failure, not a crash."""
    assert _parse_exec_exit_code({"ExitCode": None, "Running": True}) != 0


# --- protocol conformance --------------------------------------------------------


def test_docker_types_satisfy_protocols() -> None:
    """The concrete types implement the same run/destroy protocol the fake satisfies."""
    from retinue.container import Container, ContainerRuntime

    runtime: ContainerRuntime = DockerRuntime()
    assert isinstance(runtime, DockerRuntime)
    # DockerContainer fulfils the Container protocol's surface (structural check).
    assert hasattr(DockerContainer, "run_command")
    assert hasattr(DockerContainer, "destroy")
    _: type[Container] = DockerContainer
    _r: RunResult = RunResult(exit_code=0)
    assert _r.ok


# --- pull fallback: a locally-built image not in any registry ---------------------


def _scripted_request(responses: dict[str, tuple[int, bytes]], calls: list[str]) -> Any:
    """An async stand-in for ``DockerRuntime._request`` keyed on the path prefix.

    ``responses`` maps a substring of the request path to a ``(status, body)`` pair;
    every call is appended to ``calls`` so a test can assert which Engine endpoints
    were hit (e.g. that the image-inspect fallback fired).
    """

    async def fake_request(method, path, body=None, headers=None):  # type: ignore[no-untyped-def]
        calls.append(f"{method} {path}")
        for needle, resp in responses.items():
            if needle in path:
                return resp
        raise AssertionError(f"unexpected request: {method} {path}")

    return fake_request


@pytest.mark.asyncio
async def test_pull_falls_back_to_local_image_when_pull_denied() -> None:
    """A registry-denied pull must NOT fail the build when the image is present locally.

    Local single-host deploys build the runner image on the host daemon and never push
    it to a registry, so ``/images/create`` is denied. The runtime must then inspect the
    daemon, find the image, and proceed (``docker run --pull=missing`` semantics) instead
    of raising. The orchestrator drives the host socket, so a local-only image is valid.
    """
    runtime = DockerRuntime()
    calls: list[str] = []
    runtime._request = _scripted_request(  # type: ignore[method-assign]
        {
            "/images/create": (500, b'{"message":"error from registry: denied"}'),
            "/json": (200, b'{"Id":"sha256:abc"}'),  # image-inspect: present locally
        },
        calls,
    )

    # Must not raise: the image is present locally.
    await runtime._pull_image("ghcr.io/the-retinue/done-check-runner:latest")

    assert any("/images/create" in c for c in calls)
    assert any("/json" in c for c in calls), "must inspect the local daemon after a failed pull"


@pytest.mark.asyncio
async def test_pull_raises_when_denied_and_image_absent() -> None:
    """A denied pull with no local image is a real failure — surface it."""
    runtime = DockerRuntime()
    calls: list[str] = []
    runtime._request = _scripted_request(  # type: ignore[method-assign]
        {
            "/images/create": (500, b'{"message":"error from registry: denied"}'),
            "/json": (404, b'{"message":"no such image"}'),
        },
        calls,
    )

    with pytest.raises(DockerError, match="image pull failed"):
        await runtime._pull_image("ghcr.io/the-retinue/done-check-runner:latest")
