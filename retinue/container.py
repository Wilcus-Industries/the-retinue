"""Disposable container seam: the runtime the done-check runs inside.

The done-check executes untrusted repo code, so it runs in a fresh, throwaway
container that is always destroyed afterward. The real Docker calls live behind the
:class:`ContainerRuntime` / :class:`Container` protocols so the orchestrator can be
tested with the container faked — no Docker daemon, no network. The contract the
orchestrator relies on:

- :meth:`ContainerRuntime.start` returns a started :class:`Container`, with the
  configured secrets already present in its environment.
- :meth:`Container.run_command` runs a command inside and returns a :class:`RunResult`.
- :meth:`Container.destroy` tears it down; the orchestrator calls it in a ``finally``
  so a container is never leaked, even when a step raises.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunResult:
    """Outcome of a command run inside a container.

    Attributes:
        exit_code: Process exit status; ``0`` means success.
        stdout: Captured standard output.
        stderr: Captured standard error.
    """

    exit_code: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        """True when the command exited successfully (exit code 0)."""
        return self.exit_code == 0


class Container(Protocol):
    """A running disposable container the orchestrator drives then destroys."""

    async def run_command(
        self, command: list[str], *, env: Mapping[str, str] | None = None
    ) -> RunResult:
        """Run ``command`` inside the container and return its captured result.

        ``env`` overrides the container's environment for this command only (e.g. the
        done-check blanks the Anthropic auth credential so it never enters pytest's env).
        """
        ...

    async def destroy(self) -> None:
        """Tear the container down, releasing its resources. Must be idempotent."""
        ...


class ContainerRuntime(Protocol):
    """Spawns fresh disposable containers. The Docker seam.

    A production implementation talks to the Docker daemon; tests inject a fake that
    records the spawn and returns an in-memory container. ``env`` carries the secrets
    to inject, so they are present in the container's environment from the start.
    """

    async def start(self, *, image: str, env: dict[str, str]) -> Container:
        """Start a fresh container from ``image`` with ``env`` in its environment."""
        ...


# --- real Docker-backed implementation -------------------------------------------
#
# The production runtime drives the Docker Engine API over its Unix socket. No Docker
# SDK is pulled in: the wire protocol is plain HTTP/1.1 over a stream socket, so a thin
# stdlib client keeps the dependency surface (and the attack surface) small. The pure,
# bug-prone seams — auth header, request payloads, stream demux, exit-code parsing —
# are factored into the free functions below so they can be tested without a daemon.

DEFAULT_DOCKER_SOCKET = "/var/run/docker.sock"
WORKSPACE_DIR = "/workspace"
# Idle entrypoint so the container stays alive between start and exec.
_KEEP_ALIVE_CMD = ["sleep", "infinity"]


class DockerError(RuntimeError):
    """A Docker Engine API call returned an unexpected status or malformed body."""


def _registry_auth_header(username: str, password: str) -> str:
    """Build Docker's ``X-Registry-Auth`` value: base64url(JSON credentials).

    Docker authenticates image pulls with a base64url-encoded JSON document in this
    header. Returns the header *value* only; the caller attaches it to the pull request.
    """
    blob = json.dumps({"username": username, "password": password}).encode()
    return base64.urlsafe_b64encode(blob).decode()


def _create_container_payload(*, image: str, env: dict[str, str]) -> dict[str, Any]:
    """Assemble the ``POST /containers/create`` body for a fresh done-check container.

    Env is rendered as Docker's ``KEY=value`` list, and the container idles on a
    keep-alive command so :meth:`DockerContainer.run_command` can exec into it.
    """
    return {
        "Image": image,
        "Env": [f"{key}={value}" for key, value in env.items()],
        "Cmd": list(_KEEP_ALIVE_CMD),
        "WorkingDir": WORKSPACE_DIR,
    }


def _exec_create_payload(
    command: list[str], env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Assemble the ``POST /containers/{id}/exec`` body, capturing both output streams.

    ``env`` is layered over the container's own environment for this exec only (Docker
    merges exec env on top of container env, last wins), so a caller can blank a
    container-level secret for a single command — e.g. the done-check unsets the
    Anthropic auth credential so it never reaches pytest's environment.
    """
    payload: dict[str, Any] = {
        "Cmd": command,
        "AttachStdout": True,
        "AttachStderr": True,
        "WorkingDir": WORKSPACE_DIR,
    }
    if env:
        payload["Env"] = [f"{key}={value}" for key, value in env.items()]
    return payload


def _demux_stream(raw: bytes) -> tuple[str, str]:
    """Split Docker's multiplexed exec stream into ``(stdout, stderr)``.

    Each frame is an 8-byte header — stream id (1=stdout, 2=stderr), three zero bytes,
    a big-endian uint32 payload length — followed by that many payload bytes. A trailing
    frame too short to be complete is dropped rather than raising, so a truncated stream
    still yields whatever was fully received.
    """
    stdout = bytearray()
    stderr = bytearray()
    offset = 0
    while offset + 8 <= len(raw):
        stream_id = raw[offset]
        size = int.from_bytes(raw[offset + 4 : offset + 8], "big")
        start = offset + 8
        end = start + size
        if end > len(raw):
            break
        if stream_id == 2:
            stderr += raw[start:end]
        else:
            stdout += raw[start:end]
        offset = end
    return stdout.decode(errors="replace"), stderr.decode(errors="replace")


def _parse_exec_exit_code(inspect: dict[str, Any]) -> int:
    """Read the exit code from a ``GET /exec/{id}/json`` payload.

    A null ``ExitCode`` (exec not finished / unknown) is treated as a failure rather
    than a success, so an indeterminate result never masquerades as a passing check.
    """
    code = inspect.get("ExitCode")
    if isinstance(code, int):
        return code
    return 1


async def _read_http_response(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """Read one HTTP/1.1 response off the socket, returning ``(status, body)``.

    Supports the two framings the Engine API uses for these calls: ``Content-Length``
    and chunked transfer-encoding. Exec output streams come back as the raw, multiplexed
    body, which falls through to read-until-EOF.
    """
    status_line = await reader.readline()
    status = int(status_line.split(b" ", 2)[1]) if b" " in status_line else 0
    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"", b"\n"):
            break
        name, _, value = line.decode().partition(":")
        headers[name.strip().lower()] = value.strip()

    if "content-length" in headers:
        body = await reader.readexactly(int(headers["content-length"]))
    elif headers.get("transfer-encoding", "").lower() == "chunked":
        body = await _read_chunked(reader)
    else:
        body = await reader.read()
    return status, body


async def _read_chunked(reader: asyncio.StreamReader) -> bytes:
    """Decode an HTTP/1.1 chunked-transfer body into its concatenated payload."""
    body = bytearray()
    while True:
        size_line = await reader.readline()
        size = int(size_line.strip().split(b";", 1)[0] or b"0", 16)
        if size == 0:
            await reader.readline()  # trailing CRLF after the final chunk
            break
        body += await reader.readexactly(size)
        await reader.readline()  # CRLF terminating the chunk
    return bytes(body)


class DockerRuntime:
    """Production :class:`ContainerRuntime` backed by the Docker Engine API.

    Talks to the daemon over its Unix socket. ``socket_path`` defaults to the value of
    ``DOCKER_SOCKET`` or the conventional ``/var/run/docker.sock``. Registry credentials,
    when supplied, authenticate the image pull.
    """

    def __init__(
        self,
        *,
        socket_path: str | None = None,
        registry_username: str | None = None,
        registry_password: str | None = None,
    ) -> None:
        self._socket_path = (
            socket_path or os.environ.get("DOCKER_SOCKET") or DEFAULT_DOCKER_SOCKET
        )
        self._registry_username = registry_username
        self._registry_password = registry_password

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
        reader, writer = await asyncio.open_unix_connection(self._socket_path)
        try:
            payload = b"" if body is None else json.dumps(body).encode()
            lines = [
                f"{method} {path} HTTP/1.1",
                "Host: docker",
                "Connection: close",
                "Content-Type: application/json",
                f"Content-Length: {len(payload)}",
            ]
            for name, value in (headers or {}).items():
                lines.append(f"{name}: {value}")
            request = ("\r\n".join(lines) + "\r\n\r\n").encode() + payload
            writer.write(request)
            await writer.drain()
            return await _read_http_response(reader)
        finally:
            writer.close()
            await writer.wait_closed()

    async def start(self, *, image: str, env: dict[str, str]) -> Container:
        await self._pull_image(image)
        status, body = await self._request(
            "POST", "/containers/create", body=_create_container_payload(image=image, env=env)
        )
        if status not in (200, 201):
            raise DockerError(
                f"container create failed ({status}): {body.decode(errors='replace')}"
            )
        container_id = json.loads(body)["Id"]
        start_status, start_body = await self._request(
            "POST", f"/containers/{container_id}/start"
        )
        if start_status not in (204, 304):
            raise DockerError(
                f"container start failed ({start_status}): {start_body.decode(errors='replace')}"
            )
        return DockerContainer(container_id, self)

    async def _pull_image(self, image: str) -> None:
        headers: dict[str, str] = {}
        if self._registry_username is not None and self._registry_password is not None:
            headers["X-Registry-Auth"] = _registry_auth_header(
                self._registry_username, self._registry_password
            )
        status, body = await self._request(
            "POST", f"/images/create?fromImage={quote(image, safe='')}", headers=headers
        )
        if status == 200:
            return
        # The pull failed. Before treating that as fatal, fall back to a locally-present
        # image — `docker run --pull=missing` semantics. A local single-host deploy builds
        # the runner image on the same daemon the worker drives and never pushes it to a
        # registry, so the pull is denied even though the image is right there. Only raise
        # when the daemon has no such image either.
        if await self._image_exists_locally(image):
            logger.warning(
                "image pull failed (%s) but %s is present locally; using the local image",
                status,
                image,
            )
            return
        raise DockerError(f"image pull failed ({status}): {body.decode(errors='replace')}")

    async def _image_exists_locally(self, image: str) -> bool:
        """Whether the daemon already has ``image`` (inspect returns 200)."""
        status, _ = await self._request(
            "GET", f"/images/{quote(image, safe='')}/json"
        )
        return status == 200


class DockerContainer:
    """A running Docker container the orchestrator execs into, then destroys."""

    def __init__(self, container_id: str, runtime: DockerRuntime) -> None:
        self._id = container_id
        self._runtime = runtime
        self._destroyed = False

    async def run_command(
        self, command: list[str], *, env: Mapping[str, str] | None = None
    ) -> RunResult:
        """Exec ``command`` inside the container, returning its captured result.

        ``env`` is layered over the container env for this exec only, so a caller can
        blank a container-level secret for one command (the done-check unsets the
        Anthropic credential so it never reaches the checked repo's test process).
        """
        status, body = await self._runtime._request(
            "POST",
            f"/containers/{self._id}/exec",
            body=_exec_create_payload(command, env),
        )
        if status not in (200, 201):
            raise DockerError(f"exec create failed ({status}): {body.decode(errors='replace')}")
        exec_id = json.loads(body)["Id"]

        _, stream = await self._runtime._request(
            "POST", f"/exec/{exec_id}/start", body={"Detach": False, "Tty": False}
        )
        stdout, stderr = _demux_stream(stream)

        inspect_status, inspect_body = await self._runtime._request(
            "GET", f"/exec/{exec_id}/json"
        )
        if inspect_status != 200:
            raise DockerError(
                f"exec inspect failed ({inspect_status}): "
                f"{inspect_body.decode(errors='replace')}"
            )
        exit_code = _parse_exec_exit_code(json.loads(inspect_body))
        return RunResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    async def destroy(self) -> None:
        """Force-remove the container. Idempotent: a second call is a no-op."""
        if self._destroyed:
            return
        self._destroyed = True
        try:
            status, body = await self._runtime._request(
                "DELETE", f"/containers/{self._id}?force=true&v=true"
            )
        except OSError as exc:
            logger.warning("container %s teardown failed: %s", self._id, exc)
            return
        if status not in (200, 204, 404):
            logger.warning(
                "container %s teardown returned %s: %s",
                self._id,
                status,
                body.decode(errors="replace"),
            )
