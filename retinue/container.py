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

from dataclasses import dataclass
from typing import Protocol


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

    async def run_command(self, command: list[str]) -> RunResult:
        """Run ``command`` inside the container and return its captured result."""
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
