"""The shared single-run lock: a non-blocking in-process guard reused by every lane.

Each lane driver (the cron tick, the ad-hoc drain) enters an injected
``AbstractAsyncContextManager`` that admits the first holder and *rejects* — rather than
blocks — a second concurrent ``__aenter__``, so the "at most one run at a time" contract
is observable to the caller. The guard is a plain in-process flag (no wall-clock, Redis,
or file lock), correct because each run lives inside a single worker process; a
cross-process lock is out of scope for the single-worker deployment.

The rejection error is per-lane (``CronBusyError``, ``AdhocDrainBusyError``) so callers
and tests catch a lane-specific type; :class:`SingleRunLock` carries the flag mechanics
once and a subclass names the error it raises via :attr:`busy_error`.
"""

from __future__ import annotations

from typing import Self


class SingleRunLock:
    """A non-blocking single-run guard: a second concurrent holder is rejected.

    A subclass sets :attr:`busy_error` to the lane's typed exception; entering while the
    lock is already held raises it instead of blocking. One instance guards one scope
    (the worker keeps a per-repo registry so distinct scopes run concurrently while a
    scope's own runs serialize through the same lock).
    """

    busy_error: type[BaseException]

    def __init__(self) -> None:
        self._held = False

    async def __aenter__(self) -> Self:
        if self._held:
            raise self.busy_error
        self._held = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        self._held = False
