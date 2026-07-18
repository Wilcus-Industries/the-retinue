"""Worker-global cron heartbeat: the safety-net sweep + the backlog cron lane (issue #35).

A single arq ``cron_jobs`` tick fires on a fixed cadence (the global tick). On each tick
the heartbeat does two things across the opted-in repos:

1. **fires the ad-hoc drain** (:func:`retinue.adhoc_drain.run_adhoc_drain`, behind the
   injected ``drain`` callable) as the safety-net sweep — the catch-up for issues labeled
   ``ready-for-agent`` while the webhook was missed or the worker was down. ``repo_config.cron``
   is the per-repo "is this repo due?" filter under the global tick (:func:`cron_due`): a
   repo only gets a scheduled sweep on a tick its cadence matches, so a repo with a sparse
   cadence is not swept on every tick;
2. **drives the backlog cron lane** (:func:`retinue.cron.run_cron_tick`, behind the injected
   ``cron_tick`` callable) for every opted-in repo — making this heartbeat the first runtime
   caller of the previously-dead cron lane. The backlog tick gates itself on the shared
   budget and picks its own work, so it runs on every tick regardless of the per-repo
   ad-hoc cadence.

The global cadence is owned by the arq ``cron_jobs`` registration (the worker fires
:func:`heartbeat_tick` every N minutes); the per-repo cadence is owned by ``repo_config.cron``.
Every collaborator — the clock, the opted-in repo enumeration, the per-repo drain, and the
backlog tick — is injected, so the heartbeat runs with no real arq, Redis, gh, Docker, or
wall-clock in tests.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from retinue.budget import Clock
from retinue.cron import CronTickResult
from retinue.repo_config import RepoConfig

logger = logging.getLogger(__name__)

# A cron cadence is the classic five whitespace-separated fields
# (minute hour day-of-month month day-of-week); mirrors repo_config's validation.
_CRON_FIELD_COUNT = 5


@dataclass(frozen=True)
class DueRepo:
    """One opted-in repo the heartbeat sweeps, with its parsed config.

    Attributes:
        repo_full_name: e.g. "owner/repo".
        config: The repo's accepted :class:`~retinue.repo_config.RepoConfig`; its ``cron``
            field is the per-repo "is this repo due?" filter under the global tick.
    """

    repo_full_name: str
    config: RepoConfig


# Enumerates the opted-in repos to sweep on a tick (each with its parsed config). Injected
# so the heartbeat does not reach for a live repo store; production binds it to the GitHub
# App's installed-repository listing, tests script a fixed set.
RepoEnumerator = Callable[[], Awaitable[list[DueRepo]]]

# Fires the safety-net ad-hoc drain for one due repo (wraps
# :func:`retinue.adhoc_drain.run_adhoc_drain` with the repo's gh seam, build, governor, and
# lock already bound). Injected so the heartbeat runs with no real gh, Docker, or network.
HeartbeatDrain = Callable[..., Awaitable[None]]

# Drives the backlog cron lane for one repo on this tick (wraps
# :func:`retinue.cron.run_cron_tick`, e.g. via :func:`retinue.wiring.bind_cron_tick`).
# Injected for the same reason.
HeartbeatCronTick = Callable[..., Awaitable[CronTickResult]]


@dataclass(frozen=True)
class HeartbeatResult:
    """What one heartbeat tick fired.

    Attributes:
        drained_repos: The repos whose safety-net ad-hoc drain ran (cron-due and did not
            raise), in sweep order.
        ticked_repos: The repos whose backlog cron lane was driven this tick, in sweep
            order — every opted-in repo whose tick did not raise.
    """

    drained_repos: list[str]
    ticked_repos: list[str]


def cron_due(cron_expr: str | None, when: datetime) -> bool:
    """Whether a repo's five-field ``cron`` cadence is due at ``when`` (the global tick).

    The per-repo "is this repo due?" filter under the worker-global heartbeat: a repo with
    no cadence (``None``) is never picked up by the scheduled sweep, and a repo with a
    cadence is swept only on a tick its cadence matches. Supports the standard cron field
    grammar — ``*`` (any), a literal value, a ``a,b`` comma list, an ``a-b`` inclusive
    range, and a ``*/step``, ``a-b/step``, or ``N/step`` step — across minute, hour,
    day-of-month, month, and day-of-week. A value-base step follows Vixie semantics: ``N/step``
    is ``N-high/step`` (``5/15`` minute is ``5,20,35,50``), not the single point ``N``.
    Sunday is accepted as both ``0`` and ``7``. The seconds/lower
    resolution is ignored: arq fires the global tick on whole minutes, so matching to the
    minute is the right grain.

    Args:
        cron_expr: The repo's five-field cron string, or ``None`` for no cadence.
        when: The tick instant to test the cadence against.

    Returns:
        True when the repo is due on this tick, else False.
    """
    if cron_expr is None:
        return False
    fields = cron_expr.split()
    if len(fields) != _CRON_FIELD_COUNT:
        # Defensive: repo_config validates the field count, but a hand-built config could
        # carry a malformed cron. A malformed cadence is treated as never-due rather than
        # raising into the worker-global tick.
        logger.warning("Ignoring malformed cron cadence %r", cron_expr)
        return False

    minute, hour, day, month, weekday = fields
    return (
        _field_matches(minute, when.minute, low=0, high=59)
        and _field_matches(hour, when.hour, low=0, high=23)
        and _field_matches(day, when.day, low=1, high=31)
        and _field_matches(month, when.month, low=1, high=12)
        and _weekday_matches(weekday, when)
    )


def _weekday_matches(field: str, when: datetime) -> bool:
    """Match a cron day-of-week field, accepting Sunday as both 0 and 7.

    Python's :meth:`datetime.weekday` is Mon=0..Sun=6; cron is Sun=0..Sat=6 with Sun also
    expressible as 7. Convert to cron's numbering and test both Sunday spellings so
    ``* * * * 0`` and ``* * * * 7`` both match a Sunday tick.
    """
    cron_dow = (when.weekday() + 1) % 7  # Mon=0..Sun=6 -> Sun=0,Mon=1..Sat=6
    if _field_matches(field, cron_dow, low=0, high=7):
        return True
    # Sunday is 0 or 7; a field written as 7 must still match a Sunday tick (cron_dow == 0).
    if cron_dow == 0:
        return _field_matches(field, 7, low=0, high=7)
    return False


def _field_matches(field: str, value: int, *, low: int, high: int) -> bool:
    """Whether one cron field (a comma list of ``*``/value/range/step terms) matches ``value``."""
    return any(_term_matches(term, value, low=low, high=high) for term in field.split(","))


def _term_matches(term: str, value: int, *, low: int, high: int) -> bool:
    """Whether one cron term (``*``, a value, ``a-b``, ``*/s``, or ``a-b/s``) matches ``value``."""
    base, _, step_text = term.partition("/")
    has_step = bool(step_text)
    step = _parse_step(step_text) if has_step else 1
    start, end = _term_range(base, has_step=has_step, low=low, high=high)
    if start is None or end is None or step is None:
        return False
    return start <= value <= end and (value - start) % step == 0


def _term_range(
    base: str, *, has_step: bool, low: int, high: int
) -> tuple[int | None, int | None]:
    """Resolve a cron term's base (before any ``/step``) into an inclusive ``[start, end]``.

    ``*`` spans the whole field; ``a-b`` is the literal inclusive range. A bare value is the
    single point ``[v, v]`` on its own, but ``[v, high]`` when a ``/step`` follows it: Vixie
    cron reads ``N/step`` as ``N-high/step`` (``5/15`` minute is ``5,20,35,50``, not just
    ``5``). A non-numeric or out-of-bounds base yields ``(None, None)`` so the term simply
    does not match rather than raising into the worker-global tick.
    """
    if base == "*":
        return low, high
    if "-" in base:
        start_text, _, end_text = base.partition("-")
        start, end = _parse_int(start_text), _parse_int(end_text)
        if start is None or end is None or start > end:
            return None, None
        return start, end
    point = _parse_int(base)
    if point is None:
        return None, None
    # A bare value with a step walks from the value to the field's max (Vixie N-high/step);
    # without a step it is the single point.
    return (point, high) if has_step else (point, point)


def _parse_int(text: str) -> int | None:
    """Parse a cron integer, or ``None`` when the text is not a plain integer."""
    try:
        return int(text)
    except ValueError:
        return None


def _parse_step(text: str) -> int | None:
    """Parse a positive cron step, or ``None`` when it is not a positive integer."""
    step = _parse_int(text)
    if step is None or step <= 0:
        return None
    return step


async def run_heartbeat(
    *,
    enumerate_repos: RepoEnumerator,
    clock: Clock,
    drain: HeartbeatDrain,
    cron_tick: HeartbeatCronTick,
    tick_number: int,
) -> HeartbeatResult:
    """Run one worker-global heartbeat: the safety-net sweep + the backlog cron lane.

    For each opted-in repo the enumerator returns, on this tick:

    * fire the **safety-net ad-hoc drain** when ``repo_config.cron`` is due at the clock's
      instant (:func:`cron_due`) — the catch-up for issues labeled while the webhook was
      missed or the worker was down;
    * drive the **backlog cron lane** (:func:`retinue.cron.run_cron_tick`) regardless of
      the per-repo cadence — the lane gates itself on the shared budget and picks its own
      work, and this is the first runtime caller of the previously-dead lane.

    A drain or tick that raises for one repo is logged and skipped so a single bad repo
    cannot starve the rest of the sweep (mirroring the worker's per-repo skip discipline).

    Args:
        enumerate_repos: Yields the opted-in repos to sweep, each with its parsed config.
        clock: The injected wall-clock seam; its ``now()`` is the tick instant the per-repo
            cadence is tested against (no real wall-clock in tests).
        drain: Fires the safety-net ad-hoc drain for one due repo.
        cron_tick: Drives the backlog cron lane for one repo on this tick.
        tick_number: This tick's sequence number, threaded into the backlog tick for its
            quota floor (every Nth tick drains the oldest low-priority backlog issue).

    Returns:
        A :class:`HeartbeatResult` listing the repos drained and ticked this sweep.
    """
    now = clock.now()
    repos = await enumerate_repos()
    drained: list[str] = []
    ticked: list[str] = []
    for repo in repos:
        if cron_due(repo.config.cron, now) and await _safe_drain(repo, drain):
            drained.append(repo.repo_full_name)
        if await _safe_cron_tick(repo, cron_tick, tick_number):
            ticked.append(repo.repo_full_name)
    logger.info(
        "Heartbeat tick %d: drained %d repo(s), ticked %d backlog lane(s)",
        tick_number,
        len(drained),
        len(ticked),
    )
    return HeartbeatResult(drained_repos=drained, ticked_repos=ticked)


async def _safe_drain(repo: DueRepo, drain: HeartbeatDrain) -> bool:
    """Fire the safety-net drain for ``repo``; log and swallow a per-repo failure.

    A drain that raises for one repo is an observable skip, not a crash of the whole sweep,
    so the remaining repos still drain and the backlog lane still ticks.
    """
    try:
        await drain(repo_full_name=repo.repo_full_name, config=repo.config)
    except Exception:
        logger.exception("Heartbeat drain failed for %s", repo.repo_full_name)
        return False
    return True


async def _safe_cron_tick(
    repo: DueRepo, cron_tick: HeartbeatCronTick, tick_number: int
) -> bool:
    """Drive the backlog cron lane for ``repo``; log and swallow a per-repo failure.

    The repo's ``config`` rides the call so the tick's trickle promotion can apply the
    repo's own ``trigger_label`` (label surgery: ``backlog`` -> ``trigger_label``).
    """
    try:
        await cron_tick(
            repo_full_name=repo.repo_full_name,
            tick_number=tick_number,
            config=repo.config,
        )
    except Exception:
        logger.exception("Heartbeat backlog tick failed for %s", repo.repo_full_name)
        return False
    return True


async def heartbeat_tick(ctx: dict[str, Any]) -> None:
    """Arq ``cron_jobs`` task: run one worker-global heartbeat from the injected ``ctx``.

    Reads the heartbeat's collaborators from ``ctx`` (populated by the worker's
    ``on_startup``): the repo enumerator, the clock, the per-repo drain, and the backlog
    cron tick. The tick number is kept on ``ctx`` and incremented each tick so the backlog
    lane's quota floor advances across heartbeats. With no collaborators wired (a bare
    worker / round-trip skeleton) the tick is a harmless no-op rather than a crash.

    Args:
        ctx: Arq worker context; may carry ``heartbeat_enumerate_repos``,
            ``heartbeat_clock``, ``heartbeat_drain``, and ``heartbeat_cron_tick``.
    """
    enumerate_repos: RepoEnumerator | None = ctx.get("heartbeat_enumerate_repos")
    clock: Clock | None = ctx.get("heartbeat_clock")
    drain: HeartbeatDrain | None = ctx.get("heartbeat_drain")
    cron_tick: HeartbeatCronTick | None = ctx.get("heartbeat_cron_tick")
    if enumerate_repos is None or clock is None or drain is None or cron_tick is None:
        logger.info("Heartbeat not wired; skipping tick")
        return

    tick_number = int(ctx.get("heartbeat_tick_number", 0)) + 1
    ctx["heartbeat_tick_number"] = tick_number
    await run_heartbeat(
        enumerate_repos=enumerate_repos,
        clock=clock,
        drain=drain,
        cron_tick=cron_tick,
        tick_number=tick_number,
    )
