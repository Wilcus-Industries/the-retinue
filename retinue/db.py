"""Shared aiosqlite connection helper for the retinue's SQLite-backed stores.

Every store keeps a long-lived aiosqlite connection, and aiosqlite runs each
connection on a worker thread with no public daemon switch. A store leaked without
``close()`` (crash paths, test teardown) would strand that non-daemon thread and
block interpreter exit forever on the thread join, so every connection opened here
marks its thread daemon before ``__await__`` starts it. This is safe: the stores
commit before returning from every write, so process exit kills no in-flight write.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aiosqlite


async def connect_daemon(db_path: Path, **kwargs: Any) -> aiosqlite.Connection:
    """Open ``db_path`` (creating parent dirs) on a daemon worker thread.

    Args:
        db_path: The SQLite file to open; missing parent directories are created.
        **kwargs: Passed through to :func:`aiosqlite.connect` (e.g. ``isolation_level``).

    Returns:
        The connected :class:`aiosqlite.Connection`, its worker thread marked daemon.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connector = aiosqlite.connect(db_path, **kwargs)
    # aiosqlite exposes no daemon option; its private ``_thread`` has not started yet
    # (that happens in ``__await__``), so flipping the flag here is race-free.
    connector._thread.daemon = True
    return await connector
