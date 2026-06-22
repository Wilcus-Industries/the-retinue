"""Arq worker: the process_prd task and WorkerSettings.

The walking-skeleton worker dequeues a PRD job and logs the repository, issue
number, and action. Later issues replace the body of :func:`process_prd` with the
real PRD pipeline; the queue contract (task name and kwargs) stays the same.

Launch the worker with:
    arq retinue.worker.WorkerSettings
or via the ``retinue-worker`` console script.
"""

from __future__ import annotations

import base64
import binascii
import enum
import logging
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from arq.connections import RedisSettings

from retinue.dedupe import PrdDedupeStore, prd_dedupe_key
from retinue.github_app import InstallationAuth
from retinue.queue import PrdJob
from retinue.repo_config import RepoConfig, load_repo_config

logger = logging.getLogger(__name__)

# Async callable that returns a repo's raw ``.github/retinue.yml`` text, or None
# when the repo has no such file. The GitHub fetch is injected at this seam so the
# gate is testable without network; :func:`github_config_fetcher` builds the real one.
ConfigFetcher = Callable[[str], Awaitable[str | None]]

# Path of the opt-in config file inside each repo, fetched over the contents API.
RETINUE_CONFIG_PATH = ".github/retinue.yml"
GITHUB_API_BASE_URL = "https://api.github.com"


class GateOutcome(enum.Enum):
    """Why the opt-in gate accepted or skipped a PRD."""

    ACCEPTED = "accepted"
    NOT_OPTED_IN = "not_opted_in"
    MALFORMED_CONFIG = "malformed_config"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class GateResult:
    """Result of gating one PRD.

    Attributes:
        outcome: Why the PRD was accepted or skipped.
        config: The parsed repo config when ``outcome`` is ``ACCEPTED``, else None.
    """

    outcome: GateOutcome
    config: RepoConfig | None = None


async def gate_prd(
    job: PrdJob,
    *,
    fetch_config: ConfigFetcher,
    dedupe: PrdDedupeStore,
) -> GateResult:
    """Decide whether a dequeued PRD should be processed, and parse its config.

    Three gates, in order: opt-in (a ``.github/retinue.yml`` exists), validity (it
    parses against the schema), and novelty (not already deduped). A repo with no
    file or a malformed file is an observable skip — logged, never raised — so one
    bad repo cannot crash the worker. The dedupe slot is claimed only for an
    accepted PRD, so a malformed config does not block a later fixed redelivery.

    Args:
        job: The dequeued PRD job.
        fetch_config: Async callable returning the repo's ``retinue.yml`` text, or
            None when the repo has no config file.
        dedupe: The SQLite-backed dedupe store.

    Returns:
        A :class:`GateResult`; ``config`` is set only when ``outcome`` is ACCEPTED.
    """
    ref = f"{job.repo_full_name}#{job.issue_number}"

    raw = await fetch_config(job.repo_full_name)
    if raw is None:
        logger.info("Skipping %s: repo not opted in (no .github/retinue.yml)", ref)
        return GateResult(GateOutcome.NOT_OPTED_IN)

    config = load_repo_config(raw)
    if config is None:
        logger.warning("Skipping %s: malformed .github/retinue.yml", ref)
        return GateResult(GateOutcome.MALFORMED_CONFIG)

    if not await dedupe.claim(prd_dedupe_key(job)):
        logger.info("Skipping %s: duplicate PRD already processed", ref)
        return GateResult(GateOutcome.DUPLICATE)

    return GateResult(GateOutcome.ACCEPTED, config=config)


async def process_prd(
    ctx: dict[str, Any],
    *,
    repo_full_name: str,
    issue_number: int,
    action: str,
) -> None:
    """Arq task: gate a dequeued PRD on opt-in + validity + novelty, then process.

    Loads the repo's ``.github/retinue.yml`` (presence = opt-in), validates it, and
    deduplicates the PRD before doing real work. A repo with no config, a malformed
    config, or an already-processed PRD is an observable skip. The config-fetcher
    and dedupe store are read from ``ctx`` (populated by ``on_startup``); when absent
    (e.g. the bare round-trip test) the task falls back to logging only.

    Args:
        ctx: Arq worker context; may carry ``fetch_config`` and ``dedupe``.
        repo_full_name: e.g. "owner/repo".
        issue_number: The GitHub issue number.
        action: The webhook action that triggered the job.
    """
    job = PrdJob(
        repo_full_name=repo_full_name, issue_number=issue_number, action=action
    )

    fetch_config: ConfigFetcher | None = ctx.get("fetch_config")
    dedupe: PrdDedupeStore | None = ctx.get("dedupe")
    if fetch_config is None or dedupe is None:
        # No gate wired in (bare worker / round-trip skeleton): log and return.
        logger.info(
            "Processing PRD for %s#%d action=%s",
            repo_full_name,
            issue_number,
            action,
        )
        return

    result = await gate_prd(job, fetch_config=fetch_config, dedupe=dedupe)
    if result.outcome is not GateOutcome.ACCEPTED:
        return

    logger.info(
        "Processing PRD for %s#%d action=%s (staging_branch=%s)",
        repo_full_name,
        issue_number,
        action,
        result.config.staging_branch if result.config else "?",
    )


def _configure_logging() -> None:
    """Send INFO-level logs to stdout so the worker's progress is observable.

    ``run_worker`` does not configure logging (only arq's CLI does), so without
    this the root logger sits at WARNING with no handler and every ``logger.info``
    line is dropped. Install a single stdout handler at INFO; ``force=True``
    rebinds any pre-existing root config to the current stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


def _contents_url(repo_full_name: str) -> str:
    """Build the GitHub contents-API URL for a repo's ``.github/retinue.yml``."""
    return f"{GITHUB_API_BASE_URL}/repos/{repo_full_name}/contents/{RETINUE_CONFIG_PATH}"


def _auth_headers(token: str) -> dict[str, str]:
    """Build the request headers authorising a GitHub contents-API read.

    Uses the documented ``Bearer`` scheme and pins the v3 contents media type and API
    version so the response shape (a base64 ``content`` field) is stable.
    """
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _decode_contents_payload(payload: dict[str, Any]) -> str:
    """Decode a GitHub contents-API payload into the raw file text.

    The contents API returns the file body base64-encoded in ``content`` (with
    embedded newlines) under ``encoding: base64``. Decode it to UTF-8 text — the same
    raw YAML the fake fetcher hands :func:`gate_prd` for parsing.

    Raises:
        ValueError: when the payload is not a base64-encoded file (unexpected shape,
            e.g. a directory listing or an unknown encoding).
    """
    encoding = payload.get("encoding")
    if encoding != "base64" or not isinstance(payload.get("content"), str):
        raise ValueError(f"unexpected contents payload encoding: {encoding!r}")
    try:
        return base64.b64decode(payload["content"]).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"undecodable contents payload: {exc}") from exc


def github_config_fetcher(
    auth: InstallationAuth, client: httpx.AsyncClient
) -> ConfigFetcher:
    """Build the production config fetcher backed by the GitHub contents API.

    The returned async callable mints an installation token for the repo, reads
    ``.github/retinue.yml`` over the contents API, and returns its raw YAML text — the
    exact shape :func:`gate_prd` expects. A 404 (no such file) maps to ``None`` so the
    gate reads the repo as not opted in, matching the injected fake. Any other HTTP
    error is raised: a transient failure must retry the job, not be silently mistaken
    for an opted-out repo.

    Args:
        auth: Mints an installation token scoped to the target repo.
        client: A shared httpx client used for the contents read.

    Returns:
        A :class:`ConfigFetcher` returning the raw config text, or ``None`` on 404.
    """

    async def fetch(repo_full_name: str) -> str | None:
        installation = await auth.installation_token(repo_full_name)
        response = await client.get(
            _contents_url(repo_full_name),
            headers=_auth_headers(installation.token),
        )
        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        response.raise_for_status()
        return _decode_contents_payload(response.json())

    return fetch


async def on_startup(ctx: dict[str, Any]) -> None:
    """Populate the worker context with the PRD gate's collaborators.

    Installs the config fetcher and the SQLite-backed dedupe store onto ``ctx`` so
    :func:`process_prd` can gate each dequeued PRD on opt-in, validity, and novelty.
    """
    global settings
    if settings is None:
        settings = _load_settings()
    wiring = _load_github_client()
    if wiring is None:
        # No GitHub App auth wired yet (the concrete InstallationAuth is another
        # layer's seam): fall back to the safe not-opted-in default so nothing is
        # processed without an explicit config, rather than crashing the worker.
        ctx["fetch_config"] = _no_config_fetcher
    else:
        auth, client = wiring
        ctx["github_client"] = client
        ctx["fetch_config"] = github_config_fetcher(auth, client)
    ctx["dedupe"] = PrdDedupeStore(settings.dedupe_db_path)


async def on_shutdown(ctx: dict[str, Any]) -> None:
    """Close the shared GitHub HTTP client opened in :func:`on_startup`, if any."""
    client: httpx.AsyncClient | None = ctx.get("github_client")
    if client is not None:
        await client.aclose()


async def _no_config_fetcher(repo_full_name: str) -> str | None:
    """Fallback fetcher used when no GitHub App auth is wired: treat repo as opted out.

    The safe default — nothing is processed without an explicit, fetchable config.
    """
    logger.debug("No GitHub auth wired; treating %s as not opted in", repo_full_name)
    return None


def _load_github_client() -> tuple[InstallationAuth, httpx.AsyncClient] | None:
    """Construct the production GitHub installation auth and HTTP client, if available.

    Returns ``None`` when no concrete :class:`~retinue.github_app.InstallationAuth`
    builder is wired yet — its production implementation is owned by a separate
    seam/layer (:mod:`retinue.github_app` exposes only the protocol today). In that
    case the worker falls back to the safe not-opted-in default rather than crashing.
    Resolved lazily so registering the task (e.g. in tests) needs no GitHub App
    credentials and opens no network client at import time.
    """
    import retinue.github_app as github_app

    builder = getattr(github_app, "build_installation_auth", None)
    if builder is None:
        return None
    return builder(), httpx.AsyncClient(timeout=30.0)


# Module-level settings, loaded lazily in main() so importing this module does
# not require the env vars to be present (e.g. when registering the task in tests).
settings: Any = None


def _load_settings() -> Any:
    from retinue.config import Settings

    return Settings()  # type: ignore[call-arg]


def main() -> None:
    """Console-script entrypoint: start the Arq worker with WorkerSettings.

    Resolves ``WorkerSettings.redis_settings`` from the configured ``REDIS_URL``
    here, at process start: arq reads that class attribute when it constructs the
    Worker, before ``on_startup`` runs, so the override must be applied now.
    """
    from arq.worker import run_worker

    _configure_logging()

    global settings
    if settings is None:
        settings = _load_settings()
    WorkerSettings.redis_settings = RedisSettings.from_dsn(settings.redis_url)

    run_worker(WorkerSettings)  # type: ignore[arg-type]


class WorkerSettings:
    """Arq WorkerSettings: registers process_prd and the Redis connection.

    Launch the worker process with:
        arq retinue.worker.WorkerSettings
    """

    functions = [process_prd]
    on_startup = on_startup
    on_shutdown = on_shutdown
    # Overridden from the configured REDIS_URL in main() at process start (arq
    # reads this attribute before on_startup runs); the localhost default keeps it
    # a valid RedisSettings for the ``arq retinue.worker.WorkerSettings`` launch.
    redis_settings: RedisSettings = RedisSettings()
