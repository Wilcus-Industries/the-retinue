# The Retinue

A signed-webhook transport spine: a FastAPI `/webhook` endpoint that verifies the
GitHub `X-Hub-Signature-256` HMAC, acts on `issues` events, enqueues a job onto an
Arq/Redis queue, and a worker that dequeues and processes it.

## Architecture

```
GitHub issues webhook
        │  POST /webhook  (HMAC-SHA256 verified, 401 on mismatch/missing)
        ▼
FastAPI app (retinue.app)  ──enqueue_prd──▶  Arq / Redis queue
        │  202 Accepted                              │
                                                     ▼
                                  Worker (retinue.worker.process_prd)
                                  gates on opt-in + validity + novelty, then processes
```

- `retinue.config` — `Settings` loaded from env / `.env` (`WEBHOOK_SECRET`, `REDIS_URL`,
  `DEDUPE_DB_PATH`).
- `retinue.webhook` — HMAC verification, `issues`-event filtering, enqueue, 202 ack.
- `retinue.queue` — the `PrdJob` model and `enqueue_prd`.
- `retinue.app` — FastAPI factory; an Arq Redis pool is created in the lifespan and
  stored on `app.state.arq_pool`.
- `retinue.repo_config` — the per-repo `.github/retinue.yml` schema (`RepoConfig`) and
  `load_repo_config`, which never raises on bad input.
- `retinue.dedupe` — `PrdDedupeStore`, SQLite-backed first-claim-wins PRD dedupe.
- `retinue.worker` — the `process_prd` Arq task, the `gate_prd` opt-in gate, and
  `WorkerSettings`.
- `retinue.github_app` — `InstallationAuth`, the seam that mints a GitHub App
  installation token (`InstallationToken`) the worker clones with.
- `retinue.container` — `ContainerRuntime` / `Container`, the disposable-container
  seam the done-check runs inside (real Docker lives behind it).
- `retinue.done_check` — `run_done_check`, which runs an accepted repo's done-check in
  a fresh container and reports the outcome.

A validly signed `issues` webhook returns 202 and enqueues exactly one job; an
invalid or missing signature returns 401 and enqueues nothing. Non-`issues` events
are acked with 204 without enqueuing.

## Per-repo opt-in (`.github/retinue.yml`)

The worker only acts on a PRD when the target repo opts in by committing a
`.github/retinue.yml`. The gate in `process_prd` applies three checks in order:

1. **Opt-in** — no file means the repo is not opted in and the PRD is skipped.
2. **Validity** — a malformed file (bad YAML or a schema violation) is skipped and
   logged; it never crashes the worker and never burns the dedupe slot, so a later
   fixed config can still run.
3. **Novelty** — the PRD is deduplicated by `owner/repo#issue` (SQLite-backed), so a
   redelivered or repeated event is processed exactly once.

Schema (all fields optional; defaults shown):

```yaml
staging_branch: staging        # branch the retinue integrates onto
retry_cap: 3                   # max retries per unit of work (>= 0)
max_parallel: 4                # optional concurrency cap (> 0)
cron: "0 */6 * * *"            # optional five-field cron cadence
models:                        # optional role -> model-id overrides
  planner: claude-opus-4
secrets:                       # optional inline secrets + external refs
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  refs:
    - vault://team/retinue/github-token
```

Unknown top-level keys are rejected (a typo'd field is a skip, not a silent drop).

> The GitHub fetch of `retinue.yml` is a later issue; the worker's `fetch_config`
> seam currently treats every repo as not opted in until that lands.

## Disposable-container done-check

Once a PRD is accepted, the worker runs the repo's done-check in a fresh, throwaway
container. `retinue.done_check.run_done_check` orchestrates, in order:

1. **auth** — mint a GitHub App installation token (`retinue.github_app`),
2. **clone** — clone the repo into the container over that token,
3. **inject** — resolve the config's `secrets` block and place it in the container env;
   a missing required secret escalates (an observable report) *before* any container
   starts, so a doomed check never runs,
4. **run** — run the done-check command read from the repo's `CLAUDE.md`,
5. **report** — post the outcome to an observable sink (commit status / issue comment),
6. **teardown** — destroy the container (guaranteed via `try/finally`, on every path).

Auth, the container runtime, the secret resolver, and the report sink are all injected,
so the orchestration is fully exercised without Docker or network; a real container is
only touched in the manual smoke. The done-check command is parsed from the first fenced
code block under a "Definition of done" heading in the repo's `CLAUDE.md`.

## Configuration

Set via environment variables or a `.env` file:

| Variable         | Required | Default                   | Description                       |
| ---------------- | -------- | ------------------------- | --------------------------------- |
| `WEBHOOK_SECRET` | yes      | —                         | GitHub webhook HMAC secret        |
| `REDIS_URL`      | no       | `redis://localhost:6379`  | Redis connection URL              |
| `DEDUPE_DB_PATH` | no       | `retinue-dedupe.sqlite3`  | SQLite file backing PRD dedupe    |

## Running

```sh
# API
uv run uvicorn retinue.app:create_app --factory

# Worker (separate process)
uv run retinue-worker
# or: uv run arq retinue.worker.WorkerSettings
```

## Development

```sh
uv run pytest
uv run ruff check .
uv run mypy .
```
