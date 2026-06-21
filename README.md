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
                                  dequeues and logs repo + issue # + action
```

- `retinue.config` — `Settings` loaded from env / `.env` (`WEBHOOK_SECRET`, `REDIS_URL`).
- `retinue.webhook` — HMAC verification, `issues`-event filtering, enqueue, 202 ack.
- `retinue.queue` — the `PrdJob` model and `enqueue_prd`.
- `retinue.app` — FastAPI factory; an Arq Redis pool is created in the lifespan and
  stored on `app.state.arq_pool`.
- `retinue.worker` — the `process_prd` Arq task and `WorkerSettings`.

A validly signed `issues` webhook returns 202 and enqueues exactly one job; an
invalid or missing signature returns 401 and enqueues nothing. Non-`issues` events
are acked with 204 without enqueuing.

## Configuration

Set via environment variables or a `.env` file:

| Variable         | Required | Default                  | Description                |
| ---------------- | -------- | ------------------------ | -------------------------- |
| `WEBHOOK_SECRET` | yes      | —                        | GitHub webhook HMAC secret |
| `REDIS_URL`      | no       | `redis://localhost:6379` | Redis connection URL       |

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
