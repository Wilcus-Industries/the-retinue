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
- `retinue.orchestrator` — `build_slice`, the single-slice orchestrator: spawn one
  implementer subagent on an `issue-<N>` branch, gate on `run_done_check`, and merge
  the green slice into the integration branch `retinue/prd-<n>` (a red check blocks it).
- `retinue.notify` — the reusable `Notifier`: fans one escalation out to a push
  channel (ntfy / Pushover), an issue comment, and a label, through injected sinks.
  Every escalation in the retinue routes through it.
- `retinue.slicer` — `slice_prd`: runs the headless Agent-SDK slicer over a PRD
  body to produce vertical-slice issues labeled `ready-for-agent` + `Part of #<prd>`
  with a resolved `## Blocked by` graph, reserving `hitl` for genuinely human-only
  slices. A thin/malformed PRD escalates through `Notifier` instead of inventing
  slices. The Agent-SDK call and the gh issue creation are injected seams.
- `retinue.reviewer` — `review_round`: the internal reviewer that runs after a PRD
  round merges, reviews the round's diff for correctness/stale docs, and files
  `review-fix` follow-up issues (`ready-for-agent` + `Part of #<prd>`) wired into
  dependents' `## Blocked by`. It never edits code. The Agent-SDK review, the gh
  issue creation (reused from the slicer), and the gh issue-body edit are injected.

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

## Single-slice orchestrator

`retinue.orchestrator.build_slice` builds one ready slice end-to-end, in order:

1. **spawn** — run one implementer subagent (the Agent SDK seam) in an isolated git
   worktree inside the disposable container; it implements TDD-first and commits to the
   slice's `issue-<N>` branch,
2. **done-check** — run the repo's done-check via `run_done_check` (the gate),
3. **merge** — only on a green done-check, ensure the integration branch
   `retinue/prd-<n>` exists (created off the config's `staging_branch` when absent) and
   merge `issue-<N>` into it. A red done-check **blocks** the merge: no failing slice is
   ever integrated, and the integration branch is left untouched.

The implementer spawn and the git operations (`GitOps`: ensure-branch + merge) are
injected alongside the done-check seams, so the whole one-slice flow is exercised with
no Agent SDK, no Docker, no gh, and no network.

## Full-PRD orchestrator

`retinue.orchestrator.build_prd` builds a whole PRD by wrapping the single-slice
primitive, looping rounds until the ready set drains:

1. **ready set** — pick every `PrdSlice` whose `blocked_by` refs are all merged this run
   (or absent from the PRD's slice set, meaning already merged/closed before it began),
2. **parallel fan-out** — spawn the round's implementers concurrently, bounded by the
   config's `max_parallel` (unbounded when unset),
3. **topological merge** — merge the green branches in dependency order under the
   done-check; a red slice is **blocked**, a merge conflict is handed to the injected
   `ConflictResolver` to resolve-and-retry or **escalate** (unresolvable, no resolver,
   or a retry that still conflicts). A blocked or escalated slice is terminal, so its
   dependents never become ready — that pruned subtree is reported as **skipped**, not
   silently dropped,
4. **loop** — repeat until no ready slice remains, then drain every still-pending slice
   into `skipped`.

The result is a `PrdBuildResult` that partitions every input slice into exactly one
bucket — `merged`, `blocked`, `escalated`, or `skipped` — so no slice ever vanishes
from the outcome.

The run executes inside an injected single-run lock, so at most one orchestrator run is
in flight at a time — a second entry raises `OrchestratorBusyError`. The lock, the
conflict resolver, the implementer spawn, and the git operations are all injected
alongside the done-check seams, so the whole parallel/topological flow is exercised with
no Agent SDK, no Docker, no gh, no network, and no concurrency races.

## Internal reviewer

After a PRD round merges (`build_prd`), `retinue.reviewer.review_round` reviews that
round before the next one starts:

1. **review** — run the headless Agent-SDK reviewer (the injected `generate` seam) over
   the round's merged diff and merged issue numbers, surfacing correctness bugs and
   stale docs as `ReviewFinding`s,
2. **file** — for each finding, file a follow-up issue via the slicer's `create_issue`
   seam, reusing the `ready-for-agent` + `Part of #<prd>` shape and adding a
   `review-fix` label so the agent loop routes it as a fix,
3. **wire** — append the new review-fix issue to the `## Blocked by` of each dependent
   open issue it flags (the injected `edit_blocked_by` gh seam), so the fix builds in a
   later round *before* the work layered on top of the defect.

A clean review files nothing. The reviewer **never edits code** — it only files and
wires issues. The Agent-SDK review, the gh issue creation, and the gh issue-body edit
are all injected, so the flow is exercised with no Agent SDK, no gh, and no network.

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
