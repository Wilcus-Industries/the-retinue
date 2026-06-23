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
                                  gates on opt-in + validity + novelty, then drives the
                                  pipeline: slice → build → open the staging PR
```

- `retinue.config` — `Settings` loaded from env / `.env` (`WEBHOOK_SECRET`, `REDIS_URL`,
  `DEDUPE_DB_PATH`, the GitHub App / Anthropic / push-channel credentials).
- `retinue.webhook` — HMAC verification then event dispatch: a `prd`-labeled `issues`
  event (action opened/reopened/edited/labeled) enqueues a PRD job, a merged
  `pull_request` (closed + merged) enqueues the reap, and a `pull_request_review` from
  heimdall's bot enqueues the loopback. Off-target events (an unlabeled or non-PRD issue,
  a review from anyone but heimdall) are acked 204 and enqueue nothing, so the slicer and
  loopback never see them. Same 401/204/202 contract for all.
- `retinue.queue` — the `PrdJob` / `ReviewJob` / `MergedPrJob` models and their
  `enqueue_prd` / `enqueue_review` / `enqueue_merged_pr` helpers.
- `retinue.pipeline` — `Pipeline`, the orchestration object the worker drives once a PRD
  is accepted: slice → build → open the staging PR, plus the `process_review` /
  `reap_pr` / `reconcile` entry points the webhook events and a restart route into.
  `build_pipeline_factory` wires it over the real adapters (the orchestrator `build_prd`
  seam stays injected, pending the implementer-spawn adapter).
- `retinue.wiring` — `bind_build_prd` (budget-gate + triage glue around the orchestrator
  build) and `bind_cron_tick` (the cron backlog lane over its real collaborators). Both
  take the implementer-spawn seam as their one injected dependency and share the
  service-level `BudgetGovernor`.
- `retinue.app` — FastAPI factory; an Arq Redis pool is created in the lifespan and
  stored on `app.state.arq_pool`.
- `retinue.repo_config` — the per-repo `.github/retinue.yml` schema (`RepoConfig`) and
  `load_repo_config`, which never raises on bad input.
- `retinue.roles` — the agent-role registry: one `ROLE_REGISTRY` table mapping each
  `Role` (slicer / implementer / resolver / reviewer) to its model id, reasoning-effort
  tier, and invocation transport. The four role adapters resolve their model and effort
  here via `resolve_model` / `resolve_effort` instead of hand-rolled constants;
  `resolve_model` applies a repo's `models` override, effort stays registry-owned.
- `retinue.dedupe` — `PrdDedupeStore`, SQLite-backed first-claim-wins PRD dedupe.
- `retinue.worker` — the `process_prd` Arq task (gate → drive the pipeline), the
  `process_review_job` / `reap_pr_job` tasks the webhook events enqueue, the `gate_prd`
  opt-in gate, and `WorkerSettings`. `on_startup` wires the real `fetch_config`, PRD-body
  fetcher, and `pipeline_factory` onto the Arq context.
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
- `retinue.pr_opener` — `open_staging_pr`: once a PRD's ready set drains, prechecks
  heimdall is installed and the staging branch exists, then brings `retinue/prd-<n>`
  up to date with staging and opens exactly one PR into it; a failed precheck escalates
  through `Notifier` and opens no PR. The gh operations are injected seams.
- `retinue.slicer` — `slice_prd`: runs the headless Agent-SDK slicer over a PRD
  body to produce vertical-slice issues labeled `ready-for-agent` + `prd-slice` +
  `Part of #<prd>` with a resolved `## Blocked by` graph, reserving `hitl` for genuinely
  human-only slices. A thin/malformed PRD escalates through `Notifier` instead of inventing
  slices. The Agent-SDK call and the gh issue creation are injected seams.
- `retinue.impl_retry` — `ImplRetryStore`, the SQLite-backed per-slice
  implementer-attempt counter that persists the retry budget across worker restarts.
- `retinue.triage` — `triage_implementer` / `decide_triage`: reasons about an
  implementer failure or returned notes and decides retry / reslice / escalate,
  bounded by the persisted retry count.
- `retinue.reviewer` — `review_round`: the internal reviewer that runs after a PRD
  round merges, reviews the round's diff for correctness/stale docs, and files
  `review-fix` follow-up issues (`ready-for-agent` + `Part of #<prd>`) wired into
  dependents' `## Blocked by`. It never edits code. The Agent-SDK review, the gh
  issue creation (reused from the slicer), and the gh issue-body edit are injected.
- `retinue.reconcile` — `reconcile_run` / `ResumePhase` / `RunStateStore`: on worker
  restart, reconciles an in-flight PRD round against GitHub (the source of truth —
  which slice issues are closed, which `issue-<N>` branches are merged, whether the
  `retinue/prd-<n>` → staging PR exists) plus the SQLite run-state, and picks the phase
  to resume at (`BUILD` finishes only the unfinished slices, `PR_OPEN`, `LOOPBACK`, or
  `DONE`) so no duplicate issue, branch, or PR is produced. A slice is finished when
  EITHER its issue is closed OR its branch is merged, so a crash between the two still
  resumes correctly. The gh queries are injected seams; `RunStateStore` is the durable
  secondary ledger (owned-slice set + PR↔PRD mapping), mirroring `PrdDedupeStore`.

A validly signed `prd`-labeled `issues` webhook (action opened/reopened/edited/labeled)
returns 202 and enqueues exactly one job; an invalid or missing signature returns 401 and
enqueues nothing. A signed issues event without the `prd` label (or with any other action),
a `pull_request_review` not from heimdall's bot, and any other off-target event are acked
with 204 without enqueuing.

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
models:                        # optional role -> model-id overrides; keys are the
  implementer: claude-opus-4-8  # registry roles: slicer/implementer/resolver/reviewer
secrets:                       # optional inline secrets + external refs
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  refs:
    - vault://team/retinue/github-token
```

Unknown top-level keys are rejected (a typo'd field is a skip, not a silent drop).

> The worker's `fetch_config` seam reads `retinue.yml` over the GitHub contents API
> (`github_config_fetcher`) once GitHub App auth is wired; absent that auth it falls back
> to treating every repo as not opted in.

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

## Reasoning failure triage

The implementer subagent does not always cleanly build a slice — it can fail (raise)
or return *notes* explaining that it could not finish or that the slice is mis-scoped.
`retinue.triage.triage_implementer` refuses to blind-loop on this: it reasons about the
signal plus how many attempts it has already spent, then carries out one decision.

1. **retry** — re-run the implementer while the *persisted* attempt count is below the
   config's `retry_cap` (default `3`; `0` means no retries). The bound is persisted in
   `retinue.impl_retry.ImplRetryStore` (a SQLite counter keyed by `owner/repo#issue`),
   so a doomed slice cannot retry forever and a slice that already spent its budget in
   an earlier run escalates without burning another attempt,
2. **reslice** — when the notes say the slice is mis-scoped, file an adjusted slice
   through the gh issue-creation seam (reusing the slicer's `create_issue`), carrying
   the implementer's reasoning into the new issue body,
3. **escalate** — hand the slice to a human by fanning a `Notification` out through the
   shared `Notifier` (push + comment + the `hitl` label).

The pure `decide_triage(failed, notes, attempts, cap)` returns the typed
`RETRY` / `RESLICE` / `ESCALATE` decision; both the failure path and the returned-notes
path reach it — notes are never silently dropped. The implementer, the notifier sinks,
the gh issue creator, and the SQLite store are all injected, so the whole loop is
exercised with no Agent SDK, no gh, no push service, and no network.

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

## Staging PR + heimdall precheck

Once a PRD's ready set drains (the full-PRD build completes), `retinue.pr_opener.open_staging_pr`
lands the work by opening a PR into `staging`, behind two prechecks applied in order:

1. **heimdall installed** — the repo must have the heimdall check installed. A repo
   without it escalates through the shared `Notifier` (push + comment + label) and opens
   no PR — landing into `staging` without the gate is unsafe.
2. **staging exists** — the target `staging` branch (`config.staging_branch`) must
   exist. A missing one escalates on its own path and opens no PR.

When both pass, the integration branch `retinue/prd-<n>` is brought up to date with the
staging branch and **exactly one** PR `retinue/prd-<n>` -> `staging` is opened. The four
gh operations — the heimdall precheck, the staging-branch existence check, the
bring-up-to-date, and the open-PR — are a single injected `PrOps` seam, so the whole
flow is exercised with no real `gh` and no network. The result is a `PrOpenResult`
whose outcome is `OPENED` (with the created PR), `HEIMDALL_MISSING`, or
`STAGING_MISSING`.

## Heimdall verdict loopback

Once the staging PR is open, heimdall posts a bot review on it.
`retinue.loopback.process_review` reads that verdict and reasons about it plus a
*persisted* per-PR rebuild-round count (`HeimdallRoundStore`, an aiosqlite counter in
the `ImplRetryStore` style), via the pure `decide_verdict`:

1. **rebuild** — heimdall raised **blocking** findings (severity at/above the `high`
   threshold) and the round count is below `RepoConfig.retry_cap` (3). Each blocking
   finding becomes a fix-issue (`ready-for-agent` + `Part of #<prd>`) that rebuilds onto
   the **same** `retinue/prd-<n>` branch and re-triggers heimdall review; the round is
   persisted so the loop survives a restart and is bounded at the cap.
2. **converge** — heimdall raised **no** blocking findings. The flow proceeds to handoff.
   Any non-blocking nits are filed as `backlog` issues carrying heimdall severity mapped
   1:1 to a `priority:<severity>` label.
3. **escalate** — the round budget is spent while still blocked. The flow stops: it
   comments the PRD, applies `hitl`, and notifies through the shared `Notifier`, leaving
   the PR open for a human.

The heimdall verdict, the gh issue creator (reused from the slicer), the
rebuild-onto-same-branch trigger, the handoff, and the notifier sinks are all injected,
so the loop is exercised with no real `gh`, heimdall, or network.

## Convergence handoff + merge reap

On convergence the loopback calls its `Handoff` seam, which `retinue.handoff` implements
as **`announce_handoff`**: a single "test & merge" notification through the shared
`Notifier` — a push heads-up plus a PR comment (and a findable `test-and-merge` label)
telling a human the PR is clean and ready. **The retinue never merges**: there is no
merge collaborator on the handoff, and a human performs the merge.

When the human merges, **`reap_merged_pr`** reacts to the `pull_request` closed+merged
signal: it closes the PR's slice issues, then *reaps* the PRD — closing it IFF every
non-`hitl` child (issues carrying `Part of #<prd>`) is closed. An open `hitl` child (a
deliberately human-only slice) does not block the reap; an open non-`hitl` child does.
The gh issue-close and child-enumeration are a single injected `Handoff` gh seam (no
merge method), so both flows run with no real `gh`, push service, or network.

## Budget governor

`retinue.budget` meters agent token spend against a **service-level** weekly budget (one
ledger shared across the orchestrator and cron lanes) and enforces a per-rolling-24h-window
ceiling — by default 12% of the weekly budget (`cap()`). The **`BudgetLedger`** is an
aiosqlite spend ledger in the `PrdDedupeStore` style: `record_spend` appends a timestamped
charge, `trailing_24h_spend` sums only the charges inside the trailing 24h read off an
**injected `Clock`** (no wall-clock, so the window is deterministic in tests).

Metering is auth-aware: an API key meters **dollars** against a weekly-$ budget,
subscription OAuth meters **tokens** against a weekly-token budget — same rolling math,
different unit. **`BudgetGovernor`** enforces at two points: `gate` **defers** a run whose
estimated charge would start it over the cap; `meter` **pauses + checkpoints** a run whose
next charge would cross the cap. Because the two lanes meter against one shared ledger
file, `meter` records through `try_record_if_within_cap`, which performs the cap check and
the insert inside a single `BEGIN IMMEDIATE` transaction: a second concurrent lane
serializes on the SQLite write lock, re-reads the updated trailing total, and pauses
instead of recording — so two charges that would jointly cross the cap can never both
land (no overshoot). The `defer_until` / `resume_at` is the instant the window
frees enough room for that specific amount — `window_frees_at(amount)` walks the in-window
charges oldest-first, accumulating freed spend, and returns the expiry of the charge that
first brings the trailing spend back under the cap (not merely the oldest charge's expiry,
which can be too early when the estimate exceeds the room it frees). `try_resume`
re-verifies the cap before clearing the pause: past `resume_at` but still over cap, it
returns `None` and leaves the run paused rather than resuming over-budget; once the window
genuinely has room it reuses the reconcile machinery (`reconcile_run` over the checkpointed
slice set), so only the unfinished slices rebuild — no duplicate issue, branch, or PR.

## Lane classifier + cron backlog drainer

`retinue.lane.classify` routes a GitHub issue to one of three lanes from its labels and
body. `ready-for-agent` is the single "build me" trigger; the `Part of #<prd>` body link
splits provenance from pickup: a `ready-for-agent` slice carrying a `Part of #<prd>` link
goes to the **orchestrator** lane (built by `build_prd`); a `ready-for-agent` issue with
**no** Part-of link goes to the **ad-hoc** lane (standalone work, not a slice of any PRD);
a loose `backlog` issue goes to the **cron** lane. PRD work runs first by default, but a
*standalone* `priority:critical` / `priority:high` issue **preempts** that ordering onto
the orchestrator lane — a critical must not wait its turn in the slow backlog drain. The
classifier is pure (labels + body only, no `gh`) and reuses the same `priority:<severity>`
vocabulary loopback emits. The slicer additionally stamps `prd-slice` alongside
`ready-for-agent` on every slice it files, so a slice is distinguishable from ad-hoc
pickup at the label layer.

`retinue.cron.run_cron_tick` is the cron lane's per-tick driver: a scheduled tick drains
loose `backlog` issues **one at a time**. Each tick runs under an injected single-run
**lock** (mirroring the orchestrator's `OrchestratorBusyError` guard, here `CronBusyError`)
so at most one cron run executes alongside at most one orchestrator run. It then **gates**
on the *same shared* service-level `BudgetGovernor` and **defers** (picking nothing,
running nothing) when the budget is spent. Otherwise it **picks** the next backlog issue by
a weighted score (priority dominates, age breaks ties within a priority), except on every
Nth tick where a **quota floor** takes the oldest low-priority issue so the low items
provably drain rather than starving behind a steady high-priority stream. The picked issue
runs the same downstream the orchestrator drives (build -> PR -> heimdall loopback ->
notify) via one injected build callable.

The clock is injected (age-weighting) and the tick counter is passed in (the quota floor),
so nothing reads the wall clock. The backlog `gh` query, the budget governor, the
single-run lock, and the downstream build are all injected and faked, so a tick runs with
no real `gh`, no Docker, and no network.

## Configuration

Set via environment variables or a `.env` file:

| Variable                     | Required | Default                   | Description                                          |
| ---------------------------- | -------- | ------------------------- | ---------------------------------------------------- |
| `WEBHOOK_SECRET`             | yes      | —                         | GitHub webhook HMAC secret                           |
| `REDIS_URL`                  | no       | `redis://localhost:6379`  | Redis connection URL                                 |
| `DEDUPE_DB_PATH`             | no       | `retinue-dedupe.sqlite3`  | SQLite file backing PRD dedupe                       |
| `AUTH_MODE`                  | no       | `api_key`                 | Metering unit: `api_key` (dollars) or `subscription` (tokens) |
| `WEEKLY_BUDGET`              | no       | `0`                       | Service-level weekly budget (dollars or tokens)      |
| `BUDGET_DB_PATH`             | no       | `retinue-budget.sqlite3`  | SQLite file backing the rolling-24h spend ledger     |
| `BUDGET_DAILY_CAP_FRACTION`  | no       | `0.12`                    | Fraction of the weekly budget spendable per 24h      |
| `GITHUB_APP_ID`              | no\*     | —                         | GitHub App numeric id (the JWT `iss` claim)          |
| `GITHUB_APP_PRIVATE_KEY_PATH`| no\*     | —                         | Path to the App RSA private key (PEM) signing app JWTs |
| `ANTHROPIC_API_KEY`          | no\*     | —                         | Anthropic API key — used in `api_key` auth mode      |
| `CLAUDE_CODE_OAUTH_TOKEN`    | no\*     | —                         | Claude subscription OAuth token (`sk-ant-oat…`) — used in `subscription` auth mode |
| `NTFY_TOPIC`                 | no\*\*   | —                         | ntfy topic for the push sink (ntfy backend)          |
| `NTFY_TOKEN`                 | no       | —                         | ntfy access token for a protected topic              |
| `PUSHOVER_TOKEN`             | no\*\*   | —                         | Pushover application API token (Pushover backend)    |
| `PUSHOVER_USER`              | no\*\*   | —                         | Pushover user/group key (Pushover backend)           |

\* Required once the worker drives the real pipeline: the GitHub App credentials mint
the per-repo tokens the gh adapters use, and the Anthropic credential for the active
`AUTH_MODE` (`ANTHROPIC_API_KEY` for `api_key`, `CLAUDE_CODE_OAUTH_TOKEN` for
`subscription`) authenticates the Agent-SDK calls. Without GitHub App auth the worker
falls back to the safe not-opted-in default and drives nothing.

\*\* Push channel: configure **either** ntfy (`NTFY_TOPIC`) **or** Pushover
(`PUSHOVER_TOKEN` + `PUSHOVER_USER`). With neither set the push heads-up is a logged
no-op; the issue comment + label (the durable escalation record) still land.

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
