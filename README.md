# The Retinue

A signed-webhook transport spine: a FastAPI `/webhook` endpoint that verifies the
GitHub `X-Hub-Signature-256` HMAC, acts on `issues` events, enqueues a job onto an
Arq/Redis queue, and a worker that dequeues and drives **one unified scheduler lane** —
list the ready work, rank it by severity, build each admitted issue in a disposable
container, and gate it through an in-session pre-PR review before opening the PR.

## Architecture

```
GitHub issues webhook
        │  POST /webhook  (HMAC-SHA256 verified, 401 on mismatch/missing)
        ▼
FastAPI app (retinue.app)  ──enqueue_adhoc_drain──▶  Arq / Redis queue
        │  202 Accepted                                    │
                                                           ▼
                                    Worker (retinue.worker.run_adhoc_drain_job)
                                    run_adhoc_drain: list trigger-labeled issues →
                                    admit → readiness gate → flight-state classify →
                                    two-queue scheduler select → build each →
                                    in-session review gate → open PR
```

There is **one** scheduler lane. No PRD lane, no orchestrator, no heimdall staging-PR
loopback, no slicer — all deleted. Every issue is standalone work built directly on its
own `issue-<N>` branch. A separate **cron lane** trickles the `backlog` (nits the review
gate files) back into the scheduler queue.

- `retinue.config` — `Settings` loaded from env / `.env` (`WEBHOOK_SECRET`, `REDIS_URL`,
  `DEDUPE_DB_PATH`, the GitHub App / Anthropic / push-channel credentials).
- `retinue.webhook` — HMAC verification then event dispatch: a `ready-for-agent` `issues`
  event (action opened/reopened/edited/labeled) kicks a single scheduler drain, and a
  merged `pull_request` (closed + merged) enqueues the reap. Off-target events (an issue
  with neither the trigger label nor a relevant action, any non-merge PR) are acked 204 and
  enqueue nothing. Same 401/204/202 contract for all. The enqueue is awaited before the ack
  so a failed enqueue surfaces as a 5xx (GitHub redelivers) rather than vanishing.
- `retinue.queue` — the `MergedPrJob` / `AdhocDrainJob` models and their `enqueue_merged_pr`
  / `enqueue_adhoc_drain` helpers. The drain kick pins a per-repo `_job_id` so a burst of
  `ready-for-agent` events collapses to one in-flight drain.
- `retinue.api` — the authed `/api/*` read/control surface: every route requires
  `Authorization: Bearer <API_SERVICE_TOKEN>` (constant-time compare, same idiom as the
  webhook's HMAC check), wired as a router-level dependency so new routes are authed by
  construction. `POST /api/drain` enqueues an ad-hoc scheduler drain via the same
  `enqueue_adhoc_drain` path the webhook uses. `GET /api/budget` returns
  `{trailing_24h_spend, cap}` read from a `BudgetLedger` the API process opens
  read-only against the same `BUDGET_DB_PATH` the worker writes (`retinue.app.create_app`)
  — its own `weekly_budget`/`BUDGET_DAILY_CAP_FRACTION` compute `cap()` with no
  dependency on the worker's in-memory governor.
- `retinue.scheduler` — the pure two-queue queue model: each candidate's tier is its
  `priority:<tier>` label matched against `config.severity_tiers`; candidates in the top
  range (`config.priority_tiers`) form the **priority queue**, the rest the **main queue**,
  both ranked by tier then ascending number. `select_to_build` honors a **reserved priority
  slot** — with a parallel-build cap `N ≥ 2` the main queue holds at most `N-1` slots so one
  is always free for priority work; the priority queue always drains first. No `gh`, no I/O.
- `retinue.readiness` — blocked-by readiness: an issue is schedulable only when every
  blocker is closed. Blockers are the **union** of the `## Blocked by #N` refs in the body
  (`parse_body_blockers`) and GitHub's native "blocked by" relations (the `ReadinessGh`
  seam). Only same-repo blockers count; the computation is pure over an injected seam.
- `retinue.adhoc_drain` — `run_adhoc_drain`, the retinue's central mechanism: the single
  drain the webhook kick and the heartbeat both invoke, one per repo under a single-run
  lock, in one stateless pass (see *Scheduler drain* below).
- `retinue.adhoc_build` — `build_adhoc_issue`, the build primitive: plan → materialize →
  implement → done-check → push-on-green → in-session review gate, in one disposable
  container (see *Ad-hoc build + review gate* below).
- `retinue.container_build` — `build_issue_in_container`, the per-issue container-build
  lifecycle: parse the done-check, resolve secrets, start the container, clone + branch off
  the target, exec the implementer, guard against a hollow (zero-commit) implement, run the
  done-check, push only on green, report. The one shared lifecycle the build primitive wraps.
- `retinue.reviewer` — the internal reviewer: the headless Messages-API seam the review gate
  runs after a green push over one `issue-<N>` diff, returning a `ReviewPlan` of
  `ReviewFinding`s each carrying a `Severity`. It never files, wires, or edits anything —
  the gate partitions and acts on its findings.
- `retinue.pipeline` — `Pipeline`, the object the drain drives per built issue:
  `process_adhoc_pr` consumes the build's review-gate outcome (blocking → escalate, no PR;
  backlog → file nits, then PR; clean → PR), plus the `reap_pr` / `reconcile` entry points.
- `retinue.pr_opener` — `open_staging_pr`: once a build pushes a green `issue-<N>` branch,
  opens exactly one PR `issue-<N>` → `config.require_target_branch()`, behind one precheck
  (the target branch must exist); a missing one escalates through `Notifier` and opens no PR.
- `retinue.cron` — `run_cron_tick`, the cron lane's per-tick driver: drains loose `backlog`
  issues one at a time under a single-run lock, gated on the shared budget, picking by a
  weighted priority+age score with an every-Nth-tick quota floor for low-priority items.
  WS1's backlog job is the **trickle promotion** (`GhCliBacklogPromoter`): one `gh issue
  edit` swaps `backlog` for the trigger label so the nit re-enters the scheduler queue.
- `retinue.heartbeat` — `run_heartbeat`, the worker-global arq `cron_jobs` tick. Each tick
  sweeps the opted-in repos: fires the safety-net **scheduler drain** for each repo whose
  `repo_config.cron` cadence is due (catching up issues labeled while the webhook was missed
  or the worker down) and drives the **backlog cron lane** for every repo.
- `retinue.wiring` — the composition root: `bind_adhoc_drain` (the scheduler lane over its
  real gh/readiness/build seams, resolving `target_branch`) and `bind_cron_tick` (the
  backlog cron lane + its promoter). The webhook kick and the heartbeat sweep fire the *same*
  bound drain.
- `retinue.notify` — the reusable `Notifier`: fans one escalation out to a push channel
  (ntfy / Pushover), an issue comment, and a label, through injected sinks.
- `retinue.issues` — `IssueDraft` / `CreatedIssue` / the `IssueCreator` seam
  (`GhCliIssueCreator`): the one `gh issue create` vocabulary every filer shares (the review
  gate's backlog nits, the escalation flows).
- `retinue.handoff` — `reap_merged_pr`: on the human's merge (`pull_request` closed+merged),
  closes the PR's owned issues. "The retinue never merges" — a human merges, the reap reacts.
- `retinue.reconcile` — `RunStateStore` (the durable PR↔issue mapping keyed by issue) plus
  the gh seam it reads truth through, so a later merge webhook resolves a PR back to its issue.
- `retinue.routing` / `retinue.level` / `retinue.classifier` — per-issue model/effort
  routing over the repo's optional `routing:` table: `resolve_level` honors a pre-existing
  `level:` label else classifies (`ClaudeIssueClassifier`, a Haiku-class Messages-API call),
  and each level's `roles:` map overrides the model/effort. A table-less repo makes zero
  classifier calls and builds at the registry defaults.
- `retinue.roles` — the agent-role registry: one `ROLE_REGISTRY` mapping each `Role`
  (implementer / reviewer / resolver / planner / classifier) to its model id,
  reasoning-effort tier, and transport. `resolve_model` / `resolve_effort` are level-aware
  over the routing table.
- `retinue.budget` — `BudgetGovernor` over a service-level weekly budget with a
  per-rolling-24h ceiling (see *Budget governor* below). The scheduler and cron lanes meter
  against one shared ledger.
- `retinue.single_run` — the shared non-blocking single-run lock: admits the first holder
  and *rejects* a second (`CronBusyError`, `AdhocDrainBusyError`), so "at most one run at a
  time" is observable.
- `retinue.done_check` — `run_done_check_commands`, which runs a repo's done-check in a
  fresh container; the command is parsed from the first fenced code block under a
  "Definition of done" heading in the repo's `CLAUDE.md`.
- `retinue.github_app` — `InstallationAuth`, the seam that mints a GitHub App installation
  token the worker clones with.
- `retinue.container` — `ContainerRuntime` / `Container`, the disposable-container seam.
- `retinue.impl_retry` — `ImplRetryStore`, the SQLite-backed per-issue attempt counter.
- `retinue.vocab` — shared labels, `Severity`, and `issue-<N>` branch naming; the bottom
  layer every lane imports.
- `retinue.gh` — the shared `gh` subprocess seams (argv runner, auth env, JSON parse).
- `retinue.app` — FastAPI factory; an Arq Redis pool is created in the lifespan and stored
  on `app.state.arq_pool`.
- `retinue.repo_config` — the per-repo `.github/retinue.yml` schema (`RepoConfig`) and
  `load_repo_config`, which never raises on bad input.
- `retinue.worker` — the `reap_pr_job` + `run_adhoc_drain_job` Arq tasks, the heartbeat, and
  `WorkerSettings`. `on_startup` binds the real collaborators under live GitHub-App auth.

A validly signed `ready-for-agent` `issues` webhook (action opened/reopened/edited/labeled)
returns 202 and enqueues exactly one scheduler-drain kick, which the worker dequeues into
`run_adhoc_drain_job`. An invalid or missing signature returns 401 and enqueues nothing. A
signed issues event without the trigger label (or on any other action) and any other
off-target event are acked 204 without enqueuing.

## Per-repo opt-in (`.github/retinue.yml`)

The worker only acts on a repo that opts in by committing a `.github/retinue.yml`. Its
presence is the opt-in signal: a repo with no file is skipped upstream, a repo whose file is
malformed is skipped by `load_repo_config` (it returns `None` and logs — a single broken
config never crashes the worker). Validation is strict (unknown keys are rejected), so a
typo'd field is a skip, not a silent drop.

Schema (all fields optional; defaults shown):

```yaml
trigger_label: ready-for-agent   # the "build me" label an issue must wear to be scheduled
target_branch: staging           # branch an issue-<N> build is cut from and PR'd against
                                 #   (None → the repo's own default branch, resolved at build)
severity_tiers: [critical, high, medium, low]   # ordered vocabulary, most-severe first
priority_tiers: [critical, high]                # top prefix that routes to the priority queue
retry_cap: 3                     # max retries per unit of work (>= 0)
max_parallel: 4                  # optional concurrent-build cap (> 0); the reserved slot needs >= 2
cron: "0 */6 * * *"              # optional five-field cron cadence for the safety-net sweep
secrets:                         # optional inline secrets + external refs
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  refs:
    - vault://team/retinue/github-token
routing:                         # optional per-issue model/effort routing table
  default: standard
  levels:
    standard:
      description: Ordinary work across a few files with tests.
      roles:
        reviewer: {model: claude-opus-4-8, effort: xhigh}
```

An issue's queue tier is the `priority:<tier>` label whose `<tier>` names one of
`severity_tiers`; an untiered issue ranks last. `priority_tiers` must be a (possibly empty)
top prefix of `severity_tiers` — "which tiers are drop-everything" has to be the top of the
order, or the reserved-slot ranking would contradict itself.

## Scheduler drain

`retinue.adhoc_drain.run_adhoc_drain` is the central mechanism — one drain per repo, under a
single-run lock, in one stateless pass. Each entry recomputes readiness and ranking from
GitHub truth rather than keeping a persistent queue store:

1. **list** every open issue wearing `config.trigger_label` (number, labels, body),
2. **admit** the ones the scheduler acts on — trigger label present, `hitl` absent (a
   human-escalated issue stays out until the `hitl` label is removed, the human "resume"),
3. **gate on readiness** — drop any issue with an open blocker (the union of body
   `## Blocked by #N` refs and native GitHub relations, `retinue.readiness.resolve_ready`),
4. **classify flight state** against GitHub truth (`FlightState`, fetched once per drain as a
   whole-repo `FlightSnapshot`): an issue with an open PR is `IN_FLIGHT` (skip, no duplicate);
   one with a pushed `issue-<N>` branch but no open PR is `STRANDED` (a prior green build
   whose PR never opened — open its PR **without rebuilding**); the rest are buildable,
5. **rank + select** the buildable set through the pure two-queue scheduler
   (`retinue.scheduler.select_to_build`, `cap=config.max_parallel`): priority queue first,
   the main queue held to at most `cap-1` slots (the reserved priority slot),
6. **build** each selected issue in a disposable container, concurrently, each metered
   against the one shared `BudgetGovernor`; a build that would cross the rolling-24h cap is
   skipped, so the shared budget is never overshot.

Every leaf I/O — the gh queries, the readiness lookups, the budget store, the downstream
build — is injected and faked, so the whole drain runs with no real `gh`, no Docker, and no
network.

## Ad-hoc build + review gate

`retinue.adhoc_build.build_adhoc_issue` builds one issue in **one disposable container**
destroyed on every path. There is no integration branch and no merge — the issue is built
directly on an `issue-<N>` branch cut off `config.require_target_branch()`:

1. **clone + branch** — the container clones over the installation token and checks out a
   fresh `issue-<N>` branch off the resolved target branch,
2. **plan** — the read-only planner (Opus on the in-container CLI) maps the code with an
   Explore subagent and emits a plan, captured from its output (it writes nothing),
3. **materialize** — the captured plan is written byte-exact into `.retinue/plan.md`,
4. **implement** — the same implementer, pointed at the plan file via its `plan_path`,
   implements TDD-first and commits to `issue-<N>`; a run that lands **zero commits** fails
   the build (the hollow-implement guard) instead of pushing an empty branch,
5. **done-check** — the repo's done-check runs in the *same* container over the real changes,
6. **push** — only on a green done-check is `issue-<N>` pushed; a red check pushes nothing,
7. **review gate** — after the green push, the in-session gate (`_run_review_gate`) runs the
   internal reviewer (Opus) over the `issue-<N>` diff. On a clean review it is a no-op. On
   findings it runs **one critique-and-fix pass**: the same implementer fixes the flagged
   findings in the same container, the done-check re-runs, and — only if it stays green — the
   branch is re-pushed and the reviewer runs again over the fixed diff.

The surviving findings are **partitioned by severity** into a `ReviewGateOutcome`:

- **blocking** (severity at or above the threshold, default `Severity.HIGH`, or a fix-pass
  regression) → the pipeline escalates the issue through one `Notifier` fan-out (push +
  `hitl` comment + label) and opens **no PR**, leaving the green branch pushed for a human;
- **backlog** (below the threshold) → the pipeline files each as a `backlog` +
  `priority:<severity>` follow-up, then opens the PR.

A fix pass that turns the done-check red is a **regression**: the gate flags it blocking and
does **not** push the red fix, so the branch stays at its green pre-fix pushed state. Every
side-effecting collaborator — the planner, implementer, reviewer, container, auth, secret
resolver, report sink — is injected, so the whole flow is exercised with no Agent SDK, no
Docker, no gh, and no network.

## PR + merge reap

After a green build (clean or backlog gate), `Pipeline.process_adhoc_pr` opens **exactly
one** PR `issue-<N>` → `config.require_target_branch()` via `pr_opener.open_staging_pr`,
behind one precheck (the target branch must exist; a missing one escalates and opens no PR). The
PR↔issue mapping is recorded in `RunStateStore` keyed by the single issue, so the merge
webhook resolves the PR back to its issue.

When the human merges, `retinue.handoff.reap_merged_pr` reacts to the `pull_request`
closed+merged signal and closes the issue(s) the PR owned. **The retinue never merges** — a
human performs the merge, and the reap reacts to it.

## Cron backlog trickle

`retinue.cron.run_cron_tick` is the cron lane's per-tick driver: a scheduled tick drains
loose `backlog` issues **one at a time**, alongside the scheduler drain. Each tick runs under
its own single-run lock (`CronBusyError`), **gates** on the *same shared* service-level
`BudgetGovernor` and defers when the budget is spent, then **picks** the next backlog issue
by a weighted score (priority dominates, age breaks ties) — except on every Nth tick where a
**quota floor** takes the oldest low-priority issue so the low items provably drain rather
than starving behind a steady high-priority stream.

WS1's backlog job is the **trickle promotion**: `GhCliBacklogPromoter` swaps `backlog` for
the repo's `trigger_label` in one `gh issue edit --add-label <trigger> --remove-label
backlog`, so the promoted nit re-enters the scheduler queue and the real build stays with the
scheduler. The clock is injected and the tick counter is passed in, so nothing reads the wall
clock; the gh query, the budget governor, the lock, and the downstream promoter are all
injected and faked.

## Runtime heartbeat

`retinue.heartbeat.run_heartbeat` fires both lanes at runtime. Registered as the
worker-global arq `cron_jobs` tick (`WorkerSettings.cron_jobs`, every Nth minute — the global
tick), each heartbeat sweeps the opted-in repos: it fires the safety-net **scheduler drain**
for each repo whose `repo_config.cron` cadence is due on this tick (`cron_due`, the per-repo
"is this repo due?" filter), catching up issues labeled while the webhook was missed or the
worker was down, and drives the **backlog cron lane** for every repo. A drain or tick that
raises for one repo is logged and skipped so a single bad repo cannot starve the sweep.

In production the worker's `on_startup` binds the collaborators under live GitHub-App auth —
the real wall-clock, the App's installed-and-opted-in repo enumeration, the *same* bound
scheduler drain the webhook kick fires, and a bound `run_cron_tick`. A bare/unauthed worker
leaves them unset and the tick stays a no-op.

## Disposable-container done-check

Each build runs the repo's done-check in a fresh, throwaway container.
`retinue.container_build.build_issue_in_container` orchestrates, in order: **auth** (mint a
GitHub App installation token), **clone + branch**, **inject** (resolve the config's
`secrets` block into the container env — a missing required secret escalates *before* any
container starts, so a doomed check never runs), **implement**, **done-check** (the command
read from the repo's `CLAUDE.md`), **push-on-green**, **report** (to an observable sink), and
**teardown** (guaranteed via `try/finally`, on every path). Auth, the container runtime, the
secret resolver, and the report sink are all injected, so the orchestration is fully
exercised without Docker or network.

## Budget governor

`retinue.budget` meters agent token spend against a **service-level** weekly budget (one
ledger shared across the scheduler and cron lanes) and enforces a per-rolling-24h-window
ceiling — by default 12% of the weekly budget (`cap()`). The `BudgetLedger` is an aiosqlite
spend ledger: `record_spend` appends a timestamped charge, `trailing_24h_spend` sums only the
charges inside the trailing 24h read off an **injected `Clock`** (no wall-clock, so the
window is deterministic in tests).

Metering is auth-aware: an API key meters **dollars** against a weekly-$ budget, subscription
OAuth meters **tokens** against a weekly-token budget — same rolling math, different unit.
`BudgetGovernor` enforces at two points: `gate` **defers** a run whose estimated charge would
start it over the cap; `meter` **pauses** a run whose next charge would cross it. Because the
two lanes meter against one shared ledger file, `meter` records through
`try_record_if_within_cap`, which performs the cap check and the insert inside a single
`BEGIN IMMEDIATE` transaction: a second concurrent lane serializes on the SQLite write lock,
re-reads the updated trailing total, and pauses instead of recording — so two charges that
would jointly cross the cap can never both land.

## Configuration

Set via environment variables or a `.env` file:

| Variable                     | Required | Default                   | Description                                          |
| ---------------------------- | -------- | ------------------------- | ---------------------------------------------------- |
| `WEBHOOK_SECRET`             | yes      | —                         | GitHub webhook HMAC secret                           |
| `API_SERVICE_TOKEN`          | yes      | —                         | Bearer token required on every `/api/*` request      |
| `REDIS_URL`                  | no       | `redis://localhost:6379`  | Redis connection URL                                 |
| `DEDUPE_DB_PATH`             | no       | `retinue-dedupe.sqlite3`  | Locates the worker's durable-state directory (run-state / retry stores) |
| `AUTH_MODE`                  | no       | `api_key`                 | Metering unit: `api_key` (dollars) or `subscription` (tokens) |
| `WEEKLY_BUDGET`              | no       | `0`                       | Service-level weekly budget (dollars or tokens)      |
| `BUDGET_DB_PATH`             | no       | `retinue-budget.sqlite3`  | SQLite file backing the rolling-24h spend ledger     |
| `BUDGET_DAILY_CAP_FRACTION`  | no       | `0.12`                    | Fraction of the weekly budget spendable per 24h      |
| `JOB_TIMEOUT_SECONDS`        | no       | `1800`                    | Arq worker-global job timeout; must outlast a full build |
| `IMPLEMENT_MAX_TURNS`        | no       | `80`                      | Hard cap on the implementer's agent loop (`claude --max-turns`) |
| `GITHUB_APP_ID`              | no\*     | —                         | GitHub App numeric id (the JWT `iss` claim)          |
| `GITHUB_APP_PRIVATE_KEY_PATH`| no\*     | —                         | Path to the App RSA private key (PEM) signing app JWTs |
| `ANTHROPIC_API_KEY`          | no\*     | —                         | Anthropic API key — used in `api_key` auth mode      |
| `CLAUDE_CODE_OAUTH_TOKEN`    | no\*     | —                         | Claude subscription OAuth token (`sk-ant-oat…`) — used in `subscription` auth mode |
| `NTFY_TOPIC`                 | no\*\*   | —                         | ntfy topic for the push sink (ntfy backend)          |
| `NTFY_TOKEN`                 | no       | —                         | ntfy access token for a protected topic              |
| `PUSHOVER_TOKEN`             | no\*\*   | —                         | Pushover application API token (Pushover backend)    |
| `PUSHOVER_USER`              | no\*\*   | —                         | Pushover user/group key (Pushover backend)           |

\* Required once the worker drives the real pipeline: the GitHub App credentials mint the
per-repo tokens the gh adapters use, and the Anthropic credential for the active `AUTH_MODE`
(`ANTHROPIC_API_KEY` for `api_key`, `CLAUDE_CODE_OAUTH_TOKEN` for `subscription`)
authenticates the agent calls. Without GitHub App auth the worker falls back to the safe
not-opted-in default and drives nothing.

\*\* Push channel: configure **either** ntfy (`NTFY_TOPIC`) **or** Pushover (`PUSHOVER_TOKEN`
+ `PUSHOVER_USER`). With neither set the push heads-up is a logged no-op; the issue comment +
label (the durable escalation record) still land.

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
