# Deploy the retinue

The stack is three core containers — a FastAPI **web** receiver, an Arq **worker**, and
**redis** — built from one shared image (`Dockerfile`) and wired by `docker-compose.yml`.
Two optional profiles put a public HTTPS edge in front of `web` so GitHub can reach the
webhook: `tunnel` (local, ephemeral URL) and `edge` (VPS, your own domain + TLS).

This runbook takes you from a fresh GitHub App to a live deployment. Follow it in order.

> **Honest state of the pipeline.** The worker drives the **real, end-to-end** pipeline
> (slice → build → open staging PR, plus the heimdall loopback and merge reap) once
> GitHub App auth is configured: `on_startup` wires the real `fetch_config`, the PRD-body
> fetcher, and a `pipeline_factory` that — per repo — sources the target's `CLAUDE.md`
> (the done-check command) and binds the live orchestrator build lane (the Agent-SDK
> implementer behind the budget gate + triage, the disposable done-check container, and
> the integration-branch merge). An issue event on an opted-in repo now slices the PRD,
> builds the slices, and opens the staging PR — it is no longer a skip.
>
> Without GitHub App credentials the `fetch_config` seam treats every repo as *not opted
> in*, so a delivered event is logged as a SKIP — still proving the webhook auth +
> transport + queue + worker path end to end, but doing no real work.
>
> **The full pipeline spends real money and needs Docker.** The build lane spawns the
> Agent SDK (real Anthropic token spend, metered against `WEEKLY_BUDGET`) and runs the
> done-check inside a disposable container, so the worker needs the **Docker socket**
> mounted and a funded `WEEKLY_BUDGET`. A full live smoke (the #17 end-to-end run) will
> therefore spend real Anthropic tokens and exercise Docker — keep that in mind before
> opening an issue on an opted-in repo.

---

## a. Register the GitHub App

Create a new GitHub App (Settings -> Developer settings -> GitHub Apps -> New).

- **Repository permissions:**
  - Contents: **Read & write**
  - Pull requests: **Read & write**
  - Issues: **Read & write**
  - Metadata: **Read-only**
- **Subscribe to events:** `Issues`, `Pull request review`, `Pull request`.
- **Webhook:** enable it. Set a **webhook secret** (generate a strong random string —
  you will put it in `.env` as `WEBHOOK_SECRET`). Leave the webhook URL blank for now;
  you set it in step (e) once you have a public URL.
- After creating the App: **generate a private key** (downloads a `.pem`) and **record
  the App ID** shown on the App's page.

## b. Create a throwaway test repo and install the App

1. Create a **public** test repository (the App only needs to be installed somewhere you
   can safely open issues).
2. Commit a `.github/retinue.yml` to opt the repo in. The schema is documented in the
   README under **"Per-repo opt-in (`.github/retinue.yml`)"** — all fields are optional,
   so an empty file is a valid opt-in. A minimal example:

   ```yaml
   staging_branch: staging
   retry_cap: 3
   ```

3. Install the App on this repo (App page -> Install App -> pick the test repo).

## c. Provision secrets

```sh
cp .env.example .env
```

- Fill `WEBHOOK_SECRET` with the exact secret you set on the App in step (a).
- Set `WEEKLY_BUDGET` to a real cap before going live (it defaults to `0`).
- Drop the private key from step (a) at the path the worker expects. The intended mount
  is `/secrets/app.pem` (see `GITHUB_APP_PRIVATE_KEY_PATH` in `.env.example`). The
  GitHub-App adapter reads this PEM at startup to mint installation tokens, so the worker
  will not do real work without it.
- Set the Anthropic credential the build lane spends: `ANTHROPIC_API_KEY` (api_key mode)
  or `CLAUDE_CODE_OAUTH_TOKEN` (subscription mode), matching `AUTH_MODE`.

`.env` is **gitignored — NEVER commit it.** Same for the `.pem`.

## d. Notifications

Pick one notification transport (the notify adapter publishes escalations to it):

- **ntfy:** choose a hard-to-guess topic name, subscribe to it in the ntfy app, and set
  `NTFY_TOPIC` in `.env`.
- **Pushover:** set `PUSHOVER_TOKEN` + `PUSHOVER_USER` in `.env`.

## e. Boot locally with a Cloudflare tunnel

```sh
docker compose --profile tunnel up --build
```

Watch the **cloudflared** logs for a line like
`https://<random>.trycloudflare.com` — that is your public URL. In the GitHub App
settings, set the **Webhook URL** to that URL with `/webhook` appended:

```
https://<random>.trycloudflare.com/webhook
```

> The trycloudflare URL is **ephemeral** — it changes every time cloudflared restarts,
> and you must re-paste it into the App. For a stable URL, switch to a named tunnel
> (`CLOUDFLARE_TUNNEL_TOKEN`) or use the VPS `edge` profile (step g).

## f. Verify the transport (and, on an opted-in repo, the pipeline)

1. Open an issue on the test repo.
2. GitHub delivers the webhook; **web** verifies the HMAC and returns **202**, enqueuing
   the event (check the App's webhook "Recent Deliveries" tab for the 202).
3. The **worker** logs that it received the event.

What happens next depends on the repo's opt-in:

- **No `.github/retinue.yml`** (or no GitHub App auth): the worker logs a **"not opted in"
  SKIP** — `fetch_config` returns `None`, nothing downstream runs. This still confirms
  auth + transport + queue + worker are wired, doing no real work. Use this as the safe
  transport-only check.
- **Opted in** (the `.github/retinue.yml` from step (b)): the worker drives the **real
  pipeline** — it slices the PRD into issues, runs the build lane (Agent-SDK implementer
  → done-check container → integration-branch merge), and opens the staging PR. This is
  the full #17 smoke: it **spends real Anthropic tokens** (metered against
  `WEEKLY_BUDGET`) and **requires the Docker socket** mounted for the done-check
  container. Only open an issue on an opted-in repo once you are ready for that.

## g. Deploy to a VPS (edge profile)

1. Point your domain's DNS **A/AAAA record** at the VPS host.
2. Put `DOMAIN=your.domain` in `.env`.
3. Bring the stack up with the edge profile:

   ```sh
   docker compose --profile edge up -d --build
   ```

   Caddy obtains a Let's-Encrypt certificate for `${DOMAIN}` automatically and
   reverse-proxies `:443` to `web:8000`.
4. Set the GitHub App **Webhook URL** to `https://${DOMAIN}/webhook`.

Re-run the step (f) verification against the live domain.

---

## State persistence note

The worker's SQLite state — the dedupe ledger (`DEDUPE_DB_PATH`) and the budget ledger
(`BUDGET_DB_PATH`) — is pinned to `/data` on the **worker-data** named volume so it
survives restarts. The pipeline's own durable stores (the heimdall round counter, the
per-slice implementer-retry counter, and the per-PRD run-state) are **co-located next to
the dedupe DB automatically** — the factory derives their directory from
`DEDUPE_DB_PATH`'s parent — so keeping `DEDUPE_DB_PATH` under `/data` lands all of them on
`worker-data` with no extra `Settings` fields to configure.
