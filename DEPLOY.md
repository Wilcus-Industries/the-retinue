# Deploy the retinue

The stack is three core containers — a FastAPI **web** receiver, an Arq **worker**, and
**redis** — built from one shared image (`Dockerfile`) and wired by `docker-compose.yml`.
Two optional profiles put a public HTTPS edge in front of `web` so GitHub can reach the
webhook: `tunnel` (local, ephemeral URL) and `edge` (VPS, your own domain + TLS).

This runbook takes you from a fresh GitHub App to a live deployment. Follow it in order.

> **Honest state of the pipeline.** The worker drives the real pipeline (slice → build →
> open staging PR, plus the heimdall loopback and merge reap) once GitHub App auth is
> configured: `on_startup` then wires the real `fetch_config`, the PRD-body fetcher, and
> the `pipeline_factory`. Without GitHub App credentials the `fetch_config` seam treats
> every repo as *not opted in*, so a delivered event is logged as a SKIP — still proving
> the webhook auth + transport + queue + worker path end to end. The one remaining
> injected seam is the orchestrator `build_prd` (the implementer-spawn adapter), bound via
> `retinue.wiring.bind_build_prd` once that layer lands; until then the build step is the
> only part that is not yet executable.

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
  is `/secrets/app.pem` (see `GITHUB_APP_PRIVATE_KEY_PATH` in `.env.example`); that mount
  is wired in the Build phase when the GitHub-App adapter lands.

`.env` is **gitignored — NEVER commit it.** Same for the `.pem`.

## d. Notifications

Pick one notification transport (consumed once the notify adapter lands):

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

## f. Verify the transport

1. Open an issue on the test repo.
2. GitHub delivers the webhook; **web** verifies the HMAC and returns **202**, enqueuing
   the event (check the App's webhook "Recent Deliveries" tab for the 202).
3. The **worker** logs that it received the event.

As noted above, the worker currently logs a **"not opted in" SKIP** because
`fetch_config` returns `None` until the adapters land — that is the expected result and
still confirms auth + transport + queue + worker are all wired correctly.

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
survives restarts. As the orchestrator, loopback, and reconcile seams are wired, **their
SQLite paths must ALSO land on `worker-data`** (`/data/...`), and the corresponding
`Settings` fields get added in the Build phase that implements each adapter.
