# CLAUDE.md

> **Read first:** [`STYLEGUIDE.md`](./STYLEGUIDE.md) — code conventions, follow on every change.

Project-specific rules only. The universal working rules — boundaries, when-stuck,
secrets, done-honesty — live in the global `~/.claude/CLAUDE.md` and apply underneath
this. Only add a rule here when it differs from, or isn't covered by, the global.

## Definition of done

The project's full check — the global done-rule points here for the exact commands. All
must pass before any task is "done":

```
uv run pytest
uv run ruff check .
uv run mypy .
```

## The scheduler + in-session review gate

There is **one** unified scheduler lane (no PRD lane, no heimdall loopback — both deleted).
The scheduler drain (`retinue/adhoc_drain.py`) lists `trigger_label` issues, gates each on
blocked-by readiness (`retinue/readiness.py`), and ranks the ready set by the two-queue
severity + reserved-priority-slot scheduler (`retinue/scheduler.py`). Each admitted issue
builds directly on its own `issue-<N>` branch in one disposable container
(`retinue/adhoc_build.py`): plan → implement → done-check → push-on-green.

After the green push, the **in-session review gate** (`_run_review_gate`, `adhoc_build.py`)
runs the internal reviewer (`retinue/reviewer.py`, Opus) over the `issue-<N>` diff, then:

- **findings** → one critique-and-fix pass by the same implementer in the same container,
  the done-check re-runs, and — only if still green — the branch is re-pushed and the
  reviewer runs again. A fix pass that turns the done-check red is a **regression**
  (blocking; the red fix is not pushed).
- surviving findings are **partitioned by severity** (`Severity.HIGH` threshold):
  - **blocking** (≥ HIGH, or a regression) → escalate: one `hitl` notification (push +
    comment + label), **no PR** (green branch left pushed for a human);
  - **backlog** (< HIGH) → filed as `backlog` + `priority:<severity>` issues, then the PR
    opens.

The **cron lane** (`retinue/cron.py`) trickles the backlog back in: each tick promotes the
top-severity `backlog` issue into the scheduler queue by label surgery (add `trigger_label`,
remove `backlog`) — the real build stays with the scheduler.

## Keep these docs current

Treat `CLAUDE.md` and `STYLEGUIDE.md` as living docs, not write-once boilerplate. As part
of a change that makes a rule here stale, wrong, or redundant, prune or rewrite it in the
same change; add a rule when a real, recurring need shows up. Keep it tight — fewer,
sharper lines beat an accreting pile. (This project directive overrides the global
ask-first-before-editing-docs rule for these two files.)
