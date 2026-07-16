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

## Heimdall staging-PR review loop

Every `retinue/prd-<n>` → staging PR is reviewed by heimdall; the retinue **awaits and
acts on** that verdict (`retinue/loopback.py`):

- **Passed** (APPROVED, or a verdict-carrying COMMENT — heimdall never approves: its
  clean pass is the "no concerns" COMMENT, nits-only reviews are COMMENTED too): findings
  are filed as `backlog` + `priority:<severity>` issues, then the PR proceeds to
  reap/handoff. A verdict-less COMMENT (heimdall's "review failed" note) is ignored.
- **Changes requested**: each blocking finding becomes a fix-issue (`ready-for-agent`,
  `Part of #<prd>`) rebuilt onto the **same** PR branch, re-triggering heimdall review —
  looped up to `retry_cap` rounds (count persisted, survives a worker restart).
- **Round cap exhausted while still blocked**: escalate — comment the PRD, apply `hitl`,
  push-notify, and leave the PR open for a human.

## Keep these docs current

Treat `CLAUDE.md` and `STYLEGUIDE.md` as living docs, not write-once boilerplate. As part
of a change that makes a rule here stale, wrong, or redundant, prune or rewrite it in the
same change; add a rule when a real, recurring need shows up. Keep it tight — fewer,
sharper lines beat an accreting pile. (This project directive overrides the global
ask-first-before-editing-docs rule for these two files.)
