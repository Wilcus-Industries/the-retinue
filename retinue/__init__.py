"""The Retinue: a signed-webhook transport spine over an Arq/Redis queue.

After a PRD round merges, :mod:`retinue.reviewer` reviews the round's diff and files
``review-fix`` follow-up issues wired into dependents' ``## Blocked by``.
"""
