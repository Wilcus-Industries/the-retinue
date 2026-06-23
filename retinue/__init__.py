"""The Retinue: a signed-webhook transport spine over an Arq/Redis queue.

After a PRD round merges, :mod:`retinue.reviewer` reviews the round's diff and files
``review-fix`` follow-up issues wired into dependents' ``## Blocked by``.

Every agent role (slicer, implementer, conflict resolver, internal reviewer) resolves
its model and reasoning-effort tier from the :mod:`retinue.roles` registry, so a repo's
``models`` override and the per-role tiers live in one table rather than scattered
constants.
"""
