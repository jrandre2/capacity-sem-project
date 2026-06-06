---
name: sem-manuscript-audit
description: Project adapter for structural equation modeling manuscript audits.
---

# SEM Manuscript Audit

Base: `/Users/jesseandrews/.codex/skills/academic-methods-manuscript-audit/SKILL.md`

Snapshot: `/Volumes/T9/Skill Builder/references/donor-snapshots/Volumes__T9__Projects__capacity-sem-project__.codex__skills__sem-manuscript-audit__SKILL.md.md`

Use when: Use when auditing `manuscript_quarto/` primary SEM manuscript or `manuscript_kaifa_archive/` archived source draft.

Keep reusable rules in the global skill; this wrapper keeps only local facts.

## Local Context

- `manuscript_quarto/`
- `manuscript_kaifa_archive/`
- latent constructs, loadings, standard errors, residuals, latent correlations, reliability, validity, and respecification logic
- moderation, mediation, causality, invariance, and multi-group claims
- QCEW, SVI, matching crosswalks, and risk indices
- partial dependence plots label risk

## Workflow

1. Load the base skill and read the local files named here.
2. Apply the global procedure with these constraints; use the snapshot only for missing historical details.
3. Report checked files, unresolved limits, and project-specific caveats.

## Verification

- Read `references/audit-checklist.md` for detailed issue prompts.
- Treat additive controls as controls, not moderators.
- Flag cross-sectional claims dressed as longitudinal inference.

## Portability Notes

Reusable instructions belong in the base skill; project facts belong here.
