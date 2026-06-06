---
name: manuscript-review-triage
description: Project adapter for capacity SEM review intake, tracker maintenance, and review artifact refresh.
---

# Manuscript Review Triage

Base: `/Users/jesseandrews/.codex/skills/academic-review-tracker-triage/SKILL.md`

Snapshot: `/Volumes/T9/Skill Builder/references/donor-snapshots/Volumes__T9__Projects__capacity-sem-project__.codex__skills__manuscript-review-triage__SKILL.md.md`

Use when: Use for review intake in the primary `manuscript_quarto/` manuscript and archived `velocity` or `kaifa` manuscripts.

Keep reusable rules in the global skill; this wrapper keeps only local facts.

## Local Context

- `python src/pipeline.py review_status --manuscript <key>`
- `quarto` primary SEM manuscript targeting PAR
- `velocity` archived null-results draft
- `kaifa` archived source SEM draft
- `REVISION_TRACKER.md` parseable tracker
- `doc/reviews/<manuscript>/REVISION_PLAN.md`
- `python src/pipeline.py review_response --manuscript <key>`
- `python src/pipeline.py review_verify --manuscript <key>`
- `VALID - ACTION NEEDED`, `ALREADY ADDRESSED`, `BEYOND SCOPE`, `INVALID`, and `PARTIALLY VALID`

## Workflow

1. Load the base skill and read the local files named here.
2. Apply the global procedure with these constraints; use the snapshot only for missing historical details.
3. Report checked files, unresolved limits, and project-specific caveats.

## Verification

- Read `references/tracker-format.md` before editing a tracker by hand.
- Keep `source_type` truthful: synthetic for AI/internal reviews and actual for real external reviews.
- Preserve parseability for `review_response` and `review_verify`.

## Portability Notes

Reusable instructions belong in the base skill; project facts belong here.
