---
name: docx-quarto-manuscript-bridge
description: Project adapter for capacity SEM DOCX and Quarto manuscript conversion.
---

# DOCX Quarto Manuscript Bridge

Base: `/Users/jesseandrews/.codex/skills/academic-docx-quarto-manuscript-bridge/SKILL.md`

Snapshot: `/Volumes/T9/Skill Builder/references/donor-snapshots/Volumes__T9__Projects__capacity-sem-project__.codex__skills__docx-quarto-manuscript-bridge__SKILL.md.md`

Use when: Use for manuscript conversion and formatting-bridge work under `manuscript_quarto/` and `manuscript_kaifa_archive/`.

Keep reusable rules in the global skill; this wrapper keeps only local facts.

## Local Context

- `manuscript_quarto/` primary manuscript
- `manuscript_kaifa_archive/` archived source
- `DOCX -> Quarto`
- `Quarto -> DOCX style sync`
- `index.qmd` plus appendices
- `manuscript_kaifa_archive/code/postprocess_word_format.py`
- journal-facing labels: `Abstract:`, `Keywords:`, and `Data Availability Statement`

## Workflow

1. Load the base skill and read the local files named here.
2. Apply the global procedure with these constraints; use the snapshot only for missing historical details.
3. Report checked files, unresolved limits, and project-specific caveats.

## Verification

- Read `references/conversion-checklist.md` before conversion.
- Use `python-docx` to inspect paragraph order, front matter, headings, table count, and references.
- Inspect rendered DOCX/PDF artifacts before closing conversion work.

## Portability Notes

Reusable instructions belong in the base skill; project facts belong here.
