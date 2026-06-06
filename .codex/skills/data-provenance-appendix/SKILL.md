---
name: data-provenance-appendix
description: Project adapter for capacity SEM provenance, source descriptions, linkage notes, harmonization text, and data-availability statements.
---

# Data Provenance Appendix

Base: `/Users/jesseandrews/.codex/skills/academic-data-provenance-reproducibility/SKILL.md`

Snapshot: `/Volumes/T9/Skill Builder/references/donor-snapshots/Volumes__T9__Projects__capacity-sem-project__.codex__skills__data-provenance-appendix__SKILL.md.md`

Use when: Use for provenance or data-availability work in `manuscript_quarto/` and `manuscript_kaifa_archive/`.

Keep reusable rules in the global skill; this wrapper keeps only local facts.

## Local Context

- `manuscript_quarto/` primary manuscript package
- `manuscript_kaifa_archive/` archived source manuscript
- HUD DRGR / QPR administrative data
- BLS QCEW `NAICS 925110`
- CDC/ATSDR SVI vintages
- Census relationship or gazetteer files
- manual organization-name matching
- minimum-quarter filters, censoring thresholds, and winsorization rules

## Workflow

1. Load the base skill and read the local files named here.
2. Apply the global procedure with these constraints; use the snapshot only for missing historical details.
3. Report checked files, unresolved limits, and project-specific caveats.

## Verification

- Check `references/appendix-checklist.md` before drafting appendix text.
- Distinguish public raw sources, public derived files, restricted derived files, and manual validation artifacts.
- Report match shares, unresolved geography cases, and confidence coding when geography matching matters.

## Portability Notes

Reusable instructions belong in the base skill; project facts belong here.
