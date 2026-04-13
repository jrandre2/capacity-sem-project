---
name: data-provenance-appendix
description: Draft and audit data-provenance appendices, source descriptions, linkage notes, harmonization text, and data-availability statements for this repository. Use when Codex needs to document QPR, QCEW, SVI, geography matching, filtering, aggregation, public-versus-restricted materials, or the reproducibility chain from raw sources to manuscript-ready analysis files.
---

# Data Provenance Appendix

Use this skill when writing or auditing data and methods appendix text for this repository, especially for `manuscript_quarto/` (primary) and `manuscript_kaifa_archive/` (archived source).

## Workflow

1. Build a source inventory.
- List each raw source, its version or vintage, time coverage, geography, unit of observation, and acquisition method.
- Distinguish public raw sources, public derived files, restricted derived files, and manual validation artifacts.

2. Document the transformation chain.
- Describe extraction, cleaning, matching, filtering, aggregation, and derived-variable construction in order.
- State thresholds and exclusions explicitly, including minimum-quarter filters, censoring thresholds, and winsorization rules.

3. Document source-specific caveats.
- For QPR or DRGR data, explain quarter definitions, activity or program granularity, and maturity limitations.
- For QCEW, explain NAICS selection, suppression or missingness handling, and why the series is treated as a proxy.
- For SVI, explain how vintages were aligned or why comparisons across releases remain valid.
- For geography matching, prefer official Census relationship or gazetteer files over generic spreadsheets unless there is a documented reason otherwise.

4. Validate the geography layer.
- Report match shares by method, unresolved cases, and any manual audit or confidence coding.
- State clearly when city-level entities are collapsed to counties or states.

5. Write the data-availability statement.
- Separate public source files from derived linkages, manual validation tables, and any restricted artifacts.
- Promise code, formulas, and nonrestricted derivatives when the raw sources are public but some derived files are controlled.

6. Keep the prose audit-ready.
- Prefer exact filenames, source agencies, release years, and transformation steps over vague summary language.
- Treat proxy validity and harmonization limits as first-class methodological issues, not footnotes.

## Repo-Specific Focus

Pay special attention to:

- HUD DRGR / QPR administrative data
- BLS QCEW `NAICS 925110`
- CDC/ATSDR SVI vintages
- Census relationship or gazetteer files
- manual organization-name matching

## References

- Read `references/appendix-checklist.md` before drafting or revising appendix text.
