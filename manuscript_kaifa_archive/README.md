# Kaifa's Original Manuscript (Archived)

**Archived:** December 26, 2024

This directory contains the original Kaifa SEM manuscript lineage, the
2026 audit-informed exploratory redraft, and a standalone Word revision
built in the original article format.

## Archived Source Files

- Historical Quarto render output: [`_output/index.docx`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/_output/index.docx)
- Updated Word manuscript received on April 9, 2026:
  [`source_docs/SEM_Manuscript_2026-04-07.docx`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/source_docs/SEM_Manuscript_2026-04-07.docx)
- Current standalone Word revision used for review:
  [`source_docs/SEM_Manuscript_2026-04-09_full_revision.docx`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/source_docs/SEM_Manuscript_2026-04-09_full_revision.docx)
- Recovered later-manuscript DOCX bundled with Kaifa's SEM code ZIP:
  [`source_docs/SEM_Manuscript_2026-04-09_code_bundle.docx`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/source_docs/SEM_Manuscript_2026-04-09_code_bundle.docx)
- Audit-informed Quarto redraft:
  [`index.qmd`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/index.qmd)
- Quantitative audit memo:
  [`doc/reviews/kaifa/QUANTITATIVE_AUDIT.md`](/Volumes/T9/Projects/capacity-sem-project/doc/reviews/kaifa/QUANTITATIVE_AUDIT.md)
- Recovered code-bundle audit:
  [`doc/reviews/kaifa/CODE_BUNDLE_AUDIT.md`](/Volumes/T9/Projects/capacity-sem-project/doc/reviews/kaifa/CODE_BUNDLE_AUDIT.md)
- Imported later-manuscript SEM export bundles:
  [`data/external_sem_exports/README.md`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/data/external_sem_exports/README.md)
- Recovered Kaifa notebook bundle:
  [`data/recovered_code_bundle/README.md`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/data/recovered_code_bundle/README.md)

The updated Word file is preserved as source material for the project. It
should not be confused with the active velocity manuscript in
`manuscript_velocity/`, and it does not replace the current canonical
analysis workflow documented in `doc/PROJECT_STATUS.md`.

## Review Workflow

This archive is now attached to the unified project review engine.

```bash
python src/pipeline.py review_ingest_docx --manuscript kaifa
python src/pipeline.py review_status --manuscript kaifa
python src/pipeline.py review_response --manuscript kaifa
python src/pipeline.py review_diff --manuscript kaifa
```

The original `SEM_Manuscript_2026-04-07.docx` file does not contain embedded
Word comments or tracked changes, so the review tracker began as a manual
AI-review scaffold and provenance record. The current review target is the
standalone full revision DOCX listed above.

## Current Status

The current Kaifa package is an archival SEM workspace, not the active
project manuscript. As of April 9, 2026:

- the later Word manuscript has been triaged through the unified review
  system;
- the Quarto source has been rewritten as an exploratory redraft that aligns
  with the archived robustness tables;
- the recovered SEM notebook ZIP has been imported and ported into a repo
  analysis stage, so the later Kaifa SEM now has recovered source logic in
  addition to saved exports;
- the useful external SEM bundles from the later manuscript workflow have
  been imported into `data/external_sem_exports/` for provenance and future
  re-audit;
- the quantitative audit concludes that the archive cannot support a
  confirmatory SEM paper with the stronger state/local effect claims in the
  Word draft.

## Methodological Issues Identified

See [`doc/reviews/kaifa/QUANTITATIVE_AUDIT.md`](/Volumes/T9/Projects/capacity-sem-project/doc/reviews/kaifa/QUANTITATIVE_AUDIT.md)
and
[`doc/archive/ANALYSIS_COMPARISON_REPORT.md`](/Volumes/T9/Projects/capacity-sem-project/doc/archive/ANALYSIS_COMPARISON_REPORT.md)
for detailed comparison.

1. **Right-censoring:** 73.7% of observations lack valid Duration at 95% threshold
2. **Mathematical circularity:** Timeliness = 1/Duration as capacity indicator creates artificial coupling with Duration outcome
3. **Grantee-level aggregation:** Averaging across disasters reduces variance and may inflate effects

## Why This Was Archived

The repository's active analytical workflow moved away from the original SEM
framing because later audit work showed that the strongest SEM result was
design-dependent and that the standardized pipeline now centers on
quarter-corrected survival analysis. The Kaifa manuscript remains valuable
as a methodological lineage document and an exploratory archival redraft, but
it is not the repository's authoritative empirical workflow.

## Notes On The Recovered 2026 SEM Bundle

The later Kaifa materials now include both a bundled DOCX and the recovered
SEM notebook ZIP. Together they show that the later manuscript was built
around a larger cross-sectional SEM workflow than the older archived draft.

Useful recovered assets:

- `SEM_model_latest_all_funds_latent_all_year_v2_final.ipynb`
- `sem_partial_dependence_outputs_4_v2_final.ipynb`
- `all_state_local_fund_latent_var_4_v2.csv`
- `SEM_Manuscript_final.docx`

The imported `baseline_sem/` export bundle remains important because it matches
the saved 573-row notebook outputs more closely than the currently bundled raw
CSV. The recovered code bundle therefore improves source transparency but also
reveals an input-version mismatch that still needs to be tracked explicitly.

That mismatch is now partially resolved: the reference 573-row SEM sample can
be recreated by excluding four grantees from the recovered 577-row CSV
(`Collier County, FL`, `KY`, `Leon County, FL`, and `Nash County, NC`), but a
few outcome columns still reflect an earlier unrecovered input version. See
[`doc/reviews/kaifa/CODE_BUNDLE_AUDIT.md`](/Volumes/T9/Projects/capacity-sem-project/doc/reviews/kaifa/CODE_BUNDLE_AUDIT.md)
for the current reconstruction status.

The smaller `baseline_sem_admin_4/` bundle remains a filtered robustness
version of the same expanded SEM family.

## Kaifa Analysis Commands

The repository now exposes two Kaifa SEM support paths:

```bash
python src/pipeline.py run_kaifa_recovered_analysis
python src/pipeline.py run_kaifa_external_replication --bundle baseline_sem
python src/pipeline.py run_kaifa_external_replication --bundle baseline_sem_admin_4
```

`run_kaifa_recovered_analysis` is the primary recovered-code workflow for the
later manuscript. It ports the recovered notebook logic, reruns it on the
imported CSV, and writes a discrepancy audit under
`data_work/diagnostics/kaifa_recovered_analysis/`.

## Provisional External Reconstruction

The repository now includes a temporary SEM reconstruction stage for these
imported bundles:

```bash
python src/pipeline.py run_kaifa_external_replication --bundle baseline_sem
python src/pipeline.py run_kaifa_external_replication --bundle baseline_sem_admin_4
```

This workflow is intentionally labeled as a `PROVISIONAL_EXTERNAL_RECONSTRUCTION`.
It rebuilds later-manuscript SEM tables from the imported ready datasets and
saved exports. It is not Kaifa's original notebook or script, and its outputs
are written with an `external_reconstruction_*` prefix under
`data_work/diagnostics/kaifa_external_replication/` to keep them distinct from
original or canonical project outputs.

It remains useful for provenance and comparison, but the recovered notebook
port is now the primary Kaifa analysis workflow in this archive.
