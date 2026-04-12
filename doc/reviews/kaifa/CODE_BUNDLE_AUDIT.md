# Kaifa Recovered Code Bundle Audit

**Date**: 2026-04-09  
**Scope**: `sem_output_and_partial_dependence_plots.zip`

## Bundle Contents

The recovered ZIP bundle imported into
[`manuscript_kaifa_archive/data/recovered_code_bundle/`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/data/recovered_code_bundle/README.md)
contains four relevant artifacts:

- [`SEM_model_latest_all_funds_latent_all_year_v2_final.ipynb`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/data/recovered_code_bundle/extracted/SEM_model_latest_all_funds_latent_all_year_v2_final.ipynb)
- [`sem_partial_dependence_outputs_4_v2_final.ipynb`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/data/recovered_code_bundle/extracted/sem_partial_dependence_outputs_4_v2_final.ipynb)
- [`all_state_local_fund_latent_var_4_v2.csv`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/data/recovered_code_bundle/extracted/all_state_local_fund_latent_var_4_v2.csv)
- [`SEM_Manuscript_2026-04-09_code_bundle.docx`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/source_docs/SEM_Manuscript_2026-04-09_code_bundle.docx)

## What This Resolves

The core SEM notebook is now recovered in source form. The repository no
longer relies only on an inferred SEM reconstruction for the later Kaifa
manuscript. The notebook specifies:

- the state/local classification rule
- the staff-scaled workload variables
- the one-factor and two-factor SEM equations
- the standardized SEM-ready dataset structure
- the expected output file family

This is enough to build a notebook-backed analysis stage in the repo.

## What Remains Inconsistent

The recovered bundle is not internally self-consistent.

- The saved SEM notebook output reports a `573 x 19` raw dataset and `573`
  SEM-ready rows.
- The recovered CSV currently bundled in the ZIP has `577 x 19` rows.
- Re-running the recovered notebook logic on the bundled CSV reproduces the
  model structure but not the exact saved coefficient set.

Practically, that means the bundle appears to contain:

1. recovered notebook source code,
2. a later or adjacent raw CSV snapshot,
3. saved notebook outputs or manuscript tables derived from a slightly
   different input version.

## Identified 573-Row Sample Membership

The missing `573`-row SEM input can now be narrowed to a specific sample
membership difference.

Relative to the recovered `577`-row CSV, the `573`-row SEM sample excludes:

- `Collier County, FL`
- `KY`
- `Leon County, FL`
- `Nash County, NC`

Dropping those four rows reproduces the reference `573`-row grantee roster
exactly and also reproduces most SEM-ready columns exactly.

Columns that match the reference SEM-ready dataset after dropping those four
grantees:

- `state_level`
- `z_avg_employment`
- `z_avg_payroll`
- `z_rev_programs_per_staff`
- `z_rev_disasters_per_staff`
- `z_Ratio_Program_Completed`
- `z_rev_Average_Duration_Program_Completion`
- `z_E_TOTPOP`
- `z_SPL_THEME1`
- `z_SPL_THEME2`
- `z_SPL_THEME3`
- `z_SPL_THEME4`

Columns that still drift after dropping those four grantees:

- `z_Ratio_disbursed_to_obligated`
- `z_Ratio_expended_to_disbursed`
- `z_Ratio_obligated_funds_fully_expended`
- `z_rev_Duration_of_completion`

This means the missing input version is not just a sample filter. It also
contains an earlier version of several outcome fields.

Targeted searches across the local Kaifa working directories did not identify
an additional raw CSV carrying that exact 573-row outcome version. So the
sample membership is now identified, but the exact earlier raw outcome file is
still unrecovered.

## Core SEM Status

For manuscript purposes, the later Kaifa SEM is now best understood as:

- **Primary executed reference outputs**:
  [`manuscript_kaifa_archive/data/external_sem_exports/baseline_sem/`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/data/external_sem_exports/baseline_sem/sem_model_comparison.csv)
- **Recovered source logic**:
  [`SEM_model_latest_all_funds_latent_all_year_v2_final.ipynb`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/data/recovered_code_bundle/extracted/SEM_model_latest_all_funds_latent_all_year_v2_final.ipynb)
- **Current rerunnable raw input**:
  [`all_state_local_fund_latent_var_4_v2.csv`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/data/recovered_code_bundle/extracted/all_state_local_fund_latent_var_4_v2.csv)

The repository command that operationalizes this is:

```bash
python src/pipeline.py run_kaifa_recovered_analysis
```

That workflow writes a discrepancy audit under
[`data_work/diagnostics/kaifa_recovered_analysis/`](/Volumes/T9/Projects/capacity-sem-project/data_work/diagnostics/kaifa_recovered_analysis/README.md).

The best current approximation to the missing raw input is written there as:

- [`recovered_notebook_candidate_573_raw_subset.csv`](/Volumes/T9/Projects/capacity-sem-project/data_work/diagnostics/kaifa_recovered_analysis/recovered_notebook_candidate_573_raw_subset.csv)
- [`recovered_notebook_candidate_573_alignment_summary.csv`](/Volumes/T9/Projects/capacity-sem-project/data_work/diagnostics/kaifa_recovered_analysis/recovered_notebook_candidate_573_alignment_summary.csv)

## Partial-Dependence / Spatial Notebook Status

The recovered partial-dependence notebook is useful for figure lineage, but it
is not yet manuscript-grade as a reproducible workflow.

Problems:

- It uses `Num_Program / avg_employment` and `Num_Disaster / avg_employment`
  in places where the recovered SEM notebook uses
  `Num_Program / (avg_employment * E_TOTPOP)` and
  `Num_Disaster / (avg_employment * E_TOTPOP)`.
- It depends on shapefiles not present in the recovered ZIP or current repo:
  `cb_2018_us_county_500k.zip` and `cb_2018_us_state_500k.zip`.
- It blends SEM-implied plots with descriptive mapping and heuristic risk/gap
  indices that remain more weakly audited than the core SEM.

Implication:

- keep the core SEM analysis in the main Kaifa evidence chain
- treat the partial-dependence and spatial notebook as secondary figure lineage
  until its inputs and denominators are aligned

## Manuscript Implication

The recovered code bundle strengthens the manuscript’s reproducibility claim
for the core SEM, but it does not fully close the workflow.

- The SEM section can now be tied to recovered source code plus reference
  outputs.
- The exact `573`-row input used in the saved notebook execution is still not
  perfectly recovered.
- The spatial/gap/risk sections remain exploratory unless their workflow is
  reconstructed to the same standard as the core SEM.
