# Pipeline Guide

This document describes the workflows that are actually current in this repository.

## Canonical Workflow

The active analysis path is the standardized survival pipeline:

```bash
python src/pipeline.py run_all
```

Equivalent stage-by-stage execution:

```bash
python src/pipeline.py ingest_data
python src/pipeline.py standardize_data
python src/pipeline.py build_panel
python src/pipeline.py build_features_std
python src/pipeline.py aggregate_program_types
python src/pipeline.py run_survival
```

`run_all` is the current default because it follows the quarter-corrected workflow. `run_all_legacy` exists only for SEM replication and comparison.

## Stage Reference

### `ingest_data`

Purpose:
- load QPR and external data
- clean QPR fields
- build quarterly rollups
- emit QA reports

Primary outputs:
- `data_work/qpr_raw.parquet`
- `data_work/qpr_clean.parquet`
- `data_work/qpr_quarterly.parquet`
- `data_work/quality/qpr_quality_report.csv`
- `data_work/quality/qpr_quarterly_quality_report.csv`

### `standardize_data`

Purpose:
- apply fixed denominators
- create monotonic clean series
- compute standardized ratios and winsorized velocity

Primary outputs:
- `data_work/qpr_standardized.parquet`
- `data_work/quality/qpr_standardized_report.csv`

### `build_panel`

Purpose:
- merge QPR with external covariates
- create the grantee-disaster analysis panel

Primary output:
- `data_work/panel.parquet`

### `build_features_std`

Purpose:
- collapse standardized activity rows to quarter-aware grantee-disaster features
- compute durations, survival covariates, velocity summaries, and interaction terms

Primary output:
- `data_work/panel_features_std.parquet`

### `aggregate_program_types`

Purpose:
- aggregate activity types into program portfolio features

Primary output:
- `data_work/panel_program_types.parquet`

### `run_survival`

Purpose:
- build time-varying survival data
- fit corrected Cox models
- write diagnostics and survival figures

Primary outputs:
- `data_work/diagnostics/survival_time_varying_cox_results.csv`
- `data_work/diagnostics/survival_hazard_ratios.csv`
- `data_work/diagnostics/survival_bootstrap_se.csv`
- `data_work/diagnostics/survival_robustness_checks.csv`

## Supporting Analysis Commands

```bash
python src/pipeline.py run_survival_threshold_sensitivity
python src/pipeline.py run_alternatives
python src/pipeline.py make_figures
python src/pipeline.py capacity_summary
```

Use these after the canonical pipeline when you need:
- threshold-sensitivity diagnostics
- alternative capacity specifications
- publication figures
- multiple-testing corrected summary outputs

## Legacy SEM Workflow

The SEM path is retained for replication, sensitivity checks, and historical comparison.

```bash
python src/pipeline.py run_all_legacy
python src/pipeline.py run_estimation --model exp_optimal_v1 --subset all
python src/pipeline.py run_robustness
python src/pipeline.py list_models
```

Important:
- legacy SEM outputs should not be treated as the primary analysis
- `compute_features` belongs to the legacy path
- many historical SEM interpretations predate the corrected duration logic

## Review Workflow

Review management now runs on the unified CENTAUR review engine. The host
`review_*` commands are the project-facing entrypoint; they default to the
velocity manuscript and also support Kaifa's archived DOCX workflow.

```bash
python src/pipeline.py review_status --manuscript velocity
python src/pipeline.py review_new --manuscript velocity --focus par_general
python src/pipeline.py review_diff --manuscript velocity
python src/pipeline.py review_response --manuscript velocity
python src/pipeline.py review_verify --manuscript velocity
python src/pipeline.py review_archive --manuscript velocity
python src/pipeline.py review_report
python src/pipeline.py review_ingest_docx --manuscript kaifa
```

Current status is tracked in:
- `manuscript_velocity/REVISION_TRACKER.md`
- `manuscript_kaifa_archive/REVISION_TRACKER.md`
- `doc/MANUSCRIPT_REVISION_CHECKLIST.md`
- `doc/reviews/`

## CENTAUR Workflow

Vendored CENTAUR tooling is exposed separately:

```bash
python src/pipeline.py centaur --help
python src/pipeline.py centaur list_stages
python src/pipeline.py centaur validate_submission --journal jeem
python src/pipeline.py centaur review_status
```

Use CENTAUR for:
- project analysis and migration planning
- the vendored manuscript scaffold in `manuscript_quarto/`
- journal validation, review tooling, and draft-generation experiments

Do not confuse this with the active Capacity-SEM pipeline in `src/stages/`.

## What Is Canonical

For current analysis and documentation:
- canonical data path: `qpr_quarterly.parquet` → `qpr_standardized.parquet` → `panel_features_std.parquet`
- canonical manuscript: `manuscript_velocity/`
- canonical findings status: see [PROJECT_STATUS.md](PROJECT_STATUS.md)

Anything that still depends on pre-fix positive velocity claims should be treated as historical until updated.
