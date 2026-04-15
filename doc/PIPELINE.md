# Pipeline Guide

This document describes the workflows that are actually current in this repository.

## Canonical Workflow

The active data pipeline feeds both the primary SEM analysis (`manuscript_quarto/`) and the complementary survival analysis:

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

## SEM Workflow (Primary Analysis)

The cross-sectional SEM is the primary methodology for the manuscript (`manuscript_quarto/`). The SEM estimation commands use the same data pipeline outputs.

```bash
python src/pipeline.py run_estimation --model exp_optimal_v1 --subset all
python src/pipeline.py run_robustness
python src/pipeline.py list_models
```

The legacy end-to-end flow is also available for replication:

```bash
python src/pipeline.py run_all_legacy
```

Note:
- `compute_features` belongs to the legacy path
- many historical SEM interpretations predate the corrected duration logic

## Review Workflow

Review management now runs on the unified CENTAUR review engine. The host
`review_*` commands are the project-facing entrypoint; they default to the
quarto manuscript and also support Kaifa's archived DOCX workflow.

```bash
python src/pipeline.py review_status --manuscript quarto
python src/pipeline.py review_new --manuscript quarto --focus par_general
python src/pipeline.py review_diff --manuscript quarto
python src/pipeline.py review_response --manuscript quarto
python src/pipeline.py review_verify --manuscript quarto
python src/pipeline.py review_archive --manuscript quarto
python src/pipeline.py review_report
python src/pipeline.py review_ingest_docx --manuscript kaifa
```

Current status is tracked in:
- `manuscript_quarto/REVISION_TRACKER.md`
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
- the manuscript scaffold and CENTAUR tooling in `manuscript_quarto/`
- journal validation, review tooling, and draft-generation experiments

Do not confuse this with the active Capacity-SEM pipeline in `src/stages/`.

## What Is Canonical

For current analysis and documentation:
- canonical data path: `qpr_quarterly.parquet` → `qpr_standardized.parquet` → `panel_features_std.parquet`
- canonical manuscript: `manuscript_quarto/`
- canonical findings status: see [PROJECT_STATUS.md](PROJECT_STATUS.md)

Anything that still depends on pre-fix positive velocity claims should be treated as historical until updated.

## Kaifa Recovered SEM Pipeline

The primary SEM in `manuscript_quarto/` operates on the Kaifa recovered-analysis bundle (`manuscript_kaifa_archive/data/recovered_code_bundle/`), processed through `src/capacity_sem/models/kaifa_recovered_analysis.py`. Stage:

```bash
python src/pipeline.py run_kaifa_recovered_analysis
```

Outputs: sensitivity summary, parameter estimates, fit statistics, geography-matching audit — stored in `data_work/diagnostics/kaifa_recovered_analysis/`. The Kaifa-derived SEM input is the 573-jurisdiction reference dataset `manuscript_kaifa_archive/data/external_sem_exports/baseline_sem/sem_2factor_ready_dataset.csv`.

## Historical SVI Data

Six CDC/ATSDR Social Vulnerability Index vintages (2010, 2014, 2016, 2018, 2020, 2022) are stored in `data_raw/svi_historical/SVI{YYYY}_US_COUNTY.csv` for the vintage-sensitivity and per-jurisdiction disaster-year re-estimation analyses reported in Appendix C.9. See `doc/DATA_DICTIONARY.md` for schema details (note: SVI 2010 uses `S_PL_THEME*` and `FIRST_STATE_ABBR`; SVI 2014–2022 use `SPL_THEME*` and `ST_ABBR`).

The SVI historical analyses are currently run inline (not as first-class pipeline stages); see `doc/reviews/quarto/response_04_2026-04-14.md` for the reproducible procedure.
