# Pipeline Documentation

## Overview

The Capacity-SEM analysis pipeline consists of six sequential stages that process raw QPR data through to publication figures.

## Pipeline Stages

### Stage 0: Data Ingestion (`s00_ingest.py`)

**Command**: `python src/pipeline.py ingest_data`

**Purpose**: Load QPR data, standardize columns, build quarterly rollups with cumulative totals, and ingest external covariates.

**Inputs**:
- `data_raw/qpr_data.csv` - Raw QPR export

**Outputs**:
- `data_work/qpr_raw.parquet` - Validated QPR data
- `data_work/qpr_clean.parquet` - Cleaned QPR data with QA flags and imputed grantee state
- `data_work/qpr_quarterly.parquet` - Quarterly rollup with flow + cumulative series
- `data_work/quality/qpr_quality_report.csv` - Row-level quality summary for QPR data
- `data_work/quality/qpr_quarterly_quality_report.csv` - Quarterly rollup quality summary
- `data_work/population.parquet` - Grantee population covariates
- `data_work/grantee_disaster_population.parquet` - Grantee-disaster population covariates (if available)
- `data_work/severity.parquet` - Disaster severity covariates
- `data_work/employment.parquet` - Employment covariates

**Key Functions**:
- `ingest_qpr_data()` - Load and validate QPR CSV
- `build_qpr_quarterly()` - Build quarterly rollup and cumulative series
- `ingest_covariates()` - Load external datasets

**Data Quality Outputs**:
- `qpr_clean.parquet` includes row-level `QA_` flags, `QPR_Date`, and `Grantee State` imputed from the grant code when missing.
- `data_work/quality/qpr_quality_report.csv` summarizes row-level QA flags and basic counts.
- `data_work/quality/qpr_quarterly_quality_report.csv` summarizes quarterly rollup issues (negative flows and cumulative decreases).

**Known Data Issues (current exports)**:
- Some rows lack `QPR Actual Quarter`, so they are excluded from the quarterly rollup.
- Some rows lack `Grantee State`; it is imputed from the grant code and tracked in `Grantee State Source`.
- Negative obligated/disbursed/expended values appear (adjustments or corrections); they are flagged but not modified.
- Cumulative totals can decrease within a grantee-disaster series (reflecting revisions); flagged in the quarterly quality report.
- Location data are limited to `Grantee State`; no sub-state geographies are present in the QPR export.

---

### Stage 1: Panel Construction (`s01_link.py`)

**Command**: `python src/pipeline.py build_panel`

**Purpose**: Link data sources and construct analysis panel.

**Inputs**:
- `data_work/qpr_quarterly.parquet` (preferred)
- `data_work/qpr_raw.parquet` (fallback if quarterly not present)
- `data_work/population.parquet`
- `data_work/grantee_disaster_population.parquet`
- `data_work/severity.parquet`
- `data_work/employment.parquet`

**Outputs**:
- `data_work/panel.parquet` - Grantee-disaster level panel

**Key Functions**:
- `create_grantee_disaster_panel()` - Aggregate to analysis unit
- `merge_population()` - Add population covariates
- `merge_severity()` - Add disaster severity
- `compute_ratios()` - Calculate financial ratios
- `scale_covariates()` - Z-score standardization

**Ratio Configuration**:
- `RATIO_DEFINITION` in `src/config.py` controls ratio construction:
  - `mean_cumulative`: mean of quarterly cumulative ratios across quarters
  - `final_cumulative`: ratio of final cumulative totals
- `QPR_DOLLAR_FIELDS_ARE_FLOW` controls whether raw QPR dollar fields are treated as quarterly net flows (default) or cumulative totals.

---

### Stage 2: Feature Engineering (`s02_features.py`)

**Command**: `python src/pipeline.py compute_features`

**Purpose**: Compute timeliness metrics, experience indicators, and program stratification.

**Inputs**:
- `data_work/panel.parquet`
- `data_work/qpr_clean.parquet` (preferred)
- `data_work/qpr_raw.parquet` (fallback)
- `data_work/qpr_quarterly.parquet`

**Outputs**:
- `data_work/panel_features.parquet` - Panel with all computed features

**Key Functions**:
- `compute_timeliness_features()` - Duration, timeliness metrics
- `compute_additional_timeliness_metrics()` - Progress rate, CV, log duration
- `build_experience_dataset()` - Prior grant experience
- `add_program_type_column()` - Activity type classification

Timeliness and spending consistency metrics are computed from `qpr_quarterly` (quarterly flows and cumulative series).

---

### Stage 3: SEM Estimation (`s03_estimation.py`)

**Command**: `python src/pipeline.py run_estimation [--model MODEL] [--subset SUBSET]`

**Purpose**: Fit structural equation models.

**Inputs**:
- `data_work/panel_features.parquet`

**Outputs**:
- `data_work/diagnostics/estimates_*.csv` - Parameter estimates
- `data_work/diagnostics/fit_stats_*.csv` - Fit statistics

**Options**:
- `--model, -m`: Model specification (default: `exp_optimal_v1`)
- `--subset, -s`: Government type (`all`, `state`, `local`)

**Key Functions**:
- `run_estimation()` - Fit and evaluate model
- `run_model_comparison()` - Compare specifications
- `run_subset_comparison()` - Compare government types

**Data Considerations**:
- SEM fitting uses complete-case observations for variables in the model specification; missing indicators (e.g., `Duration_log`, `Spending_CV`) can materially reduce sample size.

---

### Stage 3b: Time-Varying Survival Estimation (`s03b_survival_estimation.py`)

**Command**: `python src/pipeline.py run_survival`

**Purpose**: Fit time-varying Cox proportional hazards models with lagged capacity covariates and comprehensive diagnostics.

**Inputs**:
- `data_work/panel_time_varying.parquet` - Time-varying survival panel with start/stop intervals
- `data_work/panel_features.parquet` - Panel with static covariates

**Outputs**:
- `data_work/diagnostics/survival_time_varying_cox_results.csv` - Cox model parameter estimates
- `data_work/diagnostics/survival_hazard_ratios.csv` - Hazard ratios with 95% CIs
- `data_work/diagnostics/survival_bootstrap_se.csv` - Bootstrap clustered standard errors
- `data_work/diagnostics/survival_robustness_checks.csv` - Alternative specifications
- `figures/survival_martingale_residuals.png` - Linearity diagnostic plots
- `figures/survival_cox_snell_residuals.png` - Overall fit diagnostic
- `figures/survival_predicted_curves_*.png` - Survival curves by capacity quartile
- `figures/survival_influence_diagnostics.png` - Influential observation diagnostics

**Key Functions**:
- `run_time_varying_cox()` - Fit Cox model with time-varying covariates
- `run_robustness_checks()` - Alternative specifications and sensitivity checks
- `generate_diagnostic_plots()` - Comprehensive diagnostic visualizations

**Robustness Checks**:

1. **Capacity Only (No Covariates)**: Tests capacity ratios without controls
2. **Full Covariates (Main Specification)**: Includes government type, grant size, experience, disaster year, population
3. **Stratified by Government Type**: Tests whether state vs. local differ
4. **Alternative Lag Structures**: Tests lag=0 (contemporaneous), lag=1 (main), lag=2 (longer lag)

**Time-Varying Panel Structure**:

The time-varying panel is automatically generated in Stage 1 ([s01_link.py](s01_link.py)) using `reshape_quarterly_to_time_varying()`. Each grantee-disaster is expanded into multiple rows representing quarterly intervals:

```
Grantee-Disaster | start | stop | E | Ratio_disb_lag1 | Gov_Type | Log_Grant
NYC-Sandy        | 0     | 3    | 0 | NaN             | Local    | 17.2
NYC-Sandy        | 3     | 6    | 0 | 0.42            | Local    | 17.2
NYC-Sandy        | 6     | 9    | 0 | 0.58            | Local    | 17.2
...
NYC-Sandy        | 45    | 48   | 1 | 0.89            | Local    | 17.2
```

**Key features**:
- Lagged capacity ratios (default: 1-quarter lag) avoid reverse causality
- E=1 only on final row if program completed
- Static covariates repeated on every row
- Bootstrap clustered SEs account for repeated grantee observations

**Diagnostics Generated**:

1. **Proportional Hazards Tests**: Schoenfeld residual tests
2. **Linearity Checks**: Martingale residuals vs. predictors
3. **Model Fit**: Cox-Snell residuals vs. theoretical exponential
4. **Influence Diagnostics**: Score residuals identify outliers
5. **Predicted Curves**: Survival probability by capacity quartile

---

### Stage 3c: Kaifa's Models Replication (`s03_manuscript_replication.py`) - EXPERIMENTAL

**Command**: `PYTHONPATH=src python3 src/stages/s03_manuscript_replication.py --subset state`

**Purpose**: Replicate Kaifa's original manuscript analysis for verification and critique.

**WARNING**: This stage is experimental and for verification only. Use the canonical pipeline (Stage 3) for production analyses.

**Key Differences from Canonical Pipeline**:

1. **Grantee-level aggregation**: Averages across disasters per grantee (N~38 state, ~40 local) instead of grantee-disaster pairs (N~156)
2. **Right-censoring**: Incomplete programs use observation time (N_Quarters) as Duration
3. **Original 3x3 model**: Includes Timeliness = 1/Duration (creates mathematical coupling)
4. **Mean-of-ratios**: Computes mean of quarterly ratios (vs. final cumulative ratio)

**Inputs**:

- `data_work/panel_features.parquet`
- `data_work/qpr_quarterly.parquet`

**Outputs**:

- `data_work/diagnostics/manuscript_replication_estimates_*.csv`
- `data_work/diagnostics/manuscript_replication_summary_*.csv`
- `data_work/diagnostics/manuscript_replication_critique_*.csv`

**Kaifa's Model Specifications**:

| Model | Description |
|-------|-------------|
| `kaifa_3x3_full` | Original 3x3 with censored Duration and Timeliness |
| `kaifa_3x3_no_duration` | 3x3 without Duration indicator |
| `kaifa_2x2_minimal` | Minimal 2x2 specification |

**Methodological Critique Points**:

- Right-censoring biases duration downward (treats incomplete as complete)
- Timeliness = 1/Duration creates deterministic relationship
- Grantee-level aggregation loses within-grantee variation
- Small sample (N<40) may produce unstable estimates

---

### Stage 4: Robustness Checks (`s04_robustness.py`)

**Command**: `python src/pipeline.py run_robustness`

**Purpose**: Run alternative specifications and sensitivity analyses.

**Inputs**:
- `data_work/panel_features.parquet`

**Outputs**:
- `data_work/diagnostics/robustness_specifications.csv`
- `data_work/diagnostics/robustness_subsets.csv`
- `data_work/diagnostics/robustness_sample_sensitivity.csv`
- `data_work/diagnostics/robustness_covariates.csv`

**Key Functions**:
- `run_alternative_specifications()` - Test model variants
- `run_subset_robustness()` - State vs. local comparison
- `run_sample_sensitivity()` - Vary minimum quarters
- `run_covariate_robustness()` - Test covariate inclusion

**Extended Analyses** (via `--extended` flag):

| Analysis | Function | Description |
|----------|----------|-------------|
| Multi-group SEM | `run_multigroup_analysis()` | State vs. local measurement invariance |
| Mediation | `run_mediation_analysis()` | Indirect effect decomposition |
| Formative vs. Reflective | `run_formative_comparison()` | Alternative capacity specification |

**Multi-group SEM Output**:
- `data_work/diagnostics/robustness_multigroup.csv` - Separate state/local estimates
- Tests configural, metric, and scalar invariance

**Mediation Analysis Output**:
- `data_work/diagnostics/robustness_mediation.csv` - Direct/indirect effects
- Bootstrap confidence intervals for indirect effects

---

### Stage 5: Figure Generation (`s05_figures.py`)

**Command**: `python src/pipeline.py make_figures`

**Purpose**: Generate publication-ready figures.

**Inputs**:
- `data_work/panel_features.parquet`
- `data_work/diagnostics/*.csv`

**Outputs**:
- `figures/fig_descriptive.png`
- `figures/fig_model_comparison.png`
- `figures/fig_subset_comparison.png`
- `figures/fig_sensitivity.png`
- `figures/fig_path_diagram.png`

**Options**:
- `--style, -s`: Figure style (`publication`, `presentation`)

---

## Full Pipeline Execution

```bash
# Run all stages sequentially
python src/pipeline.py run_all

# With options
python src/pipeline.py run_all --model exp_optimal_v1 --subset all
```

---

## Model Specifications

### Primary Models

| Name | Description | Indicators |
|------|-------------|------------|
| `exp_optimal_v1` | Recommended 2x2 model | Ratios → Capacity; Log Duration, CV → Outcome |
| `full` | Original 3x3 model | Includes Timeliness (has redundancy) |
| `improved_3x3` | Enhanced 3x3 | Adds Startup Lag, Time to 50% |

### List Available Models

```bash
python src/pipeline.py list_models
```

---

---

## Review Management Commands

**Note**: These commands are separate from the data pipeline stages above. They manage the manuscript review process for PAR submission.

### Check Review Status

```bash
python src/pipeline.py review_status
```

Displays current review cycle information, summary statistics, and verification progress.

### Start New Review Cycle

```bash
python src/pipeline.py review_new --focus par_general  # Comprehensive PAR review
python src/pipeline.py review_new --focus methods      # Methodology focus
python src/pipeline.py review_new --focus policy       # Practitioner relevance
python src/pipeline.py review_new --focus clarity      # Writing/presentation
```

Creates a new `manuscript_quarto/REVISION_TRACKER.md` with:
- Auto-incremented review number
- Focus-specific PAR review prompt
- Template for tracking responses
- Summary statistics table

**Available Focus Areas**:

| Focus | Description |
|-------|-------------|
| `par_general` | Comprehensive PAR review (practitioner relevance, methodology, validity, rigor, robustness, literature, clarity, PAR style) |
| `methods` | Deep methodological review (model specification, censoring, sample size, validity, endogeneity, robustness, diagnostics) |
| `policy` | Practitioner relevance (Evidence for Practice, policy recommendations, generalizability, accessibility, examples, implementation barriers) |
| `clarity` | Writing quality (abstract, intro, lit review, methods, results, discussion, conclusions, flow, PAR style) |

### Verify Review Completion

```bash
python src/pipeline.py review_verify
```

Checks:
- Verification checklist progress (13 items)
- PAR compliance checks:
  - Word count ≤ 8,000 (currently ~7,851)
  - No "this study" self-references (currently 0)
  - Evidence for Practice section present
  - Abstract ≤ 150 words

### Archive Completed Review

```bash
python src/pipeline.py review_archive
```

Archives current review to `doc/reviews/archive/review_NN_YYYY-MM-DD_FOCUS.md` and resets tracker for next cycle.

### Generate Review Report

```bash
python src/pipeline.py review_report
```

Summary of all review cycles (archived + active), including:
- Total cycles completed
- Comments per cycle
- Active review status

### Review Workflow

1. **Generate**: `python src/pipeline.py review_new --focus par_general`
2. **Obtain LLM Review**: Send manuscript + embedded prompt to Claude/GPT-4
3. **Triage**: Classify comments as VALID/ADDRESSED/SCOPE/INVALID in `manuscript_quarto/REVISION_TRACKER.md`
4. **Implement**: Address valid concerns, update manuscript, re-render
5. **Verify**: `python src/pipeline.py review_verify`
6. **Archive**: `python src/pipeline.py review_archive` when complete

See [SYNTHETIC_REVIEW_PROCESS.md](SYNTHETIC_REVIEW_PROCESS.md) for detailed methodology.

---

## Troubleshooting

### Missing semopy

```bash
pip install semopy
```

### Missing data files

Run ingestion stage first:
```bash
python src/pipeline.py ingest_data
```

### Empty panel

Check that QPR data file exists in `data_raw/` and has expected columns.
