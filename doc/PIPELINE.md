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

### Stage 0b: Data Standardization (`s00b_standardize.py`) ✨ NEW

**Command**: `python src/pipeline.py standardize_data`

**Purpose**: Standardize quarterly data with fixed denominators and winsorized velocity to eliminate computational artifacts in time-varying analyses.

**Inputs**:
- `data_work/qpr_clean.parquet` - Cleaned QPR data from Stage 0

**Outputs**:
- `data_work/qpr_standardized.parquet` - Standardized quarterly data (130,605 rows, 35 columns)
- `data_work/quality/qpr_standardized_report.csv` - Standardization quality report

**Key Functions**:
- `compute_stable_denominator()` - Compute final or max obligated amount per grantee-disaster
- `standardize_grantee_disaster()` - Apply fixed-denominator standardization per group
- `apply_winsorization()` - Winsorize velocity at 1%/99% percentiles dataset-wide
- `generate_standardization_report()` - Create quality metrics report

**Standardization Logic**:

1. **Fixed denominators**: Use final obligated amount as stable denominator across all quarters
   ```python
   Ratio_Disbursed_Std = Disbursed_Clean / Obligated_Final
   Velocity_Disb_Std = Ratio_Disbursed_Std - Ratio_Disbursed_Std_lag1
   ```

2. **Monotonic series**: Apply cummax to ensure cumulative totals never decrease
   ```python
   Obligated_Clean = QPR_Fund_Obligated.clip(lower=0).cummax()
   ```

3. **Winsorization**: Cap velocity at 1st/99th percentiles to handle outliers
   ```python
   Velocity_winsor = np.clip(Velocity, lower=p01, upper=p99)
   ```

**Quality Assurance**:
- `QA_Extreme_Velocity`: Flags observations with velocity >100 pp/quarter (before winsorization)
- `QA_Obligated_Jump`: Flags quarters where obligated changed >10%
- `QA_Negative_Adjustment`: Flags quarters with negative flows

**Impact**:
- Extreme velocity observations reduced from 0.6% to 0.24%
- Velocity standard deviation reduced by 68% (48 → 15 pp/quarter)
- Eliminates spurious velocity swings from changing denominators

**Documentation**: See `doc/ETL_STANDARDIZATION.md` for full methodology

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

### Stage 1b: Standardized Feature Engineering (`s01b_features.py`) ✨ NEW

**Command**: `python src/pipeline.py build_features_std`

**Purpose**: Aggregate standardized quarterly data to grantee-disaster level and compute analysis-ready features using fixed-denominator velocity measures.

**Inputs**:
- `data_work/qpr_standardized.parquet` - Standardized quarterly data from Stage 0b
- `data_work/panel.parquet` - Base panel from Stage 1

**Outputs**:
- `data_work/panel_features_std.parquet` - Standardized features panel (156 rows, 182 columns)

**Key Functions**:
- `aggregate_standardized_velocity()` - Aggregate velocity to grantee-disaster level (mean, median, std)
- `compute_timeliness_features_std()` - Duration to 17 completion thresholds (20%-100%)
- `compute_experience_features()` - Prior grant experience indicators using chronological disaster ordering
- `add_survival_covariates()` - Government type, log obligated, disaster year
- `add_capacity_indices()` - Composite capacity measures
- `add_interaction_terms()` - Ratio × velocity interactions

**Feature Categories** (182 total):
- **Velocity features** (106): Mean/median/std of standardized velocity, rolling averages, scaled versions
- **Duration features** (20): Time to reach 20%, 25%, ..., 100% completion thresholds
- **Interaction terms** (22): Ratio × velocity interactions at various thresholds
- **Experience indicators** (4): Prior_Grant_Count, Prior_Grant_Dollars, Years_Experience, Experience_Index
- **Survival covariates** (6): Government_Type_State, Log_Obligated, Prior_Grant_Count, Prior_Grant_Dollars_log, Disaster_Year, Population_log
- **Other features** (24): Capacity ratios, indices, quartiles

**Experience Computation**:
- Uses `build_experience_dataset()` from `experience_indicators.py` with DRGR_DISASTER_YEARS chronological mapping
- "Prior" grants = disasters occurring in earlier years than current disaster
- First-time grantees assigned 0 for all experience variables
- Integrated after timeliness features, before survival covariate engineering
- Sample: 47% of grantee-disasters (73/156) have prior grant experience

**Key Advantages Over Legacy Pipeline**:
- Uses pre-computed standardized velocity from s00b (fixed denominators, winsorized)
- Single source of truth for velocity calculations
- Proper handling of computational artifacts
- Backward-compatible column names (Duration_of_completion, N_Quarters)

**Documentation**: See `doc/ETL_STANDARDIZATION.md` for velocity methodology

---

### Stage 1c: Aggregate Program Types (`s01c_program_types.py`)

**Command**: `python src/pipeline.py aggregate_program_types`

**Purpose**: Aggregate activity-level data to grantee-disaster level to create program portfolio features.

**Inputs**:
- `data_work/qpr_standardized.parquet` - Quarterly data with Activity Type column
- `data_work/panel_features_std.parquet` - Base panel (for validation)

**Outputs**:
- `data_work/panel_program_types.parquet`
  - 156 grantee-disaster pairs
  - 18 columns: grantee/disaster identifiers + 6 dollar amounts + 6 percentages + 4 derived features

**Processing Steps**:

1. **Activity Classification**: Map 51 raw activity types to 6 categories using PROGRAM_TYPE_MAPPING
   - Housing (15 activities)
   - Infrastructure (14 activities)
   - Economic Development (7 activities)
   - Acquisition (7 activities)
   - Administration (4 activities)
   - Other (4 activities)

2. **Dollar Aggregation**: For each grantee-disaster pair:
   - Get maximum obligated amount per activity (across all quarters)
   - Sum by category to get total obligated per category
   - Calculate total obligated across all categories

3. **Portfolio Features**:
   - Category percentages: `Housing_Pct`, `Infrastructure_Pct`, etc.
   - Primary program type: Category with highest obligated dollars
   - Program diversity index: Herfindahl index (1 - Σ(share_i²))
   - Number of active categories: Count of categories with >5% of obligated

**Features Created**:
- `Housing`, `Infrastructure`, `Economic Development`, `Acquisition`, `Administration`, `Other` - Dollar amounts
- `Housing_Pct`, `Infrastructure_Pct`, etc. - Percentage of total obligated
- `Primary_Program_Type` - Categorical: largest category
- `Program_Diversity_Index` - Continuous: 0 (single category) to ~0.83 (perfectly diverse)
- `N_Active_Categories` - Integer: number of categories with >5% share
- `Total_Obligated_by_Category` - Total obligated summed across categories (for validation)

**Usage in Analysis**:
- Used by `scripts/run_program_type_analysis.py` to stratify velocity effects by program type
- Enables testing of heterogeneity: Do velocity effects vary by what grantees do?

**Runtime**: ~2-3 seconds

**Dependencies**:
- `config.PROGRAM_TYPE_MAPPING` - Activity type classification scheme
- `fastparquet` - Parquet serialization (fallback to pyarrow)

---

### Stage 2: Feature Engineering (`s02_features.py`) ⚠️ DEPRECATED

> **⚠️ DEPRECATED**: Use Stage 1b (`build_features_std`) for all new analyses.
> Stage 2 is retained only for replication of legacy results. Do not use for new research.

**Status**: **DEPRECATED** - Use Stage 1b (s01b_features.py) for new analyses

**Command**: `python src/pipeline.py compute_features`

**Purpose**: Legacy feature engineering with dynamic-denominator velocity (retained for replication only).

**Deprecation Reason**: Uses dynamic denominators for velocity calculation, which creates computational artifacts (extreme outliers ±1,933 pp/quarter). Replaced by standardized pipeline (Stages 0b + 1b) with fixed denominators.

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

**Additional capacity measures** (exploratory):
- Absolute dollars (log): `Obligated_log`, `Disbursed_log`, `Expended_log`
- Velocity: `Disbursement_Velocity`, `Expenditure_Velocity`, `Capacity_Velocity_Index`
- Velocity (percent-per-quarter): `Disbursement_Velocity_pp`, `Expenditure_Velocity_pp`, `Capacity_Velocity_Index_pp`
- Velocity (winsorized): `Disbursement_Velocity_winsor`, `Expenditure_Velocity_winsor`, `Capacity_Velocity_Index_winsor`
- Standardized velocity (z-score): `Disbursement_Velocity_scaled`, `Expenditure_Velocity_scaled`, `Capacity_Velocity_Index_scaled`
- Early-window velocity (first 2/3/4/6 quarters): `Disbursement_Velocity_early_2q`/`3q`/`4q`/`6q`, `Expenditure_Velocity_early_2q`/`3q`/`4q`/`6q` (pp variants), and `Capacity_Velocity_Index_early_*` (pp, winsor, scaled variants)
- Fixed calendar windows (first 12/18 months): `Disbursement_Velocity_fixed_12m`/`18m`, `Expenditure_Velocity_fixed_12m`/`18m`, and `Capacity_Velocity_Index_fixed_*` (pp, winsor, scaled variants)
- Ratio x velocity interactions (centered and threshold/spline): `Ratio_disbursed_to_obligated_c`, `Ratio_disbursed_to_obligated_high`, `Ratio_disbursed_to_obligated_above`, alternative cutoffs (`*_high_q25/q33/q67/q75`, `*_above_q25/q33/q67/q75`), `Disbursement_Velocity_pp_c`, `Capacity_Velocity_Index_pp_c`, plus interaction terms
- Composite indices: `Capacity_Absolute_Index`, `Capacity_PCA1`
- Quartile dummies (non-parametric): `Capacity_Index_Q2-Q4`, `Capacity_Absolute_Index_Q2-Q4`, `Capacity_Velocity_Index_Q2-Q4`, `Capacity_Velocity_Index_pp_Q2-Q4`, `Capacity_Velocity_Index_winsor_Q2-Q4`, `Capacity_Velocity_Index_scaled_Q2-Q4`

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
3. **Velocity (Percent-Per-Quarter)**: Lagged velocity and velocity index specifications
4. **Rolling/Cumulative Velocity**: Rolling-window and cumulative velocity robustness checks
5. **Stratified by Government Type**: Tests whether state vs. local differ
6. **Alternative Lag Structures**: Tests lag=0 (contemporaneous), lag=1 (main), lag=2 (longer lag)

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

### Stage 6: Alternative Modeling (`s06_alternatives.py`)

**Command**: `python src/pipeline.py run_alternatives [--methods METHODS] [--capacity-sets all]`

**Purpose**: Run alternative survival capacity sets and SEM robustness checks.

**Outputs**:
- `data_work/diagnostics/alternatives_survival.csv` (single-set survival)
- `data_work/diagnostics/alternatives_survival_capacity_sets.csv`
- `data_work/diagnostics/alternatives_survival_stratified_velocity.csv`
- `data_work/diagnostics/alternatives_survival_velocity_strata_models.csv`
- `data_work/diagnostics/alternatives_threshold_sensitivity.csv`
- `data_work/diagnostics/alternatives_duration_free.csv`
- `data_work/diagnostics/alternatives_milestone.csv`
- `data_work/diagnostics/alternatives_comparison.csv`

**Notes**:
- Stratified outputs include penalized Cox fits for low-event strata.
- Pooled/stratified interaction models test differential velocity effects
  across ratio quartiles.

---

### Stage 7: Capacity Summary (`s07_capacity_summary.py`)

**Command**: `python src/pipeline.py capacity_summary`

**Purpose**: Apply multiple-testing corrections across capacity-set survival
models (static + time-varying) and produce a compact table and figure.

**Outputs**:
- `data_work/diagnostics/multiple_testing_capacity_sets_time_varying.csv`
- `data_work/diagnostics/capacity_corrected_table.csv`
- `figures/fig_capacity_corrected.png`

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
