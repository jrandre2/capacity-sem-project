# Data Dictionary

## Overview

This document defines all variables used in the Capacity-SEM analysis.
QPR dollar fields can be quarterly net flows or cumulative totals depending on the export; the pipeline standardizes them using `QPR_DOLLAR_FIELDS_ARE_FLOW` and builds `qpr_quarterly.parquet` with both flow and cumulative series.
Raw column names are normalized using `COLUMN_MAPPING` (e.g., `QPR Funds Obligated $` -> `QPR Fund Obligated $`).

---

## Identification Variables

| Variable | Type | Description |
|----------|------|-------------|
| `Grantee` | String | Name of the grantee (state or local government) |
| `Grantee State` | String | State or territory associated with the grantee (may be missing in raw export) |
| `Disaster Type` | String | DRGR disaster type identifier |
| `Disaster_Year` | Integer | Year of the disaster event |
| `Grant` | String | Grant identifier |
| `Appropriation` | String | Congressional appropriation |

---

## Raw QPR Variables

| Variable | Type | Description |
|----------|------|-------------|
| `QPR Fund Obligated $` | Float | Raw export field; treated as quarterly net flow when `QPR_DOLLAR_FIELDS_ARE_FLOW=True`, otherwise cumulative obligated total |
| `QPR Fund Disbursed $` | Float | Raw export field; treated as quarterly net flow when `QPR_DOLLAR_FIELDS_ARE_FLOW=True`, otherwise cumulative disbursed total |
| `QPR Fund Expended $` | Float | Raw export field; treated as quarterly net flow when `QPR_DOLLAR_FIELDS_ARE_FLOW=True`, otherwise cumulative expended total |
| `QPR Fund Expended Q $` | Float | Quarterly net change in expenditures (if provided by export; otherwise derived in `qpr_quarterly`) |
| `QPR Actual Quarter` | String | Quarter identifier (e.g., "2020 Q3") |
| `Activity Type` | String | Type of recovery activity |

---

## Cleaned QPR Variables (`qpr_clean.parquet`)

| Variable | Type | Description |
|----------|------|-------------|
| `Grantee State Raw` | String | Original `Grantee State` value before imputation |
| `Grant_State_Code` | String | State/territory code parsed from `Grant` (FIPS-style two-digit code) |
| `Grant_State_Name` | String | State/territory name mapped from `Grant_State_Code` |
| `Grantee State Source` | String | `raw` when original state present, `grant_code` when imputed from grant code |
| `QPR_Date` | Date | Parsed date for `QPR Actual Quarter` (quarter end) |

---

## Quarterly Derived QPR Variables (`qpr_quarterly.parquet`)

| Variable | Type | Description |
|----------|------|-------------|
| `QPR Fund Obligated Q $` | Float | Quarterly net change in obligated funds (flow) |
| `QPR Fund Disbursed Q $` | Float | Quarterly net change in disbursed funds (flow) |
| `QPR Fund Expended Q $` | Float | Quarterly net change in expended funds (flow) |
| `QPR Fund Obligated $` | Float | Cumulative obligated total (constructed if raw fields are flows) |
| `QPR Fund Disbursed $` | Float | Cumulative disbursed total (constructed if raw fields are flows) |
| `QPR Fund Expended $` | Float | Cumulative expended total (constructed if raw fields are flows) |
| `QPR_Date` | Date | Date representation of `QPR Actual Quarter` (quarter start) |

---

## Standardized QPR Variables (`qpr_standardized.parquet`)

**Purpose**: Quarterly data with fixed denominators and winsorized velocity to eliminate computational artifacts.

**Created by**: Stage 00b (s00b_standardize.py)

**Documentation**: See `doc/ETL_STANDARDIZATION.md` for methodology

| Variable | Type | Description |
|----------|------|-------------|
| `Obligated_Final` | Float | Final obligated amount (used as stable denominator for all quarters) |
| `Obligated_Clean` | Float | Monotonic obligated amount (cummax of QPR Fund Obligated $) |
| `Disbursed_Clean` | Float | Monotonic disbursed amount (cummax of QPR Fund Disbursed $) |
| `Expended_Clean` | Float | Monotonic expended amount (cummax of QPR Fund Expended $) |
| `Ratio_Disbursed_Std` | Float | Standardized disbursement ratio: Disbursed_Clean / Obligated_Final |
| `Ratio_Expended_Std` | Float | Standardized expenditure ratio: Expended_Clean / Obligated_Final |
| `Velocity_Disb_Std` | Float | Standardized disbursement velocity (quarterly change, fraction) |
| `Velocity_Exp_Std` | Float | Standardized expenditure velocity (quarterly change, fraction) |
| `Velocity_Disb_Std_pp` | Float | Standardized disbursement velocity (percentage points/quarter) |
| `Velocity_Exp_Std_pp` | Float | Standardized expenditure velocity (percentage points/quarter) |
| `Velocity_Disb_Std_pp_winsor` | Float | **PRIMARY VELOCITY MEASURE**: Winsorized at 1%/99% percentiles |
| `Velocity_Exp_Std_pp_winsor` | Float | Winsorized expenditure velocity |
| `Velocity_Index_Std_pp` | Float | Capacity velocity index (average of disbursement and expenditure velocity) |
| `Velocity_Index_Std_pp_winsor` | Float | Winsorized capacity velocity index |
| `Velocity_Disb_Std_pp_roll2` | Float | 2-quarter rolling mean velocity (disbursement) |
| `Velocity_Disb_Std_pp_roll4` | Float | 4-quarter rolling mean velocity (disbursement) |
| `Velocity_Exp_Std_pp_roll2` | Float | 2-quarter rolling mean velocity (expenditure) |
| `Velocity_Exp_Std_pp_roll4` | Float | 4-quarter rolling mean velocity (expenditure) |
| `QA_Extreme_Velocity` | Bool | Raw velocity exceeds ±100 pp/quarter (before winsorization) |
| `QA_Obligated_Jump` | Bool | Obligated amount changed >10% from prior quarter |
| `QA_Negative_Adjustment` | Bool | Disbursed or expended decreased (negative flow) |

**Key features**:
- **Fixed denominators**: Uses final obligated amount for all quarters, eliminating spurious velocity swings
- **Monotonic series**: Clean series ensure cumulative totals never decrease
- **Winsorization**: Primary velocity measures capped at 1%/99% percentiles to handle outliers
- **Rolling averages**: Smooth out quarterly volatility
- **QA flags**: Track data quality issues for investigation

**Impact**: Extreme velocity observations reduced from 0.6% to 0.24%; velocity std dev reduced 68%

---

## Quality Flags (`qpr_clean.parquet`)

All `QA_` fields are boolean flags indicating row-level data quality issues.

| Variable | Type | Description |
|----------|------|-------------|
| `QA_missing_grantee` | Bool | Missing `Grantee` value |
| `QA_missing_grant` | Bool | Missing `Grant` value |
| `QA_missing_disaster_type` | Bool | Missing `Disaster Type` value |
| `QA_missing_qpr_actual_quarter` | Bool | Missing `QPR Actual Quarter` |
| `QA_invalid_qpr_actual_quarter` | Bool | `QPR Actual Quarter` present but unparsable |
| `QA_missing_grantee_state` | Bool | Missing `Grantee State` after imputation |
| `QA_unknown_grantee_state` | Bool | `Grantee State` not in canonical state/territory list |
| `QA_grant_state_mismatch` | Bool | `Grant` state code conflicts with `Grantee State` |
| `QA_negative_obligated` | Bool | Negative `QPR Fund Obligated $` value |
| `QA_negative_disbursed` | Bool | Negative `QPR Fund Disbursed $` value |
| `QA_negative_expended` | Bool | Negative `QPR Fund Expended $` value |
| `QA_duplicate_row` | Bool | Duplicate full row detected |

---

## Quality Report Files (`data_work/quality/*.csv`)

Both quality report files use the same schema:

| Column | Type | Description |
|--------|------|-------------|
| `metric` | String | Metric name (counts or checks) |
| `value` | Float | Count or value for the metric |
| `percent` | Float | Share of rows/groups (blank when not applicable) |

`qpr_quality_report.csv` summarizes row-level `QA_` flags and basic counts.
`qpr_quarterly_quality_report.csv` summarizes issues in the quarterly rollup (negative flows, cumulative decreases).

---

## Time-Varying Panel Variables (`panel_time_varying.parquet`)

The time-varying survival panel restructures data into interval format for Cox proportional hazards models with time-varying covariates.

| Variable | Type | Description |
|----------|------|-------------|
| `Grantee` | String | Grantee identifier |
| `Disaster Type` | String | DRGR disaster type identifier |
| `start` | Float | Interval start time (months) |
| `stop` | Float | Interval end time (months) |
| `E` | Integer | Event indicator (1=completion in this interval, 0=censored/ongoing) |
| `Ratio_disbursed_to_obligated_lag1` | Float | Lagged (1 quarter) disbursement ratio |
| `Ratio_expended_to_disbursed_lag1` | Float | Lagged (1 quarter) expenditure ratio |
| `Ratio_disbursed_to_obligated_lag0` | Float | Contemporaneous disbursement ratio (lag=0, for sensitivity) |
| `Ratio_expended_to_disbursed_lag0` | Float | Contemporaneous expenditure ratio (lag=0, for sensitivity) |
| `Ratio_disbursed_to_obligated_lag2` | Float | 2-quarter lagged disbursement ratio (for sensitivity) |
| `Ratio_expended_to_disbursed_lag2` | Float | 2-quarter lagged expenditure ratio (for sensitivity) |

**Panel structure**:
- One row per (grantee-disaster, quarter) interval
- E=1 only on final row if program completed
- Lagged ratios avoid reverse causality
- Static covariates repeated on every row

---

## Survival Covariates

Additional covariates engineered for time-varying survival models:

| Variable | Type | Description |
|----------|------|-------------|
| `Government_Type_State` | Integer | Dummy variable (1=State government, 0=Local government) |
| `Log_Obligated` | Float | Log-transformed grant size: log(1 + Max_Obligated) |
| `Prior_Grant_Count` | Integer | Number of prior CDBG-DR grants (missing treated as 0) |
| `Prior_Grant_Dollars_log` | Float | Log-transformed prior grant dollars: log(1 + cumulative prior obligated) |
| `Disaster_Year` | Integer | Year of disaster event (parsed from Disaster Type) |
| `Population_log` | Float | Log-transformed jurisdiction population: log(1 + Population) |

**Covariate sources**:
- Government_Type_State: Derived from grantee name matching STATE_GOVERNMENTS list
- Log_Obligated: Computed from Max_Obligated in panel
- Prior experience: Computed from experience dataset
- Disaster_Year: Parsed from Disaster Type string with manual fallback for exceptions
- Population_log: From external population data

---

## Aggregated Panel Variables

| Variable | Type | Description |
|----------|------|-------------|
| `Total_Obligated` | Float | Sum of quarterly obligated flows |
| `Total_Disbursed` | Float | Sum of quarterly disbursed flows |
| `Total_Expended` | Float | Sum of quarterly expended flows |
| `Max_Obligated` | Float | Final cumulative obligated total (max across quarters) |
| `Max_Disbursed` | Float | Final cumulative disbursed total (max across quarters) |
| `Max_Expended` | Float | Final cumulative expended total (max across quarters) |
| `N_Quarters` | Integer | Number of reporting quarters |

---

## Capacity Indicators

### Primary Ratios

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `Ratio_disbursed_to_obligated` | Float | [0, 1+] | Total Disbursed / Total Obligated |
| `Ratio_expended_to_disbursed` | Float | [0, 1+] | Total Expended / Total Disbursed |
| `Ratio_expended_to_obligated` | Float | [0, 1+] | Total Expended / Total Obligated |

`Ratio_disbursed_to_obligated` and `Ratio_expended_to_disbursed` use either the mean of quarterly cumulative ratios (`mean_cumulative`) or the final cumulative ratio (`final_cumulative`), controlled by `RATIO_DEFINITION` in `src/config.py`.
`Ratio_expended_to_obligated` always uses final cumulative totals.

### Alternative Capacity Measures

| Variable | Type | Description |
|----------|------|-------------|
| `Progress_Rate` | Float | Completion percentage per quarter |
| `Startup_Lag` | Integer | Quarters before first expenditure |
| `Capacity_Index` | Float | Mean of disbursement and expenditure ratios (formative models) |

### Phase-Specific Velocity Features (Week 5)

**Purpose**: Decompose velocity effects across program timeline to identify WHEN velocity matters.

**Method**: Each program's observation period is divided into thirds chronologically, and velocity statistics are computed separately for each phase.

| Feature | Description | Units | Typical Range |
|---------|-------------|-------|---------------|
| `Velocity_Early` | Mean velocity in first third of program duration | pp/quarter | -5 to 15 |
| `Velocity_Early_median` | Median velocity in first third | pp/quarter | -3 to 10 |
| `Velocity_Mid` | Mean velocity in middle third of program duration | pp/quarter | -5 to 15 |
| `Velocity_Mid_median` | Median velocity in middle third | pp/quarter | -3 to 10 |
| `Velocity_Late` | Mean velocity in final third of program duration | pp/quarter | -5 to 15 |
| `Velocity_Late_median` | Median velocity in final third | pp/quarter | -3 to 10 |
| `Velocity_Acceleration` | Change in velocity from early to late phase (Late - Early) | pp/quarter | -10 to 10 |

**Missing Data**: Programs with <3 quarters of data may have missing phase-specific features.

**Usage**: Used in `run_phase_specific_analysis.py` to test Cox PH models with phase-specific predictors.

**Key Finding**: Late-phase velocity (HR=5.00, p=0.040) dominates when all phases are included simultaneously.

### Program Type Features (panel_program_types.parquet)

**Purpose**: Characterize program portfolio composition to test heterogeneity by activity type.

**Method**: Activity-level obligated dollars aggregated to grantee-disaster level across 6 program categories.

**Dollar Amounts** (in USD):

| Feature | Description | Typical Range |
|---------|-------------|---------------|
| `Housing` | Total obligated for housing activities | $0 - $8B |
| `Infrastructure` | Total obligated for infrastructure activities | $0 - $3B |
| `Economic Development` | Total obligated for economic development activities | $0 - $500M |
| `Acquisition` | Total obligated for acquisition activities | $0 - $1B |
| `Administration` | Total obligated for administrative activities | $0 - $200M |
| `Other` | Total obligated for uncategorized activities | $0 - $100M |
| `Total_Obligated_by_Category` | Sum across all categories (validation field) | $1M - $8B |

**Portfolio Composition** (percentages):

| Feature | Description | Units | Typical Range |
|---------|-------------|-------|---------------|
| `Housing_Pct` | Housing as % of total obligated | proportion | 0.00 - 1.00 |
| `Infrastructure_Pct` | Infrastructure as % of total obligated | proportion | 0.00 - 1.00 |
| `Economic Development_Pct` | Economic development as % of total obligated | proportion | 0.00 - 0.50 |
| `Acquisition_Pct` | Acquisition as % of total obligated | proportion | 0.00 - 0.40 |
| `Administration_Pct` | Administration as % of total obligated | proportion | 0.00 - 0.20 |
| `Other_Pct` | Other activities as % of total obligated | proportion | 0.00 - 0.20 |

**Derived Features**:

| Feature | Description | Units | Typical Range |
|---------|-------------|-------|---------------|
| `Primary_Program_Type` | Category with highest obligated dollars | categorical | Housing, Infrastructure, Administration |
| `Program_Diversity_Index` | Herfindahl index: 1 - Σ(share²) | 0-1 scale | 0.00 - 0.83 |
| `N_Active_Categories` | Count of categories with >5% of obligated | integer | 1 - 6 |

**Distribution**:
- Housing-dominant: 85 programs (54%)
- Infrastructure-dominant: 58 programs (37%)
- Administration-dominant: 10 programs (6%)
- Other: 3 programs (2%)

**Usage**: Merged with panel_features_std.parquet in `run_program_type_analysis.py` for stratified survival analysis.

**Key Finding**: Administration programs show extreme velocity effects (HR=30.74, p=0.004), suggesting rapid spending is critical for administrative efficiency.

### Multi-Stage Lag Features (panel_features_std.parquet)

**Purpose**: Identify bottlenecks in the administrative pipeline (Obligate→Disburse→Expend).

**Method**: Compute time lag (in quarters) from first non-zero value in each stage to the next stage.

| Feature | Description | Units | Typical Range |
|---------|-------------|-------|---------------|
| `Lag_Obligate_to_Disburse` | Quarters from first obligated to first disbursed > 0 | quarters | 0 - 8 |
| `Lag_Disburse_to_Expend` | Quarters from first disbursed to first expended > 0 | quarters | 0 - 12 |
| `Lag_Total_Pipeline` | Total lag from first obligated to first expended | quarters | 0 - 20 |
| `Stage1_Efficiency` | Mean(Disbursed/Obligated) across all quarters | ratio | 0.2 - 1.0 |
| `Stage2_Efficiency` | Mean(Expended/Disbursed) across all quarters | ratio | 0.3 - 1.0 |

**Missing Data**: Programs that never disburse or never expend have missing lag values (treated as censored).

**Usage**: Used in `run_multistage_analysis.py` for competing risks survival analysis.

**Key Finding**: Stage 1 (obligate→disburse) typically 1-2 quarters, Stage 2 (disburse→expend) typically 3-5 quarters. Stage 2 bottlenecks are more common.

---

## Outcome Indicators

### Duration Measures

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `Duration_of_completion` | Float | Months | Time to reach 95% expenditure |
| `Duration_log` | Float | ln(months) | Log-transformed duration |
| `Timeliness` | Float | 1/months | Inverse of duration (deprecated: use `Duration_log` instead; Timeliness = 1/Duration creates mathematical redundancy with duration measures) |

### Multi-Threshold Duration Variables

Duration computed at multiple completion thresholds (30% to 100% in 5% increments):

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `Duration_30pct` | Float | Quarters | Time to reach 30% completion |
| `Duration_35pct` | Float | Quarters | Time to reach 35% completion |
| `Duration_40pct` | Float | Quarters | Time to reach 40% completion |
| `Duration_45pct` | Float | Quarters | Time to reach 45% completion |
| `Duration_50pct` | Float | Quarters | Time to reach 50% completion |
| `Duration_55pct` | Float | Quarters | Time to reach 55% completion |
| `Duration_60pct` | Float | Quarters | Time to reach 60% completion |
| `Duration_65pct` | Float | Quarters | Time to reach 65% completion |
| `Duration_70pct` | Float | Quarters | Time to reach 70% completion |
| `Duration_75pct` | Float | Quarters | Time to reach 75% completion |
| `Duration_80pct` | Float | Quarters | Time to reach 80% completion |
| `Duration_85pct` | Float | Quarters | Time to reach 85% completion |
| `Duration_90pct` | Float | Quarters | Time to reach 90% completion |
| `Duration_95pct` | Float | Quarters | Time to reach 95% completion |
| `Duration_100pct` | Float | Quarters | Time to reach 100% completion |

Log-transformed versions are available with `_log` suffix (e.g., `Duration_50pct_log`).

**Expected Sample Sizes by Threshold**:
- 30-50%: ~140-150 observations (most programs reach early stages)
- 60-80%: ~80-120 observations (moderate completion)
- 90-100%: ~30-50 observations (full completion is rare)

### Kaifa's Censored Duration Variables (Experimental)

Variables used in Kaifa's manuscript replication with right-censoring applied:

| Variable | Type | Description |
|----------|------|-------------|
| `Duration_censored` | Float | Duration with right-censoring (incomplete = N_Quarters) |
| `Timeliness_censored` | Float | 1/Duration_censored |
| `Duration_censored_log` | Float | log(1 + Duration_censored) |
| `Is_Censored` | Bool | True if Duration was imputed from N_Quarters |

**WARNING**: Right-censoring biases duration downward by treating incomplete programs as if they completed at their last observation. Use canonical Duration variables for primary analyses.

### Spending Consistency

| Variable | Type | Description |
|----------|------|-------------|
| `Quarter_by_quarter_variance_expended` | Float | Normalized std of quarterly expended flows (scaled by max) |
| `Spending_CV` | Float | Coefficient of variation of quarterly expended flows |
| `Spending_Gini` | Float | Gini coefficient of quarterly expended flows |
| `Spending_Acceleration` | Float | Mean change in quarterly expended flows, normalized by mean absolute spending |
| `Completion_Pct` | Float | Final cumulative expended / final cumulative obligated |

### Alternative Outcome Measures

| Variable | Type | Description |
|----------|------|-------------|
| `Time_to_50pct` | Float | Normalized quarters to 50% completion |
| `Completion_Velocity` | Float | Mean quarterly change in completion % |

---

## Covariate Variables

### Population

| Variable | Type | Description |
|----------|------|-------------|
| `Population` | Integer | Grantee jurisdiction population |
| `Population_scaled` | Float | Z-score standardized population |
| `Population_log` | Float | Log-transformed population |
| `Population_log_scaled` | Float | Z-score of log population |

### Disaster Severity

| Variable | Type | Description |
|----------|------|-------------|
| `Severity_Index` | Float | Composite disaster severity (0-1) |
| `Severity_Index_scaled` | Float | Z-score standardized severity |

### Experience

Organizational learning proxies based on prior CDBG-DR grant management history.

| Variable | Type | Description | Range |
|----------|------|-------------|-------|
| `Years_Experience` | Float | Years since first CDBG-DR grant | 0-25 |
| `Prior_Grant_Count` | Integer | Number of prior CDBG-DR disasters managed before current | 0-7 |
| `Prior_Grant_Dollars` | Float | Total obligated dollars from prior disasters ($) | 0-$10B |
| `Experience_Index` | Float | Composite experience score (normalized 0-1) | 0-1 |
| `Experience_Index_scaled` | Float | Z-score standardized experience | -2 to +3 |

**Computation**: Uses DRGR_DISASTER_YEARS mapping to determine chronological order. "Prior" means disasters that occurred in earlier years than the current disaster. Computed by `build_experience_dataset()` in `experience_indicators.py` and integrated into standardized features at Stage 1b.

**Missing Data**: First-time grantees (no prior CDBG-DR experience) have all values = 0.

**Sample Statistics** (standardized panel):
- 47% of grantee-disasters (73/156) have prior grant experience
- Mean Prior_Grant_Count: 0.93
- Mean Prior_Grant_Dollars: $1.28B

### Employment

| Variable | Type | Description |
|----------|------|-------------|
| `Employment` | Float | Employment level for grantee jurisdiction |
| `Employment_scaled` | Float | Z-score standardized employment |

---

## Program Type Classification

| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| `Program_Type` | String | Housing, Infrastructure, Administration, Economic Development, Acquisition, Other | Major activity category |

---

## Grantee Classification

| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| `Government_Type` | String | State, Local | Derived from `Grantee` membership in `STATE_GOVERNMENTS`/`LOCAL_GOVERNMENTS` (not stored in parquet outputs) |

State governments include U.S. states and territories.
Local governments include cities, counties, and special districts.

---

## Estimation Output Variables

### Parameter Estimates

| Variable | Type | Description |
|----------|------|-------------|
| `LHS` | String | Left-hand side variable |
| `Operator` | String | `=~` (loading), `~` (regression), `~~` (covariance) |
| `RHS` | String | Right-hand side variable |
| `Estimate` | Float | Parameter estimate |
| `Std. Err` | Float | Standard error |
| `z-value` | Float | z-statistic |
| `p-value` | Float | p-value for hypothesis test |

### Fit Statistics

| Variable | Type | Description |
|----------|------|-------------|
| `chi2` | Float | Chi-square test statistic |
| `dof` | Integer | Degrees of freedom |
| `chi2 p-value` | Float | Chi-square p-value |
| `CFI` | Float | Comparative Fit Index |
| `TLI` | Float | Tucker-Lewis Index |
| `RMSEA` | Float | Root Mean Square Error of Approximation |
| `AIC` | Float | Akaike Information Criterion |
| `BIC` | Float | Bayesian Information Criterion |

---

## Survival Analysis Output Variables

### Cox Model Results (`survival_time_varying_cox_results.csv`)

| Variable | Type | Description |
|----------|------|-------------|
| `Variable` | String | Predictor variable name |
| `Coefficient` | Float | Log hazard ratio (β) |
| `SE` | Float | Standard error |
| `z` | Float | z-statistic |
| `p_value` | Float | p-value for hypothesis test |
| `model` | String | Model specification identifier |

### Hazard Ratios (`survival_hazard_ratios.csv`)

| Variable | Type | Description |
|----------|------|-------------|
| `Variable` | String | Predictor variable name |
| `HR` | Float | Hazard ratio (exp(β)) |
| `HR_Lower` | Float | Lower 95% CI for HR |
| `HR_Upper` | Float | Upper 95% CI for HR |
| `p_value` | Float | p-value for hypothesis test |
| `model` | String | Model specification identifier |

**Interpretation**:
- HR > 1: Higher values increase completion hazard (faster completion)
- HR < 1: Higher values decrease completion hazard (slower completion)
- HR = 1: No effect

### Bootstrap Standard Errors (`survival_bootstrap_se.csv`)

| Variable | Type | Description |
|----------|------|-------------|
| `Variable` | String | Predictor variable name |
| `Coefficient` | Float | Point estimate (β) |
| `Bootstrap_SE` | Float | Bootstrap clustered standard error |
| `Bootstrap_Lower` | Float | Lower 95% CI from bootstrap distribution |
| `Bootstrap_Upper` | Float | Upper 95% CI from bootstrap distribution |
| `n_bootstrap` | Integer | Number of bootstrap iterations |
| `model` | String | Model specification identifier |

### Robustness Checks (`survival_robustness_checks.csv`)

| Variable | Type | Description |
|----------|------|-------------|
| `model` | String | Model specification (capacity_only, full_covariates, stratified_gov_type, lag0, lag2) |
| `variable` | String | Predictor variable name |
| `HR` | Float | Hazard ratio |
| `HR_Lower` | Float | Lower 95% CI for HR |
| `HR_Upper` | Float | Upper 95% CI for HR |
| `p_value` | Float | p-value |
| `concordance` | Float | Concordance index (C-statistic) |
| `n_obs` | Integer | Number of observations (intervals) |
| `n_events` | Integer | Number of events (completions) |
