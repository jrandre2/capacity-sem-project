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

| Variable | Type | Description |
|----------|------|-------------|
| `Years_Experience` | Float | Years since first CDBG-DR grant |
| `Prior_Grant_Count` | Integer | Number of prior disaster grants |
| `Prior_Grant_Dollars` | Float | Cumulative prior obligated dollars |
| `Experience_Index` | Float | Composite experience score (0-1) |
| `Experience_Index_scaled` | Float | Z-score standardized experience |

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
