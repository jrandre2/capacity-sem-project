# Data Dictionary

## Overview

This document defines all variables used in the Capacity-SEM analysis.

---

## Identification Variables

| Variable | Type | Description |
|----------|------|-------------|
| `Grantee` | String | Name of the grantee (state or local government) |
| `Disaster Type` | String | DRGR disaster type identifier |
| `Disaster_Year` | Integer | Year of the disaster event |
| `Grant` | String | Grant identifier |
| `Appropriation` | String | Congressional appropriation |

---

## Raw QPR Variables

| Variable | Type | Description |
|----------|------|-------------|
| `QPR Fund Obligated $` | Float | Funds obligated in the quarter |
| `QPR Fund Disbursed $` | Float | Funds disbursed in the quarter |
| `QPR Fund Expended $` | Float | Funds expended in the quarter |
| `QPR Fund Expended Q $` | Float | Quarterly net change in expenditures |
| `QPR Actual Quarter` | String | Quarter identifier (e.g., "2020 Q3") |
| `Activity Type` | String | Type of recovery activity |

---

## Aggregated Panel Variables

| Variable | Type | Description |
|----------|------|-------------|
| `Total_Obligated` | Float | Sum of quarterly obligated amounts |
| `Total_Disbursed` | Float | Sum of quarterly disbursed amounts |
| `Total_Expended` | Float | Sum of quarterly expended amounts |
| `N_Quarters` | Integer | Number of reporting quarters |

---

## Capacity Indicators

### Primary Ratios

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `Ratio_disbursed_to_obligated` | Float | [0, 1+] | Total Disbursed / Total Obligated |
| `Ratio_expended_to_disbursed` | Float | [0, 1+] | Total Expended / Total Disbursed |
| `Ratio_expended_to_obligated` | Float | [0, 1+] | Total Expended / Total Obligated |

### Alternative Capacity Measures

| Variable | Type | Description |
|----------|------|-------------|
| `Progress_Rate` | Float | Completion percentage per quarter |
| `Startup_Lag` | Integer | Quarters before first expenditure |

---

## Outcome Indicators

### Duration Measures

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `Duration_of_completion` | Float | Months | Time to reach 95% expenditure |
| `Duration_log` | Float | ln(months) | Log-transformed duration |
| `Timeliness` | Float | 1/months | Inverse of duration (deprecated: use `Duration_log` instead; Timeliness = 1/Duration creates mathematical redundancy with duration measures) |

### Spending Consistency

| Variable | Type | Description |
|----------|------|-------------|
| `Quarter_by_quarter_variance_expended` | Float | Normalized std of quarterly spending |
| `Spending_CV` | Float | Coefficient of variation of quarterly spending |
| `Completion_Pct` | Float | Final expenditure / final obligated |

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

---

## Program Type Classification

| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| `Program_Type` | String | Housing, Infrastructure, Administration, Economic Development, Acquisition, Other | Major activity category |

---

## Grantee Classification

| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| `Government_Type` | String | State, Local | Government level |

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
