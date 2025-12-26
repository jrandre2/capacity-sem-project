# Time-Varying Survival Analysis: Technical Guide

## Overview

This document provides a comprehensive technical guide to the time-varying survival analysis implementation for the Capacity-SEM project.

## The Methodological Problem

### Reverse Causality with Static Ratios

The original analysis computed capacity ratios as:

```python
ratios = qpr_quarterly.groupby(['Grantee', 'Disaster Type'])[ratio_cols].mean()
```

This creates **post-outcome bias**:

1. **Static ratios include post-completion data**: For a program completing at month 48, the ratio is the mean across all quarters including Q1-Q16 (months 0-48).

2. **Reverse causality**: Completion timing affects predictor values rather than predictors affecting completion timing.

3. **Upward bias**: Programs that complete quickly have fewer quarters to accumulate high ratios, while programs that complete slowly have more time to improve ratios. This biases effect estimates upward.

**Example**:
- Program A completes in 12 months with mean ratio 0.70
- Program B completes in 60 months with mean ratio 0.85
- Program B's high ratio partially reflects it having more time to improve

### The Time-Varying Solution

Use **time-varying covariates** with lagging:

- At time t, capacity ratio reflects cumulative performance through time t-1 (1-quarter lag)
- Predictor values cannot include information from the contemporaneous or future outcome period
- Eliminates reverse causality

## Data Structure Transformation

### Input: Quarterly Panel

One row per (grantee-disaster, quarter):

```
Grantee | Disaster | Quarter | Obligated_Cum | Disbursed_Cum | Expended_Cum
NYC     | Sandy    | 1       | 100M          | 40M           | 20M
NYC     | Sandy    | 2       | 100M          | 55M           | 35M
NYC     | Sandy    | 3       | 100M          | 70M           | 50M
...
NYC     | Sandy    | 16      | 100M          | 95M           | 95M (COMPLETED)
```

### Output: Time-Varying Survival Panel

One row per (grantee-disaster, interval):

```
Grantee | Disaster | start | stop | E | Ratio_disb_lag1 | Ratio_exp_lag1 | Gov_Type | Log_Grant
NYC     | Sandy    | 0     | 3    | 0 | NaN             | NaN            | Local    | 17.2
NYC     | Sandy    | 3     | 6    | 0 | 0.40            | 0.50           | Local    | 17.2
NYC     | Sandy    | 6     | 9    | 0 | 0.55            | 0.64           | Local    | 17.2
...
NYC     | Sandy    | 45    | 48   | 1 | 0.95            | 0.98           | Local    | 17.2
```

**Key features**:
- `start`/`stop`: Interval start/end times in months
- `E`: Event indicator (1=completion in this interval, 0=censored/ongoing)
- `Ratio_disb_lag1`: Lagged ratio (at time t, uses data through t-3 months)
- Static covariates repeated on every row

## Implementation Details

### Core Transformation Function

Located in `src/capacity_sem/models/time_varying_survival.py`:

```python
def reshape_quarterly_to_time_varying(
    qpr_quarterly: pd.DataFrame,
    panel_features: pd.DataFrame,
    lag_quarters: int = 1
) -> pd.DataFrame:
    """
    Transform quarterly panel to time-varying survival format.

    Parameters
    ----------
    qpr_quarterly : pd.DataFrame
        Quarterly data with cumulative totals
    panel_features : pd.DataFrame
        Grantee-disaster panel with outcomes
    lag_quarters : int
        Number of quarters to lag capacity ratios (default: 1)

    Returns
    -------
    pd.DataFrame
        Time-varying panel with start/stop/E columns
    """
```

**Steps**:

1. **Compute quarterly ratios** from cumulative totals
2. **Apply lag** using `.shift(lag_quarters)` within each grantee-disaster
3. **Expand to intervals**: Each quarter becomes [start, stop] interval in months
4. **Set event indicator**: E=1 only on final row if program completed
5. **Add static covariates**: Government type, grant size, experience, disaster year, population

### Lagging Mechanism

```python
# Compute ratio at each quarter
qpr_quarterly['Ratio_disbursed'] = (
    qpr_quarterly['QPR Fund Disbursed $'] /
    qpr_quarterly['QPR Fund Obligated $']
)

# Apply lag within each grantee-disaster group
qpr_quarterly['Ratio_disbursed_lag1'] = (
    qpr_quarterly.groupby(['Grantee', 'Disaster Type'])['Ratio_disbursed']
    .shift(1)  # 1-quarter lag
)
```

**Why lag=1 quarter?**
- Balances timeliness (recent data) with avoiding contemporaneous correlation
- Sensitivity checks test lag=0 (contemporaneous) and lag=2 (longer lag)

### Static Covariate Engineering

Located in `src/stages/s02_features.py`:

```python
def create_survival_covariates(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Create survival analysis covariates.

    Covariates created:
    - Government_Type_State: 1=State, 0=Local
    - Log_Obligated: log(1 + Max_Obligated)
    - Prior_Grant_Count: Number of prior grants (missing → 0)
    - Prior_Grant_Dollars_log: log(1 + prior dollars)
    - Disaster_Year: Parsed from disaster name
    - Population_log: log(1 + population)
    """
```

## Cox Proportional Hazards with Time-Varying Covariates

### Model Specification

$$h(t|X(t)) = h_0(t) \exp(\beta_1 X_1(t) + \beta_2 X_2(t) + \gamma' Z)$$

Where:
- $h_0(t)$: Baseline hazard (unspecified)
- $X_1(t), X_2(t)$: Time-varying capacity ratios (lagged)
- $Z$: Static covariates (government type, grant size, experience, etc.)
- $\beta_1, \beta_2$: Log hazard ratios for capacity
- $\gamma$: Log hazard ratios for static covariates

### Fitting with lifelines

```python
from lifelines import CoxPHFitter

cph = CoxPHFitter(penalizer=0.1)  # L2 regularization for stability
cph.fit(
    tv_data,
    duration_col='stop',
    event_col='E',
    start_col='start',  # CRITICAL for time-varying
    robust=False  # Will use bootstrap for SEs instead
)
```

**Key parameters**:
- `start_col='start'`: Indicates time-varying data structure
- `penalizer=0.1`: L2 regularization (Ridge penalty) for numerical stability
- `robust=False`: Standard ML SEs (bootstrap SEs computed separately)

### Interpreting Hazard Ratios

| HR | Interpretation |
|----|----------------|
| HR = 1.15 | 15% increase in completion hazard (15% faster) |
| HR = 0.85 | 15% decrease in completion hazard (15% slower) |
| HR = 1.00 | No effect |

**Example**:
- Disbursement ratio HR = 1.14 (per 0.1-unit increase)
- Interpretation: "A 0.1-unit increase in disbursement ratio (e.g., 0.5 → 0.6) is associated with a 14% increase in completion hazard, or equivalently, a 12% reduction in expected completion time."

## Bootstrap Clustered Standard Errors

### Why Bootstrap?

1. **Repeated observations**: Same grantee appears multiple times (different disasters, different time intervals)
2. **Within-cluster correlation**: Observations from same grantee are not independent
3. **Standard ML SEs**: Underestimate true uncertainty

### Algorithm

```python
def compute_bootstrap_se(
    tv_data: pd.DataFrame,
    cluster_col: str = 'Grantee',
    n_bootstrap: int = 1000
) -> pd.DataFrame:
    """
    Compute bootstrap clustered standard errors.

    Steps:
    1. Resample clusters (grantees) with replacement
    2. Include all observations for sampled grantees
    3. Refit Cox model on bootstrap sample
    4. Repeat n_bootstrap times
    5. Compute SE from bootstrap distribution
    """
```

**Implementation**:

```python
bootstrap_coefs = []
for b in range(n_bootstrap):
    # Resample grantees with replacement
    sampled_grantees = np.random.choice(
        grantees,
        size=len(grantees),
        replace=True
    )

    # Include all observations for sampled grantees
    bootstrap_sample = tv_data[tv_data['Grantee'].isin(sampled_grantees)]

    # Refit model
    cph_boot = CoxPHFitter(penalizer=0.1)
    cph_boot.fit(bootstrap_sample, duration_col='stop', event_col='E', start_col='start')

    # Store coefficients
    bootstrap_coefs.append(cph_boot.params_)

# Compute SE from bootstrap distribution
bootstrap_se = np.std(bootstrap_coefs, axis=0)
```

**Typical n_bootstrap**: 1000 iterations (configurable in `config.py`)

## Robustness Checks

### 1. Capacity Only (No Covariates)

```python
cph.fit(tv_data, duration_col='stop', event_col='E', start_col='start')
# Predictors: Ratio_disbursed_lag1, Ratio_expended_lag1 only
```

**Purpose**: Isolate capacity effects without confounding

### 2. Full Covariates (Main Specification)

```python
# Predictors: Capacity ratios + Government_Type_State + Log_Obligated +
#             Prior_Grant_Count + Prior_Grant_Dollars_log + Disaster_Year + Population_log
```

**Purpose**: Control for jurisdiction characteristics, program scale, experience, and disaster timing

### 3. Stratified by Government Type

```python
cph.fit(tv_data, duration_col='stop', event_col='E', start_col='start', strata='Government_Type')
```

**Purpose**: Allow separate baseline hazards for state vs. local governments

### 4. Alternative Lag Structures

- **lag=0**: Contemporaneous ratios (reverse causality present)
- **lag=1**: 1-quarter lag (main specification)
- **lag=2**: 2-quarter lag (longer lag)

**Purpose**: Assess sensitivity to lagging choice

## Diagnostic Checks

### 1. Proportional Hazards Assumption

**Test**: Schoenfeld residuals correlation with time

```python
from lifelines.statistics import proportional_hazard_test

ph_test = proportional_hazard_test(cph, tv_data, time_transform='rank')
```

**Interpretation**:
- Null hypothesis: Proportional hazards holds
- p > 0.05: PH assumption reasonable
- p < 0.05: Consider stratification or AFT models

### 2. Linearity of Continuous Predictors

**Test**: Martingale residuals vs. predictors

```python
martingale_residuals = cph.compute_residuals(tv_data, 'martingale')
plt.scatter(tv_data['Ratio_disbursed_lag1'], martingale_residuals)
```

**Interpretation**:
- Should scatter around zero with no pattern
- Non-random pattern → non-linear relationship → consider transformation

### 3. Overall Model Fit

**Test**: Cox-Snell residuals vs. theoretical exponential

```python
cox_snell_residuals = cph.compute_residuals(tv_data, 'cox-snell')
kmf = KaplanMeierFitter()
kmf.fit(cox_snell_residuals, event_observed=tv_data['E'])
plt.plot(kmf.survival_function_)
plt.plot([0, max(kmf.survival_function_.index)], [1, np.exp(-max(kmf.survival_function_.index))])
```

**Interpretation**:
- KM curve should follow theoretical exponential (45° line)
- Deviation indicates poor fit

### 4. Influential Observations

**Test**: Score residuals (dfbeta)

```python
score_residuals = cph.compute_residuals(tv_data, 'score')
influential = (np.abs(score_residuals) > 3).any(axis=1)
```

**Interpretation**:
- |score residual| > 3 → influential observation
- Check if results driven by outliers

### 5. Predicted Survival Curves

**Visualization**: Survival probability by capacity quartile

```python
# Stratify by capacity quartile
tv_data['capacity_quartile'] = pd.qcut(tv_data['Ratio_disbursed_lag1'], q=4)

# Plot survival curves for each quartile
for q in [1, 2, 3, 4]:
    subset = tv_data[tv_data['capacity_quartile'] == q]
    kmf.fit(subset['stop'], event_observed=subset['E'])
    kmf.plot_survival_function(label=f'Q{q}')
```

**Interpretation**:
- Higher capacity quartiles → steeper decline (faster completion)
- Separation between curves indicates strong effect

## Expected Results vs. Current Results

### Current Results (Static Ratios - FLAWED)

| Model | Disbursement HR | p-value | Notes |
|-------|----------------|---------|-------|
| Cox PH | 4.367 | 0.006 | Upward bias from reverse causality |

**Interpretation**: "337% increase in completion hazard" (implausibly large)

### Expected Results (Time-Varying - UNBIASED)

| Model | Disbursement HR (per 0.1) | p-value | Notes |
|-------|---------------------------|---------|-------|
| Cox (No Covariates) | 1.12-1.16 | <0.05 | Capacity effect persists |
| Cox (Full Covariates) | 1.10-1.14 | <0.05 | Robust to controls |
| AFT Lognormal | 0.88-0.92 | <0.05 | TR < 1 = faster completion |

**Interpretation**: "12-16% increase in completion hazard per 0.1-unit increase in disbursement ratio" (realistic)

**Key insight**: Effect persists but is smaller and more realistic after addressing reverse causality and adding controls.

## Running the Pipeline

### Generate Time-Varying Panel

The time-varying panel is automatically generated in Stage 1:

```bash
source .venv/bin/activate
python src/pipeline.py build_panel
```

This creates `data_work/panel_time_varying.parquet`.

### Run Survival Analysis

```bash
python src/pipeline.py run_survival
```

**Outputs**:
- `data_work/diagnostics/survival_time_varying_cox_results.csv`
- `data_work/diagnostics/survival_hazard_ratios.csv`
- `data_work/diagnostics/survival_bootstrap_se.csv`
- `data_work/diagnostics/survival_robustness_checks.csv`
- `figures/survival_*.png`

**Runtime**: ~10-15 minutes (bootstrap with 1000 iterations)

## Configuration

Edit `src/config.py`:

```python
# Time-varying survival analysis parameters
TV_LAG_QUARTERS = 1  # Lag for capacity ratios (1 = 1-quarter lag)
BOOTSTRAP_ITERATIONS = 1000  # Number of bootstrap samples
BOOTSTRAP_CLUSTER_COL = 'Grantee'  # Cluster variable for bootstrap

# Covariates for survival models
SURVIVAL_COVARIATE_COLS = [
    'Government_Type_State',
    'Log_Obligated',
    'Prior_Grant_Count',
    'Prior_Grant_Dollars_log',
    'Disaster_Year',
    'Population_log',
]
```

## Troubleshooting

### Error: "KeyError: 'start'"

**Cause**: Time-varying panel not generated

**Solution**:
```bash
python src/pipeline.py build_panel  # Regenerate panel
```

### Error: "ConvergenceWarning: Newton-Raphson failed to converge"

**Cause**: Numerical instability in Cox model fitting

**Solutions**:
1. Increase penalizer: `penalizer=0.5` (more regularization)
2. Check for multicollinearity: Remove highly correlated covariates
3. Standardize continuous covariates

### Warning: "PH assumption violated (p < 0.05)"

**Cause**: Effect of predictor changes over time

**Solutions**:
1. Stratify by offending variable
2. Use AFT models instead of Cox
3. Include time interactions (advanced)

### Very wide confidence intervals

**Cause**: Small sample size or high variability

**Solutions**:
1. Increase bootstrap iterations: `BOOTSTRAP_ITERATIONS = 2000`
2. Check for influential observations (influence diagnostics)
3. Consider AFT models (more efficient if correctly specified)

## References

### Key Papers

- **Time-varying covariates in Cox models**: Therneau & Grambsch (2000), *Modeling Survival Data: Extending the Cox Model*
- **Bootstrap clustered SEs**: Cameron, Gelbach, & Miller (2008), "Bootstrap-Based Improvements for Inference with Clustered Errors"
- **Survival analysis diagnostics**: Harrell (2015), *Regression Modeling Strategies*

### Software Documentation

- **lifelines**: https://lifelines.readthedocs.io/en/latest/
- **Time-varying covariates syntax**: https://lifelines.readthedocs.io/en/latest/Time%20varying%20survival%20regression.html

## Summary

The time-varying survival analysis addresses the critical reverse causality flaw in the original static ratio approach by:

1. **Restructuring data** into interval format with start/stop times
2. **Lagging capacity ratios** by 1 quarter to avoid contemporaneous correlation
3. **Including full covariate set** to control for confounders
4. **Bootstrap clustered SEs** to account for repeated grantee observations
5. **Comprehensive diagnostics** to validate model assumptions

This produces **unbiased estimates** of capacity effects on disaster recovery completion timing, suitable for publication in Public Administration Review.
