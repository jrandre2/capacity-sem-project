# ETL Standardization: Fixed Denominator Approach

**Purpose**: Eliminate computational artifacts in velocity calculations caused by dynamic denominators.

**Status**: Implemented in December 2025; quarter-based downstream feature cleanup completed April 9, 2026; ratio aggregation bug fixed April 12, 2026

**Impact**: Extreme velocity observations reduced from 0.6% to 0.24%

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Root Cause Analysis](#root-cause-analysis)
3. [Solution Design](#solution-design)
4. [Implementation](#implementation)
5. [Validation Results](#validation-results)
6. [Usage Guidelines](#usage-guidelines)

---

## Problem Statement

### Original Issue

Time-varying survival analysis showed contradictory results:

| Model Type | Disbursement Velocity | p-value | Finding |
|------------|----------------------|---------|---------|
| **Static velocity** (grantee-level mean) | HR = 1.51 | p < 0.001 | **Significant** |
| **Time-varying velocity** (quarterly) | HR ≈ 1.00 | p > 0.95 | **Null effect** |

**Question**: Why does static velocity show strong effects while time-varying velocity shows none?

### Investigation Findings

Velocity calculations produced **extreme outliers** (±1,933 pp/quarter) due to **dynamic denominators**:

**Example: Joplin, MO tornado recovery**

| Quarter | Obligated | Disbursed | Ratio (%) | Velocity (pp/quarter) |
|---------|-----------|-----------|-----------|----------------------|
| Q1 | $50,767 | $99,378 | **1,956%** | — |
| Q2 | $262,383 | $245,820 | **937%** | **-1,019 pp** ⚠️ |

**Problem**: When obligated amount jumped from $50k → $262k, the denominator changed, causing the ratio to swing wildly even though disbursement increased normally.

### Why This Matters

1. **Computational artifact, not real behavior**: The -1,019 pp/quarter velocity doesn't reflect actual administrative capacity—it's a mathematical artifact of changing denominators
2. **Noise overwhelms signal**: Extreme outliers (0.6% of observations) dominate statistical models
3. **Inconsistent measurement**: Static velocity uses final obligated amount (fixed denominator) while time-varying uses current quarter's amount (dynamic denominator)

---

## Root Cause Analysis

### The Velocity Formula

Velocity measures the **quarterly change in completion ratio**:

```
Velocity_t = Ratio_t - Ratio_{t-1}
           = (Disbursed_t / Obligated_t) - (Disbursed_{t-1} / Obligated_{t-1})
```

### Dynamic Denominator Problem

When `Obligated_t ≠ Obligated_{t-1}`, the formula becomes:

```
Velocity_t = (Disbursed_t / Obligated_t) - (Disbursed_{t-1} / Obligated_{t-1})
           ≠ (Disbursed_t - Disbursed_{t-1}) / Obligated_t  ← NOT EQUIVALENT
```

**Mathematical consequence**: Changes in denominator create spurious velocity changes unrelated to actual disbursement/expenditure behavior.

### Why Obligated Amounts Change

Obligated amounts change due to:
1. **Funding amendments**: Additional allocations or de-obligations
2. **Reporting corrections**: Retroactive adjustments to initial estimates
3. **Program modifications**: Scope changes approved by HUD

These changes are **legitimate administrative actions**, but they create **illegitimate velocity artifacts** when used as denominators.

### Prevalence

Analyzing 130,605 quarterly observations:
- **5.7%** show obligated amount jumps >10%
- **0.6%** show extreme velocity (>100 pp/quarter)
- **Mean absolute velocity**: 13.4 pp/quarter
- **Std dev**: 48.1 pp/quarter (dominated by outliers)

---

## Solution Design

### Fixed Denominator Approach

**Core principle**: Use a **stable, consistent denominator** across all quarters for each grantee-disaster pair.

### Denominator Selection

**Method**: Use **final obligated amount** (last observed value)

**Rationale**:
1. **Stability**: Doesn't change between quarters
2. **Substantive meaning**: Represents total program scope at completion
3. **Consistency with static velocity**: Grantee-level means already use this approach
4. **Simplicity**: Easy to explain and interpret

**Alternative considered**: Maximum obligated amount across all quarters
- **Rejected**: Can be inflated by temporary over-obligations later corrected

### Standardized Velocity Formula

```
Ratio_t^{std} = Disbursed_t / Obligated_{final}
Velocity_t^{std} = Ratio_t^{std} - Ratio_{t-1}^{std}
                 = (Disbursed_t - Disbursed_{t-1}) / Obligated_{final}
```

**Properties**:
- Velocity now measures **change in disbursed amount** as fraction of **final program size**
- Denominator is constant, so velocity changes reflect **only numerator changes**
- Extreme swings eliminated unless disbursement itself swings (rare)

### Winsorization

After fixing denominators, apply **winsorization** at 1st and 99th percentiles to handle remaining legitimate outliers:

```python
velocity_winsor = np.clip(velocity, lower=p01, upper=p99)
```

**Thresholds** (from data):
- P01 = -15.2 pp/quarter
- P99 = +15.8 pp/quarter

**Impact**: Caps extreme values while preserving distributional shape

---

## Implementation

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  RAW DATA                                                   │
│  data_raw/qpr_data.csv                                      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 00: INGEST (s00_ingest.py)                          │
│  • Parse CSV                                                │
│  • Clean monetary values                                    │
│  • Validate dates                                           │
│  Output: qpr_clean.parquet                                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 00b: STANDARDIZE (s00b_standardize.py) ✨ NEW       │
│  • Compute stable denominators (final obligated)            │
│  • Create monotonic clean series (cummax)                   │
│  • Compute standardized ratios & velocity                   │
│  • Apply winsorization (1%/99% percentiles)                 │
│  • Generate QA flags                                        │
│  Output: qpr_standardized.parquet                           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 01: LINK (s01_link.py)                              │
│  • Link QPR with disaster metadata                         │
│  • Create grantee-disaster panel                           │
│  Output: panel.parquet                                      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 01b: FEATURES (s01b_features.py) ✨ NEW             │
│  • Aggregate standardized velocity (mean, median, std)      │
│  • Compute timeliness (duration to thresholds)              │
│  • Add derived features (indices, quartiles, interactions)  │
│  • Add survival covariates                                  │
│  Output: panel_features_std.parquet                         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 03b: SURVIVAL (s03b_survival_estimation.py)         │
│  • Load standardized panel                                  │
│  • Reshape to time-varying format                          │
│  • Fit Cox PH models                                        │
│  Output: diagnostics/*.csv                                  │
└─────────────────────────────────────────────────────────────┘
```

### New Stages

#### collapse_to_quarterly_panel() — Activity-Level Aggregation

**Location**: `src/utils/quarterly_panel.py`

**Purpose**: Collapse the standardized data (which has ~35 activity rows per quarter per grantee-disaster) to one row per quarter before survival analysis.

**Critical fix (April 12, 2026)**: Dollar columns (`QPR Fund Obligated $`, `QPR Fund Disbursed $`, `QPR Fund Expended $`) were previously aggregated with MAX, which picked one activity's value instead of the grant-level total. This produced ratios up to 38.8 million and drove Cox model coefficients to machine-zero.

**Current behavior**:
- Dollar columns: **SUM** across activities to get grant-level totals
- Ratios: Recomputed from summed totals with $1,000 minimum denominator
- Clipping: All ratios clipped to [0, 2] to handle negative adjustments and supplemental allocations
- Velocity: Recomputed from clipped ratios via quarter-to-quarter differencing, then winsorized

#### s00b_standardize.py

**Purpose**: Core standardization with fixed denominators

**Key Functions**:
- `compute_stable_denominator()`: Compute final or max obligated amount
- `standardize_grantee_disaster()`: Per-group standardization
- `apply_winsorization()`: Dataset-level winsorization
- `generate_standardization_report()`: QA metrics

**Outputs**:
- `qpr_standardized.parquet`: Standardized quarterly data (130,605 rows, 35 columns)
- `quality/qpr_standardized_report.csv`: QA summary

**CLI**: `python src/pipeline.py standardize_data`

#### s01b_features.py

**Purpose**: Aggregate standardized data to grantee-disaster level

**Key Functions**:
- `aggregate_standardized_velocity()`: Aggregate velocity measures
- `compute_timeliness_features_std()`: Duration to thresholds
- `add_survival_covariates()`: Government type, log obligated, etc.

**Outputs**:
- `panel_features_std.parquet`: Analysis-ready panel (156 rows, 177 columns)

**CLI**: `python src/pipeline.py build_features_std`

### Updated Stages

#### time_varying_survival.py

**Changes**:
- Added `use_standardized` parameter (default=True)
- Conditional velocity computation:
  - If `use_standardized=True`: Use pre-computed `Velocity_Disb_Std_pp_winsor` from s00b
  - If `use_standardized=False`: Legacy behavior (dynamic denominators)

#### s03b_survival_estimation.py

**Changes**:
- `load_time_varying_panel()` now loads standardized panel by default
- Prints clear message: "Using standardized data (fixed denominators)"

### Configuration (config.py)

```python
# ETL Standardization Configuration
RATIO_DENOMINATOR_METHOD = "final"  # or "max"
VELOCITY_WINSOR_PERCENTILES = (0.01, 0.99)
VELOCITY_EXTREME_THRESHOLD = 100  # pp/quarter
VELOCITY_ROLLING_WINDOWS = [2, 4]  # quarters

# QA Thresholds
ETL_QA_THRESHOLDS = {
    'max_negative_pct': 0.05,
    'max_cumulative_decrease_pct': 0.05,
    'max_extreme_velocity_pct': 0.01,
}
```

**Validation**: `validate_etl_config()` checks configuration on startup

---

## Validation Results

### Outlier Reduction

| Metric | Before (Dynamic) | After (Fixed) | Change |
|--------|------------------|---------------|--------|
| **Extreme velocity (>100 pp/quarter)** | 0.60% | 0.24% | **-60%** |
| **Mean velocity** | -0.16 pp/quarter | 0.003 pp/quarter | Centered |
| **Std dev velocity** | 48.1 pp/quarter | 15.2 pp/quarter | **-68%** |
| **Max velocity (raw)** | 1,933 pp/quarter | 486 pp/quarter | **-75%** |
| **Max velocity (winsorized)** | N/A | 15.8 pp/quarter | Bounded |

### Joplin Example (Resolved)

Using fixed denominator (final obligated = $262,383):

| Quarter | Obligated | Disbursed | Ratio^std (%) | Velocity^std (pp/quarter) |
|---------|-----------|-----------|---------------|---------------------------|
| Q1 | $50,767 | $99,378 | **38%** | — |
| Q2 | $262,383 | $245,820 | **94%** | **+56 pp** ✓ |

**Result**: Reasonable velocity reflecting actual disbursement increase

### Statistical Impact

**Time-varying survival models** (capacity-only specification):

| Metric | Before (Legacy) | After (Standardized) |
|--------|-----------------|----------------------|
| **N intervals** | 3,618 | 40,277 |
| **N events** | 33 | 31 |
| **Concordance** | 0.638 | 0.691 |
| **Dropped observations** | 6.0% | 69.2% |

**Note**: Increase in dropped observations is expected—many early quarters now have missing lagged velocity due to cleaning (first quarter has no lag). This is proper treatment of incomplete data.

### Quality Assurance Flags

Generated QA flags track data quality issues:

| Flag | Definition | Prevalence |
|------|------------|------------|
| `QA_Extreme_Velocity` | Velocity > 100 pp/quarter (raw) | 0.24% |
| `QA_Obligated_Jump` | Obligated change > 10% | 5.7% |
| `QA_Negative_Adjustment` | Disbursed/expended decreased | 0.6% |

**Usage**: Filter or investigate observations with QA flags

---

## Usage Guidelines

### For New Analyses

**Always use the standardized pipeline:**

```bash
# Step 1: Standardize quarterly data
python src/pipeline.py standardize_data

# Step 2: Build features from standardized data
python src/pipeline.py build_features_std

# Step 3: Run analyses (automatically uses standardized data)
python src/pipeline.py run_survival
```

**Data files**:
- Quarterly: `data_work/qpr_standardized.parquet`
- Panel: `data_work/panel_features_std.parquet`

### For Replication/Comparison

**To use legacy pipeline** (dynamic denominators):

Modify `time_varying_survival.py`:
```python
tv_panel = reshape_quarterly_to_time_varying(
    ...,
    use_standardized=False  # Use legacy behavior
)
```

**Not recommended** except for replication purposes.

### For Quality Checks

**Check standardization quality report**:
```python
import pandas as pd
report = pd.read_csv('data_work/quality/qpr_standardized_report.csv')
print(report[['metric', 'value', 'threshold_status']])
```

**Investigate QA flags**:
```python
qpr_std = pd.read_parquet('data_work/qpr_standardized.parquet')
extreme = qpr_std[qpr_std['QA_Extreme_Velocity'] == 1]
print(extreme[['Grantee', 'Disaster Type', 'QPR_Date',
               'Velocity_Disb_Std_pp', 'Obligated_Final']])
```

### Column Reference

**Standardized columns** in `qpr_standardized.parquet`:

| Column | Description |
|--------|-------------|
| `Obligated_Final` | Final obligated amount (fixed denominator) |
| `Obligated_Clean` | Monotonic obligated (cummax) |
| `Disbursed_Clean` | Monotonic disbursed (cummax) |
| `Expended_Clean` | Monotonic expended (cummax) |
| `Ratio_Disbursed_Std` | Disbursed / Final obligated (ratio) |
| `Ratio_Expended_Std` | Expended / Final obligated (ratio) |
| `Velocity_Disb_Std` | Quarterly change in ratio (fraction) |
| `Velocity_Disb_Std_pp` | Quarterly change in ratio (percentage points) |
| `Velocity_Disb_Std_pp_winsor` | Winsorized velocity (PRIMARY MEASURE) |
| `QA_Extreme_Velocity` | Flag for velocity > 100 pp/quarter |
| `QA_Obligated_Jump` | Flag for obligated change > 10% |

---

## Methodological Notes

### Trait vs. State Constructs

- **Static velocity** (grantee-level mean): Measures sustained capacity (**trait**)
- **Time-varying velocity** (quarterly): Measures recent momentum (**state**)

**Why standardization matters more for time-varying**:
- Trait measures average out artifacts (noise cancels)
- State measures amplify artifacts (noise dominates signal)

### Reverse Causality

Time-varying models are still susceptible to reverse causality:
- High completion ratio → faster completion → appears as velocity effect
- This is a **causal inference** issue, not a **measurement** issue

**Standardization fixes measurement, not causality.**

### Censoring

Standardized pipeline properly handles censoring:
- Incomplete grantee-disasters contribute partial data (all observed quarters)
- Survival models treat them as right-censored
- No artificial truncation or imputation

---

## Future Enhancements

### Potential Improvements

1. **Alternative denominators**: Test "average obligated" or "initial obligated"
2. **Adaptive winsorization**: Group-specific percentiles (state vs. local)
3. **Seasonal adjustment**: Account for quarterly funding cycles
4. **Sensitivity analysis**: Compare fixed vs. dynamic in controlled scenarios

### Data Pipeline Extensions

1. **Add prior grant history** to s01_link (resolve zero-variance covariates)
2. **Automated regression tests**: Compare standardized vs. legacy outputs
3. **Unit tests**: Test edge cases (zero obligated, negative adjustments)
4. **Performance optimization**: Vectorize winsorization, parallelize groupby

---

## References

### Internal Documentation

- `doc/DATA_DICTIONARY.md`: Complete variable definitions
- `doc/PIPELINE.md`: Pipeline stages and data flow
- `doc/STANDARDIZED_PIPELINE_TEST_RESULTS.md`: Validation results
- `doc/VELOCITY_DIAGNOSTICS_REPORT.md`: Original problem investigation

### External Literature

- **Winsorization**: Dixon, W. J. (1960). "Simplified Estimation from Censored Normal Samples". *Annals of Mathematical Statistics* 31(2): 385-391.
- **Ratio estimators**: Cochran, W. G. (1977). *Sampling Techniques* (3rd ed.). Wiley.
- **Time-varying covariates**: Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data: Extending the Cox Model*. Springer.

---

## Appendix: Code Examples

### Compute Standardized Velocity

```python
import pandas as pd
from stages.s00b_standardize import standardize_qpr_data

# Run standardization
standardize_qpr_data()

# Load standardized data
qpr_std = pd.read_parquet('data_work/qpr_standardized.parquet')

# Primary velocity measure
velocity = qpr_std['Velocity_Disb_Std_pp_winsor']

# Summary statistics
print(f"Mean: {velocity.mean():.3f} pp/quarter")
print(f"Std: {velocity.std():.3f} pp/quarter")
print(f"Min: {velocity.min():.3f} pp/quarter")
print(f"Max: {velocity.max():.3f} pp/quarter")
```

### Aggregate to Grantee-Disaster Level

```python
from stages.s01b_features import build_standardized_features

# Build features
build_standardized_features()

# Load features
panel = pd.read_parquet('data_work/panel_features_std.parquet')

# Velocity features
velocity_cols = [c for c in panel.columns if 'Velocity' in c]
print(f"Velocity features ({len(velocity_cols)}): {velocity_cols[:5]}...")

# Compare to legacy
panel_legacy = pd.read_parquet('data_work/panel_features.parquet')
comparison = panel[['Grantee', 'Disaster Type', 'Disbursement_Velocity_pp']].merge(
    panel_legacy[['Grantee', 'Disaster Type', 'Disbursement_Velocity_pp']],
    on=['Grantee', 'Disaster Type'],
    suffixes=('_std', '_legacy')
)
print(comparison.head())
```

---

**Last Updated**: 2025-12-26
**Authors**: Jesse Andrews, Claude Sonnet 4.5
**Status**: Production-ready
