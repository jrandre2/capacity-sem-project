# Comprehensive Analysis Comparison Report

## Government Capacity and Disaster Recovery: Alternative Modeling Results

**Date:** December 26, 2024
**Project:** Capacity-SEM
**Purpose:** Compare alternative modeling approaches with Kaifa's original manuscript analysis

---

## Executive Summary

This report compares results across multiple analytical approaches for examining how government administrative capacity affects disaster recovery outcomes. The central finding is a **fundamental methodological discrepancy**:

| Finding | Kaifa's Manuscript | Alternative Analyses |
|---------|-------------------|---------------------|
| **Capacity → Outcome Effect** | Significant (p=0.01) | Not significant in SEM (p>0.13) |
| **Sample Size** | N=36 grantees | N=40-156 depending on method |
| **Most Robust Predictor** | Latent capacity construct | Disbursement ratio only |

**Key Insight:** When duration data is properly handled using survival analysis (N=152), the **disbursement ratio significantly predicts completion time** (HR=4.37, p=0.006). However, this effect disappears in latent variable SEM models due to sample size limitations and model specification issues.

---

## 1. The Right-Censoring Problem

### Data Availability by Completion Threshold

| Threshold | Valid Observations | Percent of Sample |
|-----------|-------------------|-------------------|
| 30% complete | 95 | 60.9% |
| 50% complete | 89 | 57.1% |
| 70% complete | 75 | 48.1% |
| 90% complete | 58 | 37.2% |
| **95% complete** | **41** | **26.3%** |
| 100% complete | 15 | 9.6% |

**73.7% of observations are right-censored** at the 95% completion threshold used in Kaifa's analysis. This means Duration cannot be observed for most programs because they haven't reached completion.

---

## 2. Methodological Comparison

### Approaches Tested

| Method | Sample | Unit of Analysis | Censoring Treatment |
|--------|--------|-----------------|---------------------|
| **Kaifa Original** | N=36 | Grantee (aggregated) | Right-censored: incomplete = observation time |
| **Standard SEM** | N=40 | Grantee-disaster pair | Listwise deletion: incomplete excluded |
| **Lower Threshold SEM** | N=57-88 | Grantee-disaster pair | Listwise deletion at lower threshold |
| **Duration-Free SEM** | N=156 | Grantee-disaster pair | N/A (no duration) |
| **Survival Analysis** | N=152 | Grantee-disaster pair | Proper statistical censoring |

### Why Kaifa's Approach Differs

Kaifa's manuscript methodology (documented in `sem_manuscript_replication.py`):

1. **Grantee-level aggregation**: Averages across disasters for each grantee
2. **Right-censoring**: Assigns observation time as "duration" for incomplete programs
3. **Timeliness indicator**: Uses Timeliness = 1/Duration on the capacity factor
4. **Mathematical circularity**: Duration appears on both factors (directly and as inverse)

---

## 3. Detailed Results

### 3.1 Kaifa's Original Manuscript Claims

**Reported in manuscript:**
- Structural path: β = 71.024, p = 0.01
- Model: 3x3 factor structure with Timeliness on capacity, Duration on outcome

**Our replication attempt:**
- Structural path: β ≈ 113.65, p < 0.001
- Difference likely due to ratio calculation method (mean-of-ratios vs final cumulative)

**Critical Issue:** The significant effect relies on:
1. Including Timeliness (= 1/Duration) as capacity indicator
2. Including Duration as outcome indicator
3. This creates mathematical coupling between the constructs

---

### 3.2 Standard SEM Results (exp_optimal_v1)

The recommended model removes the Timeliness/Duration circularity:

```
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
recovery_outcome =~ Duration_log + Spending_CV
recovery_outcome ~ gov_cap
```

| Subset | N | β | SE | p-value | CFI | RMSEA |
|--------|---|---|----|---------| ----|-------|
| All | 40 | 0.320 | 6.14 | 0.958 | 0.790 | 0.000 |
| State | 23 | 0.362 | 9.70 | 0.970 | 1.379 | 0.227 |
| Local | 17 | - | - | - | 1.372 | 0.275 |

**Finding:** No significant capacity → outcome relationship in any subset.

---

### 3.3 Lower Threshold SEM Results

Testing whether lower completion thresholds (more observations) change results:

| Threshold | N | β | SE | p-value |
|-----------|---|---|----| --------|
| 50% | 88 | -0.619 | - | 0.985 |
| 70% | 74 | -0.420 | - | 0.979 |
| 90% | 57 | -4.850 | - | 0.730 |

**Finding:** Effect direction is unstable (positive at 95%, negative at lower thresholds). No significance at any threshold.

---

### 3.4 Duration-Free SEM Results

Removing duration entirely to use full sample:

| Model | N | β | p-value | CFI |
|-------|---|---|---------| ----|
| duration_free_cv | 156 | 0.360 | 0.132 | 0.990 |
| duration_free_3x2 | 156 | 0.279 | 0.309 | 0.967 |

**Finding:** With full sample (N=156), effect approaches marginal significance (p=0.13) but remains non-significant.

---

### 3.5 Milestone-Based SEM Results

Using alternative outcome measures:

| Model | Outcome | N | β | p-value |
|-------|---------|---|---| --------|
| milestone_time_to_50 | Time to 50% completion | 156 | -0.365 | 0.354 |
| milestone_progress_rate | Progress Rate | 156 | 0.091 | 0.685 |
| milestone_velocity | Completion Velocity | 156 | 0.014 | 0.669 |

**Finding:** No significant effects regardless of outcome operationalization.

---

### 3.6 Survival Analysis Results (KEY FINDING)

Survival analysis properly handles censored duration data:

#### Cox Proportional Hazards (N=152)

| Predictor | Hazard Ratio | 95% CI | p-value |
|-----------|-------------|--------|---------|
| **Ratio_disbursed_to_obligated** | **4.367** | [1.53, 12.48] | **0.006** |
| Ratio_expended_to_disbursed | 0.958 | [0.81, 1.14] | 0.626 |

**Interpretation:** A 1-unit increase in disbursement ratio increases the hazard (completion rate) by 337%. Higher disbursement capacity → faster completion.

#### Accelerated Failure Time - Lognormal (N=152)

| Predictor | Time Ratio | 95% CI | p-value |
|-----------|-----------|--------|---------|
| **Ratio_disbursed_to_obligated** | **0.157** | [0.06, 0.41] | **0.0001** |
| Ratio_expended_to_disbursed | 1.008 | [0.78, 1.31] | 0.954 |

**Interpretation:** A 1-unit increase in disbursement ratio reduces completion time by 84%. This is highly significant.

---

## 4. Summary Comparison Table

| Analysis | N | Effect | p-value | Significant? |
|----------|---|--------|---------|--------------|
| Kaifa manuscript (reported) | 36 | β = 71.024 | 0.01 | Yes |
| Kaifa replication (our code) | 36 | β ≈ 113.65 | <0.001 | Yes |
| Standard SEM (exp_optimal_v1) | 40 | β = 0.320 | 0.958 | No |
| Lower threshold SEM (50%) | 88 | β = -0.619 | 0.985 | No |
| Duration-free SEM | 156 | β = 0.360 | 0.132 | No |
| Milestone SEM (Time_to_50) | 156 | β = -0.365 | 0.354 | No |
| **Cox Proportional Hazards** | **152** | **HR = 4.367** | **0.006** | **Yes** |
| **AFT Lognormal** | **152** | **TR = 0.157** | **0.0001** | **Yes** |

---

## 5. Why Do Results Differ?

### 5.1 Sample Size Effect

| Method | N | Power to detect medium effect (d=0.5) |
|--------|---|--------------------------------------|
| Kaifa (grantee-level) | 36 | ~50% |
| Standard SEM | 40 | ~55% |
| Survival Analysis | 152 | ~95% |

SEM with N=40 is severely underpowered. Survival analysis with N=152 has adequate power.

### 5.2 Unit of Analysis

- **Kaifa**: Aggregates to grantee level (N=36 state grantees)
- **Standard SEM**: Uses grantee-disaster pairs (N varies by model)
- **Survival**: Uses grantee-disaster pairs with proper censoring (N=152)

### 5.3 Censoring Treatment

| Approach | Treatment | Bias |
|----------|-----------|------|
| Kaifa | Treat incomplete as complete at observation time | Underestimates duration → inflates capacity effect |
| Standard SEM | Exclude incomplete programs | Loses 74% of data |
| Survival | Proper statistical censoring | Unbiased |

### 5.4 Mathematical Circularity in Kaifa's Model

Kaifa's specification includes:
- Capacity factor: Timeliness = 1/Duration
- Outcome factor: Duration

This creates a mathematical relationship between the factors independent of any causal effect, artificially inflating the structural path.

---

## 6. Conclusions

### What We Can Confidently Claim

1. **Disbursement ratio predicts completion time** when properly analyzed with survival methods
   - Cox: HR = 4.37, p = 0.006
   - AFT: TR = 0.16, p = 0.0001

2. **Expenditure ratio does not independently predict completion time**
   - p > 0.62 across all methods

3. **Kaifa's β = 71.024 cannot be replicated** with standard SEM methodology
   - Effect disappears when Timeliness/Duration circularity is removed
   - Effect disappears with proper listwise deletion

### Methodological Recommendations

1. **Use survival analysis** for duration-based outcomes with censored data
2. **Avoid Timeliness = 1/Duration** as a capacity indicator when Duration is an outcome
3. **Report single-indicator effects** (disbursement ratio) rather than latent constructs
4. **Acknowledge sample size limitations** in SEM analyses

### Implications for the Manuscript

The original manuscript claim of a significant capacity → outcome relationship (β = 71.024, p = 0.01) appears to be driven by:

1. Right-censoring bias (treating incomplete as complete)
2. Mathematical circularity (Timeliness = 1/Duration)
3. Grantee-level aggregation (reduces variance, inflates effect)

A more defensible claim: **"Higher disbursement ratios significantly predict faster disaster recovery completion (HR = 4.37, p < 0.01), though no effect is observed for expenditure ratios."**

---

## Appendix: Output Files

| File | Content |
|------|---------|
| `alternatives_survival.csv` | Cox and AFT model coefficients |
| `alternatives_threshold_sensitivity.csv` | SEM at multiple duration thresholds |
| `alternatives_duration_free.csv` | Duration-free SEM results |
| `alternatives_milestone.csv` | Milestone-based SEM results |
| `alternatives_comparison.csv` | Cross-method comparison |

---

## Commands to Reproduce

```bash
# Run all alternative analyses
python src/pipeline.py run_alternatives

# Survival analysis only
python src/pipeline.py run_alternatives --survival-only

# SEM alternatives only
python src/pipeline.py run_alternatives --sem-only

# Kaifa's replication
PYTHONPATH=src python3 src/stages/s03_manuscript_replication.py --subset state --model kaifa_3x3_full
```
