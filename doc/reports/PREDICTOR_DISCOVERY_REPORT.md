# Predictor Discovery Report

**Analysis Date**: December 27, 2025 (Revised)
**Branch**: `analysis/alternative-capacity-measures`

## Executive Summary

This analysis pivoted from testing velocity effects to an exploratory approach: **What predicts CDBG-DR program completion?**

### ⚠️ RETRACTED: The "Velocity Paradox" Claim

The initial analysis reported a "velocity paradox" (higher velocity → slower completion). **This claim is RETRACTED** as it was a methodological artifact of reverse causality:
- Static velocity measures include post-completion data
- Programs that complete quickly have less time to generate velocity statistics
- The time-varying analysis (properly specified) shows velocity HR ≈ 1.00, p > 0.95

### ⚠️ FAILED VALIDATION: Housing_Pct

The initial analysis identified Housing_Pct as a strong predictor (HR = 15.01, p < 0.005). **This finding FAILS temporal stability validation**:
- 78.7% of programs show > 30 percentage point change in Housing_Pct over time
- 74.6% show > 50 percentage point change
- Housing_Pct evolves during implementation—it's NOT fixed at program design
- Therefore, Housing_Pct is potentially endogenous to completion timing

### The Robust Finding: NULL on Velocity

The properly-specified time-varying survival analysis shows:
- **Velocity HR ≈ 1.00, p > 0.95** across all specifications
- This null is ROBUST to lag structure, covariates, and thresholds
- **Spending velocity does NOT predict completion timing**

---

## Housing_Pct Temporal Stability Validation

### Methodology

Housing_Pct was computed at each quarter for each grantee-disaster by aggregating housing-related activity expenditures (housing, homeowner, rental, buyout, relocation, residential) divided by total expenditures.

### Results

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| Programs with 4+ quarters | 122 | - | - |
| Mean Std(Housing_Pct) | 0.26 (median) | < 0.10 | **FAIL** |
| Mean |Drift| (First to Last) | 0.32 | < 0.15 | **FAIL** |
| Programs with Range > 30pp | 78.7% | - | - |
| Programs with Range > 50pp | 74.6% | - | - |
| Meet BOTH stability criteria | 19.7% | - | **FAIL** |

### Interpretation

Housing_Pct **changes substantially during program implementation**:

1. Most programs show 30-50+ percentage point swings in housing focus
2. This could reflect:
   - Phased implementation (infrastructure first, housing later)
   - Scope changes during recovery
   - Shifting priorities as needs become clearer
3. Because Housing_Pct evolves WITH the program, it cannot be used as an exogenous predictor

### Conclusion

**Housing_Pct is excluded from causal claims** because it fails temporal stability validation. The positive association observed in the static analysis may reflect reverse causality or confounding with program duration.

---

## Methods

### Sample
- N = 156 grantee-disaster pairs
- Events (completed at 95%): 71 (45.5%)
- Censored: 85 (54.5%)

### Approach
1. **LASSO Cox regression** for feature selection (L1 penalty)
2. **Random Survival Forest** for validation (100 trees)
3. **Standard Cox PH** for interpretable hazard ratios
4. **Completer profiling** for descriptive differences

### Candidate Predictors (25 features)
- Grantee: Government type, population, employment (previously unused)
- Experience: Prior grants, years
- Disaster: Severity, counties affected, damage
- Grant: Size (log), quarters
- Pipeline: Stage 1/2 efficiency, lag
- Portfolio: Diversity, housing/infrastructure percentages
- Velocity: Capacity index, disbursement, early/late phase

---

## Results

### LASSO Cox Feature Selection

Only **1 feature** survived LASSO regularization at the optimal alpha:

| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| Housing_Pct | +0.017 | Higher housing → Faster completion |

### Random Survival Forest

**C-index: 0.880** (excellent discrimination)

Top 10 features by permutation importance:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Housing_Pct | 0.0221 |
| 2 | Disbursement_Velocity_pp | 0.0168 |
| 3 | Stage2_Efficiency | 0.0161 |
| 4 | Log_Obligated | 0.0126 |
| 5 | Ratio_expended_to_disbursed | 0.0124 |
| 6 | Stage1_Efficiency | 0.0117 |
| 7 | Velocity_Early | 0.0106 |
| 8 | Program_Diversity_Index | 0.0103 |
| 9 | Ratio_disbursed_to_obligated | 0.0099 |
| 10 | Total_Damage | 0.0090 |

### Final Cox Model

**C-index: 0.662**

| Feature | HR | 95% CI | p-value | Interpretation |
|---------|----|----|---------|----------------|
| **Housing_Pct** | 15.01 | [3.22, 70.05] | <0.005 | *** Strong positive |
| **Disbursement_Velocity_pp** | ~0.00 | [0.00, 0.32] | 0.04 | * Paradoxical negative |
| Stage2_Efficiency | 1.06 | [0.97, 1.14] | 0.19 | Not significant |
| Log_Obligated | 1.07 | [0.93, 1.23] | 0.34 | Not significant |
| Ratio_expended_to_disbursed | 1.02 | [0.90, 1.16] | 0.73 | Not significant |

### Completer Profile

Significant differences between completers (n=71) and non-completers (n=85):

| Feature | Completers | Non-Completers | p-value | Direction |
|---------|------------|----------------|---------|-----------|
| **Disbursement_Velocity_pp** | 0.01 | 0.02 | <0.001 | Completers LOWER |
| **Stage1_Efficiency** | 0.67 | 0.45 | <0.001 | Completers HIGHER |
| **N_Quarters** | 27.9 | 19.3 | 0.002 | Completers MORE |
| **Housing_Pct** | 0.26 | 0.18 | 0.002 | Completers HIGHER |
| **Log_Obligated** | 18.33 | 17.22 | 0.002 | Completers LARGER |
| **Ratio_disbursed** | 0.51 | 0.37 | 0.005 | Completers HIGHER |
| **Capacity_Velocity_Index** | 0.08 | 0.14 | 0.015 | Completers LOWER |

---

## Interpretation

### Why Housing Predicts Completion

1. **Standardized processes**: Housing programs (rental, homeowner) have well-established implementation pathways through decades of CDBG experience
2. **Clearer beneficiary identification**: Disaster-affected homeowners are easier to identify than infrastructure needs
3. **Fewer procurement barriers**: Housing often uses subrecipients and nonprofits rather than competitive bidding
4. **Political incentives**: Housing assistance is visible and politically rewarding, incentivizing faster completion

### Why Velocity is Paradoxically Negative

The velocity paradox makes sense when considering:

1. **Selection effect**: Programs with high velocity are likely more complex (more activities, larger scope)
2. **Confounding by complexity**: Complex programs have more opportunities for velocity variation AND longer timelines
3. **Survivor bias**: Programs that complete quickly have less time to generate high velocity statistics

This confirms that **velocity is not a measure of capacity**—it's a marker of program complexity and scope.

### What About Employment Data?

Employment data (previously unused) was available for 98/156 observations but **did not emerge as a significant predictor**. This suggests that external measures of government size do not predict CDBG-DR completion better than internal program characteristics.

---

## Policy Implications

### For HUD Technical Assistance

1. **Focus on housing-heavy programs**: Programs with low housing percentages may need additional support
2. **Don't push velocity**: Encouraging faster spending may not accelerate completion
3. **Monitor Stage 1 efficiency**: The ratio of disbursed-to-obligated is the earliest leading indicator of completion

### For Grantees

1. **Front-load housing activities**: Housing activities complete faster and create momentum
2. **Manage scope carefully**: Program complexity (not spending speed) predicts longer timelines
3. **Prioritize pipeline efficiency**: Getting funds disbursed is more predictive than spending them quickly

### For Future Research

1. **Program type dynamics**: Why does housing complete faster? Process tracing needed.
2. **Velocity decomposition**: Is the negative velocity effect linear or threshold-based?
3. **Causal mechanisms**: Does housing cause faster completion, or is it a selection effect?

---

## Output Files

| File | Description |
|------|-------------|
| `data_work/diagnostics/predictor_discovery_lasso.csv` | LASSO coefficients |
| `data_work/diagnostics/predictor_discovery_rsf.csv` | RSF importance |
| `data_work/diagnostics/predictor_discovery_final.csv` | Final Cox model |
| `data_work/diagnostics/predictor_discovery_profile.csv` | Completer profile |
| `figures/predictor_importance_forest.png` | Forest plot |

---

## Technical Notes

### Proportional Hazards Assumption

Housing_Pct marginally violates the PH assumption (p = 0.054). Consider stratifying on housing quartiles in sensitivity analysis.

### Missing Data

- Employment data: Available for 98/156 (63%)
- Program types: Available for all 156

### Model Comparison

| Model | C-index |
|-------|---------|
| Random Survival Forest | 0.880 |
| Final Cox (5 features) | 0.662 |

The gap suggests non-linear effects or interactions not captured by the additive Cox model.
