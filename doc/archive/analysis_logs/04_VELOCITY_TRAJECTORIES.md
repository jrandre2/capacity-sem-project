# Phase 2 Week 5: Temporal Dynamics - Phase-Specific Velocity & Trajectory Clustering

**Date**: December 26, 2025
**Branch**: `analysis/alternative-capacity-measures`
**Objective**: Test if velocity effects differ across program phases (early/mid/late) and identify distinct velocity trajectory patterns

---

## Executive Summary

Phase 2 Week 5 investigated **when** velocity matters by decomposing program timelines into phases and clustering velocity trajectories. Key findings:

1. **Late-phase velocity dominates**: When all three phases are included, late velocity (HR=5.00, p=0.040) is the strongest predictor of completion, while early and mid become non-significant
2. **Fast-Consistent trajectory completes faster**: Programs with sustained high velocity throughout (N=15, 11%) complete 23 quarters faster (45 vs 68 quarters) with higher completion rates (86.7% vs 80.0%)
3. **Temporal heterogeneity confirmed**: Velocity effects are not constant across program timeline - they intensify in the closeout phase

**Implication**: Administrative capacity matters most during late-stage program execution, suggesting interventions should focus on sustaining momentum through closeout rather than just early launch.

---

## Analysis 1: Phase-Specific Velocity

### Method

- **Timeline segmentation**: Divided each program's observation period into thirds
  - Early phase: First 33% of observed quarters
  - Mid phase: Middle 33% of observed quarters
  - Late phase: Final 33% of observed quarters

- **Phase velocity metrics**:
  - `Velocity_Early`: Mean expenditure velocity in first third
  - `Velocity_Mid`: Mean expenditure velocity in middle third
  - `Velocity_Late`: Mean expenditure velocity in final third
  - `Velocity_Acceleration`: Change from early to late (Late - Early)

- **Cox PH models**:
  1. Overall velocity (baseline replication)
  2. Early velocity only
  3. All three phases simultaneously
  4. Velocity acceleration

### Results

| Model | N | Events | Key Finding | HR | p-value |
|-------|---|--------|-------------|-----|---------|
| Overall Velocity | 143 | 106 | Baseline effect | 4.44 | 0.012 |
| Early Velocity Only | 138 | 106 | Early momentum significant | 2.51 | 0.008 |
| **Three-Phase** | **138** | **106** | **Late velocity dominates** | | |
| ↳ Early | | | Non-significant when late included | 2.04 | 0.157 |
| ↳ Mid | | | Negative, non-significant | 0.37 | 0.184 |
| ↳ **Late** | | | **Strongest effect** | **5.00** | **0.040** |
| Acceleration | 138 | 106 | Not significant | 0.59 | 0.262 |

**Key Insight**: In the three-phase model, late velocity (HR=5.00) has a significant effect while early and mid velocity become non-significant. This suggests:
- Early velocity matters **only** when late velocity is unknown (univariate model)
- When competing, late velocity dominates
- Program completion is driven by late-stage acceleration, not early launch speed

### Interpretation

**Why late velocity matters more**:
1. **Closeout bottlenecks**: Final phase involves administrative hurdles (compliance, reporting, fund reconciliation)
2. **Cumulative learning**: Experienced programs accelerate through final tasks
3. **Political pressure**: Completion deadlines create urgency in late phase
4. **Sunk cost commitment**: Programs near completion mobilize resources to finish

**Policy implication**: Target technical assistance and monitoring to programs in late stages (>67% of timeline) rather than just early launch.

---

## Analysis 2: Trajectory Clustering

### Method

- **K-means clustering** on velocity time series (quarterly velocity sequences)
  - Limited to first 80 quarters (20 years) to focus on meaningful recovery period
  - Filtered to programs with ≥12 quarters of data (N=131)
  - k=3 clusters (optimal for sample size)
  - Standardized velocity values before clustering

- **Cluster labeling**: Based on early vs late velocity profiles
  - Fast-Consistent: High velocity throughout (early >0.2, late >0.2)
  - Slow-Ramp: Low early, high late (early <-0.1, late >0.2)
  - Accelerating: Increasing velocity (late - early >0.3)
  - Moderate: Typical velocity profile

### Results

| Trajectory Cluster | N | % | Events | Completion Rate | Median Survival Time |
|--------------------|---|---|--------|-----------------|---------------------|
| Moderate | 115 | 88% | 92 | 80.0% | 68 quarters |
| **Fast-Consistent** | **15** | **11%** | **13** | **86.7%** | **45 quarters** |
| Accelerating | 1 | 1% | 0 | 0% | - (outlier) |

**Key Finding**: Fast-Consistent trajectory completes **23 quarters faster** (45 vs 68 quarters) with **6.7 percentage points higher completion rate** (86.7% vs 80.0%).

### Cluster Profiles

#### Moderate (N=115, 88%)
- **Early velocity**: -0.044 (slightly negative, slow start)
- **Late velocity**: -0.048 (slightly negative)
- **Overall**: -0.024 (slightly below average)
- **Pattern**: Typical slow, steady progression
- **Outcome**: 80% complete, 68 quarters median survival

#### Fast-Consistent (N=15, 11%)
- **Early velocity**: 0.370 (high positive)
- **Late velocity**: 0.377 (high positive, sustained)
- **Overall**: 0.140 (above average)
- **Pattern**: Sustained high spending velocity throughout
- **Outcome**: 86.7% complete, 45 quarters median survival (1.5x faster)

#### Accelerating (N=1, outlier)
- **Early velocity**: -0.466 (very low)
- **Late velocity**: -0.148 (low but improving)
- **Overall**: 0.612 (very high, driven by late surge)
- **Pattern**: Extreme acceleration, likely data artifact
- **Outcome**: 0% complete, 148 quarters (censored)

### Kaplan-Meier Survival Analysis

The survival curves show:
- **Fast-Consistent trajectory**: Steepest hazard slope, earliest median survival (45 quarters)
- **Moderate trajectory**: Gradual hazard, delayed median survival (68 quarters)
- **Accelerating**: Flat hazard (no events), censored at 148 quarters

**Survival probability gap**: By quarter 45, Fast-Consistent programs have ~50% survival (50% completed), while Moderate programs have ~75% survival (25% completed).

---

## Reconciliation with Prior Findings

### Phase-Specific Velocity vs Overall Velocity

| Finding | Overall Velocity | Phase-Specific | Trajectory Clustering |
|---------|------------------|----------------|----------------------|
| Velocity effect | HR=4.44, p=0.012 | Late HR=5.00, p=0.040 | Fast-Consistent completes 1.5x faster |
| Timing | Assumed constant | Late phase dominates | Sustained velocity throughout |
| Mechanism | Unclear | Closeout acceleration | Consistent administrative capacity |

**Consistency**: All three approaches confirm velocity predicts completion, but temporal dynamics reveal:
1. **Univariate velocity** captures a **late-phase effect** (not early launch)
2. **Fast-Consistent programs** sustain high velocity **across all phases**, not just early
3. **Acceleration** (late - early) is not significant, suggesting sustained capacity > volatile capacity

### Comparison to Phase 2 Week 4 (Stage-Specific Bottlenecks)

| Phase 2 Week 4 | Phase 2 Week 5 |
|----------------|----------------|
| **WHERE**: Bottlenecks in obligate→disburse stage | **WHEN**: Late phase (closeout) |
| Stage1_Efficiency moderation | Late velocity dominates |
| High-capacity contexts (Stage1 >0.70) | Fast-Consistent trajectory (sustained velocity) |

**Integration**: High Stage1_Efficiency (disbursement capacity) enables **sustained velocity**, which manifests as late-phase acceleration. Programs that can maintain high velocity through closeout complete faster.

---

## Data Outputs

### Feature Engineering
- **File**: `src/stages/s01b_features.py` (lines 377-471)
- **Function**: `compute_phase_specific_velocity()`
- **New columns** (7):
  - `Velocity_Early`, `Velocity_Early_median`
  - `Velocity_Mid`, `Velocity_Mid_median`
  - `Velocity_Late`, `Velocity_Late_median`
  - `Velocity_Acceleration`

### Panel Data
- **File**: `data_work/panel_features_std.parquet`
- **Updated**: 202 columns (up from 195)
- **Records**: 156 grantee-disaster pairs

### Analysis Scripts
1. **run_phase_specific_analysis.py**: Cox PH models by program phase
2. **run_trajectory_clustering.py**: K-means clustering + Kaplan-Meier curves

### Results Files
- `data_work/diagnostics/phase_specific_velocity.csv`: Phase-specific Cox PH results
- `data_work/diagnostics/temporal_dynamics_trajectory_clusters.csv`: Cluster summary statistics
- `data_work/diagnostics/trajectory_cluster_assignments.csv`: Grantee-disaster cluster labels

### Visualizations
- `figures/velocity_trajectories_kmeans.png`: Cluster velocity profiles (3 panels)
- `figures/kaplan_meier_by_trajectory.png`: Survival curves by trajectory cluster

---

## Limitations

1. **Small Fast-Consistent cluster**: Only 15 programs (11%) with sustained high velocity - limits generalizability
2. **Single Accelerating outlier**: Cannot test if acceleration trajectory is viable pattern (N=1)
3. **Unbalanced clustering**: 88% of programs in "Moderate" cluster suggests limited velocity heterogeneity
4. **Phase definition**: Dividing by timeline thirds may not capture meaningful program milestones (e.g., disaster phases vs administrative phases)
5. **Right-censoring**: Survival times may be underestimated for incomplete programs

---

## Next Steps (Phase 2 Week 6)

**Learning Curves & Experience Effects**:
- Test if prior CDBG-DR experience amplifies velocity effects
- Experience × Velocity interaction models
- Stratified analysis: Novice (N=83) vs Experienced (N=73) grantees
- Learning curve: Does velocity improve over successive grants?

**Research question**: Do experienced grantees leverage velocity more effectively to complete faster?

**Hypothesis**: If experience × velocity interaction HR > 1.0 → experienced grantees convert velocity into completion more efficiently.

---

## Conclusion

Phase 2 Week 5 established **when** velocity matters through two complementary analyses:

1. **Phase-specific velocity**: Late-phase velocity (HR=5.00, p=0.040) is the strongest predictor when all phases are modeled together
2. **Trajectory clustering**: Fast-Consistent programs (sustained high velocity, N=15) complete 23 quarters faster (45 vs 68 quarters)

**Mechanistic insight**: Administrative capacity matters most during **program closeout**, not early launch. Programs that sustain high spending velocity throughout (especially late phase) complete significantly faster.

**Policy recommendation**: Technical assistance should prioritize **late-stage support** (compliance, reconciliation, reporting) rather than just early-stage planning. Monitoring systems should flag programs with declining late-phase velocity as at-risk for non-completion.
