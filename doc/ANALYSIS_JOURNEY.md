# Analysis Journey: From SEM to Time-Varying Survival

## Timeline of Methodological Evolution

### Phase 1: Original SEM Approach (Kaifa's Manuscript)
- **Date**: Pre-December 2024
- **Method**: Structural equation modeling with latent constructs
- **Key finding**: β = 71.024, p = 0.01 (N=36-40 grantees)
- **Issues identified**:
  - Right-censoring: 73.7% incomplete at 95% threshold
  - Mathematical circularity: Timeliness = 1/Duration
  - Sample size: N=36-40 (grantee-level aggregation)
- **Status**: Archived in `manuscript_kaifa_archive/`

### Phase 2: Initial Survival Analysis (Static Ratios)
- **Date**: December 2024
- **Method**: Cox proportional hazards with static capacity ratios
- **Key finding**: HR = 4.37, p = 0.006 (N=152 grantee-disasters)
- **Issue identified**: Post-outcome bias (ratios computed across all quarters including post-completion)
- **Status**: Recognized as methodologically flawed

### Phase 3: Time-Varying Survival Analysis (December 26, 2024)
- **Date**: December 26, 2024
- **Method**: Cox PH with time-varying lagged capacity covariates
- **Implementation**:
  - 1-quarter lag on capacity ratios (eliminates reverse causality)
  - Full covariate set (7 predictors)
  - Bootstrap clustered SEs (1000 iterations, grantee-level)
  - Threshold sensitivity: 20%-100% in 5% intervals
- **Key finding**: **NULL RESULT**
  - Time-varying HR ≈ 1.05-1.15 (p = 0.22-0.69) across all thresholds
  - No significant effects at any threshold
  - Original significant result was artifact of methodological flaw
- **Status**: v0.2.0-time-varying-null-findings tag

### Phase 4: Velocity Mechanisms & Heterogeneity (Phases 2-4 Research Extension)
- **Date**: December 2025
- **Method**: Phase-specific velocity, trajectory clustering, heterogeneity analysis
- **Key findings (BEFORE bug discovery)**:
  - Late-phase velocity: HR=5.00, p=0.040
  - Novice grantees: HR=4.61, p=0.043
  - Administration programs: HR=30.74, p=0.004
  - Wildfire disasters: HR=51.09, p=0.002
- **Status**: Results invalidated by bug discovery (see Phase 5)

### Phase 5: Bug Discovery During Synthetic Peer Review (December 27, 2025)

This phase documents a critical discovery that invalidated all prior velocity findings.

#### 5.1 Synthetic Peer Review Initiated
- **Context**: Velocity manuscript prepared for PAR submission
- **Process**: Generated synthetic peer review with 7 major methodological concerns
- **Goal**: Stress-test claims before actual submission

#### 5.2 Sample Size Audit Reveals Anomaly
- **Reviewer Concern #1**: Internal inconsistencies in sample sizes (33 vs 106 events)
- **Investigation**: Why do different scripts show different event counts?
- **Discovery**: Duration column contained impossible values (326, 1852, 3977 "quarters")
- **Implication**: 326 quarters = 81 years - impossible for CDBG-DR programs

#### 5.3 Root Cause Identification
- **Bug Location**: `src/stages/s01b_features.py`, function `compute_timeliness_features_std()`
- **Error**: `n_quarters = len(group)` counted activity rows, not unique quarters
- **Data Structure**: QPR data has ~9 activity rows per quarter per grantee-disaster
- **Result**: Duration=326 rows was interpreted as Duration=326 quarters

```python
# BEFORE (BROKEN):
n_quarters = len(group)  # Counts ~9 rows per quarter

# AFTER (FIXED):
quarterly_agg = group.groupby(quarter_col).agg({...})
n_quarters = group[quarter_col].nunique()  # Counts unique quarters
```

#### 5.4 Impact Assessment
After fixing the bug and re-running all analyses:

| Metric | BEFORE (buggy) | AFTER (fixed) |
|--------|----------------|---------------|
| Duration range | 1-3977 "quarters" | 1-32 quarters |
| N_Events at 95% | 106 | 71 |
| Overall Velocity HR | 4.37 (p=0.006) | ~1.00 (p≈1.00) |
| Phase-Specific Effects | Significant | All null (p>0.35) |
| Heterogeneity Effects | Strong (HR 0.82-51) | Mostly null |

**Root Cause of Original "Significant" Results**: The buggy Duration calculation created spurious correlation between program complexity (more activities → more rows → higher "Duration") and velocity.

#### 5.5 Strategic Pivot
- **Decision**: Reframe manuscript as null finding paper
- **Rationale**:
  1. Challenges throughput assumptions in disaster recovery
  2. Questions technical assistance focused on "spending faster"
  3. Raises question: What DOES predict completion?
- **Title Change**: "When Velocity Matters" → "When Velocity Doesn't Matter"

### Phase 6: Alternative Capacity Measures (Current)
- **Date**: December 2025-ongoing
- **Branch**: `analysis/alternative-capacity-measures`
- **Method**: Explore non-ratio operationalizations
- **Planned approaches**:
  - Absolute dollars (log-transformed)
  - Categorical quartiles (non-parametric)
  - Spline/polynomial effects (non-linear)
  - Alternative velocity measures
- **Status**: Active development

## Lessons Learned

### Methodological
1. **Reverse causality is insidious**: Static ratios created 337% effect estimate entirely from bias
2. **Lagging is essential**: Time-varying with 1-quarter lag eliminates the bias
3. **Null results are valid**: Properly specified models may show no effect
4. **Power matters**: 95% threshold gives only 33-71 events (EPV=4.7-10, borderline power)
5. **Threshold choice is arbitrary**: No threshold shows significant effects

### Data Pipeline
1. **Unit of analysis matters**: Counting rows vs. quarters creates spurious correlations
2. **Sanity checks are essential**: Duration > 50 quarters should trigger warnings
3. **Synthetic peer review works**: Review process led directly to bug discovery
4. **Trace data through pipeline**: Verify aggregation at each step

### Substantive
1. **Velocity does not predict completion**: With correct data, HR ≈ 1.00 across all specifications
2. **Throughput assumption challenged**: "Spending faster" may not lead to faster completion
3. **Capacity ratios may not capture "capacity"**: Ratios might not be the right operationalization
4. **Measurement validity**: Financial flow ratios may not reflect administrative capacity

### Statistical
1. **EPV rule of thumb validated**: Need 10+ events per predictor for stable estimates
2. **Bootstrap SEs necessary**: Repeated grantee observations violate independence
3. **Diagnostics critical**: Time-varying residuals help validate assumptions
4. **Check impossible values**: 326 quarters (81 years) should have been flagged

## Next Steps

1. **Null finding paper**: Reframe velocity manuscript around the null result
2. **Alternative measures**: Explore non-ratio operationalizations of capacity
3. **What predicts completion?**: Investigate other factors that may matter
