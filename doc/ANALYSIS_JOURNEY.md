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

### Phase 4: Alternative Capacity Measures (Next)
- **Date**: TBD
- **Method**: Explore non-ratio operationalizations
- **Planned approaches**:
  - Absolute dollars (log-transformed)
  - Categorical quartiles (non-parametric)
  - Spline/polynomial effects (non-linear)
  - Rate of change (velocity measures)
- **Status**: To be implemented in `analysis/alternative-capacity-measures` branch

## Lessons Learned

### Methodological
1. **Reverse causality is insidious**: Static ratios created 337% effect estimate entirely from bias
2. **Lagging is essential**: Time-varying with 1-quarter lag eliminates the bias
3. **Null results are valid**: Properly specified models may show no effect
4. **Power matters**: 95% threshold gives only 33 events (EPV=4.7, underpowered)
5. **Threshold choice is arbitrary**: No threshold shows significant effects

### Substantive
1. **Capacity ratios may not capture "capacity"**: Ratios might not be the right operationalization
2. **Timing vs. completion**: Perhaps capacity affects completion probability, not speed
3. **Measurement validity**: Financial flow ratios may not reflect administrative capacity

### Statistical
1. **EPV rule of thumb validated**: Need 10+ events per predictor for stable estimates
2. **Bootstrap SEs necessary**: Repeated grantee observations violate independence
3. **Diagnostics critical**: Time-varying residuals help validate assumptions

## Next Steps

Explore whether alternative operationalizations of capacity show effects when ratios do not.
