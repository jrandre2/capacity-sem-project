# Experimental Branch Results (Alternative Capacity Measures)

This document summarizes the robustness tests and results from the
`analysis/alternative-capacity-measures` branch. There is no manuscript
in this branch; this is a working summary to capture methods, outputs,
and findings.

## Data and Pipeline Context

- Raw QPR rows: 130,605
- Analysis unit: 156 grantee-disaster observations
- Completion at 95% threshold: 41/156 (26.3%)
- Static survival sample: 151-156 observations, 40-41 events
- Time-varying survival sample: 3,618 intervals, 151 grantee-disasters, 33 events

Core outputs:
- Static survival results: `data_work/diagnostics/alternatives_survival_capacity_sets.csv`
- Time-varying survival results: `data_work/diagnostics/survival_hazard_ratios.csv`
- Threshold sensitivity results: `data_work/diagnostics/survival_threshold_sensitivity.csv`
- Stratified velocity results: `data_work/diagnostics/alternatives_survival_stratified_velocity.csv`
- Pooled/stratified interaction results: `data_work/diagnostics/alternatives_survival_velocity_strata_models.csv`
- Multiple-testing corrected table: `data_work/diagnostics/capacity_corrected_table.csv`
- Summary figure: `figures/fig_capacity_corrected.png`

## Executive Summary

- Static models show a strong ratio effect (HR ~4.37) and consistent
  velocity signals in percent-per-quarter units (expenditure velocity
  HR ~1.51; velocity index HR ~1.37; Q4 vs Q1 HR ~3.13), but BH-FDR now
  retains velocity measures and the median ratio-high x velocity index
  interaction, not the ratio effect itself.
- Early-window velocity (2-6 quarters) is modestly positive for
  disbursement velocity (3q/6q) and the velocity index (4q/6q), but
  effects do not survive BH-FDR correction.
- Fixed calendar windows (12/18 months) show positive velocity index
  effects (HR ~1.12-1.15; p=0.014-0.031), while component velocities are
  null; these fixed-window effects do not survive BH-FDR correction.
- Alternative cutoffs/knots (q25/q33/q67/q75, spline) show positive
  velocity index interactions; q25 threshold/spline terms are
  significant (p≈0.011) but do not survive BH-FDR, and q75 interactions
  are not significant.
- Stratified models (penalized for low-event strata) show much stronger
  velocity effects in higher ratio strata; low-ratio strata have zero
  events and remain unestimable, and mid-tercile estimates shrink to
  null under penalization.
- Pooled and stratified-baseline interaction models reject common
  velocity slopes across ratio strata (LRT p≈0.001-0.007), driven by
  stronger Q3/Q4 slopes.
- Time-varying velocity (including rolling/cumulative) is null across
  lags; time-varying ratio effects remain non-significant across
  completion thresholds (20-100%).

## Robustness Tests Added

1) **Velocity rescaling**
- Percent-per-quarter measures (`*_pp`)
- Z-score standardized measures (`*_scaled`)
- Winsorized measures (`*_winsor`, 1%/99% bounds)

2) **Early-window velocity (first N quarters)**
- 2-, 3-, 4-, and 6-quarter windows to reduce post-outcome contamination
- Disbursement/expenditure velocity and velocity index variants

3) **Fixed calendar windows (first N months)**
- 12- and 18-month windows based on calendar time from first report
- Disbursement/expenditure velocity and velocity index variants

4) **Static survival with expanded capacity sets**
- Ratios, absolute log dollars, velocity, composite indices, and quartiles
- New percent/winsorized velocity sets included in the capacity-set grid

5) **Time-varying velocity models**
- Lagged velocity covariates (lag 0/1/2) in the time-varying Cox models
- Rolling-window and cumulative velocity variants
- Both velocity pair and velocity index specifications

6) **Ratio x velocity interaction tests**
- Centered, threshold, and spline-based interactions between baseline
  disbursement ratio and velocity measures
- Alternative cutoffs/knots at the 25th, 33rd, 67th, and 75th percentiles

7) **Stratified velocity models**
- Re-estimated velocity effects within baseline ratio strata (median,
  terciles, quartiles), using penalized Cox for low-event strata

8) **Pooled / hierarchical interaction models**
- Pooled Cox with ratio-stratum interactions and stratified-baseline
  Cox to formally test heterogeneous velocity effects

9) **Completion threshold sensitivity**
- Time-varying survival models across completion thresholds (20-100%)

10) **Multiple-testing correction (family-wide)**
- BH-FDR and Bonferroni applied across all capacity-set tests (Cox + AFT)
  plus time-varying Cox results

## Key Static Survival Findings (Cox PH)

Static survival reproduces a strong ratio effect and shows a robust
velocity signal after rescaling. Velocity results below are reported
in percent-per-quarter units for comparability:

| Measure | HR (95% CI) | p-value | N | Events |
| --- | --- | --- | --- | --- |
| Disbursement ratio | 4.37 (1.53-12.48) | 0.0059 | 152 | 41 |
| Expenditure velocity (pp/quarter) | 1.51 (1.18-1.94) | 0.00093 | 151 | 40 |
| Velocity index (pp/quarter) | 1.37 (1.11-1.68) | 0.0027 | 151 | 40 |
| Velocity index Q4 vs Q1 (pp) | 3.13 (1.42-6.89) | 0.0045 | 156 | 41 |

Interpretation:
- A 1 percentage-point per quarter increase in expenditure velocity
  corresponds to ~1.5x higher completion hazard.
- A 1 percentage-point per quarter increase in the velocity index
  corresponds to ~1.37x higher completion hazard.
- Disbursement velocity alone is not significant in the static model
  (p=0.18), but the index remains significant.
- Raw velocity measures yield extreme HRs due to unit scale; percent-per-quarter
  scaling is used for interpretation.

Other static capacity sets (absolute dollars, PCA composites) remain
non-significant in Cox models.

## Early-Window Velocity Findings (Cox PH)

Early-window velocity retains a positive signal for disbursement
velocity and the velocity index, but effects are weaker and sample sizes
shrink in the early-window panels:

| Measure | HR (95% CI) | p-value | N | Events |
| --- | --- | --- | --- | --- |
| Disbursement velocity (2q, pp) | 1.06 (0.98-1.15) | 0.158 | 151 | 40 |
| Disbursement velocity (3q, pp) | 1.10 (1.00-1.21) | 0.042 | 146 | 40 |
| Disbursement velocity (4q, pp) | 1.10 (1.00-1.22) | 0.053 | 139 | 40 |
| Disbursement velocity (6q, pp) | 1.20 (1.02-1.40) | 0.024 | 125 | 40 |
| Velocity index (2q, pp) | 1.04 (0.97-1.12) | 0.270 | 151 | 40 |
| Velocity index (3q, pp) | 1.08 (0.98-1.20) | 0.125 | 146 | 40 |
| Velocity index (4q, pp) | 1.12 (1.00-1.25) | 0.048 | 139 | 40 |
| Velocity index (6q, pp) | 1.15 (1.03-1.29) | 0.016 | 125 | 40 |

Expenditure velocity remains null in early windows (p=0.60-0.94).
AFT lognormal time ratios align with Cox for disbursement (3-6q) and
velocity index (4-6q), with time ratios 0.90-0.95 and p=0.001-0.042.

## Fixed Calendar Window Velocity Findings (Cox PH)

Fixed 12- and 18-month windows show positive effects for the velocity
index but not for the component velocity measures:

| Measure | HR (95% CI) | p-value | N | Events |
| --- | --- | --- | --- | --- |
| Velocity index (12m, pp) | 1.12 (1.01-1.24) | 0.031 | 150 | 39 |
| Velocity index (18m, pp) | 1.15 (1.03-1.28) | 0.014 | 151 | 40 |

Disbursement and expenditure fixed-window velocities are not significant
(p=0.12-0.36). Scaled velocity index results mirror the pp findings.

## Ratio x Velocity Interaction Findings (Cox PH)

Interaction tests now include centered, threshold, and spline versions.
Ratio(high) indicates above-cutoff disbursement ratios; ratio(above)
is the spline hinge term max(ratio - cutoff, 0).

Threshold interactions (velocity index, pp):

| Cutoff | HR (95% CI) | p-value | N | Events |
| --- | --- | --- | --- | --- |
| 25th pct | 1.34 (1.07-1.68) | 0.011 | 151 | 40 |
| Median | 1.49 (1.15-1.92) | 0.0027 | 151 | 40 |
| 33rd pct | 1.43 (1.11-1.83) | 0.0052 | 151 | 40 |
| 67th pct | 1.35 (1.02-1.79) | 0.0368 | 151 | 40 |
| 75th pct | 1.25 (0.92-1.69) | 0.155 | 151 | 40 |

Spline interactions (velocity index, pp):

| Knot | HR (95% CI) | p-value | N | Events |
| --- | --- | --- | --- | --- |
| 25th pct hinge | 1.66 (1.12-2.46) | 0.0117 | 151 | 40 |
| 33rd pct hinge | 1.76 (1.12-2.77) | 0.015 | 151 | 40 |
| 67th pct hinge | 2.11 (0.71-6.28) | 0.180 | 151 | 40 |
| 75th pct hinge | 2.14 (0.58-7.91) | 0.256 | 151 | 40 |

Disbursement velocity interactions are weaker (p>=0.05) and none of the
quartile/tercile spline or threshold interactions survive BH-FDR. The
only interaction term that remains BH-FDR significant is the median
ratio-high x velocity index term. Centered continuous interactions
remain nonsignificant (p=0.10-0.32).

## Stratified Velocity Findings (Cox PH)

Median, tercile, and quartile stratification show stronger velocity
effects in higher ratio strata:

| Scheme | Stratum | Velocity var | HR (95% CI) | p-value | N | Events |
| --- | --- | --- | --- | --- | --- | --- |
| Median | High | Expenditure velocity (pp) | 2.26 (1.63-3.14) | <0.001 | 77 | 40 |
| Median | High | Velocity index (pp) | 1.61 (1.25-2.07) | <0.001 | 77 | 40 |
| Tercile | Mid (0.21-0.62) | Expenditure velocity (pp) | 1.07 (0.91-1.26) | 0.414 | 52 | 7 |
| Tercile | Mid (0.21-0.62) | Velocity index (pp) | 1.09 (0.91-1.30) | 0.375 | 52 | 7 |
| Tercile | High (>0.62) | Expenditure velocity (pp) | 2.48 (1.62-3.80) | <0.001 | 51 | 33 |
| Tercile | High (>0.62) | Velocity index (pp) | 1.47 (1.11-1.95) | 0.006 | 51 | 33 |
| Quartile | Q3 (0.44-0.69) | Expenditure velocity (pp) | 2.41 (1.49-3.90) | 0.00032 | 39 | 13 |
| Quartile | Q3 (0.44-0.69) | Velocity index (pp) | 2.43 (1.52-3.89) | 0.00020 | 39 | 13 |
| Quartile | Q4 (>0.69) | Expenditure velocity (pp) | 2.27 (1.38-3.73) | 0.0012 | 38 | 27 |
| Quartile | Q4 (>0.69) | Velocity index (pp) | 1.35 (1.00-1.83) | 0.050 | 38 | 27 |

Low-ratio strata (median/tercile low, quartile Q1/Q2) have zero events
and are not estimable. Mid-tercile results are penalized due to only
7 events and shrink to null; higher strata (tercile high, quartile
Q3/Q4) retain large, significant velocity effects.

## Pooled / Stratified Interaction Findings (Cox PH)

Pooled Cox models with ratio-stratum interactions and stratified-baseline
models (separate baseline hazards by ratio quartile) both reject a
common velocity slope across strata:

- Quartile cutoffs: Q1 ≤ 0.115, Q2 0.115-0.444, Q3 0.444-0.686, Q4 > 0.686.
- **Pooled interaction LRT**: expenditure velocity χ2(3)=15.84 (p=0.0012);
  velocity index χ2(3)=12.39 (p=0.0062).
- **Stratified-baseline LRT**: expenditure velocity χ2(3)=13.41 (p=0.0038);
  velocity index χ2(3)=12.14 (p=0.0069).
- Q3 interaction terms are strongest: expenditure velocity HR 2.04
  (p=0.0008) and velocity index HR 1.96 (p=0.00067).
- Q4 interaction terms remain positive but weaker: expenditure velocity
  HR 2.14 (p=0.0039 pooled; 1.94, p=0.015 stratified baseline); velocity
  index Q4 p=0.083-0.116.

## Time-Varying Velocity Findings (Lagged)

Time-varying velocity effects are **null**, even after rescaling and
adding lag 0/2 variants:

| Model | HR (95% CI) | p-value |
| --- | --- | --- |
| Disbursement velocity pp (lag0) | 1.00 (0.995-1.005) | 0.995 |
| Disbursement velocity pp (lag1) | 1.00 (0.995-1.005) | 0.973 |
| Disbursement velocity pp (lag2) | 1.00 (0.995-1.005) | 0.996 |
| Expenditure velocity pp (lag0) | 1.00 (0.998-1.002) | 0.995 |
| Expenditure velocity pp (lag1) | 1.00 (0.998-1.002) | 0.978 |
| Expenditure velocity pp (lag2) | 1.00 (0.998-1.002) | 1.000 |
| Velocity index (lag0, z-score) | 1.00 (0.900-1.111) | 0.998 |
| Velocity index (lag1, z-score) | 1.00 (0.896-1.112) | 0.973 |
| Velocity index (lag2, z-score) | 1.00 (0.896-1.117) | 0.998 |

This suggests the static velocity signal is not driven by
quarter-to-quarter changes; early-window effects are modest but
time-varying lags remain null, pointing to cumulative or late-stage
dynamics.

Rolling (2- and 4-quarter) and cumulative velocity measures are also
null across lags 0/1/2 (HR ~1.00; p>=0.88 for rolling, p>=0.93 for
cumulative), including scaled velocity index variants.

## Completion Threshold Sensitivity (Time-Varying Cox)

Across completion thresholds (20-100%), the lagged disbursement ratio is
consistently positive but not significant. The smallest p-values occur
around the 25% threshold (p ~0.21 unadjusted; p ~0.22 adjusted). For
thresholds with adequate EPV (>=10; 20-70%), adjusted HRs range from
~1.05 to ~1.15 with p-values 0.22-0.69.

See `data_work/diagnostics/survival_threshold_sensitivity.csv` for the
full table.

## Multiple-Testing Corrections

Corrections are applied across all capacity-set tests plus time-varying Cox:

- **Bonferroni**: no Cox or time-varying results remain significant.
- **BH-FDR (q<0.05)**: expenditure velocity and velocity index measures
  remain significant (scaled, pp, winsorized, raw), along with the
  median ratio-high x velocity index interaction. The disbursement ratio
  no longer survives BH-FDR under the expanded test set. Early windows,
  fixed windows, q25/q33/q67/q75 cutoffs/knots, and time-varying velocity
  remain null. Pooled/stratified interaction LRTs are reported
  separately and are not included in the family-wide correction table.

See:
- `data_work/diagnostics/multiple_testing_capacity_sets_time_varying.csv`
- `data_work/diagnostics/capacity_corrected_table.csv`
- `figures/fig_capacity_corrected.png`

## Interpretation and Implications

- **Pace > levels**: static models repeatedly point to spending velocity
  as the strongest alternative capacity signal; the disbursement ratio
  effect weakens under the expanded multiple-testing set.
- **Early-window signal is modest**: early velocity still predicts
  faster completion for disbursement velocity (3-6q) and the velocity
  index (4-6q), but effect sizes shrink and do not survive family-wide
  correction.
- **Fixed-window signals are weaker**: 12-18 month velocity index effects
  are positive but do not survive family-wide correction.
- **Conditional velocity effects**: the median ratio-high x velocity
  index interaction is significant and survives BH-FDR, stratified
  models show large effects in higher ratio strata, and pooled/stratified
  interaction models confirm heterogeneous velocity slopes; low-stratum
  results remain unestimable.
- **Temporal ordering matters**: lagged, rolling, and cumulative
  time-varying velocity effects remain null across lags, suggesting the
  static signal may capture cumulative or late-stage dynamics rather
  than within-quarter changes.

## Suggested Next Steps

1) Address zero-event low-ratio strata by combining low bins or using a
   lower completion threshold to recover events, and consider Bayesian
   or parametric survival models with partial pooling.
2) Probe stage-specific dynamics with piecewise early/late velocity
   slopes or spline-based velocity terms to isolate when the signal
   emerges.
3) Validate robustness to alternative ratio definitions (final-cumulative
   ratios, lagged ratios) and covariate sets to ensure the velocity
   findings are not driven by measurement choice.
