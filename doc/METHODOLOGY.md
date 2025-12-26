# Methodology

## Primary Methodology: Survival Analysis

The manuscript uses survival analysis to examine how government administrative capacity affects disaster recovery program completion timing.

### Why Survival Analysis?

With 73.7% of CDBG-DR programs incomplete at the 95% expenditure threshold, standard regression approaches face a fundamental problem:

1. **Listwise deletion**: Excluding incomplete programs discards 74% of observations, severely reducing statistical power and potentially introducing selection bias.

2. **Right-censoring as complete**: Treating observation time as completion time underestimates true completion times and biases estimates.

Survival analysis properly handles right-censored observations by incorporating the information that a program has *not yet* completed without treating observation time as completion time.

### Cox Proportional Hazards Model

The Cox model estimates the hazard function—the instantaneous probability of completion at time $t$:

$$h(t|X) = h_0(t) \exp(\beta_1 X_1 + \beta_2 X_2)$$

Where:
- $h_0(t)$ is the baseline hazard (unspecified)
- $X_1, X_2$ are capacity predictors (disbursement ratio, expenditure ratio)
- $\exp(\beta_i)$ gives the **hazard ratio** for a one-unit increase in $X_i$

**Interpretation:**
- HR > 1: Higher values increase completion hazard (faster completion)
- HR < 1: Higher values decrease completion hazard (slower completion)
- HR = 1: No effect

### Accelerated Failure Time (AFT) Models

AFT models directly model the log of completion time:

$$\log(T) = \mu + \beta_1 X_1 + \beta_2 X_2 + \sigma \epsilon$$

Where $\epsilon$ follows a specified distribution (Weibull, lognormal, or log-logistic).

**Interpretation:**
- TR < 1: Higher values reduce expected time (faster completion)
- TR > 1: Higher values increase expected time (slower completion)
- TR = 1: No effect

### Model Selection

| Distribution | Hazard Shape | When to Use |
|--------------|--------------|-------------|
| Weibull | Monotonic (increasing or decreasing) | When hazard changes consistently over time |
| Lognormal | Peaks then decreases | When hazard initially rises then falls |
| Log-logistic | Similar to lognormal | Alternative with different tail behavior |

Model selection is based on AIC (lower is better) and concordance index (higher is better).

### Key Results

| Model | Disbursement HR/TR | p-value | Expenditure HR/TR | p-value |
|-------|-------------------|---------|-------------------|---------|
| Cox PH | 4.367 | 0.006 | 0.958 | 0.626 |
| AFT Weibull | 0.201 | <0.001 | 0.988 | 0.917 |
| AFT Lognormal | 0.157 | <0.001 | 1.008 | 0.954 |
| AFT Log-logistic | 0.178 | <0.001 | 1.019 | 0.881 |

**Central finding**: Disbursement capacity significantly predicts completion timing (HR = 4.37, p = 0.006). A one-unit increase in disbursement ratio reduces expected completion time by 80-84%. Expenditure capacity shows no significant effect.

### Model Evaluation

| Metric | Interpretation |
|--------|----------------|
| Concordance Index | C = 0.5 random; C > 0.7 acceptable; C > 0.8 good |
| AIC | Lower is better (for comparing AFT distributions) |
| Schoenfeld residuals | Test proportional hazards assumption |

---

## SEM Framework (Sensitivity Analysis)

The SEM codebase is retained for robustness checks in Appendix C. This section documents the SEM methodology for reference.

### Why SEM Is Not the Primary Methodology

Standard SEM approaches face challenges with CDBG-DR data:

1. **Sample size after listwise deletion**: Only N=40 observations at 95% threshold
2. **Right-censoring**: Cannot properly handle incomplete programs
3. **Power limitations**: ~55% power for medium effects vs. ~95% with survival analysis

SEM results are reported in appendix-c-robustness.qmd as sensitivity analysis.

---

### Latent Variables

**Government Capacity (gov_cap)**

An unobserved construct representing the administrative efficiency and capability of the grantee in managing CDBG-DR funds.

**Recovery Outcome (recovery_outcome)**

An unobserved construct representing the overall effectiveness and timeliness of disaster recovery fund expenditure.

### Measurement Model

The measurement model links observable indicators to latent constructs:

**Capacity Indicators**:
- `Ratio_disbursed_to_obligated`: Proportion of obligated funds disbursed to recipients
- `Ratio_expended_to_disbursed`: Proportion of disbursed funds expended by recipients

**Outcome Indicators**:
- `Duration_log`: Log-transformed months to reach 95% expenditure
- `Spending_CV`: Coefficient of variation of quarterly spending

### Structural Model

The structural model specifies the relationship between latent variables:

$$
\text{recovery\_outcome} = \beta \cdot \text{gov\_cap} + \zeta
$$

where $\beta$ is the structural path coefficient and $\zeta$ is the structural disturbance.

---

## Model Specifications

### Optimal V1 Model (Recommended)

```
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
recovery_outcome =~ Duration_log + Spending_CV
recovery_outcome ~ gov_cap
```

This specification:
- Uses 2 indicators per factor (minimal but identified)
- Avoids redundancy (no Timeliness = 1/Duration)
- Uses log duration to address skewness
- Uses CV for stable variance measurement

### Full Model (Original)

```
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness
recovery_outcome =~ Duration_of_completion + Ratio_obligated_funds_fully_expended + Quarter_by_quarter_variance_expended
recovery_outcome ~ gov_cap
```

**Known Issues**:
- Timeliness = 1/Duration creates mathematical coupling
- Ratio_obligated has r=0.95 cross-factor correlation

### Improved 3x3 Model

```
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Startup_Lag
recovery_outcome =~ Duration_log + Spending_CV + Time_to_50pct
recovery_outcome ~ gov_cap
```

This specification:
- 3 indicators per factor (more degrees of freedom)
- Non-overlapping indicators
- Uses Startup_Lag (time to first expenditure) for capacity

### Additional Model Specifications

The codebase includes 24+ model specifications for robustness testing. Key additional models include:

| Name | Description |
|------|-------------|
| `reduced` | Model without Duration indicator |
| `exp_optimal_v2` | Optimal V1 with 3 outcome indicators |
| `exp_progress_rate` | Uses Progress Rate as timeliness measure |
| `exp_time_to_milestone` | Uses time to 50% milestone |
| `improved_3x3_progress` | 3x3 with Progress Rate instead of Startup Lag |
| `improved_3x3_covariates` | 3x3 with population, severity, experience controls |
| `formative_capacity` | Formative (vs reflective) capacity specification |

For the complete list of available models, run:

```bash
python src/pipeline.py list_models
```

Or see `src/capacity_sem/models/sem_specifications.py` for full specifications.

---

## Variable Transformations

### Log Duration

Duration is right-skewed; log transformation normalizes the distribution:

$$
\text{Duration\_log} = \ln(\text{Duration\_months})
$$

### Spending CV

The coefficient of variation provides a scale-independent measure of spending consistency:

$$
\text{Spending\_CV} = \frac{\sigma_{\text{quarterly}}}{\mu_{\text{quarterly}}}
$$

### Ratio Construction

Capacity ratios are computed from cumulative QPR series derived from quarterly flows.
`RATIO_DEFINITION` in `src/config.py` controls whether ratios use the mean of quarterly cumulative ratios (`mean_cumulative`) or the final cumulative ratio (`final_cumulative`).
`Ratio_expended_to_obligated` always uses the final cumulative totals.
Use `QPR_DOLLAR_FIELDS_ARE_FLOW` to indicate whether raw QPR dollar fields are quarterly net flows (default) or cumulative totals.

### Data Quality Controls

The pipeline creates a cleaned QPR dataset (`qpr_clean.parquet`) with QA flags and derived fields:

- `Grantee State` is imputed from the grant code when missing, and the original value is preserved in `Grantee State Raw`.
- `QPR_Date` is derived from `QPR Actual Quarter`; rows without a valid quarter are excluded from quarterly rollups.
- Negative flow values and cumulative decreases are retained (to reflect adjustments and revisions) but flagged for review.
- Location data are limited to `Grantee State` in the QPR export (no county/city/FIPS/lat-lon fields).

Quality summaries are written to `data_work/quality/qpr_quality_report.csv` and `data_work/quality/qpr_quarterly_quality_report.csv`.

### Covariate Scaling

All covariates (population, severity, experience, employment) are z-score standardized:

$$
x_{\text{scaled}} = \frac{x - \bar{x}}{s_x}
$$

---

## Estimation

### Maximum Likelihood

Models are estimated using maximum likelihood (ML), which minimizes:

$$
F_{ML} = \ln|\Sigma(\theta)| + \text{tr}(S\Sigma^{-1}(\theta)) - \ln|S| - p
$$

where:
- $\Sigma(\theta)$ = model-implied covariance matrix
- $S$ = sample covariance matrix
- $p$ = number of observed variables

### Software

Estimation uses the `semopy` Python package, which implements the Structural After Measurement (SAM) approach.

### Sample Definition

SEM estimation uses complete-case observations for the variables in the model specification. Missing duration or spending consistency indicators (often caused by incomplete quarterly reporting) can materially reduce the estimation sample. Use the diagnostics output to confirm the sample size for each model run.

---

## Model Fit Assessment

### Fit Indices

| Index | Good | Acceptable | Description |
|-------|------|------------|-------------|
| CFI | ≥ 0.95 | ≥ 0.90 | Comparative Fit Index |
| TLI | ≥ 0.95 | ≥ 0.90 | Tucker-Lewis Index |
| RMSEA | ≤ 0.05 | ≤ 0.08 | Root Mean Square Error of Approximation |
| SRMR | ≤ 0.05 | ≤ 0.08 | Standardized Root Mean Square Residual |

### Chi-Square Test

The chi-square test assesses exact fit. However, it is sensitive to sample size, so fit indices are preferred for large samples.

---

## Robustness Analyses

### Alternative Specifications

We test multiple model formulations to ensure results are not specification-dependent.

### Government Subsets

Models are estimated separately for state and local governments to assess heterogeneity.

### Sample Sensitivity

We vary the minimum quarters requirement (3, 4, 5, 6) to assess sensitivity to sample restrictions.

### Covariate Inclusion

We test models with and without population, severity, and experience covariates.

---

## Interpretation

### Structural Coefficient

The key parameter is $\beta$, the effect of capacity on outcomes:

- **Negative $\beta$**: Higher capacity associated with faster recovery (lower duration)
- **Positive $\beta$**: Higher capacity associated with slower recovery

Given that Duration is log-transformed, the interpretation is in percentage terms.

### Factor Loadings

Factor loadings indicate how strongly each indicator reflects its latent construct. Standardized loadings > 0.5 are considered adequate.

### R-squared

The R² for recovery_outcome indicates the proportion of variance explained by capacity.

---

## Kaifa's Original Methodology (For Reference)

Kaifa's original manuscript analysis differs from the canonical pipeline in several key ways. This section documents the approach for verification and critique.

### Key Methodological Choices

1. **Unit of Analysis**: Grantee level (N~38 state, ~40 local)
   - Averages across all disasters per grantee
   - Treats capacity as a stable grantee trait

2. **Duration Censoring**: Right-censored
   - Incomplete programs use observation time (N_Quarters) as Duration
   - Treats incomplete as if complete at observation point

3. **Factor Structure**: 3x3 with Timeliness = 1/Duration
   - Government Capacity: 3 indicators (including Timeliness)
   - Recovery Outcome: 3 indicators (including Duration)
   - Creates mathematical coupling between factors

### Critique Points

| Issue | Severity | Description |
|-------|----------|-------------|
| Right-censoring bias | High | Biases duration downward; treats incomplete programs as faster than they may actually be |
| Mathematical coupling | High | Timeliness = 1/Duration creates deterministic relationship with Duration_of_completion |
| Aggregation loss | Medium | Grantee-level averaging loses within-grantee variation across disasters |
| Small sample instability | Medium | SEM with N<40 produces unstable estimates; large β values (e.g., 71.024) are common |
| Pseudo-replication | Medium | If using grantee-disaster pairs without clustering, same grantee appears multiple times |

### Replication Results

Using Kaifa's methodology on the same data produces similar results:
- Original manuscript: β ≈ 71.024 (p = 0.01)
- Replication: β ≈ 113.65 (p < 0.001)

The canonical pipeline (grantee-disaster level, no censoring) produces:
- β ≈ 0.32 (p = 0.96)

The discrepancy is explained by methodological differences, not data differences.

---

## Multi-Group SEM

Tests measurement invariance and structural path differences between state and local governments.

### Rationale

State and local governments have different:
- Administrative structures and staffing
- Prior experience with federal grants
- Budget constraints and matching fund availability
- Political pressures and accountability mechanisms

A single model may mask important heterogeneity in how capacity relates to outcomes.

### Invariance Levels

| Level | Constraint | Interpretation |
|-------|-----------|----------------|
| Configural | Same structure | Both groups have the same factor pattern |
| Metric | Equal loadings | Indicators measure constructs equivalently across groups |
| Scalar | Equal intercepts | Groups can be compared on latent mean levels |
| Structural | Equal paths | Capacity-outcome relationship is the same across groups |

### Testing Protocol

1. Fit model separately in each group (baseline)
2. Test configural invariance (joint model, no constraints)
3. Test metric invariance (constrain factor loadings equal)
4. Test scalar invariance (constrain intercepts equal)
5. If invariance fails, report separate structural estimates

### Interpretation

- If full invariance holds: Report pooled estimates
- If metric invariance fails: Factor means cannot be compared
- If structural invariance fails: Capacity-outcome relationship differs by government type

---

## Mediation Analysis

Decomposes capacity effects into direct and indirect pathways.

### Rationale

The capacity → outcome relationship may operate through specific mechanisms:
- Spending consistency (stable quarterly expenditure)
- Startup speed (time to first activity)
- Administrative efficiency (processing speed)

### Effect Types

| Effect | Definition | Interpretation |
|--------|------------|----------------|
| Total | c = c' + a×b | Overall effect of capacity on outcome |
| Direct | c' | Effect controlling for mediator |
| Indirect | a×b | Effect through mediator |

### Mediation Models

**Through Spending Consistency**:
```
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
Spending_CV ~ gov_cap           # a path
Duration_log ~ gov_cap + Spending_CV  # c' and b paths
```

**Through Startup Speed**:
```
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
Startup_Lag ~ gov_cap           # a path
Duration_log ~ gov_cap + Startup_Lag  # c' and b paths
```

### Bootstrap Confidence Intervals

Indirect effects are tested using bootstrap confidence intervals (1000 iterations) because the product of two normal distributions is not normal.

---

## Longitudinal SEM (Future)

### Cross-Lagged Panel Model

Tests temporal precedence: does capacity precede outcomes, or vice versa?

**Model Structure**:
```
# Autoregressive paths
Capacity_t ~ Capacity_t-1
Outcome_t ~ Outcome_t-1

# Cross-lagged paths
Outcome_t ~ Capacity_t-1   # Does prior capacity predict current outcome?
Capacity_t ~ Outcome_t-1   # Does prior outcome predict current capacity?
```

### Latent Growth Curve Model

Models trajectories of completion over time:
- **Intercept**: Initial completion level
- **Slope**: Rate of change over time
- **Capacity predicts both**: High capacity → higher start AND faster improvement

---

## Bayesian SEM (Future)

### Rationale

With small samples (N=40-156), frequentist estimates are unreliable. Bayesian methods:
- Incorporate prior information
- Provide credible intervals (not confidence intervals)
- Work better with small samples
- Enable complex model comparison via WAIC/LOO

### Priors

Weakly informative priors for structural paths:
- Factor loadings: N(0.7, 0.3) - expect moderate to strong loadings
- Structural path: N(0, 1) - regularized, allows positive or negative

---

## Model Comparison

### Information Criteria

| Criterion | Formula | Lower is Better |
|-----------|---------|-----------------|
| AIC | -2LL + 2k | Yes |
| BIC | -2LL + k×ln(n) | Yes |
| WAIC | Bayesian analog | Yes |

### Chi-Square Difference Test

For nested models:
$$\Delta\chi^2 = \chi^2_{\text{constrained}} - \chi^2_{\text{free}}$$

with $\Delta df = df_{\text{constrained}} - df_{\text{free}}$

---

## Effect Size Interpretation

### Standardized Coefficients

| Range | Interpretation |
|-------|----------------|
| |β| < 0.10 | Negligible |
| 0.10 ≤ |β| < 0.30 | Small |
| 0.30 ≤ |β| < 0.50 | Medium |
| |β| ≥ 0.50 | Large |

### Variance Explained (R²)

| Range | Interpretation |
|-------|----------------|
| R² < 0.02 | Negligible |
| 0.02 ≤ R² < 0.13 | Small |
| 0.13 ≤ R² < 0.26 | Medium |
| R² ≥ 0.26 | Large |

### Cohen's f²

$$f^2 = \frac{R^2}{1 - R^2}$$

| Range | Interpretation |
|-------|----------------|
| f² < 0.02 | Negligible |
| 0.02 ≤ f² < 0.15 | Small |
| 0.15 ≤ f² < 0.35 | Medium |
| f² ≥ 0.35 | Large |
