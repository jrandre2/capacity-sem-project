# Methodology

## Structural Equation Modeling Framework

This project employs Structural Equation Modeling (SEM) to analyze the relationship between government capacity and disaster recovery outcomes.

## Conceptual Model

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

### Covariate Scaling

All covariates are z-score standardized:

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
