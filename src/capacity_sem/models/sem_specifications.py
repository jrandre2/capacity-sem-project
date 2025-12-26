"""
SEM model specification definitions.

This module contains the structural equation model specifications
for analyzing government capacity and disaster recovery outcomes.
"""

from typing import Dict

# Full latent variable model
MODEL_FULL_LATENT = """
# Measurement model
# Government Capacity latent variable
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness

# Recovery Outcome latent variable
recovery_outcome =~ Duration_of_completion + Ratio_obligated_funds_fully_expended + Quarter_by_quarter_variance_expended

# Structural model
recovery_outcome ~ gov_cap
"""

# Model without Duration of Completion indicator
MODEL_WITHOUT_DURATION = """
# Measurement model
# Government Capacity latent variable
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness

# Recovery Outcome latent variable (reduced)
recovery_outcome =~ Ratio_obligated_funds_fully_expended + Quarter_by_quarter_variance_expended

# Structural model
recovery_outcome ~ gov_cap
"""

# Alternative model with direct paths
MODEL_DIRECT_EFFECTS = """
# Measurement model
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness

# Direct effects from government capacity to outcome indicators
Duration_of_completion ~ gov_cap
Ratio_obligated_funds_fully_expended ~ gov_cap
Quarter_by_quarter_variance_expended ~ gov_cap
"""

# Full model with population covariate
MODEL_WITH_POPULATION = """
# Measurement model
# Government Capacity latent variable
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness

# Recovery Outcome latent variable
recovery_outcome =~ Duration_of_completion + Ratio_obligated_funds_fully_expended + Quarter_by_quarter_variance_expended

# Structural model with population control
recovery_outcome ~ gov_cap + Population_scaled
gov_cap ~ Population_scaled
"""

# Reduced model with population covariate
MODEL_REDUCED_WITH_POPULATION = """
# Measurement model
# Government Capacity latent variable
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness

# Recovery Outcome latent variable (reduced)
recovery_outcome =~ Ratio_obligated_funds_fully_expended + Quarter_by_quarter_variance_expended

# Structural model with population control
recovery_outcome ~ gov_cap + Population_scaled
gov_cap ~ Population_scaled
"""

# Model with population predicting outcome only
MODEL_POPULATION_OUTCOME_ONLY = """
# Measurement model
# Government Capacity latent variable
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness

# Recovery Outcome latent variable
recovery_outcome =~ Duration_of_completion + Ratio_obligated_funds_fully_expended + Quarter_by_quarter_variance_expended

# Structural model - population predicts outcome only
recovery_outcome ~ gov_cap + Population_scaled
"""

# Full model with all covariates (Population, Severity, Experience)
MODEL_WITH_ALL_COVARIATES = """
# Measurement model
# Government Capacity latent variable
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness

# Recovery Outcome latent variable
recovery_outcome =~ Duration_of_completion + Ratio_obligated_funds_fully_expended + Quarter_by_quarter_variance_expended

# Structural model with all covariates
# Recovery predicted by capacity and covariates
recovery_outcome ~ gov_cap + Population_scaled + Severity_Index_scaled + Experience_Index_scaled

# Capacity predicted by covariates
gov_cap ~ Population_scaled + Experience_Index_scaled
"""

# Model with severity as control
MODEL_WITH_SEVERITY = """
# Measurement model
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness
recovery_outcome =~ Duration_of_completion + Ratio_obligated_funds_fully_expended + Quarter_by_quarter_variance_expended

# Structural model - severity as control
recovery_outcome ~ gov_cap + Severity_Index_scaled
"""

# Model with experience as capacity predictor
MODEL_WITH_EXPERIENCE = """
# Measurement model
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness
recovery_outcome =~ Duration_of_completion + Ratio_obligated_funds_fully_expended + Quarter_by_quarter_variance_expended

# Structural model - experience predicts capacity
recovery_outcome ~ gov_cap
gov_cap ~ Experience_Index_scaled
"""

# =============================================================================
# EXPERIMENTAL MODELS - Testing Optimized Variable Transformations
# =============================================================================
# These models address measurement issues identified in diagnostics:
# 1. Timeliness = 1/Duration creates mathematical redundancy
# 2. Ratio_obligated_funds_fully_expended correlates r=0.95 with Ratio_disbursed
# 3. Quarter_by_quarter_variance uses unstable max-normalization
# 4. Duration is right-skewed and should be log-transformed

# Experiment 1: Log-transformed Duration only
MODEL_EXP_LOG_DURATION = """
# Measurement model with log-transformed duration
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness
recovery_outcome =~ Duration_log + Ratio_obligated_funds_fully_expended + Quarter_by_quarter_variance_expended

# Structural model
recovery_outcome ~ gov_cap
"""

# Experiment 2: CV instead of variance
MODEL_EXP_CV_VARIANCE = """
# Measurement model with CV instead of max-normalized variance
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness
recovery_outcome =~ Duration_of_completion + Ratio_obligated_funds_fully_expended + Spending_CV

# Structural model
recovery_outcome ~ gov_cap
"""

# Experiment 3: Remove Timeliness (redundant with Duration)
MODEL_EXP_NO_TIMELINESS = """
# Measurement model without Timeliness
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
recovery_outcome =~ Duration_of_completion + Ratio_obligated_funds_fully_expended + Quarter_by_quarter_variance_expended

# Structural model
recovery_outcome ~ gov_cap
"""

# Experiment 4: Remove redundant ratio (r=0.95 cross-factor correlation)
MODEL_EXP_NO_REDUNDANT_RATIO = """
# Measurement model without Ratio_obligated_funds_fully_expended
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness
recovery_outcome =~ Duration_of_completion + Quarter_by_quarter_variance_expended

# Structural model
recovery_outcome ~ gov_cap
"""

# Experiment 5: Combined - Log Duration + CV
MODEL_EXP_LOG_CV = """
# Measurement model with log duration and CV
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness
recovery_outcome =~ Duration_log + Ratio_obligated_funds_fully_expended + Spending_CV

# Structural model
recovery_outcome ~ gov_cap
"""

# Experiment 6: Minimal - Remove both Timeliness and redundant ratio
MODEL_EXP_MINIMAL = """
# Minimal measurement model (2 indicators per factor)
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
recovery_outcome =~ Duration_of_completion + Quarter_by_quarter_variance_expended

# Structural model
recovery_outcome ~ gov_cap
"""

# Experiment 7: Optimal combination (all fixes)
MODEL_EXP_OPTIMAL_V1 = """
# Optimized measurement model:
# - No Timeliness (= 1/Duration, redundant)
# - Log Duration (normalizes right skew)
# - CV instead of max-normalized variance (stable measure)
# - Keep Ratio_obligated for now (test discriminant validity)
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
recovery_outcome =~ Duration_log + Spending_CV

# Structural model
recovery_outcome ~ gov_cap
"""

# Experiment 8: Optimal with 3 indicators per factor
MODEL_EXP_OPTIMAL_V2 = """
# Optimized model with 3 indicators per factor
# - No Timeliness (redundant)
# - Log Duration
# - CV instead of variance
# - Keep Ratio_obligated (despite collinearity, for comparison)
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
recovery_outcome =~ Duration_log + Ratio_obligated_funds_fully_expended + Spending_CV

# Structural model
recovery_outcome ~ gov_cap
"""

# Experiment 9: Optimal with log population covariate
MODEL_EXP_OPTIMAL_POP = """
# Optimized model with log-transformed population
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
recovery_outcome =~ Duration_log + Spending_CV

# Structural model with log population
recovery_outcome ~ gov_cap + Population_log_scaled
"""

# Experiment 10: Optimal with all covariates
MODEL_EXP_OPTIMAL_FULL = """
# Optimized model with all covariates
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
recovery_outcome =~ Duration_log + Spending_CV

# Structural model with covariates
recovery_outcome ~ gov_cap + Population_log_scaled + Severity_Index_scaled + Experience_Index_scaled
gov_cap ~ Experience_Index_scaled
"""

# Experiment 11: Single factor model (test discriminant validity)
MODEL_EXP_SINGLE_FACTOR = """
# Single factor model to test discriminant validity
# If this fits well, the two-factor structure may not be supported
fund_effectiveness =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Duration_log + Spending_CV
"""

# Experiment 12: Gini coefficient instead of CV
MODEL_EXP_GINI = """
# Test Gini coefficient as alternative to CV
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
recovery_outcome =~ Duration_log + Spending_Gini

# Structural model
recovery_outcome ~ gov_cap
"""

# =============================================================================
# ALTERNATIVE TIMELINESS MODELS
# =============================================================================
# These models test alternative timeliness measures that avoid the mathematical
# inverse problem of Timeliness = 1/Duration

# Experiment 13: Progress Rate as timeliness measure on gov_cap
MODEL_EXP_PROGRESS_RATE = """
# Progress Rate on gov_cap factor (measures administrative velocity)
# Progress_Rate = Completion_Pct / N_Quarters
# Higher = faster progress (intuitive direction)
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Progress_Rate
recovery_outcome =~ Duration_log + Spending_CV

# Structural model
recovery_outcome ~ gov_cap
"""

# Experiment 14: Progress Rate replaces Duration on outcome
MODEL_EXP_PROGRESS_OUTCOME = """
# Progress Rate on recovery_outcome factor (alternative to Duration)
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
recovery_outcome =~ Progress_Rate + Spending_CV

# Structural model
recovery_outcome ~ gov_cap
"""

# Alternate spec for short-series inclusion (same measurement model)
MODEL_EXP_PROGRESS_OUTCOME_SHORT_SERIES = MODEL_EXP_PROGRESS_OUTCOME

# Experiment 15: Time to 50% milestone as timeliness
MODEL_EXP_TIME_TO_MILESTONE = """
# Time to 50% milestone (normalized, lower = faster)
# Different from Duration - captures early vs late progress
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
recovery_outcome =~ Time_to_50pct + Spending_CV

# Structural model
recovery_outcome ~ gov_cap
"""

# Experiment 16: Spending Acceleration
MODEL_EXP_ACCELERATION = """
# Spending Acceleration captures spending dynamics
# Positive = ramping up, Negative = winding down
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Spending_Acceleration
recovery_outcome =~ Duration_log + Spending_CV

# Structural model
recovery_outcome ~ gov_cap
"""

# Experiment 17: Completion Velocity
MODEL_EXP_VELOCITY = """
# Completion Velocity = mean quarterly completion % change
# First derivative of progress curve
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
recovery_outcome =~ Completion_Velocity + Spending_CV

# Structural model
recovery_outcome ~ gov_cap
"""

# Experiment 18: Progress Rate + Optimal V1 (3 indicators on gov_cap)
MODEL_EXP_OPTIMAL_PROGRESS = """
# Optimal V1 enhanced with Progress Rate
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Progress_Rate
recovery_outcome =~ Duration_log + Spending_CV

# Structural model
recovery_outcome ~ gov_cap
"""

# Experiment 19: Startup Lag as capacity indicator
MODEL_EXP_STARTUP_LAG = """
# Startup Lag captures administrative readiness
# Lower = faster start (may need reversal for interpretation)
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
recovery_outcome =~ Startup_Lag + Duration_log + Spending_CV

# Structural model
recovery_outcome ~ gov_cap
"""

# Experiment 20: Full alternative timeliness model
MODEL_EXP_ALT_TIMELINESS = """
# Full model using Progress Rate and Completion Velocity
# No Duration or Timeliness (both problematic)
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Progress_Rate
recovery_outcome =~ Completion_Velocity + Spending_CV

# Structural model
recovery_outcome ~ gov_cap
"""

# =============================================================================
# IMPROVED MODELS - Addressing Reviewer Critiques (2025-12-21)
# =============================================================================
# These models address key methodological issues:
# 1. More than 2 indicators per latent (increases df, improves reliability)
# 2. Non-overlapping indicators (avoids mathematical coupling)
# 3. Uses available but previously unused indicators

# Experiment 21: 3x3 Model with Non-Overlapping Indicators
MODEL_IMPROVED_3X3 = """
# Improved 3x3 model addressing reviewer critiques:
# - 3 indicators per latent variable (df = 8, properly over-identified)
# - Non-overlapping indicators (no mathematical coupling)
# - Uses Startup_Lag (time to first expenditure) for capacity
# - Uses Time_to_50pct (normalized milestone time) for outcome

# Measurement model
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Startup_Lag
recovery_outcome =~ Duration_log + Spending_CV + Time_to_50pct

# Structural model
recovery_outcome ~ gov_cap
"""

# Experiment 22: 3x3 Model with Progress Rate
MODEL_IMPROVED_3X3_PROGRESS = """
# 3x3 model using Progress_Rate instead of Startup_Lag
# Progress_Rate = Completion % per quarter (administrative velocity)
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Progress_Rate
recovery_outcome =~ Duration_log + Spending_CV + Time_to_50pct

# Structural model
recovery_outcome ~ gov_cap
"""

# Experiment 23: 3x3 Model with Covariates
MODEL_IMPROVED_3X3_COVARIATES = """
# 3x3 model with exogenous predictors to control for confounding
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Startup_Lag
recovery_outcome =~ Duration_log + Spending_CV + Time_to_50pct

# Structural model with covariates
recovery_outcome ~ gov_cap + Population_log_scaled + Severity_Index_scaled
gov_cap ~ Experience_Index_scaled
"""

# Experiment 24: Formative Capacity Model
MODEL_FORMATIVE_CAPACITY = """
# Formative model where capacity is FORMED BY (not reflected in) indicators
# This addresses the critique that ratios "form" capacity rather than reflect it
# Uses composite variable approach

# Formative capacity specification (single indicator with error fixed)
capacity_composite =~ 1*Capacity_Index

# Reflective outcome
recovery_outcome =~ Duration_log + Spending_CV + Time_to_50pct

# Structural model
recovery_outcome ~ capacity_composite
"""

# Model registry
MODEL_REGISTRY: Dict[str, str] = {
    # Original models
    'full': MODEL_FULL_LATENT,
    'reduced': MODEL_WITHOUT_DURATION,
    'direct': MODEL_DIRECT_EFFECTS,
    'full_with_population': MODEL_WITH_POPULATION,
    'reduced_with_population': MODEL_REDUCED_WITH_POPULATION,
    'population_outcome': MODEL_POPULATION_OUTCOME_ONLY,
    'full_with_all_covariates': MODEL_WITH_ALL_COVARIATES,
    'with_severity': MODEL_WITH_SEVERITY,
    'with_experience': MODEL_WITH_EXPERIENCE,
    # Experimental models (variable transformations)
    'exp_log_duration': MODEL_EXP_LOG_DURATION,
    'exp_cv_variance': MODEL_EXP_CV_VARIANCE,
    'exp_no_timeliness': MODEL_EXP_NO_TIMELINESS,
    'exp_no_redundant_ratio': MODEL_EXP_NO_REDUNDANT_RATIO,
    'exp_log_cv': MODEL_EXP_LOG_CV,
    'exp_minimal': MODEL_EXP_MINIMAL,
    'exp_optimal_v1': MODEL_EXP_OPTIMAL_V1,
    'exp_optimal_v2': MODEL_EXP_OPTIMAL_V2,
    'exp_optimal_pop': MODEL_EXP_OPTIMAL_POP,
    'exp_optimal_full': MODEL_EXP_OPTIMAL_FULL,
    'exp_single_factor': MODEL_EXP_SINGLE_FACTOR,
    'exp_gini': MODEL_EXP_GINI,
    # Alternative timeliness models
    'exp_progress_rate': MODEL_EXP_PROGRESS_RATE,
    'exp_progress_outcome': MODEL_EXP_PROGRESS_OUTCOME,
    'exp_progress_outcome_short_series': MODEL_EXP_PROGRESS_OUTCOME_SHORT_SERIES,
    'exp_time_to_milestone': MODEL_EXP_TIME_TO_MILESTONE,
    'exp_acceleration': MODEL_EXP_ACCELERATION,
    'exp_velocity': MODEL_EXP_VELOCITY,
    'exp_optimal_progress': MODEL_EXP_OPTIMAL_PROGRESS,
    'exp_startup_lag': MODEL_EXP_STARTUP_LAG,
    'exp_alt_timeliness': MODEL_EXP_ALT_TIMELINESS,
    # Improved models (addressing reviewer critiques)
    'improved_3x3': MODEL_IMPROVED_3X3,
    'improved_3x3_progress': MODEL_IMPROVED_3X3_PROGRESS,
    'improved_3x3_covariates': MODEL_IMPROVED_3X3_COVARIATES,
    'formative_capacity': MODEL_FORMATIVE_CAPACITY,
}

# Model descriptions
MODEL_DESCRIPTIONS: Dict[str, str] = {
    'full': (
        "Full latent variable model with two measurement models:\n"
        "- Government Capacity (gov_cap): measured by disbursement ratio, "
        "expenditure ratio, and timeliness\n"
        "- Recovery Outcome (recovery_outcome): measured by duration, "
        "completion ratio, and spending variance\n"
        "Structural path: recovery_outcome ~ gov_cap"
    ),
    'reduced': (
        "Reduced model without Duration_of_completion indicator:\n"
        "- Government Capacity (gov_cap): measured by disbursement ratio, "
        "expenditure ratio, and timeliness\n"
        "- Recovery Outcome (recovery_outcome): measured by completion ratio "
        "and spending variance only\n"
        "Structural path: recovery_outcome ~ gov_cap"
    ),
    'direct': (
        "Direct effects model:\n"
        "- Government Capacity (gov_cap): measured by disbursement ratio, "
        "expenditure ratio, and timeliness\n"
        "- Direct paths from gov_cap to each outcome indicator"
    ),
    'full_with_population': (
        "Full latent variable model with population covariate:\n"
        "- Government Capacity (gov_cap): measured by disbursement ratio, "
        "expenditure ratio, and timeliness\n"
        "- Recovery Outcome (recovery_outcome): measured by duration, "
        "completion ratio, and spending variance\n"
        "- Population (scaled): controls for jurisdiction size\n"
        "Structural paths: recovery_outcome ~ gov_cap + Population_scaled\n"
        "                  gov_cap ~ Population_scaled"
    ),
    'reduced_with_population': (
        "Reduced model with population covariate:\n"
        "- Government Capacity (gov_cap): measured by disbursement ratio, "
        "expenditure ratio, and timeliness\n"
        "- Recovery Outcome (recovery_outcome): measured by completion ratio "
        "and spending variance only\n"
        "- Population (scaled): controls for jurisdiction size\n"
        "Structural paths: recovery_outcome ~ gov_cap + Population_scaled\n"
        "                  gov_cap ~ Population_scaled"
    ),
    'population_outcome': (
        "Full model with population predicting outcome only:\n"
        "- Government Capacity (gov_cap): measured by disbursement ratio, "
        "expenditure ratio, and timeliness\n"
        "- Recovery Outcome (recovery_outcome): measured by duration, "
        "completion ratio, and spending variance\n"
        "- Population (scaled): directly predicts outcome\n"
        "Structural path: recovery_outcome ~ gov_cap + Population_scaled"
    ),
    'full_with_all_covariates': (
        "Full model with all covariates (Population, Severity, Experience):\n"
        "- Government Capacity (gov_cap): measured by disbursement ratio, "
        "expenditure ratio, and timeliness\n"
        "- Recovery Outcome (recovery_outcome): measured by duration, "
        "completion ratio, and spending variance\n"
        "- Covariates: Population, Disaster Severity, Grantee Experience\n"
        "Structural paths:\n"
        "  recovery_outcome ~ gov_cap + Pop + Severity + Experience\n"
        "  gov_cap ~ Population + Experience"
    ),
    'with_severity': (
        "Model with disaster severity as control:\n"
        "- Government Capacity (gov_cap): measured by disbursement ratio, "
        "expenditure ratio, and timeliness\n"
        "- Recovery Outcome (recovery_outcome): measured by duration, "
        "completion ratio, and spending variance\n"
        "- Disaster Severity (scaled): controls for disaster magnitude\n"
        "Structural path: recovery_outcome ~ gov_cap + Severity_Index_scaled"
    ),
    'with_experience': (
        "Model with grantee experience predicting capacity:\n"
        "- Government Capacity (gov_cap): measured by disbursement ratio, "
        "expenditure ratio, and timeliness\n"
        "- Recovery Outcome (recovery_outcome): measured by duration, "
        "completion ratio, and spending variance\n"
        "- Experience Index (scaled): predicts capacity\n"
        "Structural paths:\n"
        "  recovery_outcome ~ gov_cap\n"
        "  gov_cap ~ Experience_Index_scaled"
    ),
    # Experimental model descriptions
    'exp_log_duration': (
        "Experiment 1: Log-transformed Duration only\n"
        "Tests whether log-transforming Duration improves fit by normalizing skewness."
    ),
    'exp_cv_variance': (
        "Experiment 2: Coefficient of Variation instead of max-normalized variance\n"
        "Tests whether CV provides more stable variance measurement."
    ),
    'exp_no_timeliness': (
        "Experiment 3: Remove Timeliness indicator\n"
        "Tests fit improvement by removing redundant indicator (Timeliness = 1/Duration)."
    ),
    'exp_no_redundant_ratio': (
        "Experiment 4: Remove Ratio_obligated_funds_fully_expended\n"
        "Tests fit improvement by removing indicator with r=0.95 cross-factor correlation."
    ),
    'exp_log_cv': (
        "Experiment 5: Log Duration + CV combined\n"
        "Tests combined effect of log transformation and CV."
    ),
    'exp_minimal': (
        "Experiment 6: Minimal model (2 indicators per factor)\n"
        "Tests fit with only core indicators: Ratio_disbursed, Ratio_expended, "
        "Duration, and Variance."
    ),
    'exp_optimal_v1': (
        "Experiment 7: Optimal V1 - All fixes combined\n"
        "- No Timeliness (redundant)\n"
        "- Log Duration (normalizes skew)\n"
        "- CV instead of variance (stable measure)\n"
        "Minimal 2x2 factor structure."
    ),
    'exp_optimal_v2': (
        "Experiment 8: Optimal V2 - 3 indicators on outcome factor\n"
        "Same as V1 but keeps Ratio_obligated for comparison.\n"
        "Tests discriminant validity with r=0.95 indicator retained."
    ),
    'exp_optimal_pop': (
        "Experiment 9: Optimal V1 + log population covariate\n"
        "Adds log-transformed population as predictor of outcomes."
    ),
    'exp_optimal_full': (
        "Experiment 10: Optimal V1 + all covariates\n"
        "Full model with log population, severity, and experience covariates."
    ),
    'exp_single_factor': (
        "Experiment 11: Single factor model\n"
        "Tests discriminant validity: if single factor fits well, "
        "two-factor structure may not be supported."
    ),
    'exp_gini': (
        "Experiment 12: Gini coefficient instead of CV\n"
        "Tests Gini as alternative concentration measure."
    ),
    # Alternative timeliness model descriptions
    'exp_progress_rate': (
        "Experiment 13: Progress Rate as timeliness measure\n"
        "Progress_Rate = Completion_Pct / N_Quarters\n"
        "Placed on gov_cap factor as measure of administrative velocity.\n"
        "Not mathematically related to Duration."
    ),
    'exp_progress_outcome': (
        "Experiment 14: Progress Rate on outcome factor\n"
        "Uses Progress_Rate instead of Duration on recovery_outcome.\n"
        "Tests whether Progress_Rate captures outcome better than Duration."
    ),
    'exp_progress_outcome_short_series': (
        "Alternate spec: Progress Rate outcome for short-series inclusion\n"
        "Same measurement model as exp_progress_outcome, intended for\n"
        "datasets that include grantees with <5 quarters of reporting."
    ),
    'exp_time_to_milestone': (
        "Experiment 15: Time to 50% milestone\n"
        "Normalized time to reach 50% completion.\n"
        "Captures early vs late progress, different from total Duration."
    ),
    'exp_acceleration': (
        "Experiment 16: Spending Acceleration\n"
        "Linear trend slope of quarterly spending.\n"
        "Positive = ramping up, Negative = winding down."
    ),
    'exp_velocity': (
        "Experiment 17: Completion Velocity\n"
        "Mean quarterly change in completion percentage.\n"
        "First derivative of the progress curve."
    ),
    'exp_optimal_progress': (
        "Experiment 18: Optimal V1 + Progress Rate\n"
        "Optimal 2x2 structure with Progress_Rate added to gov_cap.\n"
        "3 indicators on gov_cap, 2 on recovery_outcome."
    ),
    'exp_startup_lag': (
        "Experiment 19: Startup Lag indicator\n"
        "Quarters before first expenditure.\n"
        "Captures administrative readiness/startup speed."
    ),
    'exp_alt_timeliness': (
        "Experiment 20: Full alternative timeliness\n"
        "No Duration or Timeliness (both problematic).\n"
        "Uses Progress_Rate and Completion_Velocity instead."
    ),
    # Improved model descriptions (addressing reviewer critiques)
    'improved_3x3': (
        "Improved 3x3 model addressing reviewer critiques:\n"
        "- 3 indicators per latent (df=8, properly over-identified)\n"
        "- Non-overlapping indicators (no mathematical coupling)\n"
        "- gov_cap: Ratio_disbursed, Ratio_expended, Startup_Lag\n"
        "- recovery_outcome: Duration_log, Spending_CV, Time_to_50pct\n"
        "Addresses: thin measurement, coupled ratios, low df"
    ),
    'improved_3x3_progress': (
        "3x3 model using Progress_Rate instead of Startup_Lag:\n"
        "- Progress_Rate = Completion % per quarter\n"
        "- Measures administrative velocity\n"
        "Alternative to improved_3x3 with different capacity indicator"
    ),
    'improved_3x3_covariates': (
        "3x3 model with exogenous predictors:\n"
        "- Controls for Population, Severity, Experience\n"
        "- Tests if capacity effect remains significant after controls\n"
        "- Addresses confounding concerns"
    ),
    'formative_capacity': (
        "Formative capacity model:\n"
        "- Capacity FORMED BY (not reflected in) indicators\n"
        "- Addresses critique that ratios 'form' capacity\n"
        "- Uses composite variable approach"
    ),
}


def get_model_spec(model_type: str = 'full') -> str:
    """
    Get SEM model specification string.

    Parameters
    ----------
    model_type : str, default 'full'
        Type of model: 'full', 'reduced', or 'direct'.

    Returns
    -------
    str
        Model specification in semopy/lavaan syntax.

    Raises
    ------
    ValueError
        If model_type is not recognized.
    """
    if model_type not in MODEL_REGISTRY:
        valid_types = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Valid types: {valid_types}"
        )

    return MODEL_REGISTRY[model_type]


def get_model_description(model_type: str = 'full') -> str:
    """
    Get human-readable description of the model.

    Parameters
    ----------
    model_type : str
        Type of model.

    Returns
    -------
    str
        Description of the model.
    """
    return MODEL_DESCRIPTIONS.get(model_type, "No description available.")


def list_available_models() -> Dict[str, str]:
    """
    List all available model specifications.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping model names to descriptions.
    """
    return MODEL_DESCRIPTIONS.copy()


def get_indicator_variables() -> Dict[str, list]:
    """
    Get lists of indicator variables for each latent construct.

    Returns
    -------
    Dict[str, list]
        Dictionary with latent variable names as keys and
        lists of indicator names as values.
    """
    return {
        'gov_cap': [
            'Ratio_disbursed_to_obligated',
            'Ratio_expended_to_disbursed',
            'Timeliness'
        ],
        'recovery_outcome': [
            'Duration_of_completion',
            'Ratio_obligated_funds_fully_expended',
            'Quarter_by_quarter_variance_expended'
        ]
    }
