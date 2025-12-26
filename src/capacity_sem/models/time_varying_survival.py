"""
Time-varying survival analysis for CDBG-DR capacity study.

This module implements time-varying survival models to address the critical
methodological flaw in the original static approach: capacity ratios computed
across ALL quarters (including post-completion) create reverse causality.

The time-varying approach:
1. Transforms quarterly panel into interval format (one row per quarter per grantee-disaster)
2. Computes capacity ratios quarter-by-quarter with lagging to avoid contemporaneous correlation
3. Merges static covariates (government type, grant size, experience)
4. Fits Cox PH and AFT models with proper temporal ordering
5. Computes bootstrap clustered standard errors

Key functions:
- reshape_quarterly_to_time_varying(): Transform quarterly data to survival intervals
- compute_lagged_capacity_ratios(): Apply lagged ratio calculation
- add_static_covariates(): Merge time-invariant predictors
- fit_time_varying_cox(): Fit Cox model with time-varying covariates
- compute_bootstrap_se(): Bootstrap SEs with cluster resampling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

# lifelines imports
try:
    from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    warnings.warn("lifelines not installed. Install with: pip install lifelines")


def reshape_quarterly_to_time_varying(
    qpr_quarterly: pd.DataFrame,
    panel_features: pd.DataFrame,
    lag_quarters: int = 1,
    min_quarters: int = 2,
    duration_col: str = 'Duration_of_completion'
) -> pd.DataFrame:
    """
    Reshape quarterly panel to time-varying survival format.

    Transforms wide quarterly data into long format with one row per interval
    (grantee-disaster × quarter). Each row represents a time interval with:
    - start/stop: Interval boundaries in months (e.g., 0-3, 3-6, 6-9)
    - E: Event indicator (1 only on final row if completion occurred)
    - Lagged capacity ratios: Predictors computed through previous quarters

    This structure avoids reverse causality by ensuring predictors at time t
    only use information available through time t-lag.

    Parameters
    ----------
    qpr_quarterly : pd.DataFrame
        Quarterly data with columns:
        - Grantee, Disaster Type (identifiers)
        - Quarter (sequential quarter number)
        - QPR Fund Obligated $, QPR Fund Disbursed $, QPR Fund Expended $ (cumulative)
    panel_features : pd.DataFrame
        Static features with columns:
        - Grantee, Disaster Type (identifiers)
        - Duration_of_completion (months to 95% threshold, NaN if censored)
        - N_Quarters (total quarters observed)
    lag_quarters : int, default=1
        Number of quarters to lag capacity ratios
    min_quarters : int, default=2
        Minimum quarters required to include grantee-disaster in analysis

    Returns
    -------
    pd.DataFrame
        Time-varying panel with columns:
        - Grantee, Disaster Type (identifiers)
        - start, stop (interval endpoints in months)
        - E (event indicator: 1 if completion occurred in this interval, else 0)
        - Ratio_disbursed_to_obligated_lag{lag_quarters}
        - Ratio_expended_to_disbursed_lag{lag_quarters}
        - Quarter (for reference)

    Notes
    -----
    - First lag_quarters intervals will have NaN for lagged ratios
    - Event E=1 only on final interval where completion occurred
    - Censored observations have E=0 on all intervals
    - Intervals with missing/invalid ratios (division by zero) are set to NaN

    Examples
    --------
    >>> qpr = pd.DataFrame({
    ...     'Grantee': ['NYC'] * 4,
    ...     'Disaster Type': ['Sandy'] * 4,
    ...     'Quarter': [1, 2, 3, 4],
    ...     'QPR Fund Obligated $': [100, 100, 100, 100],
    ...     'QPR Fund Disbursed $': [20, 50, 80, 95],
    ...     'QPR Fund Expended $': [5, 25, 60, 90]
    ... })
    >>> panel = pd.DataFrame({
    ...     'Grantee': ['NYC'],
    ...     'Disaster Type': ['Sandy'],
    ...     'Duration_of_completion': [12],  # completed at month 12
    ...     'N_Quarters': [4]
    ... })
    >>> tv = reshape_quarterly_to_time_varying(qpr, panel, lag_quarters=1)
    >>> tv[['start', 'stop', 'E', 'Ratio_disbursed_to_obligated_lag1']]
       start  stop  E  Ratio_disbursed_to_obligated_lag1
    0      0     3  0                                NaN
    1      3     6  0                               0.20
    2      6     9  0                               0.50
    3      9    12  1                               0.80
    """

    # Validate inputs
    if 'Grantee' not in qpr_quarterly.columns or 'Disaster Type' not in qpr_quarterly.columns:
        raise ValueError("qpr_quarterly must have 'Grantee' and 'Disaster Type' columns")

    required_cols = ['QPR Fund Obligated $', 'QPR Fund Disbursed $', 'QPR Fund Expended $']
    missing = [col for col in required_cols if col not in qpr_quarterly.columns]
    if missing:
        raise ValueError(f"qpr_quarterly missing required columns: {missing}")

    if lag_quarters < 0:
        raise ValueError("lag_quarters must be non-negative")

    # Sort by grantee-disaster and quarter
    qpr = qpr_quarterly.copy()
    qpr = qpr.sort_values(['Grantee', 'Disaster Type', 'QPR Actual Quarter']).reset_index(drop=True)

    # Create quarter index within each grantee-disaster
    qpr['Quarter_Index'] = qpr.groupby(['Grantee', 'Disaster Type']).cumcount()

    # Compute capacity ratios at each quarter
    qpr['Ratio_disbursed_to_obligated'] = np.where(
        qpr['QPR Fund Obligated $'] > 0,
        qpr['QPR Fund Disbursed $'] / qpr['QPR Fund Obligated $'],
        np.nan
    )

    qpr['Ratio_expended_to_disbursed'] = np.where(
        qpr['QPR Fund Disbursed $'] > 0,
        qpr['QPR Fund Expended $'] / qpr['QPR Fund Disbursed $'],
        np.nan
    )

    # Apply lagging within each grantee-disaster group
    lag_suffix = f'_lag{lag_quarters}'
    qpr[f'Ratio_disbursed_to_obligated{lag_suffix}'] = qpr.groupby(['Grantee', 'Disaster Type'])['Ratio_disbursed_to_obligated'].shift(lag_quarters)
    qpr[f'Ratio_expended_to_disbursed{lag_suffix}'] = qpr.groupby(['Grantee', 'Disaster Type'])['Ratio_expended_to_disbursed'].shift(lag_quarters)

    # Create start/stop intervals (quarters × 3 = months)
    qpr['start'] = qpr['Quarter_Index'] * 3
    qpr['stop'] = (qpr['Quarter_Index'] + 1) * 3

    # Merge with panel features to get completion info
    qpr = qpr.merge(
        panel_features[['Grantee', 'Disaster Type', duration_col, 'N_Quarters']],
        on=['Grantee', 'Disaster Type'],
        how='left'
    )

    # Determine event indicator E
    # E=1 only on the row where completion occurred (stop time >= duration)
    # For censored cases (duration_col is NaN), E=0 on all rows
    qpr['E'] = 0
    completed_mask = qpr[duration_col].notna()
    qpr.loc[completed_mask, 'E'] = (
        (qpr.loc[completed_mask, 'stop'] >= qpr.loc[completed_mask, duration_col]) &
        (qpr.loc[completed_mask, 'start'] < qpr.loc[completed_mask, duration_col])
    ).astype(int)

    # Filter to grantee-disasters with minimum quarters
    quarter_counts = qpr.groupby(['Grantee', 'Disaster Type'])['Quarter_Index'].transform('count')
    qpr = qpr[quarter_counts >= min_quarters].copy()

    # Select final columns
    output_cols = [
        'Grantee',
        'Disaster Type',
        'start',
        'stop',
        'E',
        f'Ratio_disbursed_to_obligated{lag_suffix}',
        f'Ratio_expended_to_disbursed{lag_suffix}',
        'Quarter_Index'
    ]

    tv_panel = qpr[output_cols].copy()

    # Quality check: ensure each grantee-disaster has at most one E=1
    event_counts = tv_panel.groupby(['Grantee', 'Disaster Type'])['E'].sum()
    if (event_counts > 1).any():
        warnings.warn(f"Found {(event_counts > 1).sum()} grantee-disasters with multiple events (E=1). This should not happen.")

    print(f"Time-varying panel created:")
    print(f"  Total intervals: {len(tv_panel):,}")
    print(f"  Unique grantee-disasters: {tv_panel.groupby(['Grantee', 'Disaster Type']).ngroups}")
    print(f"  Events (completions): {tv_panel['E'].sum()}")
    print(f"  Censored: {tv_panel.groupby(['Grantee', 'Disaster Type'])['E'].max().eq(0).sum()}")

    return tv_panel


def add_static_covariates(
    tv_data: pd.DataFrame,
    panel_features: pd.DataFrame,
    covariate_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Merge static covariates to time-varying data.

    Static covariates are time-invariant characteristics that don't change
    across quarters for a given grantee-disaster. These are repeated on
    every row (interval) for the same grantee-disaster.

    Parameters
    ----------
    tv_data : pd.DataFrame
        Time-varying survival data with start/stop/E columns
    panel_features : pd.DataFrame
        Panel with static features for each grantee-disaster
    covariate_cols : list of str, optional
        Specific covariates to include. If None, uses default set:
        - Government_Type_State
        - Log_Obligated
        - Prior_Grant_Count
        - Prior_Grant_Dollars_log
        - Disaster_Year
        - Population_log

    Returns
    -------
    pd.DataFrame
        Time-varying data with static covariates added

    Notes
    -----
    If covariate_cols is None, the function will include all available
    covariates from the default list that exist in panel_features.
    Missing covariates will generate a warning but won't fail.
    """

    # Default covariate set
    if covariate_cols is None:
        covariate_cols = [
            'Government_Type_State',
            'Log_Obligated',
            'Prior_Grant_Count',
            'Prior_Grant_Dollars_log',
            'Disaster_Year',
            'Population_log'
        ]

    # Check which covariates are available
    available_covariates = [col for col in covariate_cols if col in panel_features.columns]
    missing_covariates = [col for col in covariate_cols if col not in panel_features.columns]

    if missing_covariates:
        warnings.warn(f"Covariates not found in panel_features: {missing_covariates}")

    if not available_covariates:
        warnings.warn("No covariates available to merge")
        return tv_data

    # Merge covariates
    merge_cols = ['Grantee', 'Disaster Type'] + available_covariates
    tv_with_covariates = tv_data.merge(
        panel_features[merge_cols],
        on=['Grantee', 'Disaster Type'],
        how='left'
    )

    print(f"Added {len(available_covariates)} static covariates: {', '.join(available_covariates)}")

    return tv_with_covariates


def fit_time_varying_cox(
    tv_data: pd.DataFrame,
    capacity_cols: List[str],
    covariate_cols: Optional[List[str]] = None,
    penalizer: float = 0.1,
    strata: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fit Cox Proportional Hazards model with time-varying covariates.

    Uses lifelines CoxPHFitter with start/stop format to handle time-varying
    covariates. The capacity ratios change over time (lagged quarter-by-quarter),
    while static covariates remain constant for each grantee-disaster.

    Parameters
    ----------
    tv_data : pd.DataFrame
        Time-varying survival data with columns:
        - start, stop: Interval endpoints
        - E: Event indicator
        - Capacity columns (time-varying)
        - Covariate columns (static)
    capacity_cols : list of str
        Time-varying capacity predictors (e.g., lagged ratios)
    covariate_cols : list of str, optional
        Static covariates to include
    penalizer : float, default=0.1
        Ridge penalty for regularization
    strata : str, optional
        Column to stratify baseline hazard (e.g., 'Government_Type' or 'Disaster_Cohort')

    Returns
    -------
    dict
        - 'model': Fitted CoxPHFitter object
        - 'summary': Coefficient table (DataFrame)
        - 'hazard_ratios': HR with confidence intervals (DataFrame)
        - 'concordance': C-index (float)
        - 'n_obs': Number of intervals (int)
        - 'n_subjects': Number of unique grantee-disasters (int)
        - 'n_events': Number of completions (int)
        - 'ph_test': Proportional hazards test results (DataFrame) if available

    Raises
    ------
    ImportError
        If lifelines is not installed
    ValueError
        If required columns are missing or data has issues

    Notes
    -----
    - Rows with missing predictors are dropped before fitting
    - Standard errors do NOT account for clustering (use compute_bootstrap_se for that)
    - If strata is specified, baseline hazard differs across strata levels
    """

    if not LIFELINES_AVAILABLE:
        raise ImportError("lifelines required. Install with: pip install lifelines")

    # Combine predictor lists
    predictor_cols = capacity_cols.copy()
    if covariate_cols:
        predictor_cols.extend(covariate_cols)

    # Check for required columns
    required_cols = ['start', 'stop', 'E'] + predictor_cols
    missing = [col for col in required_cols if col not in tv_data.columns]
    if missing:
        raise ValueError(f"tv_data missing required columns: {missing}")

    # Prepare data for fitting - only select numeric columns needed for Cox model
    # Keep Grantee/Disaster Type for post-hoc analysis but don't pass to cph.fit()
    fit_data_full = tv_data[required_cols + (['Grantee', 'Disaster Type'] if 'Grantee' in tv_data.columns else [])]

    # Drop rows with missing predictors
    n_before = len(fit_data_full)
    fit_data_full = fit_data_full.dropna(subset=predictor_cols)
    n_after = len(fit_data_full)
    if n_after < n_before:
        print(f"Dropped {n_before - n_after} intervals with missing predictors ({100*(n_before-n_after)/n_before:.1f}%)")

    if len(fit_data_full) == 0:
        raise ValueError("No valid observations after dropping missing values")

    # Select only numeric columns for Cox fitting (exclude string identifiers)
    fit_data = fit_data_full[required_cols].copy()

    # Initialize Cox model
    cph = CoxPHFitter(penalizer=penalizer)

    # Fit model
    print(f"Fitting Cox PH model with {len(predictor_cols)} predictors...")
    print(f"  Time-varying capacity: {capacity_cols}")
    if covariate_cols:
        print(f"  Static covariates: {covariate_cols}")
    if strata:
        print(f"  Stratified by: {strata}")

    try:
        if strata:
            cph.fit(
                fit_data,
                duration_col='stop',
                event_col='E',
                entry_col='start',
                strata=strata,
                formula=" + ".join(predictor_cols) if len(predictor_cols) > 1 else predictor_cols[0]
            )
        else:
            cph.fit(
                fit_data,
                duration_col='stop',
                event_col='E',
                entry_col='start'
            )
    except Exception as e:
        print(f"Error fitting Cox model: {e}")
        raise

    # Extract results
    summary = cph.summary
    hazard_ratios = pd.DataFrame({
        'Variable': summary.index,
        'HR': np.exp(summary['coef']),
        'HR_Lower': np.exp(summary['coef'] - 1.96 * summary['se(coef)']),
        'HR_Upper': np.exp(summary['coef'] + 1.96 * summary['se(coef)']),
        'p_value': summary['p']
    })

    # Compute summary statistics
    n_subjects = fit_data_full.groupby(['Grantee', 'Disaster Type']).ngroups if 'Grantee' in fit_data_full.columns else fit_data['stop'].nunique()
    n_events = fit_data['E'].sum()

    print(f"  Concordance: {cph.concordance_index_:.3f}")
    print(f"  N intervals: {len(fit_data):,}")
    print(f"  N grantee-disasters: {n_subjects}")
    print(f"  N events: {n_events}")

    # Test proportional hazards assumption
    ph_test = None
    try:
        ph_test = cph.check_assumptions(fit_data, show_plots=False)
        print("  Proportional hazards test completed")
    except Exception as e:
        warnings.warn(f"Could not compute PH test: {e}")

    return {
        'model': cph,
        'summary': summary,
        'hazard_ratios': hazard_ratios,
        'concordance': cph.concordance_index_,
        'n_obs': len(fit_data),
        'n_subjects': n_subjects,
        'n_events': n_events,
        'ph_test': ph_test
    }


def compute_bootstrap_se(
    tv_data: pd.DataFrame,
    capacity_cols: List[str],
    covariate_cols: Optional[List[str]] = None,
    cluster_col: str = 'Grantee',
    n_bootstrap: int = 1000,
    penalizer: float = 0.1,
    seed: int = 42
) -> pd.DataFrame:
    """
    Compute bootstrap standard errors with cluster resampling.

    lifelines doesn't support native clustered SEs, so we use bootstrap
    resampling at the cluster level (typically grantee) to account for
    within-cluster correlation.

    Parameters
    ----------
    tv_data : pd.DataFrame
        Time-varying survival data
    capacity_cols : list of str
        Time-varying capacity predictors
    covariate_cols : list of str, optional
        Static covariates
    cluster_col : str, default='Grantee'
        Column to cluster on (resample entire clusters)
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
    penalizer : float, default=0.1
        Ridge penalty for Cox models
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Bootstrap results with columns:
        - Variable
        - coef_bootstrap (mean coefficient across bootstrap samples)
        - se_bootstrap (bootstrap standard error)
        - HR_bootstrap (mean hazard ratio)
        - HR_Lower_bootstrap (2.5th percentile HR)
        - HR_Upper_bootstrap (97.5th percentile HR)
        - p_value_bootstrap (two-tailed test using bootstrap SE)

    Notes
    -----
    - Resamples clusters with replacement (some clusters appear multiple times, others not at all)
    - Fits Cox model on each bootstrap sample
    - Standard errors from distribution of bootstrap estimates
    - Computationally expensive (consider starting with n=100 for testing)
    """

    if not LIFELINES_AVAILABLE:
        raise ImportError("lifelines required. Install with: pip install lifelines")

    if cluster_col not in tv_data.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in tv_data")

    np.random.seed(seed)

    # Get unique clusters
    clusters = tv_data[cluster_col].unique()
    n_clusters = len(clusters)

    print(f"Computing bootstrap SEs with cluster resampling...")
    print(f"  Cluster variable: {cluster_col}")
    print(f"  N clusters: {n_clusters}")
    print(f"  Bootstrap iterations: {n_bootstrap}")
    print(f"  This may take several minutes...")

    # Combine predictor lists
    predictor_cols = capacity_cols.copy()
    if covariate_cols:
        predictor_cols.extend(covariate_cols)

    # Store bootstrap estimates
    bootstrap_coefs = []

    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"  Iteration {i+1}/{n_bootstrap}...")

        # Resample clusters with replacement
        bootstrap_clusters = np.random.choice(clusters, size=n_clusters, replace=True)

        # Create bootstrap sample
        bootstrap_sample = pd.concat([
            tv_data[tv_data[cluster_col] == cluster]
            for cluster in bootstrap_clusters
        ], ignore_index=True)

        # Fit Cox model on bootstrap sample
        try:
            result = fit_time_varying_cox(
                tv_data=bootstrap_sample,
                capacity_cols=capacity_cols,
                covariate_cols=covariate_cols,
                penalizer=penalizer
            )
            bootstrap_coefs.append(result['summary']['coef'].values)
        except Exception as e:
            # Some bootstrap samples may fail to converge
            warnings.warn(f"Bootstrap iteration {i+1} failed: {e}")
            continue

    # Convert to array
    bootstrap_coefs = np.array(bootstrap_coefs)

    print(f"  Successful iterations: {len(bootstrap_coefs)}/{n_bootstrap}")

    if len(bootstrap_coefs) < 50:
        warnings.warn(f"Only {len(bootstrap_coefs)} successful bootstrap iterations. Results may be unreliable.")

    # Compute bootstrap statistics
    bootstrap_results = pd.DataFrame({
        'Variable': predictor_cols,
        'coef_bootstrap': bootstrap_coefs.mean(axis=0),
        'se_bootstrap': bootstrap_coefs.std(axis=0),
    })

    # Compute bootstrap HR and confidence intervals
    bootstrap_results['HR_bootstrap'] = np.exp(bootstrap_results['coef_bootstrap'])
    bootstrap_results['HR_Lower_bootstrap'] = np.exp(np.percentile(bootstrap_coefs, 2.5, axis=0))
    bootstrap_results['HR_Upper_bootstrap'] = np.exp(np.percentile(bootstrap_coefs, 97.5, axis=0))

    # Compute p-values using bootstrap SE
    z_stats = bootstrap_results['coef_bootstrap'] / bootstrap_results['se_bootstrap']
    bootstrap_results['p_value_bootstrap'] = 2 * (1 - np.abs(z_stats).apply(lambda z: np.minimum(1, np.exp(-0.717*z - 0.416*z**2))))

    print("Bootstrap SE computation complete")

    return bootstrap_results
