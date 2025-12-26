"""
Alternative modeling approaches for handling right-censored duration data.

This module provides:
1. Survival analysis (Cox PH, AFT models) for censored duration
2. Threshold sensitivity analysis utilities
3. Duration-free model support
4. Cross-method comparison functions

The primary motivation is that 73.7% of observations lack valid Duration at
the 95% completion threshold. These alternative approaches either:
- Properly model censored data (survival analysis)
- Use lower thresholds with more complete cases
- Avoid duration entirely with ratio-based outcomes
- Use milestone-based metrics (Time_to_50pct, Progress_Rate)
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import warnings

# Optional imports for survival analysis
LIFELINES_AVAILABLE = False
try:
    from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_lifelines():
    """Check if lifelines is available for survival analysis."""
    if not LIFELINES_AVAILABLE:
        raise ImportError(
            "lifelines is required for survival analysis.\n"
            "Install with: pip install lifelines>=0.27.0"
        )


def get_available_duration_thresholds(data: pd.DataFrame) -> Dict[str, int]:
    """
    Get count of non-missing observations for each Duration threshold.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with Duration columns.

    Returns
    -------
    Dict[str, int]
        Mapping of threshold column to count of valid observations.
        Example: {'Duration_30pct': 95, 'Duration_50pct': 89, ...}
    """
    thresholds = {}
    duration_cols = [c for c in data.columns if c.startswith('Duration_') and 'pct' in c]

    for col in sorted(duration_cols):
        valid_count = data[col].notna().sum()
        thresholds[col] = valid_count

    return thresholds


# =============================================================================
# SURVIVAL DATA PREPARATION
# =============================================================================

def prepare_survival_data(
    data: pd.DataFrame,
    duration_col: str = 'Duration_of_completion',
    n_quarters_col: str = 'N_Quarters',
    capacity_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Prepare data for survival analysis with proper censoring indicator.

    Censored observations (incomplete programs) get:
    - T = total observation time (N_Quarters * 3 months)
    - E = 0 (not completed)

    Complete observations get:
    - T = actual duration
    - E = 1 (completed)

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with duration and capacity indicators.
    duration_col : str
        Column with duration values (NaN for censored).
    n_quarters_col : str
        Column with number of quarters observed.
    capacity_cols : list, optional
        Columns for capacity indicators. If None, uses defaults.

    Returns
    -------
    pd.DataFrame
        Data with columns:
        - T: Observed time (duration if complete, observation_time if censored)
        - E: Event indicator (1=completed, 0=censored)
        - Capacity predictors
    """
    if capacity_cols is None:
        capacity_cols = [
            'Ratio_disbursed_to_obligated',
            'Ratio_expended_to_disbursed',
        ]

    df = data.copy()

    # Create event indicator: 1 if completed (duration not missing), 0 if censored
    df['E'] = df[duration_col].notna().astype(int)

    # Create time variable: actual duration if complete, else observation time
    df['T'] = df[duration_col].copy()

    # For censored observations, use N_Quarters * 3 as observation time
    if n_quarters_col in df.columns:
        censored_mask = df[duration_col].isna()
        df.loc[censored_mask, 'T'] = df.loc[censored_mask, n_quarters_col] * 3  # months

    # Ensure T > 0 (required for survival models)
    df['T'] = df['T'].clip(lower=0.1)

    # Select columns for survival analysis
    keep_cols = ['T', 'E'] + [c for c in capacity_cols if c in df.columns]

    # Add identifier columns if present
    for id_col in ['Grantee', 'Disaster Type', 'Disaster_Year']:
        if id_col in df.columns:
            keep_cols.append(id_col)

    # Drop rows with missing capacity values
    result = df[keep_cols].dropna(subset=[c for c in capacity_cols if c in df.columns])

    return result


# =============================================================================
# COX PROPORTIONAL HAZARDS MODEL
# =============================================================================

def fit_cox_model(
    data: pd.DataFrame,
    capacity_cols: List[str],
    duration_col: str = 'T',
    event_col: str = 'E',
    penalizer: float = 0.1
) -> Dict[str, Any]:
    """
    Fit Cox Proportional Hazards model.

    Cox PH models the hazard (instantaneous completion rate):
    h(t) = h0(t) * exp(beta * X)

    Interpretation:
    - Hazard Ratio (HR) = exp(beta)
    - HR > 1: Higher value increases completion rate (faster completion)
    - HR < 1: Higher value decreases completion rate (slower completion)

    Parameters
    ----------
    data : pd.DataFrame
        Prepared survival data with T, E columns.
    capacity_cols : list
        Predictor variables.
    duration_col : str, default 'T'
        Time variable column.
    event_col : str, default 'E'
        Event indicator column.
    penalizer : float, default 0.1
        Ridge penalty for small samples.

    Returns
    -------
    Dict[str, Any]
        - 'model': Fitted CoxPHFitter
        - 'summary': Model summary DataFrame
        - 'hazard_ratios': HR with CI
        - 'concordance': C-index
        - 'proportional_hazards_test': Schoenfeld test results (if available)
    """
    check_lifelines()

    # Prepare data for fitting
    fit_cols = [duration_col, event_col] + capacity_cols
    fit_data = data[fit_cols].dropna()

    if len(fit_data) < 10:
        return {
            'error': f'Insufficient data: N={len(fit_data)}',
            'n_obs': len(fit_data),
        }

    # Fit Cox model
    cph = CoxPHFitter(penalizer=penalizer)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(
            fit_data,
            duration_col=duration_col,
            event_col=event_col
        )

    # Extract results
    summary = cph.summary.copy()
    summary['hazard_ratio'] = np.exp(summary['coef'])

    # Concordance index
    c_index = cph.concordance_index_

    # Test proportional hazards assumption
    ph_test = None
    try:
        ph_test = cph.check_assumptions(fit_data, show_plots=False)
    except Exception:
        pass

    results = {
        'model': cph,
        'summary': summary,
        'hazard_ratios': summary[['coef', 'exp(coef)', 'coef lower 95%', 'coef upper 95%', 'p']].copy(),
        'concordance': c_index,
        'proportional_hazards_test': ph_test,
        'n_obs': len(fit_data),
        'n_events': int(fit_data[event_col].sum()),
        'model_type': 'Cox_PH',
    }

    return results


# =============================================================================
# ACCELERATED FAILURE TIME MODELS
# =============================================================================

def fit_aft_model(
    data: pd.DataFrame,
    capacity_cols: List[str],
    duration_col: str = 'T',
    event_col: str = 'E',
    distribution: str = 'weibull'
) -> Dict[str, Any]:
    """
    Fit Accelerated Failure Time model.

    AFT models the survival time directly:
    log(T) = beta * X + sigma * epsilon

    Interpretation:
    - Time Ratio (TR) = exp(beta)
    - TR > 1: Higher value extends time (SLOWER completion)
    - TR < 1: Higher value shortens time (FASTER completion)

    Note: AFT interpretation is OPPOSITE to Cox hazard ratio.

    Parameters
    ----------
    data : pd.DataFrame
        Prepared survival data with T, E columns.
    capacity_cols : list
        Predictor variables.
    duration_col : str, default 'T'
        Time variable column.
    event_col : str, default 'E'
        Event indicator column.
    distribution : str, default 'weibull'
        Survival distribution: 'weibull', 'lognormal', or 'loglogistic'.

    Returns
    -------
    Dict[str, Any]
        - 'model': Fitted AFT model
        - 'summary': Coefficient summary
        - 'time_ratios': Acceleration factors with CI
        - 'aic': Model AIC for comparison
    """
    check_lifelines()

    # Select AFT fitter based on distribution
    fitters = {
        'weibull': WeibullAFTFitter,
        'lognormal': LogNormalAFTFitter,
        'loglogistic': LogLogisticAFTFitter,
    }

    if distribution not in fitters:
        raise ValueError(f"Unknown distribution: {distribution}. Choose from: {list(fitters.keys())}")

    # Prepare data for fitting
    fit_cols = [duration_col, event_col] + capacity_cols
    fit_data = data[fit_cols].dropna()

    if len(fit_data) < 10:
        return {
            'error': f'Insufficient data: N={len(fit_data)}',
            'n_obs': len(fit_data),
        }

    # Fit AFT model
    aft = fitters[distribution]()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aft.fit(
            fit_data,
            duration_col=duration_col,
            event_col=event_col
        )

    # Extract results
    summary = aft.summary.copy()

    # Compute time ratios (exp of coefficients)
    # Note: For AFT, we need to extract mu_ coefficients
    time_ratios = summary.copy()
    time_ratios['time_ratio'] = np.exp(time_ratios['coef'])

    # AIC for model comparison
    aic = aft.AIC_

    results = {
        'model': aft,
        'summary': summary,
        'time_ratios': time_ratios,
        'aic': aic,
        'concordance': aft.concordance_index_,
        'n_obs': len(fit_data),
        'n_events': int(fit_data[event_col].sum()),
        'model_type': f'AFT_{distribution}',
        'distribution': distribution,
    }

    return results


def compare_survival_models(
    data: pd.DataFrame,
    capacity_cols: List[str],
    distributions: List[str] = ['weibull', 'lognormal', 'loglogistic']
) -> pd.DataFrame:
    """
    Fit and compare multiple survival model specifications.

    Parameters
    ----------
    data : pd.DataFrame
        Prepared survival data with T, E columns.
    capacity_cols : list
        Predictor variables.
    distributions : list
        AFT distributions to compare.

    Returns
    -------
    pd.DataFrame
        Comparison table with model fit statistics.
    """
    check_lifelines()

    results = []

    # Fit Cox PH
    cox_result = fit_cox_model(data, capacity_cols)
    if 'error' not in cox_result:
        results.append({
            'Model': 'Cox_PH',
            'Distribution': 'semi-parametric',
            'N': cox_result['n_obs'],
            'N_Events': cox_result['n_events'],
            'Concordance': cox_result['concordance'],
            'AIC': np.nan,  # Cox doesn't have comparable AIC
        })

    # Fit AFT models
    for dist in distributions:
        try:
            aft_result = fit_aft_model(data, capacity_cols, distribution=dist)
            if 'error' not in aft_result:
                results.append({
                    'Model': f'AFT_{dist}',
                    'Distribution': dist,
                    'N': aft_result['n_obs'],
                    'N_Events': aft_result['n_events'],
                    'Concordance': aft_result['concordance'],
                    'AIC': aft_result['aic'],
                })
        except Exception as e:
            print(f"  Warning: AFT {dist} failed: {e}")

    return pd.DataFrame(results)


# =============================================================================
# SURVIVAL RESULTS EXTRACTION
# =============================================================================

def extract_survival_coefficients(
    cox_result: Optional[Dict] = None,
    aft_results: Optional[List[Dict]] = None,
    capacity_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract coefficients from survival models into a unified format.

    Parameters
    ----------
    cox_result : dict, optional
        Result from fit_cox_model().
    aft_results : list of dict, optional
        Results from fit_aft_model() for different distributions.
    capacity_cols : list, optional
        Capacity variable names for filtering.

    Returns
    -------
    pd.DataFrame
        Unified coefficient table with columns:
        Model, Variable, Coefficient, SE, Effect_Ratio, Effect_Lower, Effect_Upper, p_value
    """
    rows = []

    # Cox PH results
    if cox_result and 'error' not in cox_result:
        summary = cox_result['summary']
        for var in summary.index:
            rows.append({
                'Model': 'Cox_PH',
                'Variable': var,
                'Coefficient': summary.loc[var, 'coef'],
                'SE': summary.loc[var, 'se(coef)'],
                'Effect_Ratio': summary.loc[var, 'exp(coef)'],
                'Effect_Lower': np.exp(summary.loc[var, 'coef lower 95%']),
                'Effect_Upper': np.exp(summary.loc[var, 'coef upper 95%']),
                'p_value': summary.loc[var, 'p'],
                'Interpretation': 'Hazard Ratio (>1 = faster completion)',
            })

    # AFT results
    if aft_results:
        for aft_result in aft_results:
            if 'error' in aft_result:
                continue

            model_type = aft_result['model_type']
            summary = aft_result['summary']

            # Filter to capacity variables (mu_ prefix in AFT models)
            for var in summary.index:
                # Extract variable name (handle tuple index from lifelines)
                if isinstance(var, tuple):
                    # lifelines AFT uses (param_type, var_name) tuples
                    param_type, var_name = var
                    if param_type != 'mu_':
                        continue  # Only extract mu_ (location) parameters
                else:
                    # String index - remove mu_ prefix if present
                    var_name = var.replace('mu_', '') if 'mu_' in str(var) else str(var)

                if capacity_cols and var_name not in capacity_cols:
                    continue

                rows.append({
                    'Model': model_type,
                    'Variable': var_name,
                    'Coefficient': summary.loc[var, 'coef'],
                    'SE': summary.loc[var, 'se(coef)'],
                    'Effect_Ratio': np.exp(summary.loc[var, 'coef']),
                    'Effect_Lower': np.exp(summary.loc[var, 'coef lower 95%']),
                    'Effect_Upper': np.exp(summary.loc[var, 'coef upper 95%']),
                    'p_value': summary.loc[var, 'p'],
                    'Interpretation': 'Time Ratio (<1 = faster completion)',
                })

    return pd.DataFrame(rows)


# =============================================================================
# THRESHOLD SENSITIVITY FOR SEM
# =============================================================================

def run_threshold_sensitivity_sem(
    data: pd.DataFrame,
    thresholds: List[str] = ['50pct', '70pct', '90pct'],
    subset: str = 'all',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run SEM at multiple duration thresholds to test sensitivity.

    For each threshold, fits the model substituting Duration_Xpct_log
    for Duration_log in the model specification.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with features.
    thresholds : list
        Threshold suffixes to test (e.g., ['50pct', '70pct', '90pct']).
    subset : str
        Government type: 'all', 'state', or 'local'.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        Comparison table with columns:
        - Threshold, N, CFI, RMSEA, Capacity_Beta, Capacity_SE, Capacity_p
    """
    from capacity_sem.models.sem_fitting import fit_sem_model, SEMOPY_AVAILABLE
    from capacity_sem.models.sem_diagnostics import extract_fit_stat
    from config import STATE_GOVERNMENTS, LOCAL_GOVERNMENTS

    if not SEMOPY_AVAILABLE:
        raise ImportError("semopy is required for threshold sensitivity analysis")

    results = []

    # Filter by subset
    if subset == 'state':
        data = data[data['Grantee'].isin(STATE_GOVERNMENTS)]
    elif subset == 'local':
        data = data[data['Grantee'].isin(LOCAL_GOVERNMENTS)]

    for threshold in thresholds:
        duration_col = f'Duration_{threshold}'
        duration_log_col = f'Duration_{threshold}_log'

        if verbose:
            print(f"  Testing threshold: {threshold}...")

        # Check if column exists
        if duration_log_col not in data.columns:
            if verbose:
                print(f"    Skipping: {duration_log_col} not found")
            continue

        # Count valid observations
        n_valid = data[duration_log_col].notna().sum()

        # Build model specification dynamically
        model_spec = f"""
# SEM with Duration at {threshold} threshold
gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
recovery_outcome =~ {duration_log_col} + Spending_CV

recovery_outcome ~ gov_cap
"""

        try:
            from semopy import Model, calc_stats

            # Prepare data
            required_cols = [
                'Ratio_disbursed_to_obligated',
                'Ratio_expended_to_disbursed',
                duration_log_col,
                'Spending_CV'
            ]
            fit_data = data[required_cols].dropna()

            if len(fit_data) < 10:
                if verbose:
                    print(f"    Skipping: insufficient data (N={len(fit_data)})")
                continue

            # Fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = Model(model_spec)
                model.fit(fit_data)

            estimates = model.inspect()
            fit_stats = calc_stats(model)

            # Extract structural path
            structural = estimates[
                (estimates['op'] == '~') &
                (estimates['lval'] == 'recovery_outcome')
            ]

            beta = np.nan
            se = np.nan
            pval = np.nan

            if not structural.empty:
                row = structural.iloc[0]
                beta = float(row['Estimate'])
                se = float(row['Std. Err']) if pd.notna(row['Std. Err']) else np.nan
                pval = float(row['p-value']) if pd.notna(row['p-value']) else np.nan

            results.append({
                'Threshold': threshold,
                'Duration_Col': duration_log_col,
                'N': len(fit_data),
                'Pct_Available': 100 * len(fit_data) / len(data),
                'CFI': extract_fit_stat(fit_stats, 'CFI'),
                'RMSEA': extract_fit_stat(fit_stats, 'RMSEA'),
                'Capacity_Beta': beta,
                'Capacity_SE': se,
                'Capacity_p': pval,
                'Significant': pval < 0.05 if pd.notna(pval) else False,
                'Subset': subset,
            })

            if verbose:
                sig_str = '*' if pval < 0.05 else ''
                print(f"    N={len(fit_data)}, Beta={beta:.3f}, p={pval:.3f}{sig_str}")

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            continue

    return pd.DataFrame(results)


# =============================================================================
# CROSS-METHOD COMPARISON
# =============================================================================

def compare_methods(
    sem_results: Optional[pd.DataFrame] = None,
    survival_results: Optional[pd.DataFrame] = None,
    threshold_results: Optional[pd.DataFrame] = None,
    duration_free_results: Optional[pd.DataFrame] = None,
    milestone_results: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Create unified comparison across all alternative approaches.

    Parameters
    ----------
    sem_results : pd.DataFrame, optional
        Standard SEM results.
    survival_results : pd.DataFrame, optional
        Survival analysis coefficient table.
    threshold_results : pd.DataFrame, optional
        Threshold sensitivity results.
    duration_free_results : pd.DataFrame, optional
        Duration-free model results.
    milestone_results : pd.DataFrame, optional
        Milestone-based model results.

    Returns
    -------
    pd.DataFrame
        Comparison with standardized effect sizes and significance.
    """
    rows = []

    # Add SEM standard results
    if sem_results is not None and not sem_results.empty:
        for _, row in sem_results.iterrows():
            rows.append({
                'Method': 'SEM',
                'Model': row.get('Model', 'standard'),
                'N': row.get('N', np.nan),
                'Effect_Estimate': row.get('Capacity_Beta', row.get('Beta', np.nan)),
                'Effect_SE': row.get('Capacity_SE', row.get('SE', np.nan)),
                'p_value': row.get('Capacity_p', row.get('p_value', np.nan)),
                'Interpretation': 'Structural path coefficient',
            })

    # Add survival results
    if survival_results is not None and not survival_results.empty:
        for _, row in survival_results.iterrows():
            rows.append({
                'Method': 'Survival',
                'Model': row.get('Model', 'unknown'),
                'N': row.get('N', np.nan),
                'Effect_Estimate': row.get('Effect_Ratio', np.nan),
                'Effect_SE': row.get('SE', np.nan),
                'p_value': row.get('p_value', np.nan),
                'Interpretation': row.get('Interpretation', 'Effect ratio'),
            })

    # Add threshold results
    if threshold_results is not None and not threshold_results.empty:
        for _, row in threshold_results.iterrows():
            rows.append({
                'Method': 'SEM_Threshold',
                'Model': f"Duration_{row['Threshold']}",
                'N': row['N'],
                'Effect_Estimate': row['Capacity_Beta'],
                'Effect_SE': row['Capacity_SE'],
                'p_value': row['Capacity_p'],
                'Interpretation': 'Structural path (lower threshold)',
            })

    # Add duration-free results
    if duration_free_results is not None and not duration_free_results.empty:
        for _, row in duration_free_results.iterrows():
            rows.append({
                'Method': 'SEM_DurationFree',
                'Model': row.get('Model', 'duration_free'),
                'N': row.get('N', np.nan),
                'Effect_Estimate': row.get('Capacity_Beta', row.get('Beta', np.nan)),
                'Effect_SE': row.get('Capacity_SE', row.get('SE', np.nan)),
                'p_value': row.get('Capacity_p', row.get('p_value', np.nan)),
                'Interpretation': 'Structural path (no duration)',
            })

    # Add milestone results
    if milestone_results is not None and not milestone_results.empty:
        for _, row in milestone_results.iterrows():
            rows.append({
                'Method': 'SEM_Milestone',
                'Model': row.get('Model', 'milestone'),
                'N': row.get('N', np.nan),
                'Effect_Estimate': row.get('Capacity_Beta', row.get('Beta', np.nan)),
                'Effect_SE': row.get('Capacity_SE', row.get('SE', np.nan)),
                'p_value': row.get('Capacity_p', row.get('p_value', np.nan)),
                'Interpretation': 'Structural path (milestone outcome)',
            })

    comparison = pd.DataFrame(rows)

    # Add significance flag
    if not comparison.empty and 'p_value' in comparison.columns:
        comparison['Significant'] = comparison['p_value'] < 0.05

    return comparison


def summarize_alternatives_findings(comparison: pd.DataFrame) -> str:
    """
    Generate narrative summary of findings across approaches.

    Parameters
    ----------
    comparison : pd.DataFrame
        Cross-method comparison table.

    Returns
    -------
    str
        Narrative summary.
    """
    if comparison.empty:
        return "No results to summarize."

    lines = [
        "=" * 60,
        "ALTERNATIVE MODELING APPROACHES: SUMMARY",
        "=" * 60,
        "",
    ]

    # Count significant findings
    if 'Significant' in comparison.columns:
        n_sig = comparison['Significant'].sum()
        n_total = len(comparison)
        lines.append(f"Total models tested: {n_total}")
        lines.append(f"Significant effects (p < 0.05): {n_sig} ({100*n_sig/n_total:.1f}%)")
        lines.append("")

    # Summary by method
    if 'Method' in comparison.columns:
        lines.append("Results by Method:")
        lines.append("-" * 40)

        for method in comparison['Method'].unique():
            method_df = comparison[comparison['Method'] == method]
            n_models = len(method_df)
            n_sig = method_df['Significant'].sum() if 'Significant' in method_df.columns else 0

            lines.append(f"\n{method}:")
            lines.append(f"  Models: {n_models}")
            lines.append(f"  Significant: {n_sig}")

            # Average effect if available
            if 'Effect_Estimate' in method_df.columns:
                effects = method_df['Effect_Estimate'].dropna()
                if len(effects) > 0:
                    lines.append(f"  Mean effect: {effects.mean():.3f}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
