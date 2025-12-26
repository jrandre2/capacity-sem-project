"""
SEM model diagnostics and evaluation functions.

This module provides functions to evaluate model fit and
compare alternative model specifications.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

try:
    from semopy import calc_stats
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False


# Fit index thresholds
FIT_THRESHOLDS = {
    'CFI': {'acceptable': 0.90, 'good': 0.95},
    'TLI': {'acceptable': 0.90, 'good': 0.95},
    'RMSEA': {'acceptable': 0.08, 'good': 0.05},  # Lower is better
    'SRMR': {'acceptable': 0.08, 'good': 0.05},  # Lower is better
    'chi2_pvalue': {'acceptable': 0.05, 'good': 0.10}  # Higher is better
}


def compute_srmr(model: Any, data: pd.DataFrame) -> float:
    """
    Compute Standardized Root Mean Square Residual (SRMR).

    SRMR measures the average discrepancy between observed and
    model-implied correlations. Values < 0.08 indicate acceptable fit.

    Parameters
    ----------
    model : semopy.Model
        Fitted SEM model.
    data : pd.DataFrame
        Original data used for fitting.

    Returns
    -------
    float
        SRMR value.
    """
    try:
        # Get observed correlation matrix
        obs_vars = model.vars['observed']
        obs_data = data[obs_vars].dropna()
        S = obs_data.corr().values

        # Get model-implied correlation matrix
        # semopy provides sigma (covariance), need to convert to correlation
        Sigma = model.sigma

        # Convert covariance to correlation
        D = np.sqrt(np.diag(Sigma))
        D_inv = np.diag(1.0 / D)
        Sigma_corr = D_inv @ Sigma @ D_inv

        # Compute SRMR
        n = S.shape[0]
        residuals = S - Sigma_corr

        # Only lower triangle (including diagonal for variances)
        lower_tri = np.tril_indices(n)
        srmr = np.sqrt(np.mean(residuals[lower_tri] ** 2))

        return srmr
    except Exception as e:
        return np.nan


def compute_rmsea_ci(chi2: float, df: int, n: int, alpha: float = 0.10) -> Dict[str, float]:
    """
    Compute RMSEA with 90% confidence interval.

    Uses non-central chi-square distribution to compute confidence
    bounds for RMSEA.

    Parameters
    ----------
    chi2 : float
        Chi-square test statistic.
    df : int
        Degrees of freedom.
    n : int
        Sample size.
    alpha : float, default 0.10
        Significance level (0.10 gives 90% CI).

    Returns
    -------
    Dict[str, float]
        Dictionary with 'rmsea', 'lower', 'upper' values.
    """
    from scipy import stats

    if df <= 0 or n <= 0:
        return {'rmsea': np.nan, 'lower': np.nan, 'upper': np.nan}

    # Point estimate of RMSEA
    if chi2 <= df:
        rmsea = 0.0
    else:
        rmsea = np.sqrt((chi2 - df) / (df * (n - 1)))

    # Non-centrality parameter bounds
    # Lower bound: find lambda such that P(chi2 > obs | lambda) = 1 - alpha/2
    # Upper bound: find lambda such that P(chi2 > obs | lambda) = alpha/2

    try:
        # Lower bound for RMSEA (upper bound on ncp)
        # Use optimization to find ncp
        from scipy.optimize import brentq

        def ncp_func_lower(ncp):
            return stats.ncx2.sf(chi2, df, ncp) - (1 - alpha / 2)

        def ncp_func_upper(ncp):
            return stats.ncx2.sf(chi2, df, ncp) - alpha / 2

        # Lower CI bound
        if chi2 <= df:
            ncp_lower = 0
        else:
            try:
                ncp_lower = brentq(ncp_func_lower, 0, chi2 * 5, xtol=1e-6)
            except ValueError:
                ncp_lower = 0

        # Upper CI bound
        try:
            ncp_upper = brentq(ncp_func_upper, 0, chi2 * 10 + 1, xtol=1e-6)
        except ValueError:
            ncp_upper = chi2  # Fallback

        # Convert ncp to RMSEA
        rmsea_lower = np.sqrt(max(0, ncp_lower) / (df * (n - 1)))
        rmsea_upper = np.sqrt(max(0, ncp_upper) / (df * (n - 1)))

    except Exception:
        rmsea_lower = np.nan
        rmsea_upper = np.nan

    return {
        'rmsea': rmsea,
        'lower': rmsea_lower,
        'upper': rmsea_upper
    }


def compute_composite_reliability(loadings: pd.DataFrame) -> Dict[str, float]:
    """
    Compute Composite Reliability (CR) and Average Variance Extracted (AVE).

    CR = (Σλ)² / [(Σλ)² + Σ(1-λ²)]
    AVE = Σλ² / n

    Parameters
    ----------
    loadings : pd.DataFrame
        Factor loadings with columns: factor, indicator, loading.

    Returns
    -------
    Dict[str, float]
        Dictionary with CR and AVE for each factor.
    """
    results = {}

    for factor in loadings['factor'].unique():
        factor_loadings = loadings[loadings['factor'] == factor]['loading'].values

        # Compute CR
        sum_lambda = np.sum(factor_loadings)
        sum_error = np.sum(1 - factor_loadings ** 2)
        cr = (sum_lambda ** 2) / (sum_lambda ** 2 + sum_error)

        # Compute AVE
        ave = np.mean(factor_loadings ** 2)

        results[factor] = {
            'CR': cr,
            'AVE': ave,
            'n_indicators': len(factor_loadings),
            'loadings': factor_loadings.tolist()
        }

    return results


def extract_fit_stat(fit_stats: Any, key: str) -> float:
    """
    Extract a fit statistic from semopy calc_stats output.

    Handles DataFrame, Series, or dict representations.
    """
    if fit_stats is None:
        return np.nan

    if isinstance(fit_stats, pd.DataFrame):
        if key in fit_stats.index:
            if 'Value' in fit_stats.columns:
                return fit_stats.loc[key, 'Value']
            row = fit_stats.loc[key]
            if isinstance(row, pd.Series):
                if 'Value' in row:
                    return row['Value']
                return row.iloc[0]
            return row
        if key in fit_stats.columns:
            col = fit_stats[key]
            if isinstance(col, pd.Series):
                if 'Value' in fit_stats.index:
                    return col.loc['Value']
                return col.iloc[0]
            return col

    if isinstance(fit_stats, pd.Series):
        return fit_stats.get(key, np.nan)

    if isinstance(fit_stats, dict):
        if key in fit_stats:
            value = fit_stats[key]
            if isinstance(value, dict):
                return value.get('Value', np.nan)
            return value
        if 'Value' in fit_stats and isinstance(fit_stats['Value'], dict):
            return fit_stats['Value'].get(key, np.nan)

    return np.nan


def evaluate_model_fit(fit_stats: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate model fit against standard thresholds.

    Parameters
    ----------
    fit_stats : pd.DataFrame
        Fit statistics from calc_stats().

    Returns
    -------
    Dict[str, Any]
        Evaluation results with fit indices and interpretations.
    """
    evaluation = {
        'indices': {},
        'interpretations': {},
        'overall_fit': 'unknown'
    }

    # Get CFI
    cfi = extract_fit_stat(fit_stats, 'CFI')
    if not pd.isna(cfi):
        evaluation['indices']['CFI'] = cfi
        evaluation['interpretations']['CFI'] = _interpret_index(
            cfi, 'CFI', higher_is_better=True
        )

    # Get TLI
    tli = extract_fit_stat(fit_stats, 'TLI')
    if not pd.isna(tli):
        evaluation['indices']['TLI'] = tli
        evaluation['interpretations']['TLI'] = _interpret_index(
            tli, 'TLI', higher_is_better=True
        )

    # Get RMSEA
    rmsea = extract_fit_stat(fit_stats, 'RMSEA')
    if not pd.isna(rmsea):
        evaluation['indices']['RMSEA'] = rmsea
        evaluation['interpretations']['RMSEA'] = _interpret_index(
            rmsea, 'RMSEA', higher_is_better=False
        )

    # Get SRMR if available
    srmr = extract_fit_stat(fit_stats, 'SRMR')
    if not pd.isna(srmr):
        evaluation['indices']['SRMR'] = srmr
        evaluation['interpretations']['SRMR'] = _interpret_index(
            srmr, 'SRMR', higher_is_better=False
        )

    # Get chi-square
    chi2 = extract_fit_stat(fit_stats, 'chi2')
    if not pd.isna(chi2):
        evaluation['indices']['chi2'] = chi2

    pval = extract_fit_stat(fit_stats, 'chi2 p-value')
    if not pd.isna(pval):
        evaluation['indices']['chi2_pvalue'] = pval
        evaluation['interpretations']['chi2_pvalue'] = _interpret_pvalue(pval)

    # Overall fit assessment
    evaluation['overall_fit'] = _assess_overall_fit(evaluation['interpretations'])

    return evaluation


def _interpret_index(value: float, index_name: str, higher_is_better: bool) -> str:
    """Interpret a single fit index."""
    if pd.isna(value):
        return 'NA'

    thresholds = FIT_THRESHOLDS.get(index_name, {})

    if higher_is_better:
        if value >= thresholds.get('good', 0.95):
            return 'good'
        elif value >= thresholds.get('acceptable', 0.90):
            return 'acceptable'
        else:
            return 'poor'
    else:
        if value <= thresholds.get('good', 0.05):
            return 'good'
        elif value <= thresholds.get('acceptable', 0.08):
            return 'acceptable'
        else:
            return 'poor'


def _interpret_pvalue(pvalue: float) -> str:
    """Interpret chi-square p-value."""
    if pd.isna(pvalue):
        return 'NA'
    if pvalue >= 0.05:
        return 'good'
    elif pvalue >= 0.01:
        return 'marginal'
    else:
        return 'poor'


def _assess_overall_fit(interpretations: Dict[str, str]) -> str:
    """Assess overall model fit based on individual indices."""
    if not interpretations:
        return 'unknown'

    # Count fit quality categories
    counts = {'good': 0, 'acceptable': 0, 'poor': 0, 'marginal': 0}
    valid_indices = 0

    for interpretation in interpretations.values():
        if interpretation in counts:
            counts[interpretation] += 1
            valid_indices += 1

    if valid_indices == 0:
        return 'unknown'

    # Decision rules
    if counts['poor'] >= 2:
        return 'poor'
    elif counts['good'] >= valid_indices / 2:
        return 'good'
    elif counts['acceptable'] + counts['good'] >= valid_indices / 2:
        return 'acceptable'
    else:
        return 'marginal'


def compare_models(
    models: List[Dict[str, Any]],
    model_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple fitted models.

    Parameters
    ----------
    models : List[Dict]
        List of model summaries from fit_and_summarize().
    model_names : List[str], optional
        Names for each model.

    Returns
    -------
    pd.DataFrame
        Comparison table with fit indices for each model.
    """
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(models))]

    comparison_data = []

    for name, model_summary in zip(model_names, models):
        fit_stats = model_summary.get('fit_stats', pd.DataFrame())

        row = {'Model': name}

        # Extract key fit indices
        if not fit_stats.empty:
            for key in ['chi2', 'chi2 p-value', 'CFI', 'TLI', 'RMSEA', 'AIC', 'BIC']:
                value = extract_fit_stat(fit_stats, key)
                if not pd.isna(value):
                    row[key] = value

        row['N'] = model_summary.get('sample_size', np.nan)
        comparison_data.append(row)

    return pd.DataFrame(comparison_data)


def get_modification_indices(model: Any) -> pd.DataFrame:
    """
    Get modification indices for potential model improvements.

    Note: This is a placeholder. semopy doesn't directly provide
    modification indices like lavaan does.

    Parameters
    ----------
    model : semopy.Model
        Fitted SEM model.

    Returns
    -------
    pd.DataFrame
        Modification indices (if available).
    """
    # semopy doesn't have built-in modification indices
    # Return empty DataFrame with expected structure
    return pd.DataFrame(columns=['lval', 'op', 'rval', 'mi', 'epc'])


def summarize_fit(evaluation: Dict[str, Any]) -> str:
    """
    Create human-readable fit summary.

    Parameters
    ----------
    evaluation : Dict
        Evaluation results from evaluate_model_fit().

    Returns
    -------
    str
        Formatted summary string.
    """
    lines = ["Model Fit Summary", "=" * 40]

    # Add indices
    for index, value in evaluation.get('indices', {}).items():
        interp = evaluation.get('interpretations', {}).get(index, '')
        lines.append(f"{index}: {value:.4f} ({interp})")

    # Add overall assessment
    overall = evaluation.get('overall_fit', 'unknown')
    lines.append("")
    lines.append(f"Overall Fit: {overall.upper()}")

    return "\n".join(lines)
