"""
Multi-group SEM analysis for comparing state vs. local governments.

This module provides infrastructure for:
1. Fitting the same model to multiple groups
2. Testing measurement invariance (configural, metric, scalar)
3. Comparing structural paths across groups

Multi-group SEM allows testing whether:
- The same factor structure holds across groups (configural invariance)
- Factor loadings are equivalent across groups (metric invariance)
- Intercepts are equivalent across groups (scalar invariance)
- Structural relationships differ by group type
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

try:
    from semopy import Model
    from semopy import calc_stats
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False

from .sem_fitting import fit_sem_model
from .sem_diagnostics import (
    evaluate_model_fit,
    get_standardized_estimates,
    get_reliability_summary,
    extract_fit_stat
)


def fit_multigroup(
    model_spec: str,
    data: pd.DataFrame,
    group_col: str = 'Government_Type',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Fit the same SEM model to multiple groups separately.

    This is the first step in multi-group analysis. Fitting separately
    establishes configural invariance (same structure across groups).

    Parameters
    ----------
    model_spec : str
        SEM model specification in lavaan/semopy syntax.
    data : pd.DataFrame
        Data with group indicator column.
    group_col : str, default 'Government_Type'
        Column name identifying groups (e.g., 'state' vs 'local').
    verbose : bool, default True
        Whether to print progress information.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys for each group containing:
        - 'model': Fitted semopy Model
        - 'estimates': Parameter estimates DataFrame
        - 'fit_stats': Fit statistics DataFrame
        - 'n': Sample size for group
        - 'converged': Whether model converged
    """
    if not SEMOPY_AVAILABLE:
        raise ImportError("semopy is required for multi-group SEM")

    if group_col not in data.columns:
        raise ValueError(f"Group column '{group_col}' not found in data")

    groups = data[group_col].unique()
    results = {}

    if verbose:
        print(f"\nMulti-group SEM Analysis")
        print(f"=" * 50)
        print(f"Groups: {list(groups)}")
        print(f"Total N: {len(data)}")

    for group in groups:
        if verbose:
            print(f"\nFitting model for group: {group}")
            print("-" * 30)

        # Subset data
        group_data = data[data[group_col] == group].copy()
        n_group = len(group_data)

        if verbose:
            print(f"  Sample size: {n_group}")

        # Fit model
        try:
            model, fit_result = fit_sem_model(model_spec, group_data)

            # Get estimates and fit stats
            estimates = model.inspect() if model else pd.DataFrame()
            fit_stats = calc_stats(model) if model else pd.DataFrame()

            converged = fit_result is not None

            results[group] = {
                'model': model,
                'estimates': estimates,
                'fit_stats': fit_stats,
                'n': n_group,
                'converged': converged,
                'data': group_data
            }

            if verbose and not estimates.empty:
                # Print key fit statistics
                cfi = extract_fit_stat(fit_stats, 'CFI')
                rmsea = extract_fit_stat(fit_stats, 'RMSEA')
                print(f"  CFI: {cfi:.3f}" if not pd.isna(cfi) else "  CFI: NA")
                print(f"  RMSEA: {rmsea:.3f}" if not pd.isna(rmsea) else "  RMSEA: NA")

        except Exception as e:
            if verbose:
                print(f"  Error fitting model: {e}")
            results[group] = {
                'model': None,
                'estimates': pd.DataFrame(),
                'fit_stats': pd.DataFrame(),
                'n': n_group,
                'converged': False,
                'error': str(e)
            }

    return results


def compare_group_parameters(
    multigroup_results: Dict[str, Any],
    parameter_type: str = 'structural'
) -> pd.DataFrame:
    """
    Compare parameter estimates across groups.

    Parameters
    ----------
    multigroup_results : Dict[str, Any]
        Output from fit_multigroup().
    parameter_type : str, default 'structural'
        Type of parameters to compare:
        - 'structural': Structural paths (~ operator)
        - 'loadings': Factor loadings (=~ operator)
        - 'all': All parameters

    Returns
    -------
    pd.DataFrame
        Comparison table with parameters for each group.
    """
    comparison_rows = []

    # Collect all unique parameters
    all_params = set()
    for group, results in multigroup_results.items():
        estimates = results.get('estimates', pd.DataFrame())
        if estimates.empty:
            continue

        if parameter_type == 'structural':
            params = estimates[estimates['op'] == '~']
        elif parameter_type == 'loadings':
            params = estimates[estimates['op'] == '=~']
        else:
            params = estimates

        for _, row in params.iterrows():
            param_key = (row['lval'], row['op'], row['rval'])
            all_params.add(param_key)

    # Create comparison rows
    for lval, op, rval in sorted(all_params):
        row = {
            'Parameter': f"{lval} {op} {rval}",
            'lval': lval,
            'op': op,
            'rval': rval
        }

        for group, results in multigroup_results.items():
            estimates = results.get('estimates', pd.DataFrame())
            if estimates.empty:
                row[f'{group}_Est'] = np.nan
                row[f'{group}_SE'] = np.nan
                row[f'{group}_pval'] = np.nan
                continue

            # Find this parameter
            mask = (
                (estimates['lval'] == lval) &
                (estimates['op'] == op) &
                (estimates['rval'] == rval)
            )
            param_row = estimates[mask]

            if not param_row.empty:
                row[f'{group}_Est'] = param_row['Estimate'].iloc[0]
                row[f'{group}_SE'] = param_row.get('Std. Err', pd.Series([np.nan])).iloc[0]
                row[f'{group}_pval'] = param_row.get('p-value', pd.Series([np.nan])).iloc[0]
            else:
                row[f'{group}_Est'] = np.nan
                row[f'{group}_SE'] = np.nan
                row[f'{group}_pval'] = np.nan

        comparison_rows.append(row)

    return pd.DataFrame(comparison_rows)


def test_parameter_difference(
    est1: float,
    se1: float,
    est2: float,
    se2: float
) -> Tuple[float, float]:
    """
    Test whether two parameter estimates differ significantly.

    Uses pooled SE approximation: z = (est1 - est2) / sqrt(se1² + se2²)

    Parameters
    ----------
    est1 : float
        Estimate from group 1.
    se1 : float
        Standard error from group 1.
    est2 : float
        Estimate from group 2.
    se2 : float
        Standard error from group 2.

    Returns
    -------
    Tuple[float, float]
        (z-statistic, two-tailed p-value)
    """
    from scipy import stats

    if any(pd.isna([est1, se1, est2, se2])):
        return np.nan, np.nan

    if se1 <= 0 or se2 <= 0:
        return np.nan, np.nan

    diff = est1 - est2
    pooled_se = np.sqrt(se1**2 + se2**2)

    if pooled_se <= 0:
        return np.nan, np.nan

    z = diff / pooled_se
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p


def compare_structural_paths(
    multigroup_results: Dict[str, Any],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare structural path coefficients across groups with significance tests.

    Parameters
    ----------
    multigroup_results : Dict[str, Any]
        Output from fit_multigroup().
    verbose : bool, default True
        Whether to print summary.

    Returns
    -------
    pd.DataFrame
        Comparison with difference tests.
    """
    comparison = compare_group_parameters(multigroup_results, 'structural')

    if comparison.empty:
        return comparison

    groups = list(multigroup_results.keys())
    if len(groups) != 2:
        if verbose:
            print("Note: Difference tests require exactly 2 groups")
        return comparison

    g1, g2 = groups

    # Add difference tests
    z_stats = []
    p_vals = []

    for _, row in comparison.iterrows():
        est1 = row.get(f'{g1}_Est', np.nan)
        se1 = row.get(f'{g1}_SE', np.nan)
        est2 = row.get(f'{g2}_Est', np.nan)
        se2 = row.get(f'{g2}_SE', np.nan)

        z, p = test_parameter_difference(est1, se1, est2, se2)
        z_stats.append(z)
        p_vals.append(p)

    comparison['Diff_z'] = z_stats
    comparison['Diff_p'] = p_vals
    comparison['Sig_Diff'] = comparison['Diff_p'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )

    if verbose:
        print("\nStructural Path Comparison")
        print("=" * 60)
        for _, row in comparison.iterrows():
            param = row['Parameter']
            print(f"\n{param}:")
            print(f"  {g1}: {row.get(f'{g1}_Est', np.nan):.3f} (SE={row.get(f'{g1}_SE', np.nan):.3f})")
            print(f"  {g2}: {row.get(f'{g2}_Est', np.nan):.3f} (SE={row.get(f'{g2}_SE', np.nan):.3f})")
            if not pd.isna(row['Diff_z']):
                sig = row['Sig_Diff']
                print(f"  Difference: z={row['Diff_z']:.2f}, p={row['Diff_p']:.3f} {sig}")

    return comparison


def compare_fit_statistics(
    multigroup_results: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compare model fit statistics across groups.

    Parameters
    ----------
    multigroup_results : Dict[str, Any]
        Output from fit_multigroup().

    Returns
    -------
    pd.DataFrame
        Fit statistics comparison table.
    """
    rows = []

    for group, results in multigroup_results.items():
        fit_stats = results.get('fit_stats', pd.DataFrame())
        n = results.get('n', 0)
        converged = results.get('converged', False)

        row = {
            'Group': group,
            'N': n,
            'Converged': converged
        }

        if not fit_stats.empty:
            for key in ['chi2', 'chi2 p-value', 'CFI', 'TLI', 'RMSEA', 'AIC', 'BIC']:
                row[key] = extract_fit_stat(fit_stats, key)

        rows.append(row)

    return pd.DataFrame(rows)


def assess_measurement_invariance(
    multigroup_results: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Assess measurement invariance across groups.

    This is a simplified assessment comparing factor loadings across groups.
    Full measurement invariance testing would require constrained models.

    Levels of invariance:
    1. Configural: Same factor structure (both models converge)
    2. Metric: Similar factor loadings (loadings within SE bounds)
    3. Scalar: Similar intercepts (not directly testable without constraints)

    Parameters
    ----------
    multigroup_results : Dict[str, Any]
        Output from fit_multigroup().
    verbose : bool, default True
        Whether to print assessment.

    Returns
    -------
    Dict[str, Any]
        Invariance assessment results.
    """
    assessment = {
        'configural': False,
        'metric_approximate': False,
        'loading_comparison': None,
        'notes': []
    }

    groups = list(multigroup_results.keys())

    # Check configural invariance (all models converged)
    all_converged = all(
        results.get('converged', False)
        for results in multigroup_results.values()
    )
    assessment['configural'] = all_converged

    if not all_converged:
        assessment['notes'].append("Configural invariance NOT supported - not all models converged")
        return assessment

    assessment['notes'].append("Configural invariance supported - same structure fits both groups")

    # Compare loadings for approximate metric invariance
    loading_comparison = compare_group_parameters(multigroup_results, 'loadings')
    assessment['loading_comparison'] = loading_comparison

    if len(groups) == 2 and not loading_comparison.empty:
        g1, g2 = groups

        # Check if loadings are similar (within 2 SE)
        similar_loadings = 0
        total_loadings = 0

        for _, row in loading_comparison.iterrows():
            est1 = row.get(f'{g1}_Est', np.nan)
            se1 = row.get(f'{g1}_SE', np.nan)
            est2 = row.get(f'{g2}_Est', np.nan)
            se2 = row.get(f'{g2}_SE', np.nan)

            if pd.notna(est1) and pd.notna(est2):
                total_loadings += 1

                # Check if difference is within 2 pooled SE
                if pd.notna(se1) and pd.notna(se2) and se1 > 0 and se2 > 0:
                    pooled_se = np.sqrt(se1**2 + se2**2)
                    if abs(est1 - est2) < 2 * pooled_se:
                        similar_loadings += 1

        if total_loadings > 0:
            prop_similar = similar_loadings / total_loadings
            assessment['metric_approximate'] = prop_similar >= 0.8  # 80% similar

            if assessment['metric_approximate']:
                assessment['notes'].append(
                    f"Approximate metric invariance supported ({similar_loadings}/{total_loadings} loadings similar)"
                )
            else:
                assessment['notes'].append(
                    f"Metric invariance may NOT hold ({similar_loadings}/{total_loadings} loadings similar)"
                )

    if verbose:
        print("\nMeasurement Invariance Assessment")
        print("=" * 50)
        for note in assessment['notes']:
            print(f"  - {note}")
        print()
        if assessment['configural']:
            print("  Configural: SUPPORTED")
        else:
            print("  Configural: NOT SUPPORTED")
        if assessment['metric_approximate']:
            print("  Metric (approximate): SUPPORTED")
        else:
            print("  Metric (approximate): NOT SUPPORTED or UNTESTED")

    return assessment


def run_multigroup_analysis(
    model_spec: str,
    data: pd.DataFrame,
    group_col: str = 'Government_Type',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete multi-group SEM analysis.

    This is the main entry point for multi-group analysis, combining:
    1. Separate model fitting
    2. Fit comparison
    3. Parameter comparison
    4. Measurement invariance assessment

    Parameters
    ----------
    model_spec : str
        SEM model specification.
    data : pd.DataFrame
        Data with group indicator.
    group_col : str, default 'Government_Type'
        Column identifying groups.
    verbose : bool, default True
        Whether to print results.

    Returns
    -------
    Dict[str, Any]
        Complete analysis results with all comparisons.
    """
    results = {
        'models': None,
        'fit_comparison': None,
        'structural_comparison': None,
        'loading_comparison': None,
        'invariance': None
    }

    # Fit models to each group
    multigroup = fit_multigroup(model_spec, data, group_col, verbose)
    results['models'] = multigroup

    # Compare fit statistics
    fit_comp = compare_fit_statistics(multigroup)
    results['fit_comparison'] = fit_comp

    if verbose:
        print("\nModel Fit Comparison")
        print("=" * 50)
        print(fit_comp.to_string(index=False))

    # Compare structural paths
    struct_comp = compare_structural_paths(multigroup, verbose)
    results['structural_comparison'] = struct_comp

    # Compare loadings
    loading_comp = compare_group_parameters(multigroup, 'loadings')
    results['loading_comparison'] = loading_comp

    # Assess invariance
    invariance = assess_measurement_invariance(multigroup, verbose)
    results['invariance'] = invariance

    return results


def add_government_type_column(
    data: pd.DataFrame,
    grantee_col: str = 'Grantee',
    state_list: Optional[List[str]] = None,
    local_list: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Add Government_Type column based on grantee classification.

    Parameters
    ----------
    data : pd.DataFrame
        Data with grantee column.
    grantee_col : str, default 'Grantee'
        Column containing grantee names.
    state_list : List[str], optional
        List of state government grantees. If None, uses config.
    local_list : List[str], optional
        List of local government grantees. If None, uses config.

    Returns
    -------
    pd.DataFrame
        Data with Government_Type column added.
    """
    from config import STATE_GOVERNMENTS, LOCAL_GOVERNMENTS

    state_list = state_list or STATE_GOVERNMENTS
    local_list = local_list or LOCAL_GOVERNMENTS

    data = data.copy()

    def classify_grantee(grantee):
        if grantee in state_list:
            return 'state'
        elif grantee in local_list:
            return 'local'
        else:
            return 'unknown'

    data['Government_Type'] = data[grantee_col].apply(classify_grantee)

    return data
