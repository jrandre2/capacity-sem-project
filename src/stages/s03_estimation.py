"""
Stage 03: SEM Estimation

Fit structural equation models to analyze government capacity effects.

Commands:
    python src/pipeline.py run_estimation [--model MODEL] [--subset SUBSET]

Outputs:
    data_work/diagnostics/      - Model results and diagnostics
"""

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from config import DATA_WORK_DIR, FIGURES_DIR, FIT_THRESHOLDS
from stages._io_utils import safe_read_parquet

# Import from existing modules
from capacity_sem.models.sem_specifications import (
    MODEL_REGISTRY,
    MODEL_DESCRIPTIONS,
    get_model_spec,
    get_model_description,
    list_available_models,
    get_indicator_variables,
)

from capacity_sem.models.sem_fitting import (
    SEMOPY_AVAILABLE,
    check_semopy,
    fit_sem_model,
    get_parameter_estimates,
    get_fit_statistics,
    fit_and_summarize,
    save_model_plot,
    extract_structural_coefficients,
    extract_measurement_coefficients,
    compute_cluster_robust_se,
    bootstrap_standard_errors,
)

from capacity_sem.models.sem_diagnostics import (
    compute_srmr,
    compute_rmsea_ci,
    compute_composite_reliability,
    evaluate_model_fit,
    compare_models,
    summarize_fit,
    extract_fit_stat,
)


def load_panel_features() -> pd.DataFrame:
    """Load panel with features."""
    path = DATA_WORK_DIR / "panel_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Panel features not found at {path}. Run compute_features first."
        )
    return safe_read_parquet(path)


def ensure_diagnostics_dir() -> Path:
    """Ensure diagnostics directory exists."""
    diag_dir = DATA_WORK_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    return diag_dir


def run_estimation(
    model_type: str = 'full',
    subset: str = 'all',
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run SEM estimation with specified model and subset.

    Parameters
    ----------
    model_type : str
        Model specification from MODEL_REGISTRY.
    subset : str
        Government type: 'all', 'state', or 'local'.
    save_results : bool
        Whether to save results to disk.

    Returns
    -------
    Dict[str, Any]
        Estimation results including model, estimates, and fit statistics.
    """
    check_semopy()

    # Load data
    data = load_panel_features()

    # Fit model
    results = fit_and_summarize(data, model_type, subset)

    # Add evaluation
    results['evaluation'] = evaluate_model_fit(results['fit_stats'])

    # Save if requested
    if save_results:
        diag_dir = ensure_diagnostics_dir()

        # Save estimates
        estimates_path = diag_dir / f"estimates_{model_type}_{subset}.csv"
        results['estimates'].to_csv(estimates_path, index=False)

        # Save fit statistics
        fit_path = diag_dir / f"fit_stats_{model_type}_{subset}.csv"
        if isinstance(results['fit_stats'], pd.DataFrame):
            results['fit_stats'].to_csv(fit_path)

    return results


def run_model_comparison(
    model_types: Optional[list] = None,
    subset: str = 'all'
) -> pd.DataFrame:
    """
    Compare multiple model specifications.

    Parameters
    ----------
    model_types : list, optional
        List of model types to compare. Defaults to core models.
    subset : str
        Government type filter.

    Returns
    -------
    pd.DataFrame
        Comparison table with fit indices.
    """
    if model_types is None:
        model_types = ['full', 'reduced', 'exp_optimal_v1', 'improved_3x3']

    results = []
    names = []

    for model_type in model_types:
        try:
            result = run_estimation(model_type, subset, save_results=False)
            results.append(result)
            names.append(model_type)
        except Exception as e:
            print(f"Failed to fit {model_type}: {e}")
            continue

    comparison = compare_models(results, names)

    return comparison


def run_subset_comparison(
    model_type: str = 'full'
) -> pd.DataFrame:
    """
    Compare model across government subsets.

    Parameters
    ----------
    model_type : str
        Model specification to use.

    Returns
    -------
    pd.DataFrame
        Comparison across all, state, and local subsets.
    """
    results = []
    names = []

    for subset in ['all', 'state', 'local']:
        try:
            result = run_estimation(model_type, subset, save_results=True)
            results.append(result)
            names.append(f"{model_type}_{subset}")
        except Exception as e:
            print(f"Failed to fit {model_type} for {subset}: {e}")
            continue

    comparison = compare_models(results, names)

    return comparison


def aggregate_to_grantee_level(
    data: pd.DataFrame,
    grantee_col: str = 'Grantee'
) -> pd.DataFrame:
    """
    Aggregate grantee-disaster data to grantee level.

    This replicates the manuscript's approach of averaging across disasters
    for each grantee, treating capacity as a stable grantee trait rather
    than disaster-specific performance.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data at grantee-disaster level.
    grantee_col : str
        Column containing grantee identifier.

    Returns
    -------
    pd.DataFrame
        Aggregated data at grantee level, with means across disasters.

    Notes
    -----
    This approach:
    - Assumes capacity is stable across disasters
    - Reduces pseudo-replication issues
    - Produces smaller N but cleaner signal
    """
    # Columns to aggregate (mean of numeric columns)
    agg_cols = [
        'Ratio_disbursed_to_obligated',
        'Ratio_expended_to_disbursed',
        'Timeliness',
        'Duration_of_completion',
        'Duration_log',
        'Ratio_obligated_funds_fully_expended',
        'Quarter_by_quarter_variance_expended',
        'Spending_CV',
        'Startup_Lag',
        'Time_to_50pct',
        'Progress_Rate',
        'Experience_Index',
        'Completion_Pct',
    ]

    # Filter to columns that exist
    available_cols = [c for c in agg_cols if c in data.columns]

    # Aggregate by grantee
    grantee_agg = data.groupby(grantee_col)[available_cols].agg('mean').reset_index()

    return grantee_agg


def run_grantee_level_analysis(
    model_type: str = 'exp_optimal_v1',
    subset: str = 'all',
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run SEM at grantee level (manuscript approach).

    This replicates the original Jupyter notebook analysis that produced
    β=71.024 by aggregating across disasters for each grantee.

    Parameters
    ----------
    model_type : str
        Model specification from MODEL_REGISTRY.
    subset : str
        Government type: 'all', 'state', or 'local'.
    save_results : bool
        Whether to save results to disk.

    Returns
    -------
    Dict[str, Any]
        Estimation results including model, estimates, and fit statistics.
    """
    check_semopy()

    # Load data
    data = load_panel_features()

    # Aggregate to grantee level
    grantee_data = aggregate_to_grantee_level(data)

    print(f"  Grantee-level N: {len(grantee_data)}")

    # Apply subset filter if needed
    from config import STATE_GOVERNMENTS, LOCAL_GOVERNMENTS

    if subset == 'state':
        grantee_data = grantee_data[grantee_data['Grantee'].isin(STATE_GOVERNMENTS)]
        print(f"  After state filter: {len(grantee_data)}")
    elif subset == 'local':
        grantee_data = grantee_data[grantee_data['Grantee'].isin(LOCAL_GOVERNMENTS)]
        print(f"  After local filter: {len(grantee_data)}")

    # Get model spec
    model_spec = get_model_spec(model_type)

    # Fit model
    model, result = fit_sem_model(model_spec, grantee_data)
    estimates = model.inspect()

    from semopy import calc_stats
    fit_stats = calc_stats(model)

    # Get sample size from fitted model (accounts for model-specific listwise deletion)
    # First try model.n_samples, then mx_data.shape[0], then fallback
    if hasattr(model, 'n_samples') and model.n_samples is not None:
        n_obs = model.n_samples
    elif hasattr(model, 'mx_data') and model.mx_data is not None:
        n_obs = model.mx_data.shape[0]
    else:
        # Fallback: extract observed variables and count complete cases
        import re
        obs_vars = re.findall(r'=~\s*([^\n]+)', model_spec)
        all_indicators = []
        for line in obs_vars:
            all_indicators.extend([v.strip() for v in line.split('+')])
        if all_indicators:
            available = [v for v in all_indicators if v in grantee_data.columns]
            n_obs = len(grantee_data[available].dropna())
        else:
            n_obs = len(grantee_data.dropna())

    results = {
        'model': model,
        'estimates': estimates,
        'fit_stats': fit_stats,
        'model_type': model_type,
        'subset': subset,
        'sample_size': n_obs,
        'unit_of_analysis': 'grantee',
    }

    # Add evaluation
    results['evaluation'] = evaluate_model_fit(results['fit_stats'])

    # Save if requested
    if save_results:
        diag_dir = ensure_diagnostics_dir()

        # Save estimates with grantee-level suffix
        estimates_path = diag_dir / f"estimates_{model_type}_{subset}_grantee.csv"
        results['estimates'].to_csv(estimates_path, index=False)

        # Save fit statistics
        fit_path = diag_dir / f"fit_stats_{model_type}_{subset}_grantee.csv"
        if isinstance(results['fit_stats'], pd.DataFrame):
            results['fit_stats'].to_csv(fit_path)

    return results


def run_dual_analysis(
    model_type: str = 'exp_optimal_v1',
    subset: str = 'all'
) -> pd.DataFrame:
    """
    Run dual analysis: both grantee-level and grantee-disaster level.

    This provides a comprehensive comparison showing:
    1. Grantee-level (manuscript approach): Treats capacity as stable trait
    2. Grantee-disaster level: Allows disaster-specific variation

    Parameters
    ----------
    model_type : str
        Model specification from MODEL_REGISTRY.
    subset : str
        Government type: 'all', 'state', or 'local'.

    Returns
    -------
    pd.DataFrame
        Comparison table with both analysis levels.

    Notes
    -----
    If results converge (both show similar effects), the finding is robust.
    If they diverge, discuss implications for capacity conceptualization.
    """
    print(f"\n  Running dual analysis with model: {model_type}, subset: {subset}")

    results = []

    # 1. Grantee-level analysis (manuscript approach)
    print("\n  --- Grantee-Level Analysis ---")
    try:
        grantee_results = run_grantee_level_analysis(
            model_type, subset, save_results=True
        )
        grantee_results['analysis_level'] = 'grantee'
        results.append(grantee_results)
        print(f"    N = {grantee_results['sample_size']}")
    except Exception as e:
        print(f"    Grantee-level analysis failed: {e}")

    # 2. Grantee-disaster level analysis (current pipeline)
    print("\n  --- Grantee-Disaster Analysis ---")
    try:
        disaster_results = run_estimation(model_type, subset, save_results=True)
        disaster_results['analysis_level'] = 'grantee_disaster'

        # Compute cluster-robust SEs (clustered by Grantee)
        panel_data = load_panel_features()
        robust_estimates = compute_cluster_robust_se(
            disaster_results['model'], panel_data, 'Grantee'
        )
        disaster_results['robust_estimates'] = robust_estimates

        results.append(disaster_results)
        print(f"    N = {disaster_results['sample_size']}")
    except Exception as e:
        print(f"    Grantee-disaster analysis failed: {e}")

    # Create comparison table
    if not results:
        return pd.DataFrame()

    comparison_rows = []
    for r in results:
        level = r.get('analysis_level', 'unknown')
        n = r.get('sample_size', np.nan)

        # Extract structural path coefficient
        estimates = r.get('estimates', pd.DataFrame())
        robust_estimates = r.get('robust_estimates', pd.DataFrame())

        beta = np.nan
        se = np.nan
        pval = np.nan
        robust_se = np.nan
        robust_pval = np.nan

        if not estimates.empty:
            # Try to extract structural path (capacity -> outcome)
            # Handle different column naming conventions
            op_col = 'op' if 'op' in estimates.columns else 'Operator'
            lval_col = 'lval' if 'lval' in estimates.columns else 'LHS'

            if op_col in estimates.columns and lval_col in estimates.columns:
                structural = estimates[
                    (estimates[op_col] == '~') &
                    (estimates[lval_col].isin(['recovery_outcome', 'Duration_log']))
                ]

                if not structural.empty:
                    row = structural.iloc[0]
                    beta = float(row['Estimate']) if pd.notna(row.get('Estimate')) else np.nan
                    se = float(row['Std. Err']) if pd.notna(row.get('Std. Err')) else np.nan
                    pval = float(row['p-value']) if pd.notna(row.get('p-value')) else np.nan

        # Extract robust estimates if available (for grantee-disaster level)
        if not robust_estimates.empty and 'Robust SE' in robust_estimates.columns:
            op_col = 'op' if 'op' in robust_estimates.columns else 'Operator'
            lval_col = 'lval' if 'lval' in robust_estimates.columns else 'LHS'

            if op_col in robust_estimates.columns and lval_col in robust_estimates.columns:
                structural_robust = robust_estimates[
                    (robust_estimates[op_col] == '~') &
                    (robust_estimates[lval_col].isin(['recovery_outcome', 'Duration_log']))
                ]

                if not structural_robust.empty:
                    row = structural_robust.iloc[0]
                    robust_se = float(row['Robust SE']) if pd.notna(row.get('Robust SE')) else np.nan
                    robust_pval = float(row['Robust p-value']) if pd.notna(row.get('Robust p-value')) else np.nan

        # Get fit indices
        evaluation = r.get('evaluation', {})
        indices = evaluation.get('indices', {})

        comparison_rows.append({
            'Analysis_Level': level,
            'Model': model_type,
            'Subset': subset,
            'N': n,
            'Beta': beta,
            'SE': se,
            'p_value': pval,
            'Robust_SE': robust_se,
            'Robust_p': robust_pval,
            'Significant': pval < 0.05 if pd.notna(pval) else False,
            'CFI': indices.get('CFI', np.nan),
            'RMSEA': indices.get('RMSEA', np.nan),
            'Overall_Fit': evaluation.get('overall_fit', 'unknown'),
        })

    comparison_df = pd.DataFrame(comparison_rows)

    # Save comparison
    diag_dir = ensure_diagnostics_dir()
    comp_path = diag_dir / f"dual_analysis_{model_type}_{subset}.csv"
    comparison_df.to_csv(comp_path, index=False)
    print(f"\n  Dual analysis saved to: {comp_path}")

    return comparison_df


def print_results_summary(results: Dict[str, Any]) -> None:
    """Print formatted results summary."""
    print("\n" + "=" * 60)
    print(f"Model: {results.get('model_type', 'unknown')}")
    print(f"Subset: {results.get('subset', 'all')}")
    print(f"Sample size: {results.get('sample_size', 'N/A')}")
    print("=" * 60)

    # Fit statistics
    print("\nFit Statistics:")
    print("-" * 40)
    evaluation = results.get('evaluation', {})
    for index, value in evaluation.get('indices', {}).items():
        interp = evaluation.get('interpretations', {}).get(index, '')
        print(f"  {index}: {value:.4f} ({interp})")

    print(f"\nOverall Fit: {evaluation.get('overall_fit', 'unknown').upper()}")

    # Structural coefficients
    print("\nStructural Path Coefficients:")
    print("-" * 40)
    estimates = results.get('estimates', pd.DataFrame())
    structural = extract_structural_coefficients(estimates)

    if not structural.empty:
        for _, row in structural.iterrows():
            lhs = row.get('LHS', row.get('lval', ''))
            rhs = row.get('RHS', row.get('rval', ''))
            est = row.get('Estimate', np.nan)
            se = row.get('Std. Err', np.nan)
            pval = row.get('p-value', np.nan)

            est = pd.to_numeric(est, errors='coerce')
            se = pd.to_numeric(se, errors='coerce')
            pval = pd.to_numeric(pval, errors='coerce')

            sig = ''
            if not pd.isna(pval):
                if pval < 0.001:
                    sig = '***'
                elif pval < 0.01:
                    sig = '**'
                elif pval < 0.05:
                    sig = '*'

            print(f"  {rhs} → {lhs}: {est:.4f} (SE={se:.4f}){sig}")

    print("\n* p<0.05, ** p<0.01, *** p<0.001")


def main(model: str = 'exp_optimal_v1', subset: str = 'all'):
    """
    Main entry point for estimation stage.

    Parameters
    ----------
    model : str
        Model type from MODEL_REGISTRY.
    subset : str
        Government subset: 'all', 'state', or 'local'.
    """
    print("=" * 60)
    print("Stage 03: SEM Estimation")
    print("=" * 60)

    # Check semopy
    if not SEMOPY_AVAILABLE:
        print("\nError: semopy is required for SEM estimation.")
        print("Install with: pip install semopy")
        return

    # Validate model type
    if model not in MODEL_REGISTRY:
        print(f"\nError: Unknown model type '{model}'")
        print(f"Available models: {list(MODEL_REGISTRY.keys())}")
        return

    print(f"\nModel: {model}")
    print(f"Description: {get_model_description(model)[:80]}...")
    print(f"Subset: {subset}")

    # Run estimation
    print("\nFitting model...")
    try:
        results = run_estimation(model, subset, save_results=True)
        print_results_summary(results)

        diag_dir = ensure_diagnostics_dir()
        print(f"\n  Results saved to: {diag_dir}")

    except Exception as e:
        print(f"\nError during estimation: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✓ Estimation complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SEM estimation")
    parser.add_argument("--model", "-m", default="exp_optimal_v1",
                        help="Model specification")
    parser.add_argument("--subset", "-s", default="all",
                        choices=["all", "state", "local"],
                        help="Government subset")

    args = parser.parse_args()
    main(model=args.model, subset=args.subset)
