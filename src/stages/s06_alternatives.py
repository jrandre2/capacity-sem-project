"""
Stage 06: Alternative Modeling Approaches

Runs alternative analyses to address right-censoring in duration data:
1. Survival analysis (Cox PH, AFT)
2. Lower threshold SEM (50%, 70%, 90%)
3. Duration-free SEM
4. Milestone-based SEM

Commands:
    python src/pipeline.py run_alternatives [--methods METHODS]
    python src/pipeline.py run_alternatives --survival-only
    python src/pipeline.py run_alternatives --sem-only

Outputs:
    data_work/diagnostics/alternatives_survival.csv
    data_work/diagnostics/alternatives_threshold_sensitivity.csv
    data_work/diagnostics/alternatives_duration_free.csv
    data_work/diagnostics/alternatives_milestone.csv
    data_work/diagnostics/alternatives_comparison.csv
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import warnings

from config import (
    DATA_WORK_DIR,
    STATE_GOVERNMENTS,
    LOCAL_GOVERNMENTS,
    SURVIVAL_CAPACITY_COLS,
    AFT_DISTRIBUTIONS,
    COX_PENALIZER,
)
from stages._io_utils import safe_read_parquet

from capacity_sem.models.sem_specifications import MODEL_REGISTRY, get_model_spec
from capacity_sem.models.sem_fitting import fit_sem_model, SEMOPY_AVAILABLE
from capacity_sem.models.sem_diagnostics import evaluate_model_fit, extract_fit_stat
from capacity_sem.models.sem_alternatives import (
    LIFELINES_AVAILABLE,
    check_lifelines,
    prepare_survival_data,
    fit_cox_model,
    fit_aft_model,
    compare_survival_models,
    extract_survival_coefficients,
    run_threshold_sensitivity_sem,
    compare_methods,
    summarize_alternatives_findings,
    get_available_duration_thresholds,
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


# =============================================================================
# SURVIVAL ANALYSIS
# =============================================================================

def run_survival_analysis(
    data: pd.DataFrame,
    capacity_cols: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive survival analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with features.
    capacity_cols : list, optional
        Capacity indicator columns. Defaults to SURVIVAL_CAPACITY_COLS.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    Dict[str, Any]
        Results including Cox and AFT models.
    """
    if not LIFELINES_AVAILABLE:
        if verbose:
            print("  Skipping survival analysis: lifelines not installed")
            print("  Install with: pip install lifelines>=0.27.0")
        return {'error': 'lifelines not available'}

    if capacity_cols is None:
        capacity_cols = SURVIVAL_CAPACITY_COLS

    if verbose:
        print("\n  --- Survival Analysis ---")
        print(f"  Capacity predictors: {capacity_cols}")

    # Prepare survival data
    surv_data = prepare_survival_data(data, capacity_cols=capacity_cols)

    if verbose:
        n_obs = len(surv_data)
        n_events = surv_data['E'].sum()
        n_censored = n_obs - n_events
        print(f"  Total observations: {n_obs}")
        print(f"  Completed (events): {n_events} ({100*n_events/n_obs:.1f}%)")
        print(f"  Censored: {n_censored} ({100*n_censored/n_obs:.1f}%)")

    results = {
        'n_obs': len(surv_data),
        'n_events': int(surv_data['E'].sum()),
        'n_censored': int(len(surv_data) - surv_data['E'].sum()),
    }

    # Fit Cox model
    if verbose:
        print("\n  Fitting Cox Proportional Hazards...")

    cox_result = fit_cox_model(surv_data, capacity_cols, penalizer=COX_PENALIZER)
    results['cox'] = cox_result

    if 'error' not in cox_result:
        if verbose:
            print(f"    Concordance: {cox_result['concordance']:.3f}")
            if 'summary' in cox_result:
                for var in capacity_cols:
                    if var in cox_result['summary'].index:
                        hr = cox_result['summary'].loc[var, 'exp(coef)']
                        p = cox_result['summary'].loc[var, 'p']
                        sig = '*' if p < 0.05 else ''
                        print(f"    {var}: HR={hr:.3f}, p={p:.3f}{sig}")

    # Fit AFT models
    results['aft'] = {}
    for dist in AFT_DISTRIBUTIONS:
        if verbose:
            print(f"\n  Fitting AFT ({dist})...")

        try:
            aft_result = fit_aft_model(surv_data, capacity_cols, distribution=dist)
            results['aft'][dist] = aft_result

            if 'error' not in aft_result:
                if verbose:
                    print(f"    AIC: {aft_result['aic']:.1f}")
                    print(f"    Concordance: {aft_result['concordance']:.3f}")
        except Exception as e:
            if verbose:
                print(f"    Failed: {e}")
            results['aft'][dist] = {'error': str(e)}

    # Model comparison
    if verbose:
        print("\n  --- Model Comparison ---")

    comparison = compare_survival_models(surv_data, capacity_cols, AFT_DISTRIBUTIONS)
    results['comparison'] = comparison

    if verbose and not comparison.empty:
        print(comparison.to_string(index=False))

    # Extract unified coefficients
    aft_results_list = [r for r in results['aft'].values() if 'error' not in r]
    results['coefficients'] = extract_survival_coefficients(
        cox_result if 'error' not in cox_result else None,
        aft_results_list,
        capacity_cols
    )

    return results


# =============================================================================
# LOWER THRESHOLD ANALYSIS
# =============================================================================

def run_lower_threshold_analysis(
    data: pd.DataFrame,
    thresholds: List[str] = ['50pct', '70pct', '90pct'],
    subset: str = 'all',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run SEM at multiple duration thresholds.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with features.
    thresholds : list
        Threshold suffixes to test.
    subset : str
        Government type: 'all', 'state', or 'local'.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        Comparison table across thresholds.
    """
    if not SEMOPY_AVAILABLE:
        if verbose:
            print("  Skipping threshold analysis: semopy not installed")
        return pd.DataFrame()

    if verbose:
        print("\n  --- Lower Threshold Analysis ---")
        print(f"  Thresholds: {thresholds}")
        print(f"  Subset: {subset}")

        # Show available observations
        avail = get_available_duration_thresholds(data)
        print("\n  Duration availability:")
        for col, n in avail.items():
            pct = 100 * n / len(data)
            print(f"    {col}: {n} ({pct:.1f}%)")

    results = run_threshold_sensitivity_sem(data, thresholds, subset, verbose)

    return results


# =============================================================================
# DURATION-FREE ANALYSIS
# =============================================================================

def run_duration_free_analysis(
    data: pd.DataFrame,
    subset: str = 'all',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run duration-free SEM models.

    Tests capacity effects using only ratio-based outcomes.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with features.
    subset : str
        Government type: 'all', 'state', or 'local'.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        Results for duration-free models.
    """
    if not SEMOPY_AVAILABLE:
        if verbose:
            print("  Skipping duration-free analysis: semopy not installed")
        return pd.DataFrame()

    if verbose:
        print("\n  --- Duration-Free Analysis ---")
        print(f"  Subset: {subset}")

    # Filter by subset
    if subset == 'state':
        data = data[data['Grantee'].isin(STATE_GOVERNMENTS)]
    elif subset == 'local':
        data = data[data['Grantee'].isin(LOCAL_GOVERNMENTS)]

    if verbose:
        print(f"  Sample size: {len(data)}")

    # Models to test
    duration_free_models = [
        'duration_free_cv',
        'duration_free_single',
        'duration_free_multiple',
        'duration_free_3x2',
    ]

    results = []

    for model_name in duration_free_models:
        if model_name not in MODEL_REGISTRY:
            if verbose:
                print(f"  Skipping {model_name}: not in registry")
            continue

        if verbose:
            print(f"\n  Testing: {model_name}")

        try:
            model_spec = get_model_spec(model_name)

            from semopy import Model, calc_stats

            # Fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = Model(model_spec)
                model.fit(data)

            estimates = model.inspect()
            fit_stats = calc_stats(model)

            # Get sample size from model
            if hasattr(model, 'mx_data') and model.mx_data is not None:
                n_obs = model.mx_data.shape[0]
            else:
                n_obs = len(data)

            # Extract structural path(s)
            structural = estimates[estimates['op'] == '~']

            beta = np.nan
            se = np.nan
            pval = np.nan

            # Look for capacity effect
            cap_paths = structural[structural['rval'] == 'gov_cap']
            if not cap_paths.empty:
                row = cap_paths.iloc[0]
                beta = float(row['Estimate'])
                se = float(row['Std. Err']) if pd.notna(row['Std. Err']) else np.nan
                pval = float(row['p-value']) if pd.notna(row['p-value']) else np.nan

            results.append({
                'Model': model_name,
                'N': n_obs,
                'CFI': extract_fit_stat(fit_stats, 'CFI'),
                'RMSEA': extract_fit_stat(fit_stats, 'RMSEA'),
                'Capacity_Beta': beta,
                'Capacity_SE': se,
                'Capacity_p': pval,
                'Significant': pval < 0.05 if pd.notna(pval) else False,
                'Subset': subset,
            })

            if verbose:
                sig = '*' if pval < 0.05 else ''
                print(f"    N={n_obs}, Beta={beta:.3f}, p={pval:.3f}{sig}")

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            continue

    return pd.DataFrame(results)


# =============================================================================
# MILESTONE-BASED ANALYSIS
# =============================================================================

def run_milestone_analysis(
    data: pd.DataFrame,
    subset: str = 'all',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run milestone-based SEM models.

    Uses Time_to_50pct, Progress_Rate, etc. as outcomes.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with features.
    subset : str
        Government type: 'all', 'state', or 'local'.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        Results for milestone-based models.
    """
    if not SEMOPY_AVAILABLE:
        if verbose:
            print("  Skipping milestone analysis: semopy not installed")
        return pd.DataFrame()

    if verbose:
        print("\n  --- Milestone-Based Analysis ---")
        print(f"  Subset: {subset}")

    # Filter by subset
    if subset == 'state':
        data = data[data['Grantee'].isin(STATE_GOVERNMENTS)]
    elif subset == 'local':
        data = data[data['Grantee'].isin(LOCAL_GOVERNMENTS)]

    if verbose:
        print(f"  Sample size: {len(data)}")

    # Models to test
    milestone_models = [
        'milestone_time_to_50',
        'milestone_progress_rate',
        'milestone_velocity',
        'milestone_direct',
        'exp_time_to_milestone',  # Also include existing milestone model
    ]

    results = []

    for model_name in milestone_models:
        if model_name not in MODEL_REGISTRY:
            if verbose:
                print(f"  Skipping {model_name}: not in registry")
            continue

        if verbose:
            print(f"\n  Testing: {model_name}")

        try:
            model_spec = get_model_spec(model_name)

            from semopy import Model, calc_stats

            # Fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = Model(model_spec)
                model.fit(data)

            estimates = model.inspect()
            fit_stats = calc_stats(model)

            # Get sample size from model
            if hasattr(model, 'mx_data') and model.mx_data is not None:
                n_obs = model.mx_data.shape[0]
            else:
                n_obs = len(data)

            # Extract structural path(s)
            structural = estimates[estimates['op'] == '~']

            beta = np.nan
            se = np.nan
            pval = np.nan

            # Look for capacity effect
            cap_paths = structural[structural['rval'] == 'gov_cap']
            if not cap_paths.empty:
                row = cap_paths.iloc[0]
                beta = float(row['Estimate'])
                se = float(row['Std. Err']) if pd.notna(row['Std. Err']) else np.nan
                pval = float(row['p-value']) if pd.notna(row['p-value']) else np.nan

            results.append({
                'Model': model_name,
                'N': n_obs,
                'CFI': extract_fit_stat(fit_stats, 'CFI'),
                'RMSEA': extract_fit_stat(fit_stats, 'RMSEA'),
                'Capacity_Beta': beta,
                'Capacity_SE': se,
                'Capacity_p': pval,
                'Significant': pval < 0.05 if pd.notna(pval) else False,
                'Subset': subset,
            })

            if verbose:
                sig = '*' if pval < 0.05 else ''
                print(f"    N={n_obs}, Beta={beta:.3f}, p={pval:.3f}{sig}")

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            continue

    return pd.DataFrame(results)


# =============================================================================
# COMBINED ANALYSIS
# =============================================================================

def run_all_alternatives(
    data: pd.DataFrame,
    subsets: List[str] = ['all', 'state', 'local'],
    methods: Optional[List[str]] = None,
    save_results: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete alternative analysis battery.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with features.
    subsets : list
        Government subsets to analyze.
    methods : list, optional
        Methods to run. If None, runs all.
        Options: 'survival', 'threshold', 'duration_free', 'milestone'
    save_results : bool
        Whether to save results to disk.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    Dict[str, Any]
        All results organized by method.
    """
    if methods is None:
        methods = ['survival', 'threshold', 'duration_free', 'milestone']

    results = {}
    diag_dir = ensure_diagnostics_dir()

    # 1. Survival Analysis (uses full sample, ignores subsets)
    if 'survival' in methods:
        if verbose:
            print("\n" + "=" * 60)
            print("SURVIVAL ANALYSIS")
            print("=" * 60)

        survival_results = run_survival_analysis(data, verbose=verbose)
        results['survival'] = survival_results

        if save_results and 'coefficients' in survival_results:
            path = diag_dir / "alternatives_survival.csv"
            survival_results['coefficients'].to_csv(path, index=False)
            if verbose:
                print(f"\n  Saved to: {path}")

    # 2. Lower Threshold Analysis
    if 'threshold' in methods:
        if verbose:
            print("\n" + "=" * 60)
            print("LOWER THRESHOLD ANALYSIS")
            print("=" * 60)

        threshold_results = []
        for subset in subsets:
            if verbose:
                print(f"\n  Subset: {subset}")
            subset_results = run_lower_threshold_analysis(
                data, subset=subset, verbose=verbose
            )
            threshold_results.append(subset_results)

        threshold_df = pd.concat(threshold_results, ignore_index=True)
        results['threshold'] = threshold_df

        if save_results and not threshold_df.empty:
            path = diag_dir / "alternatives_threshold_sensitivity.csv"
            threshold_df.to_csv(path, index=False)
            if verbose:
                print(f"\n  Saved to: {path}")

    # 3. Duration-Free Analysis
    if 'duration_free' in methods:
        if verbose:
            print("\n" + "=" * 60)
            print("DURATION-FREE ANALYSIS")
            print("=" * 60)

        duration_free_results = []
        for subset in subsets:
            if verbose:
                print(f"\n  Subset: {subset}")
            subset_results = run_duration_free_analysis(
                data, subset=subset, verbose=verbose
            )
            duration_free_results.append(subset_results)

        duration_free_df = pd.concat(duration_free_results, ignore_index=True)
        results['duration_free'] = duration_free_df

        if save_results and not duration_free_df.empty:
            path = diag_dir / "alternatives_duration_free.csv"
            duration_free_df.to_csv(path, index=False)
            if verbose:
                print(f"\n  Saved to: {path}")

    # 4. Milestone-Based Analysis
    if 'milestone' in methods:
        if verbose:
            print("\n" + "=" * 60)
            print("MILESTONE-BASED ANALYSIS")
            print("=" * 60)

        milestone_results = []
        for subset in subsets:
            if verbose:
                print(f"\n  Subset: {subset}")
            subset_results = run_milestone_analysis(
                data, subset=subset, verbose=verbose
            )
            milestone_results.append(subset_results)

        milestone_df = pd.concat(milestone_results, ignore_index=True)
        results['milestone'] = milestone_df

        if save_results and not milestone_df.empty:
            path = diag_dir / "alternatives_milestone.csv"
            milestone_df.to_csv(path, index=False)
            if verbose:
                print(f"\n  Saved to: {path}")

    # 5. Cross-Method Comparison
    if verbose:
        print("\n" + "=" * 60)
        print("CROSS-METHOD COMPARISON")
        print("=" * 60)

    comparison = compare_methods(
        survival_results=results.get('survival', {}).get('coefficients'),
        threshold_results=results.get('threshold'),
        duration_free_results=results.get('duration_free'),
        milestone_results=results.get('milestone'),
    )
    results['comparison'] = comparison

    if save_results and not comparison.empty:
        path = diag_dir / "alternatives_comparison.csv"
        comparison.to_csv(path, index=False)
        if verbose:
            print(f"\n  Saved to: {path}")

    # Summary
    if verbose:
        summary = summarize_alternatives_findings(comparison)
        print("\n" + summary)

    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(
    methods: Optional[List[str]] = None,
    subset: str = 'all',
    save_results: bool = True
):
    """
    Main entry point for alternative analyses.

    Parameters
    ----------
    methods : list, optional
        Which methods to run: 'survival', 'threshold', 'duration_free', 'milestone'.
        If None, runs all.
    subset : str
        Government subset (used for selecting single subset).
    save_results : bool
        Whether to save to disk.
    """
    print("=" * 60)
    print("Stage 06: Alternative Modeling Approaches")
    print("=" * 60)

    # Load data
    print("\nLoading panel features...")
    try:
        data = load_panel_features()
        print(f"  Loaded {len(data)} observations")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Determine subsets
    if subset == 'all':
        subsets = ['all', 'state', 'local']
    else:
        subsets = [subset]

    # Run analyses
    results = run_all_alternatives(
        data,
        subsets=subsets,
        methods=methods,
        save_results=save_results,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Alternative analyses complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run alternative modeling approaches")
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        default=None,
        choices=['survival', 'threshold', 'duration_free', 'milestone', 'all'],
        help="Methods to run (default: all)"
    )
    parser.add_argument(
        "--subset", "-s",
        default="all",
        choices=["all", "state", "local"],
        help="Government subset"
    )
    parser.add_argument(
        "--survival-only",
        action="store_true",
        help="Run only survival analysis"
    )
    parser.add_argument(
        "--sem-only",
        action="store_true",
        help="Run only SEM alternatives (no survival)"
    )

    args = parser.parse_args()

    # Handle convenience flags
    if args.survival_only:
        methods = ['survival']
    elif args.sem_only:
        methods = ['threshold', 'duration_free', 'milestone']
    elif args.methods and 'all' in args.methods:
        methods = None
    else:
        methods = args.methods

    main(methods=methods, subset=args.subset)
