"""
Stage 04: Robustness Checks

Run alternative specifications and sensitivity analyses.

Commands:
    python src/pipeline.py run_robustness [--models MODELS]

Outputs:
    data_work/diagnostics/robustness_*.csv  - Robustness check results
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

from config import DATA_WORK_DIR, STATE_GOVERNMENTS, LOCAL_GOVERNMENTS
from stages._io_utils import safe_read_parquet

# Import from existing modules
from capacity_sem.models.sem_specifications import MODEL_REGISTRY, get_model_spec
from capacity_sem.models.sem_fitting import (
    SEMOPY_AVAILABLE,
    check_semopy,
    fit_and_summarize,
    bootstrap_standard_errors,
    compute_cluster_robust_se,
)
from capacity_sem.models.sem_diagnostics import (
    evaluate_model_fit,
    compare_models,
    extract_fit_stat,
    get_standardized_estimates,
    get_reliability_summary,
)
from capacity_sem.models.sem_multigroup import (
    fit_multigroup,
    compare_structural_paths,
    compare_fit_statistics,
    assess_measurement_invariance,
    add_government_type_column,
)
from capacity_sem.models.sem_mediation import (
    compute_mediation_effects,
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


def run_alternative_specifications(
    data: pd.DataFrame,
    base_models: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Run alternative model specifications for robustness.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with features.
    base_models : List[str], optional
        List of model specifications to run.

    Returns
    -------
    pd.DataFrame
        Comparison table across specifications.
    """
    if base_models is None:
        base_models = [
            'full',
            'reduced',
            'exp_optimal_v1',
            'exp_optimal_v2',
            'improved_3x3',
            'improved_3x3_progress',
        ]

    results = []
    names = []

    for model_type in base_models:
        if model_type not in MODEL_REGISTRY:
            print(f"Skipping unknown model: {model_type}")
            continue

        try:
            result = fit_and_summarize(data, model_type, 'all')
            result['evaluation'] = evaluate_model_fit(result['fit_stats'])
            results.append(result)
            names.append(model_type)
            print(f"  Fitted {model_type}: n={result['sample_size']}")
        except Exception as e:
            print(f"  Failed {model_type}: {e}")
            continue

    if not results:
        return pd.DataFrame()

    return compare_models(results, names)


def run_subset_robustness(
    data: pd.DataFrame,
    model_type: str = 'exp_optimal_v1'
) -> pd.DataFrame:
    """
    Run model across government subsets.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with features.
    model_type : str
        Model specification to use.

    Returns
    -------
    pd.DataFrame
        Comparison across subsets.
    """
    results = []
    names = []

    for subset in ['all', 'state', 'local']:
        try:
            result = fit_and_summarize(data, model_type, subset)
            result['evaluation'] = evaluate_model_fit(result['fit_stats'])
            results.append(result)
            names.append(f"{subset}")
            print(f"  Fitted {subset}: n={result['sample_size']}")
        except Exception as e:
            print(f"  Failed {subset}: {e}")
            continue

    if not results:
        return pd.DataFrame()

    return compare_models(results, names)


def run_sample_sensitivity(
    data: pd.DataFrame,
    model_type: str = 'exp_optimal_v1',
    min_quarters_range: List[int] = [3, 4, 5, 6]
) -> pd.DataFrame:
    """
    Test sensitivity to minimum quarters requirement.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    model_type : str
        Model specification.
    min_quarters_range : List[int]
        Range of minimum quarters to test.

    Returns
    -------
    pd.DataFrame
        Results across different sample restrictions.
    """
    results = []

    for min_q in min_quarters_range:
        if 'N_Quarters' in data.columns:
            subset_data = data[data['N_Quarters'] >= min_q].copy()
        else:
            subset_data = data.copy()

        try:
            result = fit_and_summarize(subset_data, model_type, 'all')
            result['evaluation'] = evaluate_model_fit(result['fit_stats'])

            row = {
                'Min_Quarters': min_q,
                'N': result['sample_size'],
                'CFI': extract_fit_stat(result['fit_stats'], 'CFI'),
                'RMSEA': extract_fit_stat(result['fit_stats'], 'RMSEA'),
                'Overall_Fit': result['evaluation'].get('overall_fit', 'unknown')
            }

            # Extract structural coefficient
            from capacity_sem.models.sem_fitting import extract_structural_coefficients
            structural = extract_structural_coefficients(result['estimates'])
            if not structural.empty:
                row['Capacity_Effect'] = structural.iloc[0].get('Estimate', np.nan)
                row['Capacity_SE'] = structural.iloc[0].get('Std. Err', np.nan)

            results.append(row)
            print(f"  min_quarters={min_q}: n={result['sample_size']}")

        except Exception as e:
            print(f"  min_quarters={min_q}: Failed - {e}")
            continue

    return pd.DataFrame(results)


def run_covariate_robustness(
    data: pd.DataFrame,
    base_model: str = 'exp_optimal_v1'
) -> pd.DataFrame:
    """
    Test robustness to inclusion of different covariates.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with covariates.
    base_model : str
        Base model specification.

    Returns
    -------
    pd.DataFrame
        Results with different covariate specifications.
    """
    covariate_models = [
        ('No covariates', 'exp_optimal_v1'),
        ('With population', 'exp_optimal_pop'),
        ('All covariates', 'exp_optimal_full'),
    ]

    results = []
    names = []

    for name, model_type in covariate_models:
        if model_type not in MODEL_REGISTRY:
            continue

        try:
            result = fit_and_summarize(data, model_type, 'all')
            result['evaluation'] = evaluate_model_fit(result['fit_stats'])
            results.append(result)
            names.append(name)
            print(f"  {name}: n={result['sample_size']}")
        except Exception as e:
            print(f"  {name}: Failed - {e}")
            continue

    if not results:
        return pd.DataFrame()

    return compare_models(results, names)


def run_bootstrap_inference(
    data: pd.DataFrame,
    model_type: str = 'exp_optimal_v1',
    n_bootstrap: int = 500
) -> pd.DataFrame:
    """
    Run bootstrap inference for robust standard errors.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    model_type : str
        Model specification.
    n_bootstrap : int
        Number of bootstrap samples.

    Returns
    -------
    pd.DataFrame
        Parameter estimates with bootstrap confidence intervals.
    """
    model_spec = get_model_spec(model_type)

    print(f"  Running {n_bootstrap} bootstrap iterations...")
    boot_results = bootstrap_standard_errors(model_spec, data, n_bootstrap)

    return boot_results


def run_multigroup_comparison(
    data: pd.DataFrame,
    model_type: str = 'multigroup_2x2',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run multi-group SEM comparison (state vs. local governments).

    This analysis tests:
    1. Whether the same model fits both groups (configural invariance)
    2. Whether factor loadings are equivalent (metric invariance)
    3. Whether structural paths differ between groups

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with grantee column.
    model_type : str, default 'multigroup_2x2'
        Model specification to use for comparison.
    verbose : bool, default True
        Whether to print progress.

    Returns
    -------
    Dict[str, Any]
        Multi-group analysis results including:
        - 'fit_comparison': Fit statistics by group
        - 'structural_comparison': Path coefficients by group with tests
        - 'invariance': Measurement invariance assessment
        - 'loading_comparison': Factor loadings by group
    """
    if model_type not in MODEL_REGISTRY:
        print(f"  Unknown model: {model_type}")
        return {}

    # Add government type column
    data = add_government_type_column(data, grantee_col='Grantee')

    # Filter to state and local (exclude unknown)
    data_mg = data[data['Government_Type'].isin(['state', 'local'])].copy()

    if verbose:
        state_n = (data_mg['Government_Type'] == 'state').sum()
        local_n = (data_mg['Government_Type'] == 'local').sum()
        print(f"  State governments: {state_n}")
        print(f"  Local governments: {local_n}")

    # Get model specification
    model_spec = get_model_spec(model_type)

    # Run multi-group analysis
    results = {}

    try:
        # Fit models to each group
        multigroup = fit_multigroup(
            model_spec, data_mg,
            group_col='Government_Type',
            verbose=verbose
        )

        # Fit comparison table
        fit_comp = compare_fit_statistics(multigroup)
        results['fit_comparison'] = fit_comp

        # Structural path comparison with significance tests
        struct_comp = compare_structural_paths(multigroup, verbose=verbose)
        results['structural_comparison'] = struct_comp

        # Measurement invariance assessment
        invariance = assess_measurement_invariance(multigroup, verbose=verbose)
        results['invariance'] = invariance

        # Loading comparison
        from capacity_sem.models.sem_multigroup import compare_group_parameters
        loading_comp = compare_group_parameters(multigroup, 'loadings')
        results['loading_comparison'] = loading_comp

    except Exception as e:
        if verbose:
            print(f"  Multi-group analysis failed: {e}")
        results['error'] = str(e)

    return results


def run_mediation_analysis(
    data: pd.DataFrame,
    model_type: str = 'mediation_spending_cv',
    bootstrap: bool = False,
    n_boot: int = 200,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run mediation analysis for indirect effects.

    Tests whether the capacity → outcome relationship is mediated by
    spending consistency (Spending_CV) or startup speed (Startup_Lag).

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with features.
    model_type : str, default 'mediation_spending_cv'
        Mediation model specification.
    bootstrap : bool, default False
        Whether to compute bootstrap CI for indirect effect.
    n_boot : int, default 200
        Number of bootstrap samples.
    verbose : bool, default True
        Whether to print results.

    Returns
    -------
    Dict[str, Any]
        Mediation analysis results.
    """
    if model_type not in MODEL_REGISTRY:
        print(f"  Unknown model: {model_type}")
        return {}

    model_spec = get_model_spec(model_type)

    # Determine mediator and outcome based on model
    if 'spending_cv' in model_type.lower():
        iv = 'gov_cap'
        mediator = 'Spending_CV'
        dv = 'Duration_log'
    elif 'startup' in model_type.lower():
        iv = 'gov_cap'
        mediator = 'Startup_Lag'
        dv = 'Duration_log'
    elif 'progress' in model_type.lower():
        iv = 'gov_cap'
        mediator = 'Progress_Rate'
        dv = 'Duration_log'
    else:
        print(f"  Could not determine mediation structure for {model_type}")
        return {}

    try:
        results = compute_mediation_effects(
            model_spec, data, iv, mediator, dv,
            bootstrap=bootstrap, n_boot=n_boot, verbose=verbose
        )
        return results
    except Exception as e:
        if verbose:
            print(f"  Mediation analysis failed: {e}")
        return {'error': str(e)}


def run_formative_comparison(
    data: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare reflective vs. formative capacity models.

    Tests whether capacity is better conceptualized as:
    - Reflective: Capacity causes indicator scores
    - Formative: Indicator scores form capacity

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    verbose : bool, default True
        Whether to print results.

    Returns
    -------
    pd.DataFrame
        Comparison of reflective vs. formative models.
    """
    models_to_compare = [
        ('Reflective (exp_optimal_v1)', 'exp_optimal_v1'),
        ('Formative capacity', 'formative_capacity'),
    ]

    results = []
    names = []

    for name, model_type in models_to_compare:
        if model_type not in MODEL_REGISTRY:
            continue

        try:
            result = fit_and_summarize(data, model_type, 'all')
            result['evaluation'] = evaluate_model_fit(result['fit_stats'])
            results.append(result)
            names.append(name)
            if verbose:
                print(f"  {name}: n={result['sample_size']}")
        except Exception as e:
            if verbose:
                print(f"  {name}: Failed - {e}")
            continue

    if not results:
        return pd.DataFrame()

    return compare_models(results, names)


def main(models: Optional[List[str]] = None, run_extended: bool = False):
    """
    Main entry point for robustness checks.

    Parameters
    ----------
    models : List[str], optional
        Models to run robustness checks for.
    run_extended : bool, default False
        Whether to run extended analyses (multi-group, mediation).
    """
    print("=" * 60)
    print("Stage 04: Robustness Checks")
    print("=" * 60)

    if not SEMOPY_AVAILABLE:
        print("\nError: semopy is required.")
        print("Install with: pip install semopy")
        return

    # Load data
    print("\nLoading data...")
    try:
        data = load_panel_features()
        print(f"  Panel: {len(data):,} observations")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return

    diag_dir = ensure_diagnostics_dir()

    # 1. Alternative specifications
    print("\n1. Alternative Model Specifications")
    print("-" * 40)
    spec_comparison = run_alternative_specifications(data, models)
    if not spec_comparison.empty:
        spec_path = diag_dir / "robustness_specifications.csv"
        spec_comparison.to_csv(spec_path, index=False)
        print(f"\n  Saved to: {spec_path}")

    # 2. Subset comparison
    print("\n2. Government Subset Comparison")
    print("-" * 40)
    subset_comparison = run_subset_robustness(data)
    if not subset_comparison.empty:
        subset_path = diag_dir / "robustness_subsets.csv"
        subset_comparison.to_csv(subset_path, index=False)
        print(f"\n  Saved to: {subset_path}")

    # 3. Sample sensitivity
    print("\n3. Sample Sensitivity (Min Quarters)")
    print("-" * 40)
    sensitivity = run_sample_sensitivity(data)
    if not sensitivity.empty:
        sens_path = diag_dir / "robustness_sample_sensitivity.csv"
        sensitivity.to_csv(sens_path, index=False)
        print(f"\n  Saved to: {sens_path}")

    # 4. Covariate robustness
    print("\n4. Covariate Robustness")
    print("-" * 40)
    cov_comparison = run_covariate_robustness(data)
    if not cov_comparison.empty:
        cov_path = diag_dir / "robustness_covariates.csv"
        cov_comparison.to_csv(cov_path, index=False)
        print(f"\n  Saved to: {cov_path}")

    # 5. Multi-group comparison (state vs. local)
    print("\n5. Multi-Group Comparison (State vs. Local)")
    print("-" * 40)
    multigroup_results = run_multigroup_comparison(data)
    if multigroup_results and 'fit_comparison' in multigroup_results:
        # Save fit comparison
        mg_fit_path = diag_dir / "robustness_multigroup_fit.csv"
        multigroup_results['fit_comparison'].to_csv(mg_fit_path, index=False)
        print(f"\n  Fit comparison saved to: {mg_fit_path}")

        # Save structural comparison
        if 'structural_comparison' in multigroup_results:
            mg_struct_path = diag_dir / "robustness_multigroup_paths.csv"
            multigroup_results['structural_comparison'].to_csv(mg_struct_path, index=False)
            print(f"  Path comparison saved to: {mg_struct_path}")

        # Save invariance assessment summary
        if 'invariance' in multigroup_results:
            inv = multigroup_results['invariance']
            inv_summary = pd.DataFrame([{
                'Configural_Invariance': inv.get('configural', False),
                'Metric_Invariance_Approx': inv.get('metric_approximate', False),
                'Notes': '; '.join(inv.get('notes', []))
            }])
            inv_path = diag_dir / "robustness_multigroup_invariance.csv"
            inv_summary.to_csv(inv_path, index=False)
            print(f"  Invariance summary saved to: {inv_path}")

    # 6. Formative vs. Reflective comparison
    print("\n6. Reflective vs. Formative Capacity Model")
    print("-" * 40)
    formative_comparison = run_formative_comparison(data)
    if not formative_comparison.empty:
        form_path = diag_dir / "robustness_formative.csv"
        formative_comparison.to_csv(form_path, index=False)
        print(f"\n  Saved to: {form_path}")

    # Extended analyses (optional, slower)
    if run_extended:
        # 7. Mediation analysis
        print("\n7. Mediation Analysis (Spending CV)")
        print("-" * 40)
        mediation_results = run_mediation_analysis(data, 'mediation_spending_cv')
        if mediation_results and 'paths' in mediation_results:
            med_summary = pd.DataFrame([{
                'Path_a_IV_to_Med': mediation_results['paths'].get('a', np.nan),
                'Path_b_Med_to_DV': mediation_results['paths'].get('b', np.nan),
                'Path_c_Direct': mediation_results['paths'].get('c_prime', np.nan),
                'Indirect_Effect': mediation_results['paths'].get('indirect', np.nan),
                'Total_Effect': mediation_results['paths'].get('total', np.nan),
                'Sobel_z': mediation_results.get('sobel', {}).get('z', np.nan),
                'Sobel_p': mediation_results.get('sobel', {}).get('p', np.nan),
                'Interpretation': mediation_results.get('interpretation', '')
            }])
            med_path = diag_dir / "robustness_mediation.csv"
            med_summary.to_csv(med_path, index=False)
            print(f"\n  Saved to: {med_path}")

    print("\n✓ Robustness checks complete")
    print(f"  Results saved to: {diag_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run robustness checks")
    parser.add_argument("--models", "-m", nargs="+", default=None,
                        help="Model specifications to run")
    parser.add_argument("--extended", "-e", action="store_true",
                        help="Run extended analyses (mediation, bootstrap)")

    args = parser.parse_args()
    main(models=args.models, run_extended=args.extended)
