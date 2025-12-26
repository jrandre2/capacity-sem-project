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
    return pd.read_parquet(path)


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
