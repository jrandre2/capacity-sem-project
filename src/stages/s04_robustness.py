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


def main(models: Optional[List[str]] = None):
    """
    Main entry point for robustness checks.

    Parameters
    ----------
    models : List[str], optional
        Models to run robustness checks for.
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

    print("\n✓ Robustness checks complete")
    print(f"  Results saved to: {diag_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run robustness checks")
    parser.add_argument("--models", "-m", nargs="+", default=None,
                        help="Model specifications to run")

    args = parser.parse_args()
    main(models=args.models)
