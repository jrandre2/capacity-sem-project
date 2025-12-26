"""
Stage 03b: Time-Varying Survival Estimation

Implements time-varying survival analysis to address the critical methodological
flaw of reverse causality in the original static ratio approach.

Commands:
    python src/pipeline.py run_survival

Outputs:
    data_work/diagnostics/survival_time_varying_cox_results.csv
    data_work/diagnostics/survival_time_varying_aft_results.csv
    data_work/diagnostics/survival_bootstrap_se.csv
    data_work/diagnostics/survival_robustness_checks.csv
    figures/survival_*.png (diagnostic plots)
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import warnings

from config import DATA_WORK_DIR, TV_LAG_QUARTERS, BOOTSTRAP_ITERATIONS, BOOTSTRAP_CLUSTER_COL, SURVIVAL_COVARIATE_COLS
from stages._io_utils import safe_to_parquet, safe_read_parquet

from capacity_sem.models.time_varying_survival import (
    fit_time_varying_cox,
    compute_bootstrap_se
)
from capacity_sem.models.survival_diagnostics import (
    plot_residual_diagnostics,
    plot_predicted_survival_curves,
    test_proportional_hazards,
    compute_influence_diagnostics
)


def load_time_varying_panel() -> pd.DataFrame:
    """Load or generate time-varying survival panel."""
    path = DATA_WORK_DIR / "panel_time_varying.parquet"

    if not path.exists():
        print("\n  Time-varying panel not found, generating...")

        # Load required data
        qpr_path = DATA_WORK_DIR / "qpr_quarterly.parquet"
        panel_path = DATA_WORK_DIR / "panel_features.parquet"

        if not qpr_path.exists() or not panel_path.exists():
            raise FileNotFoundError(
                f"Required data not found. "
                f"Run 'python src/pipeline.py build_panel' and "
                f"'python src/pipeline.py compute_features' first."
            )

        qpr_quarterly = safe_read_parquet(qpr_path)
        panel_features = safe_read_parquet(panel_path)

        # Generate time-varying panel
        from capacity_sem.models.time_varying_survival import reshape_quarterly_to_time_varying, add_static_covariates

        tv_panel = reshape_quarterly_to_time_varying(
            qpr_quarterly=qpr_quarterly,
            panel_features=panel_features,
            lag_quarters=TV_LAG_QUARTERS
        )

        tv_panel = add_static_covariates(tv_panel, panel_features)

        # Save for future use
        safe_to_parquet(tv_panel, path)
        print(f"  Generated and saved time-varying panel → {path}")

    return safe_read_parquet(path)


def run_time_varying_cox(
    tv_data: pd.DataFrame,
    capacity_cols: List[str],
    covariate_cols: Optional[List[str]] = None,
    model_name: str = 'cox_baseline',
    bootstrap_se: bool = False,
    n_bootstrap: int = 100
) -> Dict[str, Any]:
    """
    Run time-varying Cox model with optional bootstrap SEs.

    Parameters
    ----------
    tv_data : pd.DataFrame
        Time-varying survival data
    capacity_cols : list of str
        Time-varying capacity predictors
    covariate_cols : list of str, optional
        Static covariates to include
    model_name : str
        Model identifier for results
    bootstrap_se : bool
        Whether to compute bootstrap clustered SEs
    n_bootstrap : int
        Number of bootstrap iterations

    Returns
    -------
    dict
        Model results including summary, HR, diagnostics
    """

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Fit Cox model
    results = fit_time_varying_cox(
        tv_data=tv_data,
        capacity_cols=capacity_cols,
        covariate_cols=covariate_cols,
        penalizer=0.1
    )

    # Add model name to results
    results['model_name'] = model_name
    results['capacity_cols'] = capacity_cols
    results['covariate_cols'] = covariate_cols if covariate_cols else []

    # Test proportional hazards assumption
    print("\nTesting proportional hazards assumption...")
    ph_test = test_proportional_hazards(results['model'], tv_data)
    results['ph_test'] = ph_test

    # Compute influence diagnostics
    print("\nComputing influence diagnostics...")
    influence = compute_influence_diagnostics(results['model'], tv_data)
    results['influence'] = influence

    # Bootstrap SEs if requested
    if bootstrap_se:
        print(f"\nComputing bootstrap SEs ({n_bootstrap} iterations)...")
        bootstrap_results = compute_bootstrap_se(
            tv_data=tv_data,
            capacity_cols=capacity_cols,
            covariate_cols=covariate_cols,
            cluster_col=BOOTSTRAP_CLUSTER_COL,
            n_bootstrap=n_bootstrap,
            penalizer=0.1
        )
        results['bootstrap_se'] = bootstrap_results

    return results


def run_robustness_checks(tv_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Run comprehensive robustness checks.

    Models:
    1. Cox - No covariates (capacity only)
    2. Cox - Full covariates (main specification)
    3. Cox - Stratified by government type
    4. Cox - Alternative lag structures (lag=0, 1, 2)

    Parameters
    ----------
    tv_data : pd.DataFrame
        Time-varying survival data

    Returns
    -------
    dict
        Dictionary of {model_name: results}
    """

    robustness_results = {}

    # Capacity predictors (time-varying with lag)
    lag_suffix = f'_lag{TV_LAG_QUARTERS}'
    capacity_cols = [
        f'Ratio_disbursed_to_obligated{lag_suffix}',
        f'Ratio_expended_to_disbursed{lag_suffix}'
    ]

    # 1. Baseline: Capacity only (no covariates)
    print("\n" + "="*60)
    print("ROBUSTNESS CHECK 1: Capacity Only (No Covariates)")
    print("="*60)
    robustness_results['capacity_only'] = run_time_varying_cox(
        tv_data=tv_data,
        capacity_cols=capacity_cols,
        covariate_cols=None,
        model_name='capacity_only'
    )

    # 2. Full covariates (main specification)
    print("\n" + "="*60)
    print("ROBUSTNESS CHECK 2: Full Covariates (Main Specification)")
    print("="*60)

    # Check which covariates are available
    available_covariates = [col for col in SURVIVAL_COVARIATE_COLS if col in tv_data.columns]
    if len(available_covariates) < len(SURVIVAL_COVARIATE_COLS):
        missing = set(SURVIVAL_COVARIATE_COLS) - set(available_covariates)
        warnings.warn(f"Missing covariates: {missing}")

    robustness_results['full_covariates'] = run_time_varying_cox(
        tv_data=tv_data,
        capacity_cols=capacity_cols,
        covariate_cols=available_covariates,
        model_name='full_covariates',
        bootstrap_se=True,  # Compute bootstrap SEs for main model
        n_bootstrap=BOOTSTRAP_ITERATIONS
    )

    # 3. Stratified by government type (if available)
    if 'Government_Type' in tv_data.columns:
        print("\n" + "="*60)
        print("ROBUSTNESS CHECK 3: Stratified by Government Type")
        print("="*60)

        # Remove Government_Type_State from covariates (can't include in stratified model)
        stratified_covariates = [c for c in available_covariates if c != 'Government_Type_State']

        try:
            from capacity_sem.models.time_varying_survival import fit_time_varying_cox as fit_cox

            # Fit stratified model
            stratified_results = fit_cox(
                tv_data=tv_data,
                capacity_cols=capacity_cols,
                covariate_cols=stratified_covariates,
                strata='Government_Type'
            )
            stratified_results['model_name'] = 'stratified_gov_type'
            robustness_results['stratified_gov_type'] = stratified_results

        except Exception as e:
            warnings.warn(f"Could not fit stratified model: {e}")

    # 4. Alternative lag structures (if data available)
    print("\n" + "="*60)
    print("ROBUSTNESS CHECK 4: Alternative Lag Structures")
    print("="*60)

    for lag in [0, 2]:
        lag_cols = [
            f'Ratio_disbursed_to_obligated_lag{lag}',
            f'Ratio_expended_to_disbursed_lag{lag}'
        ]

        # Check if these columns exist
        if all(col in tv_data.columns for col in lag_cols):
            try:
                robustness_results[f'lag{lag}'] = run_time_varying_cox(
                    tv_data=tv_data,
                    capacity_cols=lag_cols,
                    covariate_cols=available_covariates,
                    model_name=f'lag{lag}'
                )
            except Exception as e:
                warnings.warn(f"Could not fit lag={lag} model: {e}")
        else:
            print(f"  Lag={lag} columns not available, skipping")

    return robustness_results


def save_results(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> None:
    """
    Save model results to CSV files.

    Parameters
    ----------
    results : dict
        Dictionary of model results
    output_dir : Path
        Output directory
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all model summaries
    all_summaries = []
    all_hrs = []
    all_bootstrap = []

    for model_name, model_results in results.items():
        if 'summary' in model_results:
            summary = model_results['summary'].copy()
            summary['model'] = model_name
            all_summaries.append(summary)

        if 'hazard_ratios' in model_results:
            hrs = model_results['hazard_ratios'].copy()
            hrs['model'] = model_name
            all_hrs.append(hrs)

        if 'bootstrap_se' in model_results:
            bootstrap = model_results['bootstrap_se'].copy()
            bootstrap['model'] = model_name
            all_bootstrap.append(bootstrap)

    # Save combined results
    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        path = output_dir / 'survival_time_varying_cox_results.csv'
        combined_summary.to_csv(path, index=False)
        print(f"\n  Saved Cox results → {path}")

    if all_hrs:
        combined_hrs = pd.concat(all_hrs, ignore_index=True)
        path = output_dir / 'survival_hazard_ratios.csv'
        combined_hrs.to_csv(path, index=False)
        print(f"  Saved hazard ratios → {path}")

    if all_bootstrap:
        combined_bootstrap = pd.concat(all_bootstrap, ignore_index=True)
        path = output_dir / 'survival_bootstrap_se.csv'
        combined_bootstrap.to_csv(path, index=False)
        print(f"  Saved bootstrap SEs → {path}")

    # Save robustness summary
    robustness_summary = []
    for model_name, model_results in results.items():
        if 'hazard_ratios' in model_results:
            hrs = model_results['hazard_ratios']
            for _, row in hrs.iterrows():
                robustness_summary.append({
                    'model': model_name,
                    'variable': row['Variable'],
                    'HR': row['HR'],
                    'HR_Lower': row['HR_Lower'],
                    'HR_Upper': row['HR_Upper'],
                    'p_value': row['p_value'],
                    'concordance': model_results.get('concordance', np.nan),
                    'n_obs': model_results.get('n_obs', np.nan),
                    'n_events': model_results.get('n_events', np.nan)
                })

    if robustness_summary:
        robustness_df = pd.DataFrame(robustness_summary)
        path = output_dir / 'survival_robustness_checks.csv'
        robustness_df.to_csv(path, index=False)
        print(f"  Saved robustness summary → {path}")


def generate_diagnostic_plots(
    model: 'CoxPHFitter',
    tv_data: pd.DataFrame,
    capacity_cols: List[str],
    output_dir: Path
) -> None:
    """
    Generate comprehensive diagnostic plots.

    Parameters
    ----------
    model : CoxPHFitter
        Fitted Cox model
    tv_data : pd.DataFrame
        Time-varying survival data
    capacity_cols : list of str
        Capacity predictor columns
    output_dir : Path
        Output directory for figures
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating diagnostic plots...")

    # Residual diagnostics
    try:
        plot_residual_diagnostics(
            model=model,
            tv_data=tv_data,
            capacity_cols=capacity_cols,
            output_dir=str(output_dir)
        )
    except Exception as e:
        warnings.warn(f"Could not generate residual diagnostics: {e}")

    # Predicted survival curves by capacity quartiles
    for col in capacity_cols:
        if col in tv_data.columns:
            try:
                plot_predicted_survival_curves(
                    model=model,
                    tv_data=tv_data,
                    predictor_col=col,
                    stratify_by='quartile',
                    save_path=str(output_dir / f'survival_curves_{col}.png')
                )
            except Exception as e:
                warnings.warn(f"Could not plot survival curves for {col}: {e}")

    print(f"  Diagnostic plots saved to {output_dir}/")


def generate_time_varying_panel_for_threshold(
    duration_col: str,
    lag_quarters: int = TV_LAG_QUARTERS
) -> pd.DataFrame:
    """
    Generate time-varying panel for a specific completion threshold.

    Parameters
    ----------
    duration_col : str
        Duration column name (e.g., 'Duration_80pct', 'Duration_of_completion')
    lag_quarters : int
        Number of quarters to lag capacity ratios

    Returns
    -------
    pd.DataFrame
        Time-varying panel with start/stop intervals
    """
    # Load required data
    qpr_path = DATA_WORK_DIR / "qpr_quarterly.parquet"
    panel_path = DATA_WORK_DIR / "panel_features.parquet"

    if not qpr_path.exists() or not panel_path.exists():
        raise FileNotFoundError(
            f"Required data not found. "
            f"Run 'python src/pipeline.py build_panel' and "
            f"'python src/pipeline.py compute_features' first."
        )

    qpr_quarterly = safe_read_parquet(qpr_path)
    panel_features = safe_read_parquet(panel_path)

    # Check if duration column exists
    if duration_col not in panel_features.columns:
        raise ValueError(f"Duration column '{duration_col}' not found in panel_features")

    # Generate time-varying panel
    from capacity_sem.models.time_varying_survival import reshape_quarterly_to_time_varying, add_static_covariates

    tv_panel = reshape_quarterly_to_time_varying(
        qpr_quarterly=qpr_quarterly,
        panel_features=panel_features,
        lag_quarters=lag_quarters,
        duration_col=duration_col
    )

    tv_panel = add_static_covariates(tv_panel, panel_features)

    return tv_panel


def run_threshold_sensitivity_analysis(
    thresholds: Optional[List[int]] = None,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Run survival analysis across multiple completion thresholds.

    Parameters
    ----------
    thresholds : list of int, optional
        Completion thresholds to test (percentages). Default: 20 to 100 by 5.
    output_dir : Path, optional
        Directory to save results. Default: DATA_WORK_DIR / 'diagnostics'

    Returns
    -------
    pd.DataFrame
        Summary of results across all thresholds
    """
    if thresholds is None:
        thresholds = list(range(20, 105, 5))  # 20%, 25%, ..., 100%

    if output_dir is None:
        output_dir = DATA_WORK_DIR / 'diagnostics'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"\nTesting completion thresholds: {thresholds}")
    print(f"Capacity covariates: Lagged {TV_LAG_QUARTERS} quarter(s)")
    print(f"Control covariates: {len(SURVIVAL_COVARIATE_COLS)}")

    # Capacity predictors (time-varying with lag)
    lag_suffix = f'_lag{TV_LAG_QUARTERS}'
    capacity_cols = [
        f'Ratio_disbursed_to_obligated{lag_suffix}',
        f'Ratio_expended_to_disbursed{lag_suffix}'
    ]

    # Check which covariates are available (do this once)
    # Load panel to check columns
    panel_path = DATA_WORK_DIR / "panel_features.parquet"
    panel_features = safe_read_parquet(panel_path)
    available_covariates = [col for col in SURVIVAL_COVARIATE_COLS if col in panel_features.columns]

    if len(available_covariates) < len(SURVIVAL_COVARIATE_COLS):
        missing = set(SURVIVAL_COVARIATE_COLS) - set(available_covariates)
        warnings.warn(f"Missing covariates: {missing}")

    results_list = []

    for threshold in thresholds:
        print(f"\n{'='*80}")
        print(f"THRESHOLD: {threshold}% completion")
        print(f"{'='*80}")

        # Determine duration column name
        if threshold == 100:
            duration_col = 'Duration_of_completion'
        else:
            duration_col = f'Duration_{threshold}pct'

        try:
            # Generate time-varying panel for this threshold
            print(f"\nGenerating time-varying panel for {threshold}% threshold...")
            tv_data = generate_time_varying_panel_for_threshold(
                duration_col=duration_col,
                lag_quarters=TV_LAG_QUARTERS
            )

            n_intervals = len(tv_data)
            n_grantee_disasters = tv_data.groupby(['Grantee', 'Disaster Type']).ngroups
            n_events = tv_data['E'].sum()
            n_censored = tv_data.groupby(['Grantee', 'Disaster Type'])['E'].max().eq(0).sum()

            print(f"  Intervals: {n_intervals:,}")
            print(f"  Grantee-disasters: {n_grantee_disasters}")
            print(f"  Events: {n_events}")
            print(f"  Censored: {n_censored}")

            # Calculate EPV ratio
            n_predictors = len(capacity_cols) + len(available_covariates)
            epv_ratio = n_events / n_predictors if n_predictors > 0 else 0

            # Run capacity-only model
            print(f"\nFitting capacity-only model...")
            try:
                capacity_only_results = run_time_varying_cox(
                    tv_data=tv_data,
                    capacity_cols=capacity_cols,
                    covariate_cols=None,
                    model_name=f'threshold_{threshold}pct_capacity_only',
                    bootstrap_se=False
                )

                # Extract disbursement HR
                hrs_capacity = capacity_only_results['hazard_ratios']
                disb_hr_capacity = hrs_capacity[hrs_capacity['Variable'] == capacity_cols[0]]

                if len(disb_hr_capacity) > 0:
                    disb_hr = disb_hr_capacity['HR'].values[0]
                    disb_ci_lower = disb_hr_capacity['HR_Lower'].values[0]
                    disb_ci_upper = disb_hr_capacity['HR_Upper'].values[0]
                    disb_p = disb_hr_capacity['p_value'].values[0]
                else:
                    disb_hr = disb_ci_lower = disb_ci_upper = disb_p = np.nan

                concordance_capacity = capacity_only_results['concordance']

            except Exception as e:
                print(f"  Capacity-only model failed: {e}")
                disb_hr = disb_ci_lower = disb_ci_upper = disb_p = concordance_capacity = np.nan

            # Run full covariate model
            print(f"\nFitting full covariate model...")
            try:
                full_results = run_time_varying_cox(
                    tv_data=tv_data,
                    capacity_cols=capacity_cols,
                    covariate_cols=available_covariates,
                    model_name=f'threshold_{threshold}pct_full',
                    bootstrap_se=False  # Skip bootstrap for speed in sensitivity analysis
                )

                # Extract disbursement HR from full model
                hrs_full = full_results['hazard_ratios']
                disb_hr_full = hrs_full[hrs_full['Variable'] == capacity_cols[0]]

                if len(disb_hr_full) > 0:
                    disb_hr_adj = disb_hr_full['HR'].values[0]
                    disb_ci_lower_adj = disb_hr_full['HR_Lower'].values[0]
                    disb_ci_upper_adj = disb_hr_full['HR_Upper'].values[0]
                    disb_p_adj = disb_hr_full['p_value'].values[0]
                else:
                    disb_hr_adj = disb_ci_lower_adj = disb_ci_upper_adj = disb_p_adj = np.nan

                concordance_full = full_results['concordance']

            except Exception as e:
                print(f"  Full model failed: {e}")
                disb_hr_adj = disb_ci_lower_adj = disb_ci_upper_adj = disb_p_adj = concordance_full = np.nan

            # Store results
            results_list.append({
                'Threshold_pct': threshold,
                'N_Intervals': n_intervals,
                'N_Grantee_Disasters': n_grantee_disasters,
                'N_Events': n_events,
                'N_Censored': n_censored,
                'N_Predictors': n_predictors,
                'EPV_Ratio': epv_ratio,
                'Disb_HR_Unadjusted': disb_hr,
                'Disb_CI_Lower_Unadjusted': disb_ci_lower,
                'Disb_CI_Upper_Unadjusted': disb_ci_upper,
                'Disb_p_Unadjusted': disb_p,
                'Concordance_Unadjusted': concordance_capacity,
                'Disb_HR_Adjusted': disb_hr_adj,
                'Disb_CI_Lower_Adjusted': disb_ci_lower_adj,
                'Disb_CI_Upper_Adjusted': disb_ci_upper_adj,
                'Disb_p_Adjusted': disb_p_adj,
                'Concordance_Adjusted': concordance_full
            })

            print(f"\n  Results for {threshold}%:")
            print(f"    EPV Ratio: {epv_ratio:.2f} (need ≥10 for stable estimates)")
            print(f"    Capacity-only: HR={disb_hr:.3f} [{disb_ci_lower:.3f}, {disb_ci_upper:.3f}], p={disb_p:.3f}")
            print(f"    Full model: HR={disb_hr_adj:.3f} [{disb_ci_lower_adj:.3f}, {disb_ci_upper_adj:.3f}], p={disb_p_adj:.3f}")

        except Exception as e:
            print(f"\nERROR at threshold {threshold}%: {e}")
            warnings.warn(f"Failed to process threshold {threshold}%: {e}")
            # Add NaN row
            n_predictors_fallback = len(capacity_cols) + len(available_covariates)
            results_list.append({
                'Threshold_pct': threshold,
                'N_Intervals': 0,
                'N_Grantee_Disasters': 0,
                'N_Events': 0,
                'N_Censored': 0,
                'N_Predictors': n_predictors_fallback,
                'EPV_Ratio': np.nan,
                'Disb_HR_Unadjusted': np.nan,
                'Disb_CI_Lower_Unadjusted': np.nan,
                'Disb_CI_Upper_Unadjusted': np.nan,
                'Disb_p_Unadjusted': np.nan,
                'Concordance_Unadjusted': np.nan,
                'Disb_HR_Adjusted': np.nan,
                'Disb_CI_Lower_Adjusted': np.nan,
                'Disb_CI_Upper_Adjusted': np.nan,
                'Disb_p_Adjusted': np.nan,
                'Concordance_Adjusted': np.nan
            })

    # Create results DataFrame
    results_df = pd.DataFrame(results_list)

    # Save results
    output_path = output_dir / 'survival_threshold_sensitivity.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Threshold sensitivity results saved to: {output_path}")
    print(f"{'='*80}")

    return results_df


def main():
    """Main entry point for time-varying survival estimation."""
    print("=" * 60)
    print("Stage 03b: Time-Varying Survival Analysis")
    print("=" * 60)

    # Create output directories
    diagnostics_dir = DATA_WORK_DIR / 'diagnostics'
    figures_dir = Path('figures')
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load time-varying panel
    print("\nLoading time-varying panel...")
    try:
        tv_data = load_time_varying_panel()
        print(f"  Loaded {len(tv_data):,} intervals")
        print(f"  Unique grantee-disasters: {tv_data.groupby(['Grantee', 'Disaster Type']).ngroups}")
        print(f"  Events: {tv_data['E'].sum()}")
        print(f"  Censored: {tv_data.groupby(['Grantee', 'Disaster Type'])['E'].max().eq(0).sum()}")
    except FileNotFoundError as e:
        print(f"\n  Error: {e}")
        print("\n  You need to generate the time-varying panel first.")
        print("  Run: python src/pipeline.py build_panel")
        return

    # Check for required columns
    lag_suffix = f'_lag{TV_LAG_QUARTERS}'
    capacity_cols = [
        f'Ratio_disbursed_to_obligated{lag_suffix}',
        f'Ratio_expended_to_disbursed{lag_suffix}'
    ]

    missing_cols = [col for col in capacity_cols if col not in tv_data.columns]
    if missing_cols:
        print(f"\n  Error: Missing required columns: {missing_cols}")
        print("  The time-varying panel may not have been generated correctly.")
        return

    # Run robustness checks
    print("\nRunning robustness checks...")
    results = run_robustness_checks(tv_data)

    # Save results
    print("\nSaving results...")
    save_results(results, diagnostics_dir)

    # Generate diagnostic plots for main model
    if 'full_covariates' in results:
        print("\nGenerating diagnostic plots for main model...")
        main_model = results['full_covariates']['model']
        generate_diagnostic_plots(
            model=main_model,
            tv_data=tv_data,
            capacity_cols=capacity_cols,
            output_dir=figures_dir
        )

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)

    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}")
        if 'hazard_ratios' in model_results:
            hrs = model_results['hazard_ratios']
            print(f"  Concordance: {model_results['concordance']:.3f}")
            print(f"  N events: {model_results['n_events']}")
            print(f"\n  Hazard Ratios:")
            for _, row in hrs.iterrows():
                sig = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
                print(f"    {row['Variable']}: HR={row['HR']:.3f} [{row['HR_Lower']:.3f}, {row['HR_Upper']:.3f}] p={row['p_value']:.3f} {sig}")

    print("\n" + "="*60)
    print("Time-varying survival analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
