#!/usr/bin/env python3
"""
Predictor Discovery Analysis for CDBG-DR Program Completion

This script performs exploratory analysis to identify what characteristics
predict CDBG-DR program completion, pivoting from the null findings on
velocity measures.

Approach:
1. LASSO-penalized Cox regression for feature selection
2. Random Survival Forest for variable importance validation
3. Final interpretable Cox model with selected predictors
4. Descriptive profiling of completers vs non-completers

Author: Generated with Claude Code
Date: December 2025
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Survival analysis imports
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# Try to import scikit-survival for LASSO Cox
try:
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False
    warnings.warn("scikit-survival not available. Using fallback methods.")

# Try to import sklearn for preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# External data functions
from capacity_sem.data.external_data import (
    get_employment_for_all_grantees,
    get_employment_for_year,
    DRGR_DISASTER_YEARS,
)

# Set up paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data_work"
DIAGNOSTICS_DIR = DATA_DIR / "diagnostics"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Ensure output directories exist
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_parquet_safe(path: Path) -> pd.DataFrame:
    """
    Load parquet file with pyarrow fallback for compatibility issues.
    """
    try:
        return pd.read_parquet(path)
    except ValueError:
        # Fallback: use pyarrow directly
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        # Convert column by column to avoid block issues
        data = {}
        for col in table.column_names:
            try:
                data[col] = table.column(col).to_pandas()
            except Exception:
                # If conversion fails, try as numpy
                data[col] = table.column(col).to_pylist()
        return pd.DataFrame(data)


def load_analysis_data() -> pd.DataFrame:
    """
    Load and merge panel features with external data.

    Returns DataFrame ready for survival analysis with:
    - All panel features
    - Employment data (previously unused)
    - Program type data
    """
    print("Loading panel features...")
    df = load_parquet_safe(DATA_DIR / "panel_features_std.parquet")
    print(f"  Panel: {df.shape[0]} rows, {df.shape[1]} columns")

    # Load program types if available
    program_types_path = DATA_DIR / "panel_program_types.parquet"
    if program_types_path.exists():
        df_types = load_parquet_safe(program_types_path)
        print(f"  Program types: {df_types.shape[0]} rows, {df_types.shape[1]} columns")
        # Merge on Grantee + Disaster Type
        merge_cols = ['Grantee', 'Disaster Type']
        if all(c in df_types.columns for c in merge_cols):
            # Get only the new columns from program types
            new_cols = [c for c in df_types.columns if c not in df.columns or c in merge_cols]
            df = df.merge(df_types[new_cols], on=merge_cols, how='left')
            print(f"  After merge: {df.shape[1]} columns")

    # Add employment data (previously unused!)
    print("Adding employment data...")
    df_emp = get_employment_for_all_grantees()

    # Match employment to disaster year
    employment_values = []
    for _, row in df.iterrows():
        grantee = row['Grantee']
        disaster_type = row['Disaster Type']
        disaster_year = DRGR_DISASTER_YEARS.get(disaster_type, 2020)

        emp_data = get_employment_for_year(grantee, disaster_year)
        if emp_data:
            employment_values.append({
                'Total_Gov_Employment': emp_data['total_gov'],
                'Local_Gov_Employment': emp_data['local_gov'],
            })
        else:
            employment_values.append({
                'Total_Gov_Employment': np.nan,
                'Local_Gov_Employment': np.nan,
            })

    df_emp_matched = pd.DataFrame(employment_values)
    df = pd.concat([df.reset_index(drop=True), df_emp_matched], axis=1)

    # Create employment per capita (scaled)
    df['Employment_Per_Capita'] = df['Total_Gov_Employment'] / df['Population'] * 1000
    df['Employment_Log'] = np.log1p(df['Total_Gov_Employment'])

    n_with_emp = df['Total_Gov_Employment'].notna().sum()
    print(f"  Employment data available for {n_with_emp}/{len(df)} observations")

    return df


def prepare_survival_data(
    df: pd.DataFrame,
    duration_col: str = 'Duration_95pct',
    event_threshold: float = 0.95
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Prepare data for survival analysis.

    Returns:
        df_analysis: DataFrame with analysis features
        duration: Array of durations (quarters)
        event: Array of event indicators (1=completed, 0=censored)
    """
    # Create event indicator based on completion percentage
    df = df.copy()

    # Duration
    if duration_col not in df.columns:
        raise ValueError(f"Duration column '{duration_col}' not found")

    duration = df[duration_col].values

    # Event: completed at threshold
    # Use N_Quarters as censoring time for non-completers
    event = (~df[duration_col].isna()).astype(int)

    # For censored observations, use N_Quarters as duration
    duration = np.where(
        np.isnan(duration),
        df['N_Quarters'].values,
        duration
    )

    # Handle any remaining NaN
    valid_mask = ~np.isnan(duration)

    print(f"\nSurvival data summary:")
    print(f"  Total observations: {len(df)}")
    print(f"  Valid observations: {valid_mask.sum()}")
    print(f"  Events (completed): {event[valid_mask].sum()}")
    print(f"  Censored: {(valid_mask).sum() - event[valid_mask].sum()}")

    return df[valid_mask], duration[valid_mask], event[valid_mask]


def select_candidate_features(df: pd.DataFrame) -> List[str]:
    """
    Select candidate features for analysis.

    Groups features by category and selects representatives to avoid
    excessive collinearity.
    """
    features = {
        # Grantee characteristics
        'grantee': [
            'Government_Type_State',
            'Population_log',
            'Employment_Per_Capita',  # NEW - previously unused
            'Employment_Log',  # NEW - previously unused
        ],
        # Experience
        'experience': [
            'Prior_Grant_Count',
            'Years_Experience',
            'Experience_Index',
        ],
        # Disaster context
        'disaster': [
            'Severity_Index',
            'Counties_Affected',
            'Total_Damage',
            'Duration_Days',  # Disaster duration, not program duration
        ],
        # Grant characteristics
        'grant': [
            'Log_Obligated',
            'N_Quarters',
        ],
        # Pipeline efficiency
        'pipeline': [
            'Stage1_Efficiency',
            'Stage2_Efficiency',
            'Lag_Total_Pipeline',
        ],
        # Program portfolio (if available)
        'portfolio': [
            'Program_Diversity_Index',
            'Housing_Pct',
            'Infrastructure_Pct',
        ],
        # Velocity measures (for reference - expect null effects)
        'velocity': [
            'Capacity_Velocity_Index_pp',
            'Disbursement_Velocity_pp',
            'Velocity_Early',
            'Velocity_Late',
        ],
        # Ratios
        'ratios': [
            'Ratio_disbursed_to_obligated',
            'Ratio_expended_to_disbursed',
        ],
    }

    # Flatten and filter to available columns
    candidate_features = []
    for category, cols in features.items():
        available = [c for c in cols if c in df.columns]
        candidate_features.extend(available)
        print(f"  {category}: {len(available)}/{len(cols)} features available")

    return candidate_features


def run_lasso_cox(
    df: pd.DataFrame,
    duration: np.ndarray,
    event: np.ndarray,
    features: List[str],
    n_alphas: int = 100
) -> pd.DataFrame:
    """
    Run LASSO-penalized Cox regression for feature selection.
    """
    if not HAS_SKSURV:
        print("scikit-survival not available. Skipping LASSO Cox.")
        return pd.DataFrame()

    print("\n" + "="*60)
    print("LASSO Cox Regression for Feature Selection")
    print("="*60)

    # Prepare feature matrix
    X = df[features].copy()

    # Handle missing values - impute with median for now
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)

    # Create survival outcome
    y = Surv.from_arrays(event.astype(bool), duration)

    # Fit LASSO Cox with cross-validation
    print(f"\nFitting LASSO Cox with {n_alphas} regularization values...")

    model = CoxnetSurvivalAnalysis(
        l1_ratio=1.0,  # Pure LASSO
        alpha_min_ratio=0.01,
        n_alphas=n_alphas,
        fit_baseline_model=True
    )

    model.fit(X_scaled, y)

    # Find optimal alpha via cross-validation
    # Use the alpha that gives non-zero coefficients for ~5-10 features
    alphas = model.alphas_
    coef_paths = model.coef_

    # Count non-zero coefficients at each alpha
    n_nonzero = (np.abs(coef_paths) > 1e-8).sum(axis=0)

    # Find alpha where we have 5-10 non-zero coefficients
    target_features = 7
    alpha_idx = np.argmin(np.abs(n_nonzero - target_features))
    optimal_alpha = alphas[alpha_idx]

    print(f"Optimal alpha: {optimal_alpha:.4f} (gives {n_nonzero[alpha_idx]} features)")

    # Get coefficients at optimal alpha
    coefs = coef_paths[:, alpha_idx]

    # Create results DataFrame
    results = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs),
        'Selected': np.abs(coefs) > 1e-8
    }).sort_values('Abs_Coefficient', ascending=False)

    print("\nSelected features:")
    selected = results[results['Selected']]
    for _, row in selected.iterrows():
        direction = "+" if row['Coefficient'] > 0 else "-"
        print(f"  {direction} {row['Feature']}: {row['Coefficient']:.4f}")

    # Save results
    results.to_csv(DIAGNOSTICS_DIR / "predictor_discovery_lasso.csv", index=False)
    print(f"\nSaved to: {DIAGNOSTICS_DIR / 'predictor_discovery_lasso.csv'}")

    return results


def run_random_survival_forest(
    df: pd.DataFrame,
    duration: np.ndarray,
    event: np.ndarray,
    features: List[str],
    n_estimators: int = 100
) -> pd.DataFrame:
    """
    Run Random Survival Forest for variable importance.
    """
    if not HAS_SKSURV:
        print("scikit-survival not available. Skipping Random Survival Forest.")
        return pd.DataFrame()

    print("\n" + "="*60)
    print("Random Survival Forest for Variable Importance")
    print("="*60)

    # Prepare feature matrix
    X = df[features].copy()

    # Handle missing values
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Create survival outcome
    y = Surv.from_arrays(event.astype(bool), duration)

    print(f"\nFitting Random Survival Forest with {n_estimators} trees...")

    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )

    rsf.fit(X, y)

    # Get permutation importance
    print("Computing permutation importance...")
    from sklearn.inspection import permutation_importance

    perm_importance = permutation_importance(
        rsf, X, y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    # Create results DataFrame
    results = pd.DataFrame({
        'Feature': features,
        'Importance_Mean': perm_importance.importances_mean,
        'Importance_Std': perm_importance.importances_std,
    }).sort_values('Importance_Mean', ascending=False)

    # Get C-index
    c_index = rsf.score(X, y)
    print(f"\nRandom Survival Forest C-index: {c_index:.3f}")

    print("\nTop 10 features by importance:")
    for i, row in results.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance_Mean']:.4f} (+/- {row['Importance_Std']:.4f})")

    # Save results
    results['RSF_C_Index'] = c_index
    results.to_csv(DIAGNOSTICS_DIR / "predictor_discovery_rsf.csv", index=False)
    print(f"\nSaved to: {DIAGNOSTICS_DIR / 'predictor_discovery_rsf.csv'}")

    return results


def build_final_model(
    df: pd.DataFrame,
    duration: np.ndarray,
    event: np.ndarray,
    selected_features: List[str]
) -> pd.DataFrame:
    """
    Build final interpretable Cox model with selected features.
    """
    print("\n" + "="*60)
    print("Final Cox Proportional Hazards Model")
    print("="*60)

    # Prepare data for lifelines
    df_surv = df[selected_features].copy()
    df_surv['T'] = duration
    df_surv['E'] = event

    # Handle missing values
    for col in selected_features:
        if df_surv[col].isna().any():
            df_surv[col] = df_surv[col].fillna(df_surv[col].median())

    # Drop any remaining NaN rows
    df_surv = df_surv.dropna()

    print(f"\nFitting Cox PH with {len(selected_features)} features on {len(df_surv)} observations...")

    cph = CoxPHFitter()
    cph.fit(df_surv, duration_col='T', event_col='E')

    # Print summary
    print("\nModel Summary:")
    cph.print_summary()

    # Extract results
    summary = cph.summary
    summary['HR'] = np.exp(summary['coef'])
    summary['HR_lower'] = np.exp(summary['coef lower 95%'])
    summary['HR_upper'] = np.exp(summary['coef upper 95%'])

    # Calculate C-index
    c_index = cph.concordance_index_
    print(f"\nConcordance Index: {c_index:.3f}")

    # Test proportional hazards assumption
    print("\nTesting proportional hazards assumption...")
    try:
        ph_test = cph.check_assumptions(df_surv, p_value_threshold=0.05, show_plots=False)
    except Exception as e:
        print(f"  PH test error: {e}")

    # Save results
    results = summary.reset_index()
    results['C_Index'] = c_index
    results.to_csv(DIAGNOSTICS_DIR / "predictor_discovery_final.csv", index=False)
    print(f"\nSaved to: {DIAGNOSTICS_DIR / 'predictor_discovery_final.csv'}")

    return results, cph


def create_forest_plot(results: pd.DataFrame, output_path: Path):
    """
    Create forest plot of hazard ratios.
    """
    print("\nCreating forest plot...")

    # Sort by HR
    results = results.sort_values('HR')

    fig, ax = plt.subplots(figsize=(10, max(6, len(results) * 0.5)))

    y_pos = range(len(results))

    # Plot point estimates
    ax.scatter(results['HR'], y_pos, s=80, c='steelblue', zorder=3)

    # Plot confidence intervals
    for i, (_, row) in enumerate(results.iterrows()):
        ax.hlines(i, row['HR_lower'], row['HR_upper'], colors='steelblue', linewidth=2)

    # Reference line at HR=1
    ax.axvline(x=1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='HR=1 (no effect)')

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results['covariate'])
    ax.set_xlabel('Hazard Ratio (95% CI)')
    ax.set_title('Predictor Discovery: Features Associated with Program Completion\n(HR > 1 = faster completion)')

    # Log scale for x-axis
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved to: {output_path}")


def profile_completers(
    df: pd.DataFrame,
    event: np.ndarray,
    features: List[str]
) -> pd.DataFrame:
    """
    Create descriptive profile comparing completers vs non-completers.
    """
    print("\n" + "="*60)
    print("Completer Profile Analysis")
    print("="*60)

    df_analysis = df.copy()
    df_analysis['Completed'] = event

    results = []

    for feature in features:
        if feature not in df_analysis.columns:
            continue
        if df_analysis[feature].isna().all():
            continue

        completers = df_analysis[df_analysis['Completed'] == 1][feature]
        non_completers = df_analysis[df_analysis['Completed'] == 0][feature]

        # Skip if all NaN
        if completers.isna().all() or non_completers.isna().all():
            continue

        # Compute statistics
        from scipy import stats

        comp_mean = completers.mean()
        non_comp_mean = non_completers.mean()

        # T-test for continuous
        try:
            t_stat, p_value = stats.ttest_ind(
                completers.dropna(),
                non_completers.dropna(),
                equal_var=False
            )
        except Exception:
            t_stat, p_value = np.nan, np.nan

        results.append({
            'Feature': feature,
            'Completers_Mean': comp_mean,
            'Completers_N': completers.notna().sum(),
            'NonCompleters_Mean': non_comp_mean,
            'NonCompleters_N': non_completers.notna().sum(),
            'Difference': comp_mean - non_comp_mean,
            'T_Statistic': t_stat,
            'P_Value': p_value,
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('P_Value')

    print("\nSignificant differences (p < 0.05):")
    sig = df_results[df_results['P_Value'] < 0.05]
    for _, row in sig.iterrows():
        direction = "higher" if row['Difference'] > 0 else "lower"
        print(f"  {row['Feature']}: Completers {direction} ({row['Completers_Mean']:.2f} vs {row['NonCompleters_Mean']:.2f}, p={row['P_Value']:.3f})")

    # Save results
    df_results.to_csv(DIAGNOSTICS_DIR / "predictor_discovery_profile.csv", index=False)
    print(f"\nSaved to: {DIAGNOSTICS_DIR / 'predictor_discovery_profile.csv'}")

    return df_results


def main():
    """Main analysis pipeline."""
    print("="*60)
    print("PREDICTOR DISCOVERY ANALYSIS")
    print("What predicts CDBG-DR program completion?")
    print("="*60)

    # Load data
    df = load_analysis_data()

    # Prepare survival data
    df_analysis, duration, event = prepare_survival_data(df)

    # Select candidate features
    print("\nSelecting candidate features...")
    features = select_candidate_features(df_analysis)
    print(f"Total candidate features: {len(features)}")

    # Phase 1: LASSO Cox for feature selection
    lasso_results = run_lasso_cox(df_analysis, duration, event, features)

    # Phase 2: Random Survival Forest validation
    rsf_results = run_random_survival_forest(df_analysis, duration, event, features)

    # Identify features selected by both methods
    if not lasso_results.empty and not rsf_results.empty:
        lasso_selected = set(lasso_results[lasso_results['Selected']]['Feature'])
        rsf_top = set(rsf_results.head(10)['Feature'])
        concordant = lasso_selected.intersection(rsf_top)

        print("\n" + "="*60)
        print("CONCORDANT FEATURES (selected by both methods)")
        print("="*60)
        for f in concordant:
            print(f"  - {f}")

        # Use concordant features + any LASSO-selected for final model
        final_features = list(lasso_selected)
        if len(final_features) < 3:
            # Add top RSF features if LASSO selected too few
            for f in rsf_results['Feature'].head(5):
                if f not in final_features:
                    final_features.append(f)
    else:
        # Fallback: use pre-defined key features
        final_features = [
            'Government_Type_State',
            'Population_log',
            'Prior_Grant_Count',
            'Severity_Index',
            'Log_Obligated',
        ]
        final_features = [f for f in final_features if f in df_analysis.columns]

    print(f"\nFinal model features: {final_features}")

    # Phase 3: Final interpretable model
    if len(final_features) >= 2:
        final_results, cph = build_final_model(df_analysis, duration, event, final_features)

        # Create forest plot
        if 'HR' in final_results.columns:
            create_forest_plot(final_results, FIGURES_DIR / "predictor_importance_forest.png")

    # Phase 4: Descriptive profiling
    profile_results = profile_completers(df_analysis, event, features)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to:")
    print(f"  - {DIAGNOSTICS_DIR / 'predictor_discovery_lasso.csv'}")
    print(f"  - {DIAGNOSTICS_DIR / 'predictor_discovery_rsf.csv'}")
    print(f"  - {DIAGNOSTICS_DIR / 'predictor_discovery_final.csv'}")
    print(f"  - {DIAGNOSTICS_DIR / 'predictor_discovery_profile.csv'}")
    print(f"  - {FIGURES_DIR / 'predictor_importance_forest.png'}")


if __name__ == "__main__":
    main()
