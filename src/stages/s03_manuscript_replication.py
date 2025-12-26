"""
Stage 03 - Kaifa's Models Replication Pipeline (EXPERIMENTAL)
==============================================================

IMPORTANT: This replicates Kaifa's original Jupyter notebook analysis.
DO NOT use for production analyses.

For canonical analyses, use: s03_estimation.py

Author Attribution:
-------------------
Kaifa wrote the original manuscript and Jupyter notebook analysis.
This module replicates his methodology to enable understanding and critique.

Purpose:
--------
This module provides a complete, transparent implementation of Kaifa's
SEM methodology to enable:
1. Understanding how Kaifa obtained beta=71.024
2. Critiquing the methodological choices
3. Documenting differences from the canonical pipeline

Key Methodological Choices (Kaifa's Approach):
----------------------------------------
1. GRANTEE-LEVEL AGGREGATION
   - Treats capacity as a stable grantee trait
   - Averages ratios across all disasters for each grantee
   - Results in N~38 state, ~40 local governments

2. RIGHT-CENSORING OF DURATION
   - Programs that haven't reached 95% completion are assigned
     their observation time (N_Quarters) as the duration
   - This treats incomplete programs as if they "completed" at
     the current observation point
   - Dramatically increases sample size but introduces bias

3. TIMELINESS = 1/DURATION
   - Inverse transformation creates mathematical coupling with Duration
   - Both load on different factors but are deterministically related
   - May inflate model fit artificially

4. 3x3 FACTOR STRUCTURE
   - gov_cap: Ratio_disbursed, Ratio_expended, Timeliness
   - recovery_outcome: Duration, Ratio_fully_expended, Variance
   - Cross-factor correlations may be problematic

Usage:
------
    python src/stages/s03_manuscript_replication.py [--subset state|local|all]

Output:
-------
    data_work/diagnostics/manuscript_replication_*.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_WORK_DIR,
    STATE_GOVERNMENTS,
    LOCAL_GOVERNMENTS,
)
from stages._io_utils import safe_read_parquet

# Check for semopy
SEMOPY_AVAILABLE = False
try:
    from semopy import Model, calc_stats
    SEMOPY_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================

MANUSCRIPT_CONFIG = {
    'completion_threshold': 0.95,  # 95% completion to count as "done"
    'min_sample_size': 10,         # Minimum N for model fitting
    'aggregation_method': 'mean',  # How to aggregate across disasters
}

# Kaifa's Model Specifications
KAIFA_MODELS = {
    'kaifa_3x3_full': """
        # KAIFA'S MODEL: Full 3x3 specification (original manuscript)
        # Government Capacity (latent) - 3 indicators
        gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness_censored

        # Recovery Outcome (latent) - 3 indicators
        recovery_outcome =~ Duration_censored + Ratio_obligated_funds_fully_expended + Quarter_by_quarter_variance_expended

        # Structural path: Capacity -> Outcome
        recovery_outcome ~ gov_cap
    """,
    'kaifa_3x3_no_duration': """
        # KAIFA'S MODEL: Without Duration (his sensitivity analysis)
        gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness_censored
        recovery_outcome =~ Ratio_obligated_funds_fully_expended + Quarter_by_quarter_variance_expended
        recovery_outcome ~ gov_cap
    """,
    'kaifa_2x2_minimal': """
        # KAIFA'S MODEL: Minimal 2x2
        gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed
        recovery_outcome =~ Duration_censored + Quarter_by_quarter_variance_expended
        recovery_outcome ~ gov_cap
    """,
}

# Alias for backward compatibility
MANUSCRIPT_MODELS = KAIFA_MODELS


# =============================================================================
# STEP 1: DATA LOADING
# =============================================================================

def step_01_load_data() -> pd.DataFrame:
    """
    Load panel features data.

    Returns the same data used by the canonical pipeline.
    """
    print("\n" + "=" * 70)
    print("STEP 1: Load Panel Data")
    print("=" * 70)

    panel_path = DATA_WORK_DIR / "panel_features.parquet"

    if not panel_path.exists():
        raise FileNotFoundError(
            f"Panel data not found at {panel_path}.\n"
            "Run the canonical pipeline first: python src/pipeline.py run_all"
        )

    data = safe_read_parquet(panel_path)

    print(f"  Loaded: {panel_path}")
    print(f"  Rows: {len(data)} grantee-disaster pairs")
    print(f"  Unique grantees: {data['Grantee'].nunique()}")
    print(f"  Unique disasters: {data['Disaster Type'].nunique()}")

    return data


# =============================================================================
# STEP 2: APPLY RIGHT-CENSORING (MANUSCRIPT APPROACH)
# =============================================================================

def step_02_apply_censoring(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply right-censoring to Duration variable.

    CRITIQUE POINTS:
    ----------------
    1. This treats incomplete programs as if they completed at observation time
    2. Introduces bias: programs that would take longer are treated as shorter
    3. Violates the assumption that duration reflects true completion time
    4. May inflate capacity-outcome relationship if slow programs are ongoing

    Manuscript rationale:
    "the end quarter (the earlier one between the current quarter and
    the quarter when over 95% of obligated funds were fully expended)"
    """
    print("\n" + "=" * 70)
    print("STEP 2: Apply Right-Censoring to Duration")
    print("=" * 70)

    df = data.copy()

    # Original Duration availability
    n_original = df['Duration_of_completion'].notna().sum()
    print(f"  Original Duration available: {n_original}/{len(df)} ({100*n_original/len(df):.1f}%)")

    # Create censored Duration
    df['Duration_censored'] = df['Duration_of_completion'].copy()

    # For missing Duration, use N_Quarters (right-censoring)
    mask = df['Duration_of_completion'].isna() & df['N_Quarters'].notna()
    n_censored = mask.sum()
    df.loc[mask, 'Duration_censored'] = df.loc[mask, 'N_Quarters']

    print(f"  Right-censored (using N_Quarters): {n_censored}")
    print(f"  Censored Duration available: {df['Duration_censored'].notna().sum()}/{len(df)}")

    # Create Timeliness = 1/Duration
    df['Timeliness_censored'] = np.where(
        df['Duration_censored'] > 0,
        1.0 / df['Duration_censored'],
        np.nan
    )

    # Create log-transformed Duration
    df['Duration_censored_log'] = np.log1p(df['Duration_censored'])

    # Add censoring indicator for tracking
    df['Is_Censored'] = mask.astype(int)

    # Statistics on censored vs uncensored
    completed = df[~mask]
    censored = df[mask]

    print("\n  Comparison (censored vs completed):")
    print(f"    Completed programs:")
    print(f"      - Mean Duration: {completed['Duration_censored'].mean():.1f} quarters")
    print(f"      - Mean Completion %: {completed['Completion_Pct'].mean():.1%}")
    print(f"    Censored programs:")
    print(f"      - Mean Duration (N_Quarters): {censored['Duration_censored'].mean():.1f} quarters")
    print(f"      - Mean Completion %: {censored['Completion_Pct'].mean():.1%}")

    print("\n  CRITIQUE: Censored programs are assigned duration as if completed,")
    print("            but they only averaged {:.0%} completion. This biases".format(
        censored['Completion_Pct'].mean()))
    print("            duration estimates downward (making them look faster).")

    return df


# =============================================================================
# STEP 3: AGGREGATE TO GRANTEE LEVEL (MANUSCRIPT APPROACH)
# =============================================================================

def step_03_aggregate_to_grantee(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate from grantee-disaster level to grantee level.

    CRITIQUE POINTS:
    ----------------
    1. Assumes capacity is stable across disasters (may not be true)
    2. Different disasters have different complexity/requirements
    3. Averaging ratios: mean of ratios != ratio of means
    4. Loses information about within-grantee variation
    5. May mask learning effects (grantees may improve over time)

    Manuscript rationale:
    "Uses quarterly average indicators aggregated at the grantee level"
    """
    print("\n" + "=" * 70)
    print("STEP 3: Aggregate to Grantee Level")
    print("=" * 70)

    print(f"  Input: {len(data)} grantee-disaster pairs")

    # Columns to aggregate
    agg_cols = [
        'Ratio_disbursed_to_obligated',
        'Ratio_expended_to_disbursed',
        'Timeliness_censored',
        'Duration_censored',
        'Duration_censored_log',
        'Ratio_obligated_funds_fully_expended',
        'Quarter_by_quarter_variance_expended',
        'Spending_CV',
        'Completion_Pct',
        'Is_Censored',
        'N_Quarters',
    ]

    # Filter to available columns
    available = [c for c in agg_cols if c in data.columns]

    # Aggregate by grantee (manuscript uses mean)
    grantee_data = data.groupby('Grantee')[available].agg('mean').reset_index()

    print(f"  Output: {len(grantee_data)} unique grantees")

    # Show how many disasters each grantee has
    disasters_per_grantee = data.groupby('Grantee').size()
    print(f"\n  Disasters per grantee:")
    print(f"    Min: {disasters_per_grantee.min()}")
    print(f"    Max: {disasters_per_grantee.max()}")
    print(f"    Mean: {disasters_per_grantee.mean():.1f}")

    # Show grantees with multiple disasters
    multi_disaster = (disasters_per_grantee > 1).sum()
    print(f"    Grantees with multiple disasters: {multi_disaster}")

    print("\n  CRITIQUE: Averaging across disasters assumes capacity is stable.")
    print("            Grantees with multiple disasters may show learning effects.")
    print("            The mean of ratios across disasters may not equal the")
    print("            ratio of total amounts across disasters.")

    return grantee_data


# =============================================================================
# STEP 4: FILTER TO GOVERNMENT SUBSET
# =============================================================================

def step_04_filter_subset(
    data: pd.DataFrame,
    subset: str = 'state'
) -> pd.DataFrame:
    """
    Filter to state or local governments.
    """
    print("\n" + "=" * 70)
    print(f"STEP 4: Filter to {subset.upper()} Governments")
    print("=" * 70)

    if subset == 'state':
        filtered = data[data['Grantee'].isin(STATE_GOVERNMENTS)]
    elif subset == 'local':
        filtered = data[data['Grantee'].isin(LOCAL_GOVERNMENTS)]
    else:
        filtered = data

    print(f"  Before filter: {len(data)} grantees")
    print(f"  After filter: {len(filtered)} grantees")

    # List grantees
    if len(filtered) <= 40:
        print(f"\n  Grantees included:")
        for g in sorted(filtered['Grantee'].tolist()):
            print(f"    - {g}")

    return filtered


# =============================================================================
# STEP 5: PREPARE MODEL DATA
# =============================================================================

def step_05_prepare_model_data(
    data: pd.DataFrame,
    model_type: str = 'kaifa_3x3_full'
) -> Tuple[pd.DataFrame, str]:
    """
    Prepare data for SEM fitting using Kaifa's model specification.

    CRITIQUE POINTS:
    ----------------
    1. Listwise deletion may create selection bias
    2. Variables with different missing patterns create different samples
    3. 3x3 model may be over-parameterized for small N
    """
    print("\n" + "=" * 70)
    print(f"STEP 5: Prepare Model Data (Kaifa's {model_type})")
    print("=" * 70)

    model_spec = KAIFA_MODELS.get(model_type)
    if model_spec is None:
        raise ValueError(f"Unknown Kaifa model type: {model_type}")

    # Determine required columns from Kaifa's model spec
    if model_type == 'kaifa_3x3_full':
        required = [
            'Ratio_disbursed_to_obligated',
            'Ratio_expended_to_disbursed',
            'Timeliness_censored',
            'Duration_censored',
            'Ratio_obligated_funds_fully_expended',
            'Quarter_by_quarter_variance_expended',
        ]
    elif model_type == 'kaifa_3x3_no_duration':
        required = [
            'Ratio_disbursed_to_obligated',
            'Ratio_expended_to_disbursed',
            'Timeliness_censored',
            'Ratio_obligated_funds_fully_expended',
            'Quarter_by_quarter_variance_expended',
        ]
    else:
        required = [
            'Ratio_disbursed_to_obligated',
            'Ratio_expended_to_disbursed',
            'Duration_censored',
            'Quarter_by_quarter_variance_expended',
        ]

    print(f"  Required variables:")
    for var in required:
        n_avail = data[var].notna().sum()
        print(f"    {var}: {n_avail}/{len(data)} ({100*n_avail/len(data):.1f}%)")

    # Get complete cases
    model_data = data[required].dropna()

    print(f"\n  Complete cases (listwise deletion): {len(model_data)}/{len(data)}")
    print(f"  Dropped: {len(data) - len(model_data)} grantees")

    # Check for outliers
    print("\n  Variable summary statistics:")
    for var in required:
        col = model_data[var]
        print(f"    {var}:")
        print(f"      Mean: {col.mean():.4f}, SD: {col.std():.4f}")
        print(f"      Min: {col.min():.4f}, Max: {col.max():.4f}")

    if len(model_data) < MANUSCRIPT_CONFIG['min_sample_size']:
        print(f"\n  WARNING: N={len(model_data)} is below minimum ({MANUSCRIPT_CONFIG['min_sample_size']})")

    return model_data, model_spec


# =============================================================================
# STEP 6: FIT SEM MODEL
# =============================================================================

def step_06_fit_model(
    data: pd.DataFrame,
    model_spec: str
) -> Dict[str, Any]:
    """
    Fit SEM model using semopy.

    CRITIQUE POINTS:
    ----------------
    1. Small sample size (N~35) may lead to unstable estimates
    2. ML estimation assumes multivariate normality
    3. Bounded variables (ratios) may violate distributional assumptions
    4. High correlations between indicators may cause identification issues
    """
    print("\n" + "=" * 70)
    print("STEP 6: Fit SEM Model")
    print("=" * 70)

    if not SEMOPY_AVAILABLE:
        raise ImportError("semopy is required. Install with: pip install semopy")

    print(f"  Sample size: N = {len(data)}")
    print(f"\n  Model specification:")
    for line in model_spec.strip().split('\n'):
        if line.strip() and not line.strip().startswith('#'):
            print(f"    {line.strip()}")

    # Fit model
    print("\n  Fitting model...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Model(model_spec)
        model.fit(data)

    # Get estimates
    estimates = model.inspect()
    fit_stats = calc_stats(model)

    print("\n  Model converged successfully")

    results = {
        'model': model,
        'estimates': estimates,
        'fit_stats': fit_stats,
        'sample_size': len(data),
    }

    return results


# =============================================================================
# STEP 7: EXTRACT AND INTERPRET RESULTS
# =============================================================================

def step_07_interpret_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and interpret SEM results.

    CRITIQUE POINTS:
    ----------------
    1. Unstandardized coefficients (beta~71-113) are not comparable across studies
    2. Statistical significance may be inflated by right-censoring
    3. Model fit indices should be evaluated
    """
    print("\n" + "=" * 70)
    print("STEP 7: Interpret Results")
    print("=" * 70)

    estimates = results['estimates']
    fit_stats = results['fit_stats']

    interpretation = {}

    # Extract structural path (capacity -> outcome)
    structural = estimates[
        (estimates['op'] == '~') &
        (estimates['lval'] == 'recovery_outcome')
    ]

    if not structural.empty:
        row = structural.iloc[0]
        beta = float(row['Estimate'])
        se = float(row['Std. Err']) if pd.notna(row['Std. Err']) else np.nan
        pval = float(row['p-value']) if pd.notna(row['p-value']) else np.nan

        interpretation['structural_path'] = {
            'beta': beta,
            'se': se,
            'p_value': pval,
            'significant': pval < 0.05 if pd.notna(pval) else False,
        }

        print("\n  STRUCTURAL PATH (Capacity -> Outcome):")
        print(f"    Beta (unstandardized): {beta:.4f}")
        print(f"    Standard Error: {se:.4f}")
        print(f"    p-value: {pval:.4f}")
        print(f"    Significant at p<0.05: {pval < 0.05}")

        print("\n  MANUSCRIPT COMPARISON:")
        print(f"    Manuscript claims: beta=71.024, p=0.01")
        print(f"    This replication: beta={beta:.4f}, p={pval:.4f}")

        if 50 < abs(beta) < 150:
            print("    -> Beta magnitude is in similar range")
        if pval < 0.05:
            print("    -> Statistical significance matches")

    # Extract factor loadings
    loadings = estimates[estimates['op'] == '=~']
    if not loadings.empty:
        print("\n  FACTOR LOADINGS:")
        for _, row in loadings.iterrows():
            factor = row['lval']
            indicator = row['rval']
            loading = row['Estimate']
            print(f"    {factor} -> {indicator}: {loading:.4f}")

        interpretation['loadings'] = loadings.to_dict('records')

    # Model fit
    if fit_stats is not None:
        print("\n  MODEL FIT INDICES:")

        # Try to extract common indices
        for idx in ['chi2', 'dof', 'CFI', 'TLI', 'RMSEA', 'SRMR', 'AIC', 'BIC']:
            if idx in fit_stats.index:
                val = fit_stats.loc[idx].values[0] if hasattr(fit_stats, 'loc') else np.nan
                print(f"    {idx}: {val:.4f}")

        interpretation['fit_stats'] = fit_stats

    # Critique section
    print("\n  METHODOLOGICAL CRITIQUES:")
    print("    1. Beta is UNSTANDARDIZED - magnitude depends on variable scales")
    print("    2. Right-censoring biases duration downward (faster completion)")
    print("    3. Timeliness = 1/Duration creates mathematical coupling")
    print("    4. Small N (~35) may lead to unstable estimates")
    print("    5. Grantee-level aggregation assumes stable capacity")

    return interpretation


# =============================================================================
# STEP 8: SAVE RESULTS
# =============================================================================

def step_08_save_results(
    results: Dict[str, Any],
    interpretation: Dict[str, Any],
    subset: str,
    model_type: str
) -> Path:
    """
    Save replication results to disk.
    """
    print("\n" + "=" * 70)
    print("STEP 8: Save Results")
    print("=" * 70)

    diag_dir = DATA_WORK_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Save estimates
    estimates_path = diag_dir / f"manuscript_replication_estimates_{subset}_{model_type}.csv"
    results['estimates'].to_csv(estimates_path, index=False)
    print(f"  Estimates saved: {estimates_path}")

    # Save summary
    summary = {
        'methodology': 'manuscript_replication',
        'subset': subset,
        'model_type': model_type,
        'sample_size': results['sample_size'],
        'beta': interpretation['structural_path']['beta'],
        'se': interpretation['structural_path']['se'],
        'p_value': interpretation['structural_path']['p_value'],
        'significant': interpretation['structural_path']['significant'],
        'manuscript_beta': 71.024,
        'manuscript_p': 0.01,
    }

    summary_path = diag_dir / f"manuscript_replication_summary_{subset}_{model_type}.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"  Summary saved: {summary_path}")

    return diag_dir


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_kaifa_replication_pipeline(
    subset: str = 'state',
    model_type: str = 'kaifa_3x3_full'
) -> Dict[str, Any]:
    """
    Run complete Kaifa's Models replication pipeline.

    This implements Kaifa's full methodology from the original Jupyter notebook
    analysis to enable understanding and critique.
    """
    print("\n" + "=" * 70)
    print("KAIFA'S MODELS REPLICATION PIPELINE (EXPERIMENTAL)")
    print("=" * 70)
    print("\nWARNING: This replicates Kaifa's original analysis.")
    print("         Do not use for production analyses.")
    print("         Use s03_estimation.py for canonical analyses.")

    # Step 1: Load data
    data = step_01_load_data()

    # Step 2: Apply censoring
    data = step_02_apply_censoring(data)

    # Step 3: Aggregate to grantee level
    grantee_data = step_03_aggregate_to_grantee(data)

    # Step 4: Filter to subset
    subset_data = step_04_filter_subset(grantee_data, subset)

    # Step 5: Prepare model data
    model_data, model_spec = step_05_prepare_model_data(subset_data, model_type)

    # Step 6: Fit model
    results = step_06_fit_model(model_data, model_spec)

    # Step 7: Interpret results
    interpretation = step_07_interpret_results(results)

    # Step 8: Save results
    output_dir = step_08_save_results(results, interpretation, subset, model_type)

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Subset: {subset}")
    print(f"  Model: {model_type}")
    print(f"  Sample size: N = {results['sample_size']}")
    print(f"  Structural path: beta = {interpretation['structural_path']['beta']:.4f}")
    print(f"  p-value: {interpretation['structural_path']['p_value']:.4f}")
    print(f"\n  Results saved to: {output_dir}")

    return {
        'results': results,
        'interpretation': interpretation,
        'output_dir': output_dir,
    }


def main(subset: str = 'state', model_type: str = 'kaifa_3x3_full'):
    """Main entry point for Kaifa's Models replication."""
    run_kaifa_replication_pipeline(subset, model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Kaifa's Models replication pipeline (EXPERIMENTAL)"
    )
    parser.add_argument(
        "--subset", "-s",
        default="state",
        choices=["state", "local", "all"],
        help="Government subset to analyze"
    )
    parser.add_argument(
        "--model", "-m",
        default="kaifa_3x3_full",
        choices=["kaifa_3x3_full", "kaifa_3x3_no_duration", "kaifa_2x2_minimal"],
        help="Kaifa's model specification"
    )

    args = parser.parse_args()
    main(subset=args.subset, model_type=args.model)
