"""
Kaifa's Models Replication Module.

IMPORTANT: This module replicates KAIFA'S ORIGINAL ANALYSIS.
It implements Kaifa's Jupyter notebook methodology to verify
his manuscript claims. DO NOT use these functions for new analyses.

Author Attribution:
-------------------
Kaifa wrote the original manuscript and Jupyter notebook analysis.
This module replicates his methodology to enable understanding and critique.

Canonical pipeline functions are in:
- sem_fitting.py (standard model fitting)
- s03_estimation.py (main estimation stage)

Key differences from canonical pipeline (Kaifa's approach):
1. Uses grantee-level aggregation (not grantee-disaster pairs)
2. Applies right-censoring to Duration (incomplete programs get N_Quarters)
3. Uses original 3x3 model with Timeliness = 1/Duration

Replication Results:
- Kaifa's manuscript claims: beta=71.024, p=0.01
- This module produces: beta~113.65, p<0.001
- Difference likely due to ratio calculation method (mean-of-ratios vs final)
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import warnings

# Check for semopy
SEMOPY_AVAILABLE = False
try:
    from semopy import Model, calc_stats
    SEMOPY_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# KAIFA'S MODEL SPECIFICATIONS
# =============================================================================

KAIFA_MODEL_3X3 = """
# Kaifa's original 3x3 model specification
# gov_cap: Capacity measured by ratios + timeliness
# recovery_outcome: Outcome measured by duration + completion ratio + variance

gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness_censored
recovery_outcome =~ Duration_censored + Ratio_obligated_funds_fully_expended + Quarter_by_quarter_variance_expended
recovery_outcome ~ gov_cap
"""

KAIFA_MODEL_REDUCED = """
# Kaifa's reduced model (without Duration)
# For his sensitivity analysis

gov_cap =~ Ratio_disbursed_to_obligated + Ratio_expended_to_disbursed + Timeliness_censored
recovery_outcome =~ Ratio_obligated_funds_fully_expended + Quarter_by_quarter_variance_expended
recovery_outcome ~ gov_cap
"""

# Aliases for backward compatibility
MANUSCRIPT_MODEL_3X3 = KAIFA_MODEL_3X3
MANUSCRIPT_MODEL_REDUCED = KAIFA_MODEL_REDUCED


# =============================================================================
# DATA PREPARATION (KAIFA'S APPROACH)
# =============================================================================

def apply_duration_censoring(
    data: pd.DataFrame,
    duration_col: str = 'Duration_of_completion',
    n_quarters_col: str = 'N_Quarters'
) -> pd.DataFrame:
    """
    Apply right-censoring to Duration as described in Kaifa's manuscript.

    EXPERIMENTAL: This replicates Kaifa's methodology where
    incomplete programs are assigned their observation time as duration.

    Kaifa's definition:
    "the end quarter (the earlier one between the current quarter and
    the quarter when over 95% of obligated funds were fully expended)"

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with Duration and N_Quarters columns.
    duration_col : str
        Column with actual completion duration (NaN for incomplete).
    n_quarters_col : str
        Column with total quarters observed.

    Returns
    -------
    pd.DataFrame
        Data with censored duration columns added.
    """
    df = data.copy()

    # Create censored Duration
    df['Duration_censored'] = df[duration_col].copy()

    # For missing Duration, use N_Quarters (right-censoring)
    mask = df[duration_col].isna() & df[n_quarters_col].notna()
    df.loc[mask, 'Duration_censored'] = df.loc[mask, n_quarters_col]

    # Create Timeliness = 1/Duration (Kaifa's definition)
    df['Timeliness_censored'] = np.where(
        df['Duration_censored'] > 0,
        1.0 / df['Duration_censored'],
        np.nan
    )

    # Log transform
    df['Duration_censored_log'] = np.log1p(df['Duration_censored'])

    # Add censoring indicator
    df['Duration_is_censored'] = df[duration_col].isna()

    return df


def aggregate_to_grantee_level_kaifa(
    data: pd.DataFrame,
    grantee_col: str = 'Grantee'
) -> pd.DataFrame:
    """
    Aggregate to grantee level using Kaifa's methodology.

    EXPERIMENTAL: This computes mean across all disasters for each grantee,
    treating capacity as a stable grantee trait rather than disaster-specific.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data at grantee-disaster level.
    grantee_col : str
        Column containing grantee identifier.

    Returns
    -------
    pd.DataFrame
        Grantee-level aggregated data.
    """
    # Columns to aggregate by mean
    agg_cols = [
        'Ratio_disbursed_to_obligated',
        'Ratio_expended_to_disbursed',
        'Timeliness_censored',
        'Duration_censored',
        'Duration_censored_log',
        'Ratio_obligated_funds_fully_expended',
        'Quarter_by_quarter_variance_expended',
        'Spending_CV',
        'N_Quarters',
        'Completion_Pct',
        'Duration_is_censored',
    ]

    # Filter to available columns
    available = [c for c in agg_cols if c in data.columns]

    # Aggregate
    grantee_agg = data.groupby(grantee_col)[available].agg('mean').reset_index()

    # For censoring indicator, use any() instead of mean()
    if 'Duration_is_censored' in available:
        censored_any = data.groupby(grantee_col)['Duration_is_censored'].any().reset_index()
        grantee_agg['Has_any_censored'] = grantee_agg[grantee_col].map(
            censored_any.set_index(grantee_col)['Duration_is_censored']
        )

    return grantee_agg


# Alias for backward compatibility
aggregate_to_grantee_level_manuscript = aggregate_to_grantee_level_kaifa


# =============================================================================
# KAIFA'S REPLICATION FUNCTIONS
# =============================================================================

def run_kaifa_replication(
    data: pd.DataFrame,
    subset: str = 'state',
    model_type: str = '3x3',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run SEM analysis replicating Kaifa's methodology.

    EXPERIMENTAL: This function replicates Kaifa's original Jupyter notebook
    analysis that produced beta=71.024. Use only for verification.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with features.
    subset : str
        Government type: 'all', 'state', or 'local'.
    model_type : str
        '3x3' for full model, 'reduced' for model without Duration.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    Dict[str, Any]
        Replication results including estimates and comparison notes.
    """
    if not SEMOPY_AVAILABLE:
        raise ImportError("semopy is required for Kaifa's replication")

    results = {
        'methodology': 'kaifa_replication',
        'notes': [],
    }

    # Step 1: Apply censoring (Kaifa's approach)
    if verbose:
        print("Applying Kaifa's censoring approach...")

    df = apply_duration_censoring(data)
    results['notes'].append("Applied right-censoring: incomplete programs use N_Quarters as Duration")

    # Step 2: Aggregate to grantee level (Kaifa's approach)
    if verbose:
        print("Aggregating to grantee level (Kaifa's method)...")

    grantee_data = aggregate_to_grantee_level_kaifa(df)
    results['n_grantees_total'] = len(grantee_data)
    results['notes'].append(f"Aggregated to {len(grantee_data)} grantees (mean across disasters)")

    # Step 3: Filter to subset
    from config import STATE_GOVERNMENTS, LOCAL_GOVERNMENTS

    if subset == 'state':
        grantee_data = grantee_data[grantee_data['Grantee'].isin(STATE_GOVERNMENTS)]
    elif subset == 'local':
        grantee_data = grantee_data[grantee_data['Grantee'].isin(LOCAL_GOVERNMENTS)]

    results['n_grantees_subset'] = len(grantee_data)
    results['subset'] = subset

    # Step 4: Select Kaifa's model specification
    if model_type == '3x3':
        model_spec = KAIFA_MODEL_3X3
        required_cols = [
            'Ratio_disbursed_to_obligated', 'Ratio_expended_to_disbursed',
            'Timeliness_censored', 'Duration_censored',
            'Ratio_obligated_funds_fully_expended', 'Quarter_by_quarter_variance_expended'
        ]
    else:
        model_spec = KAIFA_MODEL_REDUCED
        required_cols = [
            'Ratio_disbursed_to_obligated', 'Ratio_expended_to_disbursed',
            'Timeliness_censored',
            'Ratio_obligated_funds_fully_expended', 'Quarter_by_quarter_variance_expended'
        ]

    results['model_type'] = model_type

    # Step 5: Fit model
    model_data = grantee_data[required_cols].dropna()
    results['n_complete_cases'] = len(model_data)

    if len(model_data) < 10:
        results['error'] = f"Insufficient complete cases: {len(model_data)}"
        return results

    if verbose:
        print(f"Fitting Kaifa's model with N={len(model_data)}...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Model(model_spec)
        model.fit(model_data)

    estimates = model.inspect()
    fit_stats = calc_stats(model)

    results['estimates'] = estimates
    results['fit_stats'] = fit_stats

    # Step 6: Extract structural path
    structural = estimates[
        (estimates['op'] == '~') &
        (estimates['lval'] == 'recovery_outcome')
    ]

    if not structural.empty:
        row = structural.iloc[0]
        results['structural_path'] = {
            'beta': float(row['Estimate']),
            'se': float(row['Std. Err']) if pd.notna(row['Std. Err']) else np.nan,
            'p_value': float(row['p-value']) if pd.notna(row['p-value']) else np.nan,
        }

        if verbose:
            beta = results['structural_path']['beta']
            pval = results['structural_path']['p_value']
            print(f"\nStructural path: beta={beta:.4f}, p={pval:.4f}")
            print(f"Kaifa's manuscript claims: beta=71.024, p=0.01")

    # Add comparison notes
    if 'structural_path' in results:
        beta = results['structural_path']['beta']
        if 50 < abs(beta) < 150:
            results['notes'].append("Beta magnitude matches Kaifa's order (~71-113)")
        if results['structural_path']['p_value'] < 0.05:
            results['notes'].append("Significant at p<0.05 (matches Kaifa's manuscript)")

    return results


# Alias for backward compatibility
run_manuscript_replication = run_kaifa_replication


def compare_methodologies(
    data: pd.DataFrame,
    subset: str = 'state',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare Kaifa's methodology vs. canonical pipeline methodology.

    EXPERIMENTAL: This function runs both approaches side-by-side
    to document the differences in results.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with features.
    subset : str
        Government type.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        Comparison table showing results from both methodologies.
    """
    if not SEMOPY_AVAILABLE:
        raise ImportError("semopy is required")

    results = []

    # 1. Kaifa's replication (grantee-level, censored duration)
    if verbose:
        print("=" * 60)
        print("KAIFA'S METHODOLOGY (Experimental)")
        print("=" * 60)

    try:
        kaifa_results = run_kaifa_replication(
            data, subset=subset, model_type='3x3', verbose=verbose
        )

        if 'structural_path' in kaifa_results:
            results.append({
                'Methodology': "Kaifa's (Grantee-level, Censored)",
                'N': kaifa_results['n_complete_cases'],
                'Beta': kaifa_results['structural_path']['beta'],
                'SE': kaifa_results['structural_path']['se'],
                'p_value': kaifa_results['structural_path']['p_value'],
                'Significant': kaifa_results['structural_path']['p_value'] < 0.05,
                'Notes': "Replication of Kaifa's analysis",
            })
    except Exception as e:
        if verbose:
            print(f"Kaifa's methodology failed: {e}")

    # 2. Canonical pipeline (grantee-disaster, no censoring)
    if verbose:
        print("\n" + "=" * 60)
        print("CANONICAL PIPELINE (Standard)")
        print("=" * 60)

    try:
        from capacity_sem.models.sem_fitting import fit_sem_model
        from capacity_sem.models.sem_specifications import get_model_spec
        from config import STATE_GOVERNMENTS, LOCAL_GOVERNMENTS

        # Filter to subset
        if subset == 'state':
            pipeline_data = data[data['Grantee'].isin(STATE_GOVERNMENTS)]
        elif subset == 'local':
            pipeline_data = data[data['Grantee'].isin(LOCAL_GOVERNMENTS)]
        else:
            pipeline_data = data

        model_spec = get_model_spec('exp_optimal_v1')
        model, result = fit_sem_model(model_spec, pipeline_data)
        estimates = model.inspect()

        structural = estimates[
            (estimates['op'] == '~') &
            (estimates['lval'] == 'recovery_outcome')
        ]

        if not structural.empty:
            row = structural.iloc[0]
            n_complete = len(pipeline_data[['Ratio_disbursed_to_obligated',
                                            'Ratio_expended_to_disbursed',
                                            'Duration_log', 'Spending_CV']].dropna())

            results.append({
                'Methodology': 'Canonical (Grantee-Disaster, No Censoring)',
                'N': n_complete,
                'Beta': float(row['Estimate']),
                'SE': float(row['Std. Err']) if pd.notna(row['Std. Err']) else np.nan,
                'p_value': float(row['p-value']) if pd.notna(row['p-value']) else np.nan,
                'Significant': row['p-value'] < 0.05 if pd.notna(row['p-value']) else False,
                'Notes': 'Standard pipeline analysis',
            })

            if verbose:
                print(f"N={n_complete}, Beta={row['Estimate']:.4f}, p={row['p-value']:.4f}")

    except Exception as e:
        if verbose:
            print(f"Canonical pipeline failed: {e}")

    # Create comparison DataFrame
    comparison = pd.DataFrame(results)

    if verbose and not comparison.empty:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(comparison.to_string(index=False))

        print("\nKEY DIFFERENCES:")
        print("- Kaifa's: Aggregates to grantee level, uses censored duration")
        print("- Pipeline: Uses grantee-disaster pairs, excludes incomplete programs")

    return comparison


# =============================================================================
# DOCUMENTATION
# =============================================================================

REPLICATION_NOTES = """
KAIFA'S MODELS REPLICATION ANALYSIS
====================================

This module documents the methodological differences between Kaifa's original
Jupyter notebook analysis and the canonical pipeline.

KAIFA'S APPROACH:
1. Unit of analysis: Grantee (N~38 state, ~40 local)
2. Duration handling: Right-censored (incomplete = current quarter)
3. Variable construction: Mean of quarterly ratios across disasters
4. Model: 3x3 with Timeliness = 1/Duration

PIPELINE APPROACH:
1. Unit of analysis: Grantee-Disaster pair (N=156)
2. Duration handling: Missing for incomplete programs
3. Variable construction: Final cumulative ratios
4. Model: 2x2 with Duration_log

RESULTS COMPARISON:
- Kaifa's manuscript: beta=71.024, p=0.01 (significant)
- Pipeline with Duration: beta~0.32, p>0.05 (not significant)
- This replication: beta~113.65, p<0.001 (significant)

The magnitude difference (71 vs 113) may be due to:
1. Exact ratio calculation method (mean-of-ratios vs final-ratio)
2. Sample composition (which grantees included)
3. Handling of extreme values

RECOMMENDATIONS:
1. For replication/verification: Use this module
2. For new analyses: Use canonical pipeline (more conservative)
3. Report both if defending methodology choices
"""


def print_replication_notes():
    """Print documentation on Kaifa's replication methodology."""
    print(REPLICATION_NOTES)
