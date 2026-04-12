"""
Stage 00b: Standardize QPR Data

Resolves computational artifacts in financial data by computing stable denominators
and standardized velocity measures.

Commands:
    python src/pipeline.py standardize_data

Inputs:
    data_work/qpr_clean.parquet                 - Cleaned QPR data from s00_ingest

Outputs:
    data_work/qpr_standardized.parquet          - Standardized quarterly data with fixed denominators
    data_work/quality/qpr_standardized_report.csv - Standardization quality report

Key Processing Steps:
1. Compute stable denominators (final obligated amount per grantee-disaster)
2. Create standardized ratios using fixed denominators
3. Compute velocity with fixed denominators (eliminates dynamic denominator artifacts)
4. Apply winsorization to handle extreme outliers
5. Add quality assurance flags for data issues
6. Generate standardization quality report
"""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

from config import (
    DATA_WORK_DIR,
    QPR_CLEAN_FILE,
    RATIO_DENOMINATOR_METHOD,
    VELOCITY_WINSOR_PERCENTILES,
    VELOCITY_EXTREME_THRESHOLD,
    VELOCITY_ROLLING_WINDOWS,
)
from stages._io_utils import safe_to_parquet, safe_read_parquet


def compute_stable_denominator(
    group: pd.DataFrame,
    method: str = 'final'
) -> float:
    """
    Compute stable denominator for ratio calculations.

    Parameters
    ----------
    group : pd.DataFrame
        Grantee-disaster group with quarterly data
    method : str
        Method to compute denominator: 'final' (last quarter) or 'max' (maximum)

    Returns
    -------
    float
        Stable denominator value
    """
    obligated_values = group['QPR Fund Obligated $'].dropna()

    if len(obligated_values) == 0:
        return np.nan

    if method == 'final':
        return obligated_values.iloc[-1]
    elif method == 'max':
        return obligated_values.max()
    else:
        raise ValueError(f"Unknown denominator method: {method}")


def standardize_grantee_disaster(
    group: pd.DataFrame,
    denominator_method: str = 'final',
    winsor_percentiles: Tuple[float, float] = (0.01, 0.99),
    velocity_extreme_threshold: float = 100.0,
) -> pd.DataFrame:
    """
    Standardize financial data for a single grantee-disaster.

    Applies fixed denominator approach to eliminate computational artifacts
    from changing obligated amounts.

    Parameters
    ----------
    group : pd.DataFrame
        Grantee-disaster quarterly data
    denominator_method : str
        Method to compute stable denominator ('final' or 'max')
    winsor_percentiles : tuple
        Lower and upper percentiles for winsorization (e.g., (0.01, 0.99))
    velocity_extreme_threshold : float
        Threshold for flagging extreme velocity (pp/quarter)

    Returns
    -------
    pd.DataFrame
        Standardized group with new columns
    """
    group = group.copy()

    # Sort by quarter to ensure temporal order
    # Use QPR_Date if available, otherwise QPR Actual Quarter
    if 'QPR_Date' in group.columns:
        group = group.sort_values('QPR_Date')
    elif 'QPR Actual Quarter' in group.columns:
        group = group.sort_values('QPR Actual Quarter')
    # If no temporal column, proceed without sorting (data should already be ordered)

    # =========================================================================
    # Step 1: Compute stable denominator
    # =========================================================================

    final_obligated = compute_stable_denominator(group, method=denominator_method)
    group['Obligated_Final'] = final_obligated

    # =========================================================================
    # Step 2: Create monotonic "clean" cumulative series
    # =========================================================================
    # These should already be created in s00_ingest, but ensure monotonicity

    if 'QPR Fund Obligated $' in group.columns:
        group['Obligated_Clean'] = group['QPR Fund Obligated $'].clip(lower=0).cummax()

    if 'QPR Fund Disbursed $' in group.columns:
        group['Disbursed_Clean'] = group['QPR Fund Disbursed $'].clip(lower=0).cummax()

    if 'QPR Fund Expended $' in group.columns:
        group['Expended_Clean'] = group['QPR Fund Expended $'].clip(lower=0).cummax()

    # =========================================================================
    # Step 3: Compute ratios with fixed denominators
    # =========================================================================

    if final_obligated > 0 and not np.isnan(final_obligated):
        group['Ratio_Disbursed_Std'] = group['Disbursed_Clean'] / final_obligated
        group['Ratio_Expended_Std'] = group['Expended_Clean'] / final_obligated
    else:
        group['Ratio_Disbursed_Std'] = np.nan
        group['Ratio_Expended_Std'] = np.nan

    # =========================================================================
    # Step 4: Compute velocity with fixed denominators (RAW)
    # =========================================================================
    # Velocity = quarter-to-quarter change in standardized ratio

    group['Velocity_Disb_Std'] = group['Ratio_Disbursed_Std'].diff()
    group['Velocity_Exp_Std'] = group['Ratio_Expended_Std'].diff()

    # Convert to percentage points per quarter
    group['Velocity_Disb_Std_pp'] = group['Velocity_Disb_Std'] * 100
    group['Velocity_Exp_Std_pp'] = group['Velocity_Exp_Std'] * 100

    # Velocity index (mean of disbursement and expenditure)
    group['Velocity_Index_Std_pp'] = (
        group['Velocity_Disb_Std_pp'] + group['Velocity_Exp_Std_pp']
    ) / 2

    # =========================================================================
    # Step 5: Winsorization (done at dataset level, not group level)
    # =========================================================================
    # Placeholder columns - will be filled after aggregating all groups
    group['Velocity_Disb_Std_pp_winsor'] = group['Velocity_Disb_Std_pp']
    group['Velocity_Exp_Std_pp_winsor'] = group['Velocity_Exp_Std_pp']
    group['Velocity_Index_Std_pp_winsor'] = group['Velocity_Index_Std_pp']

    # =========================================================================
    # Step 6: Rolling and cumulative velocity
    # =========================================================================

    # Use configured rolling windows (default: [2, 4])
    rolling_windows = VELOCITY_ROLLING_WINDOWS if 'VELOCITY_ROLLING_WINDOWS' in dir() else [2, 4]

    for window in rolling_windows:
        group[f'Velocity_Disb_Std_pp_roll{window}'] = (
            group['Velocity_Disb_Std_pp'].rolling(window, min_periods=1).mean()
        )
        group[f'Velocity_Exp_Std_pp_roll{window}'] = (
            group['Velocity_Exp_Std_pp'].rolling(window, min_periods=1).mean()
        )
        group[f'Velocity_Index_Std_pp_roll{window}'] = (
            group['Velocity_Index_Std_pp'].rolling(window, min_periods=1).mean()
        )

    # Cumulative (expanding mean from program start)
    group['Velocity_Disb_Std_pp_cum'] = (
        group['Velocity_Disb_Std_pp'].expanding(min_periods=1).mean()
    )
    group['Velocity_Exp_Std_pp_cum'] = (
        group['Velocity_Exp_Std_pp'].expanding(min_periods=1).mean()
    )
    group['Velocity_Index_Std_pp_cum'] = (
        group['Velocity_Index_Std_pp'].expanding(min_periods=1).mean()
    )

    # =========================================================================
    # Step 7: QA Flags
    # =========================================================================

    # Flag 1: Extreme velocity (>threshold pp/quarter)
    group['QA_Extreme_Velocity'] = (
        (group['Velocity_Disb_Std_pp'].abs() > velocity_extreme_threshold) |
        (group['Velocity_Exp_Std_pp'].abs() > velocity_extreme_threshold)
    )

    # Flag 2: Obligated amount jumped >50% quarter-over-quarter
    if 'QPR Fund Obligated $' in group.columns:
        obligated_pct_change = group['QPR Fund Obligated $'].pct_change().abs()
        group['QA_Obligated_Jump'] = obligated_pct_change > 0.5
    else:
        group['QA_Obligated_Jump'] = False

    # Flag 3: Negative quarterly flows (retroactive adjustments)
    group['QA_Negative_Adjustment'] = False
    if 'QPR Fund Obligated Q $' in group.columns:
        group['QA_Negative_Adjustment'] |= group['QPR Fund Obligated Q $'] < 0
    if 'QPR Fund Disbursed Q $' in group.columns:
        group['QA_Negative_Adjustment'] |= group['QPR Fund Disbursed Q $'] < 0
    if 'QPR Fund Expended Q $' in group.columns:
        group['QA_Negative_Adjustment'] |= group['QPR Fund Expended Q $'] < 0

    return group


def apply_winsorization(
    df: pd.DataFrame,
    columns: list,
    percentiles: Tuple[float, float] = (0.01, 0.99)
) -> pd.DataFrame:
    """
    Apply winsorization to specified columns across the full dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    columns : list
        Columns to winsorize
    percentiles : tuple
        Lower and upper percentiles (e.g., (0.01, 0.99) for 1%/99%)

    Returns
    -------
    pd.DataFrame
        Dataset with winsorized columns
    """
    df = df.copy()

    if percentiles is None:
        return df

    lower, upper = percentiles
    limits = (lower, 1 - upper)  # scipy winsorize expects (lower_limit, upper_limit)

    for col in columns:
        if col not in df.columns:
            continue

        # Get non-null values
        values = df[col].values
        non_null_mask = ~pd.isna(values)

        if non_null_mask.sum() == 0:
            continue

        # Winsorize non-null values
        winsor_col = f'{col}_winsor'
        df[winsor_col] = df[col].copy()
        df.loc[non_null_mask, winsor_col] = winsorize(
            values[non_null_mask],
            limits=limits
        )

    return df


def generate_standardization_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate quality report for standardized data.

    Parameters
    ----------
    df : pd.DataFrame
        Standardized quarterly data

    Returns
    -------
    pd.DataFrame
        Quality metrics summary
    """
    n_quarters = len(df)
    n_grantee_disasters = df.groupby(['Grantee', 'Disaster Type']).ngroups

    # Compute statistics
    report_rows = []

    # Overall statistics
    report_rows.append({
        'Metric': 'n_grantee_disasters',
        'Value': n_grantee_disasters,
        'Description': 'Number of grantee-disaster pairs'
    })

    report_rows.append({
        'Metric': 'n_quarters',
        'Value': n_quarters,
        'Description': 'Total quarterly observations'
    })

    # QA flags
    if 'QA_Extreme_Velocity' in df.columns:
        pct_extreme = (df['QA_Extreme_Velocity'].sum() / n_quarters) * 100
        report_rows.append({
            'Metric': 'pct_extreme_velocity',
            'Value': pct_extreme,
            'Description': '% observations with extreme velocity (>100 pp/quarter)'
        })

    if 'QA_Obligated_Jump' in df.columns:
        pct_jump = (df['QA_Obligated_Jump'].sum() / n_quarters) * 100
        report_rows.append({
            'Metric': 'pct_obligated_jumps',
            'Value': pct_jump,
            'Description': '% observations with >50% obligated jump'
        })

    if 'QA_Negative_Adjustment' in df.columns:
        pct_negative = (df['QA_Negative_Adjustment'].sum() / n_quarters) * 100
        report_rows.append({
            'Metric': 'pct_negative_adjustments',
            'Value': pct_negative,
            'Description': '% observations with negative quarterly flows'
        })

    # Velocity statistics (raw)
    for vel_col in ['Velocity_Disb_Std_pp', 'Velocity_Exp_Std_pp', 'Velocity_Index_Std_pp']:
        if vel_col not in df.columns:
            continue

        values = df[vel_col].dropna()
        if len(values) == 0:
            continue

        report_rows.append({
            'Metric': f'{vel_col}_mean',
            'Value': values.mean(),
            'Description': f'Mean {vel_col} (raw)'
        })

        report_rows.append({
            'Metric': f'{vel_col}_std',
            'Value': values.std(),
            'Description': f'Std {vel_col} (raw)'
        })

        report_rows.append({
            'Metric': f'{vel_col}_min',
            'Value': values.min(),
            'Description': f'Min {vel_col} (raw)'
        })

        report_rows.append({
            'Metric': f'{vel_col}_max',
            'Value': values.max(),
            'Description': f'Max {vel_col} (raw)'
        })

    # Velocity statistics (winsorized)
    for vel_col in ['Velocity_Disb_Std_pp_winsor', 'Velocity_Exp_Std_pp_winsor', 'Velocity_Index_Std_pp_winsor']:
        if vel_col not in df.columns:
            continue

        values = df[vel_col].dropna()
        if len(values) == 0:
            continue

        report_rows.append({
            'Metric': f'{vel_col}_mean',
            'Value': values.mean(),
            'Description': f'Mean {vel_col} (winsorized)'
        })

        report_rows.append({
            'Metric': f'{vel_col}_std',
            'Value': values.std(),
            'Description': f'Std {vel_col} (winsorized)'
        })

    return pd.DataFrame(report_rows)


def standardize_qpr_data(
    denominator_method: str = None,
    winsor_percentiles: Tuple[float, float] = None,
    velocity_extreme_threshold: float = None,
) -> None:
    """
    Main function to standardize QPR data.

    Reads cleaned QPR data, applies fixed denominator approach to eliminate
    computational artifacts, and outputs standardized data with QA flags.

    Parameters
    ----------
    denominator_method : str
        Method to compute stable denominator ('final' or 'max')
    winsor_percentiles : tuple
        Percentiles for winsorization (e.g., (0.01, 0.99) for 1%/99%)
        Set to None to disable winsorization
    velocity_extreme_threshold : float
        Threshold for flagging extreme velocity (pp/quarter)
    """
    # Use config defaults if not specified
    if denominator_method is None:
        denominator_method = RATIO_DENOMINATOR_METHOD
    if winsor_percentiles is None:
        winsor_percentiles = VELOCITY_WINSOR_PERCENTILES
    if velocity_extreme_threshold is None:
        velocity_extreme_threshold = VELOCITY_EXTREME_THRESHOLD

    print("=" * 80)
    print("Stage 00b: Standardize QPR Data")
    print("=" * 80)
    print()

    # =========================================================================
    # Load cleaned QPR data
    # =========================================================================

    input_path = DATA_WORK_DIR / QPR_CLEAN_FILE
    print(f"Loading cleaned QPR data: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned QPR data not found: {input_path}\n"
            "Run 'python src/pipeline.py ingest_data' first."
        )

    qpr_clean = safe_read_parquet(input_path)
    print(f"Loaded {len(qpr_clean):,} quarterly observations")
    print(f"  {qpr_clean['Grantee'].nunique()} grantees")
    print(f"  {qpr_clean.groupby(['Grantee', 'Disaster Type']).ngroups} grantee-disaster pairs")
    print()

    # =========================================================================
    # Apply standardization to each grantee-disaster
    # =========================================================================

    print("Applying standardization...")
    print(f"  Denominator method: {denominator_method}")
    print(f"  Winsorization: {winsor_percentiles if winsor_percentiles else 'disabled'}")
    print(f"  Extreme velocity threshold: {velocity_extreme_threshold} pp/quarter")
    print()

    qpr_std = qpr_clean.groupby(['Grantee', 'Disaster Type'], group_keys=False).apply(
        lambda g: standardize_grantee_disaster(
            g,
            denominator_method=denominator_method,
            winsor_percentiles=winsor_percentiles,
            velocity_extreme_threshold=velocity_extreme_threshold,
        ),
        include_groups=False
    )

    # Reset index to avoid multi-index issues
    qpr_std = qpr_std.reset_index(drop=True)

    print(f"Standardization complete: {len(qpr_std):,} observations")
    print()

    # =========================================================================
    # Apply winsorization at dataset level
    # =========================================================================

    if winsor_percentiles is not None:
        print("Applying winsorization at dataset level...")

        velocity_cols = [
            'Velocity_Disb_Std_pp',
            'Velocity_Exp_Std_pp',
            'Velocity_Index_Std_pp',
        ]

        qpr_std = apply_winsorization(qpr_std, velocity_cols, winsor_percentiles)

        print(f"  Winsorized at {winsor_percentiles[0]*100:.1f}% / {winsor_percentiles[1]*100:.1f}% percentiles")
        print()

    # =========================================================================
    # Generate quality report
    # =========================================================================

    print("Generating standardization quality report...")

    report = generate_standardization_report(qpr_std)

    # Save report
    quality_dir = DATA_WORK_DIR / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    report_path = quality_dir / "qpr_standardized_report.csv"
    report.to_csv(report_path, index=False)
    print(f"  Saved report: {report_path}")
    print()

    # Display key metrics
    print("Key quality metrics:")
    for _, row in report.iterrows():
        if row['Metric'].startswith('pct_') or row['Metric'].startswith('n_'):
            print(f"  {row['Metric']}: {row['Value']:.2f}")
    print()

    # =========================================================================
    # Save standardized data
    # =========================================================================

    output_path = DATA_WORK_DIR / "qpr_standardized.parquet"
    print(f"Saving standardized data: {output_path}")

    safe_to_parquet(qpr_std, output_path)

    print(f"  Saved {len(qpr_std):,} observations")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()

    # =========================================================================
    # Summary
    # =========================================================================

    print("=" * 80)
    print("Standardization Complete")
    print("=" * 80)
    print()
    print("Outputs:")
    print(f"  - {output_path}")
    print(f"  - {report_path}")
    print()

    # Show column summary
    new_cols = [c for c in qpr_std.columns if c not in qpr_clean.columns]
    print(f"Added {len(new_cols)} new columns:")
    for col in sorted(new_cols)[:20]:  # Show first 20
        print(f"  - {col}")
    if len(new_cols) > 20:
        print(f"  ... and {len(new_cols) - 20} more")
    print()


if __name__ == "__main__":
    # Run with default configuration
    standardize_qpr_data()
