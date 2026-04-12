"""
Stage 01b: Build Standardized Features

Aggregate standardized quarterly data to grantee-disaster level with consistent
feature engineering. Uses pre-computed standardized velocity from s00b_standardize.

Commands:
    python src/pipeline.py build_features_std

Inputs:
    data_work/qpr_standardized.parquet  - Standardized quarterly data from s00b
    data_work/panel.parquet             - Base panel from s01_link

Outputs:
    data_work/panel_features_std.parquet  - Grantee-disaster panel with standardized features

Key Features:
1. Standardized velocity measures (aggregated from s00b)
2. Timeliness metrics (duration to thresholds)
3. Experience indicators
4. Government type and covariates
5. Capacity indices and interactions
"""

from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

from config import (
    DATA_WORK_DIR,
    COMPLETION_THRESHOLD,
    DURATION_THRESHOLDS,
    EARLY_VELOCITY_WINDOWS,
    FIXED_VELOCITY_WINDOWS_MONTHS,
    RATIO_INTERACTION_QUANTILES,
)
from stages._io_utils import safe_to_parquet, safe_read_parquet
from utils.quarterly_panel import collapse_to_quarterly_panel

# Import from existing modules
from capacity_sem.features.timeliness import (
    calculate_duration_of_completion,
    calculate_multi_threshold_duration,
)
from capacity_sem.features.experience_indicators import (
    compute_grantee_experience,
    build_experience_dataset,
)


def load_standardized_qpr() -> pd.DataFrame:
    """Load standardized quarterly QPR data from s00b."""
    std_path = DATA_WORK_DIR / "qpr_standardized.parquet"
    if not std_path.exists():
        raise FileNotFoundError(
            f"Standardized QPR data not found: {std_path}\n"
            "Run 'python src/pipeline.py standardize_data' first."
        )
    return safe_read_parquet(std_path)


def load_base_panel() -> pd.DataFrame:
    """Load base panel from s01_link."""
    panel_path = DATA_WORK_DIR / "panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Base panel not found: {panel_path}\n"
            "Run 'python src/pipeline.py build_panel' first."
        )
    return safe_read_parquet(panel_path)


def aggregate_standardized_velocity(
    qpr_std: pd.DataFrame,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type'
) -> pd.DataFrame:
    """
    Aggregate standardized velocity measures to grantee-disaster level.

    Uses pre-computed winsorized velocity from s00b_standardize as the
    primary velocity measure.

    Parameters
    ----------
    qpr_std : pd.DataFrame
        Standardized quarterly data with velocity columns
    grantee_col : str
        Column name for grantee identifier
    disaster_col : str
        Column name for disaster identifier

    Returns
    -------
    pd.DataFrame
        Grantee-disaster aggregated velocity features
    """
    velocity_features = []

    for (grantee, disaster), group in qpr_std.groupby([grantee_col, disaster_col]):
        record = {
            grantee_col: grantee,
            disaster_col: disaster,
        }

        # =====================================================================
        # Static velocity (full program average)
        # =====================================================================
        # Use winsorized velocity as primary measure (outliers already handled)

        for vel_type in ['Disb', 'Exp', 'Index']:
            vel_col = f'Velocity_{vel_type}_Std_pp_winsor'

            if vel_col not in group.columns:
                continue

            vel_values = group[vel_col].dropna()

            if len(vel_values) == 0:
                record[f'Disbursement_Velocity' if vel_type == 'Disb' else f'Expenditure_Velocity' if vel_type == 'Exp' else 'Capacity_Velocity_Index'] = np.nan
                record[f'Disbursement_Velocity_median' if vel_type == 'Disb' else f'Expenditure_Velocity_median' if vel_type == 'Exp' else 'Capacity_Velocity_Index_median'] = np.nan
                record[f'Disbursement_Velocity_std' if vel_type == 'Disb' else f'Expenditure_Velocity_std' if vel_type == 'Exp' else 'Capacity_Velocity_Index_std'] = np.nan
                continue

            # Primary measure: mean (for consistency with existing code)
            if vel_type == 'Disb':
                record['Disbursement_Velocity'] = vel_values.mean()
                record['Disbursement_Velocity_median'] = vel_values.median()
                record['Disbursement_Velocity_std'] = vel_values.std()
                record['Disbursement_Velocity_pp'] = vel_values.mean()  # Already in pp
            elif vel_type == 'Exp':
                record['Expenditure_Velocity'] = vel_values.mean()
                record['Expenditure_Velocity_median'] = vel_values.median()
                record['Expenditure_Velocity_std'] = vel_values.std()
                record['Expenditure_Velocity_pp'] = vel_values.mean()  # Already in pp
            elif vel_type == 'Index':
                record['Capacity_Velocity_Index'] = vel_values.mean()
                record['Capacity_Velocity_Index_median'] = vel_values.median()
                record['Capacity_Velocity_Index_std'] = vel_values.std()
                record['Capacity_Velocity_Index_pp'] = vel_values.mean()  # Already in pp

        # =====================================================================
        # Early-window velocity (first N quarters)
        # =====================================================================

        n_quarters = len(group)

        for window in EARLY_VELOCITY_WINDOWS:
            if n_quarters < window:
                record[f'Disbursement_Velocity_early_{window}q'] = np.nan
                record[f'Expenditure_Velocity_early_{window}q'] = np.nan
                record[f'Disbursement_Velocity_early_{window}q_pp'] = np.nan
                record[f'Expenditure_Velocity_early_{window}q_pp'] = np.nan
                record[f'Capacity_Velocity_Index_early_{window}q_pp'] = np.nan
                continue

            window_group = group.iloc[:window]

            # Disbursement velocity
            if 'Velocity_Disb_Std_pp_winsor' in window_group.columns:
                vel_values = window_group['Velocity_Disb_Std_pp_winsor'].dropna()
                record[f'Disbursement_Velocity_early_{window}q'] = vel_values.mean() if len(vel_values) > 0 else np.nan
                record[f'Disbursement_Velocity_early_{window}q_pp'] = vel_values.mean() if len(vel_values) > 0 else np.nan
            else:
                record[f'Disbursement_Velocity_early_{window}q'] = np.nan
                record[f'Disbursement_Velocity_early_{window}q_pp'] = np.nan

            # Expenditure velocity
            if 'Velocity_Exp_Std_pp_winsor' in window_group.columns:
                vel_values = window_group['Velocity_Exp_Std_pp_winsor'].dropna()
                record[f'Expenditure_Velocity_early_{window}q'] = vel_values.mean() if len(vel_values) > 0 else np.nan
                record[f'Expenditure_Velocity_early_{window}q_pp'] = vel_values.mean() if len(vel_values) > 0 else np.nan
            else:
                record[f'Expenditure_Velocity_early_{window}q'] = np.nan
                record[f'Expenditure_Velocity_early_{window}q_pp'] = np.nan

            # Velocity index
            if 'Velocity_Index_Std_pp_winsor' in window_group.columns:
                vel_values = window_group['Velocity_Index_Std_pp_winsor'].dropna()
                record[f'Capacity_Velocity_Index_early_{window}q_pp'] = vel_values.mean() if len(vel_values) > 0 else np.nan
            else:
                record[f'Capacity_Velocity_Index_early_{window}q_pp'] = np.nan

        # =====================================================================
        # Fixed calendar window velocity (first N months)
        # =====================================================================

        if 'QPR_Date' in group.columns and n_quarters > 1:
            start_date = group['QPR_Date'].min()

            for months in FIXED_VELOCITY_WINDOWS_MONTHS:
                end_date = start_date + pd.DateOffset(months=months)
                window_group = group[group['QPR_Date'] <= end_date]

                if len(window_group) < 2:
                    record[f'Disbursement_Velocity_fixed_{months}m'] = np.nan
                    record[f'Expenditure_Velocity_fixed_{months}m'] = np.nan
                    record[f'Disbursement_Velocity_fixed_{months}m_pp'] = np.nan
                    record[f'Expenditure_Velocity_fixed_{months}m_pp'] = np.nan
                    record[f'Capacity_Velocity_Index_fixed_{months}m_pp'] = np.nan
                    continue

                # Disbursement velocity
                if 'Velocity_Disb_Std_pp_winsor' in window_group.columns:
                    vel_values = window_group['Velocity_Disb_Std_pp_winsor'].dropna()
                    record[f'Disbursement_Velocity_fixed_{months}m'] = vel_values.mean() if len(vel_values) > 0 else np.nan
                    record[f'Disbursement_Velocity_fixed_{months}m_pp'] = vel_values.mean() if len(vel_values) > 0 else np.nan
                else:
                    record[f'Disbursement_Velocity_fixed_{months}m'] = np.nan
                    record[f'Disbursement_Velocity_fixed_{months}m_pp'] = np.nan

                # Expenditure velocity
                if 'Velocity_Exp_Std_pp_winsor' in window_group.columns:
                    vel_values = window_group['Velocity_Exp_Std_pp_winsor'].dropna()
                    record[f'Expenditure_Velocity_fixed_{months}m'] = vel_values.mean() if len(vel_values) > 0 else np.nan
                    record[f'Expenditure_Velocity_fixed_{months}m_pp'] = vel_values.mean() if len(vel_values) > 0 else np.nan
                else:
                    record[f'Expenditure_Velocity_fixed_{months}m'] = np.nan
                    record[f'Expenditure_Velocity_fixed_{months}m_pp'] = np.nan

                # Velocity index
                if 'Velocity_Index_Std_pp_winsor' in window_group.columns:
                    vel_values = window_group['Velocity_Index_Std_pp_winsor'].dropna()
                    record[f'Capacity_Velocity_Index_fixed_{months}m_pp'] = vel_values.mean() if len(vel_values) > 0 else np.nan
                else:
                    record[f'Capacity_Velocity_Index_fixed_{months}m_pp'] = np.nan

        # =====================================================================
        # QA Flag Aggregation (for measurement validation)
        # =====================================================================

        # Count of extreme velocity observations
        if 'QA_Extreme_Velocity' in group.columns:
            record['Flag_Count_Extreme_Velocity'] = group['QA_Extreme_Velocity'].sum()
            record['Flag_Proportion_Extreme'] = group['QA_Extreme_Velocity'].mean()
        else:
            record['Flag_Count_Extreme_Velocity'] = 0
            record['Flag_Proportion_Extreme'] = 0.0

        # Count of obligated jumps (>10% quarterly change)
        if 'QA_Obligated_Jump' in group.columns:
            record['Flag_Count_Obligated_Jump'] = group['QA_Obligated_Jump'].sum()
            record['Flag_Proportion_Obligated_Jump'] = group['QA_Obligated_Jump'].mean()
        else:
            record['Flag_Count_Obligated_Jump'] = 0
            record['Flag_Proportion_Obligated_Jump'] = 0.0

        # Count of negative adjustments (decreases in cumulative series)
        if 'QA_Negative_Adjustment' in group.columns:
            record['Flag_Count_Negative_Adjustment'] = group['QA_Negative_Adjustment'].sum()
            record['Flag_Proportion_Negative_Adjustment'] = group['QA_Negative_Adjustment'].mean()
        else:
            record['Flag_Count_Negative_Adjustment'] = 0
            record['Flag_Proportion_Negative_Adjustment'] = 0.0

        # High-flag program indicator (placeholder, will be computed after aggregation)
        # This is recalculated after aggregation using Q3 thresholds
        record['QA_High_Flag_Program_Temp'] = (
            (record['Flag_Count_Extreme_Velocity'] > 2) |
            (record['Flag_Count_Obligated_Jump'] > 2)
        )

        velocity_features.append(record)

    return pd.DataFrame(velocity_features)


def compute_stage_lags(
    qpr_std: pd.DataFrame = None,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type'
) -> pd.DataFrame:
    """
    Compute lag between obligate→disburse→expend stages.

    Identifies bottlenecks in the administrative pipeline by measuring:
    - Time lag from first obligated to first disbursed
    - Time lag from first disbursed to first expended
    - Overall pipeline duration
    - Stage-specific efficiency ratios

    Parameters
    ----------
    qpr_std : pd.DataFrame, optional
        Not used - function loads qpr_quarterly.parquet internally
    grantee_col : str
        Column name for grantee identifier
    disaster_col : str
        Column name for disaster identifier

    Returns
    -------
    pd.DataFrame
        Stage lag features at grantee-disaster level
    """
    # Load quarterly aggregated data (grantee-disaster-quarter level)
    # This avoids activity-level ratio artifacts
    qpr_qtr_path = DATA_WORK_DIR / "qpr_quarterly.parquet"
    if not qpr_qtr_path.exists():
        warnings.warn(f"qpr_quarterly.parquet not found at {qpr_qtr_path}. Run ingest_data first.")
        return pd.DataFrame()

    qpr_qtr = safe_read_parquet(qpr_qtr_path)

    stage_lag_features = []

    for (grantee, disaster), group in qpr_qtr.groupby([grantee_col, disaster_col]):
        # Sort by date to ensure chronological order
        group = group.sort_values('QPR_Date')

        # Create quarter index (0-based from first quarter)
        group = group.copy()
        group['_quarter_idx'] = range(len(group))

        record = {
            grantee_col: grantee,
            disaster_col: disaster,
        }

        # Get cumulative series (use Clean columns from quarterly rollup)
        obligated_series = group['QPR Fund Obligated Clean $']
        disbursed_series = group['QPR Fund Disbursed Clean $']
        expended_series = group['QPR Fund Expended Clean $']

        # Find first quarter with Obligated > 0
        first_obl_idx = None
        if (obligated_series > 0).any():
            first_obl_idx = group[obligated_series > 0]['_quarter_idx'].min()

        # Find first quarter with Disbursed > 0
        first_disb_idx = None
        if (disbursed_series > 0).any():
            first_disb_idx = group[disbursed_series > 0]['_quarter_idx'].min()

        # Find first quarter with Expended > 0
        first_exp_idx = None
        if (expended_series > 0).any():
            first_exp_idx = group[expended_series > 0]['_quarter_idx'].min()

        # Compute lags
        if pd.notna(first_obl_idx) and pd.notna(first_disb_idx):
            record['Lag_Obligate_to_Disburse'] = first_disb_idx - first_obl_idx
        else:
            record['Lag_Obligate_to_Disburse'] = np.nan

        if pd.notna(first_disb_idx) and pd.notna(first_exp_idx):
            record['Lag_Disburse_to_Expend'] = first_exp_idx - first_disb_idx
        else:
            record['Lag_Disburse_to_Expend'] = np.nan

        if pd.notna(first_obl_idx) and pd.notna(first_exp_idx):
            record['Lag_Total_Pipeline'] = first_exp_idx - first_obl_idx
        else:
            record['Lag_Total_Pipeline'] = np.nan

        # Stage efficiency ratios (use final cumulative ratios from aggregated quarterly data)
        # Stage 1: Disbursed / Obligated (final ratio)
        if obligated_series.iloc[-1] > 0:
            record['Stage1_Efficiency'] = disbursed_series.iloc[-1] / obligated_series.iloc[-1]
        else:
            record['Stage1_Efficiency'] = np.nan

        # Stage 2: Expended / Disbursed (final ratio)
        if disbursed_series.iloc[-1] > 0:
            record['Stage2_Efficiency'] = expended_series.iloc[-1] / disbursed_series.iloc[-1]
        else:
            record['Stage2_Efficiency'] = np.nan

        stage_lag_features.append(record)

    return pd.DataFrame(stage_lag_features)


def compute_phase_specific_velocity(
    qpr_std: pd.DataFrame,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type'
) -> pd.DataFrame:
    """
    Compute velocity in early/mid/late program phases.

    Identifies temporal dynamics by measuring spending acceleration in different
    phases of program implementation (launch, recovery, closeout).

    Parameters
    ----------
    qpr_std : pd.DataFrame
        Standardized quarterly data with velocity columns
    grantee_col : str
        Column name for grantee identifier
    disaster_col : str
        Column name for disaster identifier

    Returns
    -------
    pd.DataFrame
        Phase-specific velocity features at grantee-disaster level
    """
    phase_velocity_features = []

    for (grantee, disaster), group in qpr_std.groupby([grantee_col, disaster_col]):
        # Sort by date
        group = group.sort_values('QPR_Date')

        record = {
            grantee_col: grantee,
            disaster_col: disaster,
        }

        # Get velocity series
        velocity_col = 'Velocity_Exp_Std_pp_winsor'  # Winsorized expenditure velocity
        if velocity_col not in group.columns:
            # Fallback to other velocity columns
            for alt_col in ['Expenditure_Velocity_pp_Std', 'Velocity_Exp_Std_pp']:
                if alt_col in group.columns:
                    velocity_col = alt_col
                    break

        if velocity_col in group.columns:
            velocity_series = group[velocity_col]
            n = len(group)

            # Divide into thirds
            third = n // 3

            if third > 0:
                # Early phase (first third)
                early = velocity_series.iloc[:third]
                record['Velocity_Early'] = early.mean() if len(early) > 0 else np.nan
                record['Velocity_Early_median'] = early.median() if len(early) > 0 else np.nan

                # Mid phase (second third)
                mid = velocity_series.iloc[third:2*third]
                record['Velocity_Mid'] = mid.mean() if len(mid) > 0 else np.nan
                record['Velocity_Mid_median'] = mid.median() if len(mid) > 0 else np.nan

                # Late phase (final third)
                late = velocity_series.iloc[2*third:]
                record['Velocity_Late'] = late.mean() if len(late) > 0 else np.nan
                record['Velocity_Late_median'] = late.median() if len(late) > 0 else np.nan

                # Velocity acceleration (late - early)
                if pd.notna(record['Velocity_Late']) and pd.notna(record['Velocity_Early']):
                    record['Velocity_Acceleration'] = record['Velocity_Late'] - record['Velocity_Early']
                else:
                    record['Velocity_Acceleration'] = np.nan
            else:
                # Too few quarters to divide
                record['Velocity_Early'] = velocity_series.mean()
                record['Velocity_Early_median'] = velocity_series.median()
                record['Velocity_Mid'] = np.nan
                record['Velocity_Mid_median'] = np.nan
                record['Velocity_Late'] = np.nan
                record['Velocity_Late_median'] = np.nan
                record['Velocity_Acceleration'] = np.nan
        else:
            # No velocity column found
            record['Velocity_Early'] = np.nan
            record['Velocity_Early_median'] = np.nan
            record['Velocity_Mid'] = np.nan
            record['Velocity_Mid_median'] = np.nan
            record['Velocity_Late'] = np.nan
            record['Velocity_Late_median'] = np.nan
            record['Velocity_Acceleration'] = np.nan

        phase_velocity_features.append(record)

    return pd.DataFrame(phase_velocity_features)


def compute_timeliness_features_std(
    qpr_std: pd.DataFrame,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type'
) -> pd.DataFrame:
    """
    Compute timeliness features from standardized quarterly data.

    Uses standardized cumulative series to compute duration to multiple
    completion thresholds.

    Parameters
    ----------
    qpr_std : pd.DataFrame
        Standardized quarterly data
    grantee_col : str
        Column name for grantee identifier
    disaster_col : str
        Column name for disaster identifier

    Returns
    -------
    pd.DataFrame
        Timeliness metrics at grantee-disaster level
    """
    timeliness_features = []

    # Determine quarter column name
    quarter_col = 'QPR Actual Quarter' if 'QPR Actual Quarter' in qpr_std.columns else 'Quarter_Index'

    for (grantee, disaster), group in qpr_std.groupby([grantee_col, disaster_col]):
        record = {
            grantee_col: grantee,
            disaster_col: disaster,
        }

        # CRITICAL FIX: Aggregate to quarter-level to handle activity-level data
        # The raw data may have multiple rows per quarter (one per activity)
        # We need to aggregate to get one ratio per quarter
        if quarter_col in group.columns:
            # Sort by quarter
            group = group.sort_values(quarter_col)

            # Aggregate to quarter level - take max ratio per quarter (cumulative)
            # For ratio columns, max makes sense as we want the cumulative progress
            ratio_col = 'Ratio_Expended_Std' if 'Ratio_Expended_Std' in group.columns else None

            if ratio_col:
                # Group by quarter and take max ratio (cumulative progress)
                quarterly_agg = group.groupby(quarter_col).agg({
                    ratio_col: 'max',  # Max ratio in quarter (cumulative)
                }).reset_index()
                quarterly_agg = quarterly_agg.sort_values(quarter_col)
            else:
                quarterly_agg = None

            # Number of unique quarters
            n_quarters = group[quarter_col].nunique()
        else:
            quarterly_agg = None
            n_quarters = len(group)  # Fallback to row count

        record['N_Quarters'] = n_quarters

        # Final values
        if 'Obligated_Final' in group.columns:
            final_obligated = group['Obligated_Final'].iloc[0]  # Constant across group
        elif 'QPR Fund Obligated $' in group.columns:
            final_obligated = group['QPR Fund Obligated $'].iloc[-1]
        else:
            final_obligated = 0

        if 'Expended_Clean' in group.columns:
            final_expended = group['Expended_Clean'].iloc[-1]
        elif 'QPR Fund Expended $' in group.columns:
            final_expended = group['QPR Fund Expended $'].iloc[-1]
        else:
            final_expended = 0

        # Completion percentage
        if final_obligated > 0:
            completion_pct = final_expended / final_obligated
        else:
            completion_pct = 0

        record['Completion_Pct'] = completion_pct
        record['Ratio_obligated_funds_fully_expended'] = completion_pct

        # Multi-threshold duration analysis
        for threshold in DURATION_THRESHOLDS:
            # Use quarterly-aggregated ratios if available
            if quarterly_agg is not None and 'Ratio_Expended_Std' in quarterly_agg.columns:
                ratio_series = quarterly_agg['Ratio_Expended_Std'].values
            elif 'Ratio_Expended_Std' in group.columns:
                # Fallback: use first value per quarter by deduplicating
                if quarter_col in group.columns:
                    deduped = group.drop_duplicates(subset=[quarter_col], keep='last')
                    ratio_series = deduped['Ratio_Expended_Std'].values
                else:
                    ratio_series = group['Ratio_Expended_Std'].values
            elif final_obligated > 0 and 'Expended_Clean' in group.columns:
                if quarter_col in group.columns:
                    deduped = group.drop_duplicates(subset=[quarter_col], keep='last')
                    ratio_series = deduped['Expended_Clean'].values / final_obligated
                else:
                    ratio_series = group['Expended_Clean'].values / final_obligated
            else:
                ratio_series = None

            if ratio_series is not None:
                # Find first quarter where ratio >= threshold
                quarters_to_threshold = np.nan
                for q_idx, ratio_val in enumerate(ratio_series):
                    if ratio_val >= threshold:
                        quarters_to_threshold = q_idx + 1  # 1-indexed
                        break

                record[f'Duration_{int(threshold*100)}pct'] = quarters_to_threshold
            else:
                record[f'Duration_{int(threshold*100)}pct'] = np.nan

        # Primary duration metric (95% threshold for consistency)
        record['Duration'] = record.get(f'Duration_{int(COMPLETION_THRESHOLD*100)}pct', np.nan)

        # Timeliness = 1 / Duration (inverse of time to completion)
        if not np.isnan(record['Duration']) and record['Duration'] > 0:
            record['Timeliness'] = 1.0 / record['Duration']
        else:
            record['Timeliness'] = np.nan

        timeliness_features.append(record)

    return pd.DataFrame(timeliness_features)


def add_capacity_ratios(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add capacity ratio measures from standardized or legacy columns.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with aggregated features

    Returns
    -------
    pd.DataFrame
        Panel with capacity ratio columns
    """
    # These should come from base panel (s01_link), but add if missing

    if 'Ratio_disbursed_to_obligated' not in panel.columns:
        # Calculate from available columns
        if 'Disbursed_Final' in panel.columns and 'Obligated_Final' in panel.columns:
            obligated = panel['Obligated_Final']
            disbursed = panel['Disbursed_Final']
            panel['Ratio_disbursed_to_obligated'] = disbursed / obligated.replace(0, np.nan)
        else:
            panel['Ratio_disbursed_to_obligated'] = np.nan

    if 'Ratio_expended_to_disbursed' not in panel.columns:
        if 'Expended_Final' in panel.columns and 'Disbursed_Final' in panel.columns:
            disbursed = panel['Disbursed_Final']
            expended = panel['Expended_Final']
            panel['Ratio_expended_to_disbursed'] = expended / disbursed.replace(0, np.nan)
        else:
            panel['Ratio_expended_to_disbursed'] = np.nan

    return panel


def add_scaled_velocity_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add scaled (z-score) versions of velocity features.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with velocity features

    Returns
    -------
    pd.DataFrame
        Panel with scaled velocity columns
    """
    velocity_cols = [
        'Disbursement_Velocity_pp',
        'Expenditure_Velocity_pp',
        'Capacity_Velocity_Index_pp',
    ]

    # Add early-window variants
    for window in EARLY_VELOCITY_WINDOWS:
        velocity_cols.append(f'Disbursement_Velocity_early_{window}q_pp')
        velocity_cols.append(f'Expenditure_Velocity_early_{window}q_pp')
        velocity_cols.append(f'Capacity_Velocity_Index_early_{window}q_pp')

    # Add fixed-window variants
    for months in FIXED_VELOCITY_WINDOWS_MONTHS:
        velocity_cols.append(f'Disbursement_Velocity_fixed_{months}m_pp')
        velocity_cols.append(f'Expenditure_Velocity_fixed_{months}m_pp')
        velocity_cols.append(f'Capacity_Velocity_Index_fixed_{months}m_pp')

    # Scale each column
    for col in velocity_cols:
        if col not in panel.columns:
            continue

        values = panel[col].values.reshape(-1, 1)
        valid_mask = ~np.isnan(values.flatten())

        if valid_mask.sum() < 2:
            continue

        scaler = StandardScaler()
        scaled_values = np.full_like(values, np.nan)
        scaled_values[valid_mask] = scaler.fit_transform(values[valid_mask])

        scaled_col = col.replace('_pp', '_scaled')
        panel[scaled_col] = scaled_values.flatten()

    return panel


def add_winsorized_velocity_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add winsorized versions of velocity features.

    Note: The velocity features from s00b are already winsorized at the
    quarterly level. This function provides additional winsorization at
    the grantee-disaster aggregated level if needed.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with velocity features

    Returns
    -------
    pd.DataFrame
        Panel with winsorized velocity columns
    """
    from scipy.stats.mstats import winsorize

    velocity_cols = [
        'Disbursement_Velocity_pp',
        'Expenditure_Velocity_pp',
        'Capacity_Velocity_Index_pp',
    ]

    for col in velocity_cols:
        if col not in panel.columns:
            continue

        values = panel[col].values
        valid_mask = ~np.isnan(values)

        if valid_mask.sum() < 2:
            continue

        winsorized_values = np.full_like(values, np.nan)
        winsorized_values[valid_mask] = winsorize(values[valid_mask], limits=(0.01, 0.01))

        winsor_col = col.replace('_pp', '_winsor')
        panel[winsor_col] = winsorized_values

    return panel


def add_capacity_indices(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add composite capacity indices.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with capacity features

    Returns
    -------
    pd.DataFrame
        Panel with capacity index columns
    """
    # Absolute capacity index (log obligated + ratios)
    if all(c in panel.columns for c in ['Log_Obligated', 'Ratio_disbursed_to_obligated', 'Ratio_expended_to_disbursed']):
        # Simple average for interpretability
        panel['Capacity_Absolute_Index'] = (
            panel['Ratio_disbursed_to_obligated'] +
            panel['Ratio_expended_to_disbursed']
        ) / 2

    # Velocity index (already computed from standardized data as average of disb + exp)
    # Just ensure the column exists
    if 'Capacity_Velocity_Index' not in panel.columns:
        if 'Disbursement_Velocity' in panel.columns and 'Expenditure_Velocity' in panel.columns:
            panel['Capacity_Velocity_Index'] = (
                panel['Disbursement_Velocity'] + panel['Expenditure_Velocity']
            ) / 2

    # PCA-based capacity index
    capacity_cols = ['Ratio_disbursed_to_obligated', 'Ratio_expended_to_disbursed']
    if all(c in panel.columns for c in capacity_cols):
        X = panel[capacity_cols].dropna()
        if len(X) > 1:
            pca = PCA(n_components=1)
            pca_values = pca.fit_transform(X)
            panel['Capacity_PCA1'] = np.nan
            panel.loc[X.index, 'Capacity_PCA1'] = pca_values.flatten()

    return panel


def add_capacity_quartiles(
    panel: pd.DataFrame,
    capacity_indices: List[str] = None
) -> pd.DataFrame:
    """
    Add quartile bins and dummy indicators for capacity indices.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with capacity features
    capacity_indices : list, optional
        List of capacity index columns to bin. If None, uses default set.

    Returns
    -------
    pd.DataFrame
        Panel with quartile columns
    """
    if capacity_indices is None:
        capacity_indices = [
            'Capacity_Absolute_Index',
            'Capacity_Velocity_Index',
            'Capacity_Velocity_Index_pp',
            'Capacity_Velocity_Index_winsor',
            'Capacity_Velocity_Index_scaled',
        ]

    for base_col in capacity_indices:
        if base_col not in panel.columns:
            continue

        series = panel[base_col]
        valid = series.dropna()

        if valid.nunique() < 2:
            continue

        try:
            bins = pd.qcut(valid, q=4, labels=False, duplicates='drop') + 1
        except ValueError:
            continue

        quartile_col = f"{base_col}_Quartile"
        panel[quartile_col] = np.nan
        panel.loc[bins.index, quartile_col] = bins.astype(int)

        max_bin = int(bins.max())
        if max_bin < 2:
            continue

        for bin_id in range(2, max_bin + 1):
            dummy_col = f"{base_col}_Q{bin_id}"
            panel[dummy_col] = (panel[quartile_col] == bin_id).astype(int)

    return panel


def add_interaction_terms(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction terms between ratios and velocity.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with capacity features

    Returns
    -------
    pd.DataFrame
        Panel with interaction terms
    """
    # Center variables for interactions
    if 'Ratio_disbursed_to_obligated' in panel.columns:
        panel['Ratio_disbursed_to_obligated_c'] = (
            panel['Ratio_disbursed_to_obligated'] - panel['Ratio_disbursed_to_obligated'].mean()
        )

    if 'Disbursement_Velocity_pp' in panel.columns:
        panel['Disbursement_Velocity_pp_c'] = (
            panel['Disbursement_Velocity_pp'] - panel['Disbursement_Velocity_pp'].mean()
        )

    if 'Capacity_Velocity_Index_pp' in panel.columns:
        panel['Capacity_Velocity_Index_pp_c'] = (
            panel['Capacity_Velocity_Index_pp'] - panel['Capacity_Velocity_Index_pp'].mean()
        )

    # Interaction: ratio × velocity
    if all(c in panel.columns for c in ['Ratio_disbursed_to_obligated_c', 'Disbursement_Velocity_pp_c']):
        panel['Ratio_disbursed_to_obligated_x_Disbursement_Velocity_pp'] = (
            panel['Ratio_disbursed_to_obligated_c'] * panel['Disbursement_Velocity_pp_c']
        )

    if all(c in panel.columns for c in ['Ratio_disbursed_to_obligated_c', 'Capacity_Velocity_Index_pp_c']):
        panel['Ratio_disbursed_to_obligated_x_Capacity_Velocity_Index_pp'] = (
            panel['Ratio_disbursed_to_obligated_c'] * panel['Capacity_Velocity_Index_pp_c']
        )

    # Threshold interactions (high ratio indicator × velocity)
    if 'Ratio_disbursed_to_obligated' in panel.columns:
        ratio_median = panel['Ratio_disbursed_to_obligated'].median()
        panel['Ratio_disbursed_to_obligated_high'] = (
            panel['Ratio_disbursed_to_obligated'] > ratio_median
        ).astype(int)

        # Interactions with threshold indicator
        if 'Disbursement_Velocity_pp_c' in panel.columns:
            panel['Ratio_disbursed_to_obligated_high_x_Disbursement_Velocity_pp'] = (
                panel['Ratio_disbursed_to_obligated_high'] * panel['Disbursement_Velocity_pp_c']
            )

        if 'Capacity_Velocity_Index_pp_c' in panel.columns:
            panel['Ratio_disbursed_to_obligated_high_x_Capacity_Velocity_Index_pp'] = (
                panel['Ratio_disbursed_to_obligated_high'] * panel['Capacity_Velocity_Index_pp_c']
            )

        # Multiple quantile thresholds
        for q in RATIO_INTERACTION_QUANTILES:
            ratio_quantile = panel['Ratio_disbursed_to_obligated'].quantile(q)
            panel[f'Ratio_disbursed_to_obligated_high_q{int(q*100)}'] = (
                panel['Ratio_disbursed_to_obligated'] > ratio_quantile
            ).astype(int)

            # Interactions
            if 'Disbursement_Velocity_pp_c' in panel.columns:
                panel[f'Ratio_disbursed_to_obligated_high_q{int(q*100)}_x_Disbursement_Velocity_pp'] = (
                    panel[f'Ratio_disbursed_to_obligated_high_q{int(q*100)}'] *
                    panel['Disbursement_Velocity_pp_c']
                )

            if 'Capacity_Velocity_Index_pp_c' in panel.columns:
                panel[f'Ratio_disbursed_to_obligated_high_q{int(q*100)}_x_Capacity_Velocity_Index_pp'] = (
                    panel[f'Ratio_disbursed_to_obligated_high_q{int(q*100)}'] *
                    panel['Capacity_Velocity_Index_pp_c']
                )

            # Spline interactions (above threshold value)
            panel[f'Ratio_disbursed_to_obligated_above_q{int(q*100)}'] = (
                panel['Ratio_disbursed_to_obligated'] - ratio_quantile
            ).clip(lower=0)

            if 'Disbursement_Velocity_pp_c' in panel.columns:
                panel[f'Ratio_disbursed_to_obligated_above_q{int(q*100)}_x_Disbursement_Velocity_pp'] = (
                    panel[f'Ratio_disbursed_to_obligated_above_q{int(q*100)}'] *
                    panel['Disbursement_Velocity_pp_c']
                )

            if 'Capacity_Velocity_Index_pp_c' in panel.columns:
                panel[f'Ratio_disbursed_to_obligated_above_q{int(q*100)}_x_Capacity_Velocity_Index_pp'] = (
                    panel[f'Ratio_disbursed_to_obligated_above_q{int(q*100)}'] *
                    panel['Capacity_Velocity_Index_pp_c']
                )

        # Simple spline at median
        ratio_median = panel['Ratio_disbursed_to_obligated'].median()
        panel['Ratio_disbursed_to_obligated_above'] = (
            panel['Ratio_disbursed_to_obligated'] - ratio_median
        ).clip(lower=0)

        if 'Disbursement_Velocity_pp_c' in panel.columns:
            panel['Ratio_disbursed_to_obligated_above_x_Disbursement_Velocity_pp'] = (
                panel['Ratio_disbursed_to_obligated_above'] * panel['Disbursement_Velocity_pp_c']
            )

        if 'Capacity_Velocity_Index_pp_c' in panel.columns:
            panel['Ratio_disbursed_to_obligated_above_x_Capacity_Velocity_Index_pp'] = (
                panel['Ratio_disbursed_to_obligated_above'] * panel['Capacity_Velocity_Index_pp_c']
            )

    return panel


def compute_experience_features(
    qpr_std: pd.DataFrame,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type'
) -> pd.DataFrame:
    """
    Compute experience/learning indicators from quarterly data.

    Uses DRGR_DISASTER_YEARS mapping to determine chronological order.
    Returns panel with Prior_Grant_Count, Prior_Grant_Dollars, etc.

    Parameters
    ----------
    qpr_std : pd.DataFrame
        Standardized quarterly QPR data
    grantee_col : str
        Column name for grantee identifier
    disaster_col : str
        Column name for disaster identifier

    Returns
    -------
    pd.DataFrame
        Experience indicators at grantee-disaster level with columns:
        - Grantee
        - Disaster Type
        - Years_Experience
        - Prior_Grant_Count
        - Prior_Grant_Dollars
        - Experience_Index
    """
    from capacity_sem.data.external_data import DRGR_DISASTER_YEARS
    import warnings

    # Build experience dataset using existing infrastructure
    experience_df = build_experience_dataset(
        df=qpr_std,
        grantee_col=grantee_col,
        disaster_col=disaster_col,
        obligated_col='QPR Fund Obligated $'
    )

    # Rename for consistency
    if 'Disaster_Type' in experience_df.columns:
        experience_df = experience_df.rename(columns={'Disaster_Type': disaster_col})

    # Warn about missing disaster year mappings
    missing_disasters = set()
    for disaster in experience_df[disaster_col].unique():
        if disaster not in DRGR_DISASTER_YEARS:
            missing_disasters.add(disaster)

    if missing_disasters:
        warnings.warn(
            f"Missing DRGR_DISASTER_YEARS mappings for: {missing_disasters}. "
            f"These disasters will default to 2020 for experience calculations."
        )

    return experience_df


def add_survival_covariates(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add survival analysis covariates to panel.

    Creates:
    - Government_Type_State: Binary indicator (1=State, 0=Local)
    - Log_Obligated: Log of obligated amount
    - Prior_Grant_Count: Count of prior grants (already in panel from s01_link)
    - Prior_Grant_Dollars_log: Log of prior grant dollars (already in panel from s01_link)
    - Disaster_Year: Year extracted from disaster type
    """
    from config import STATE_GOVERNMENTS, LOCAL_GOVERNMENTS
    import numpy as np
    import warnings

    # 1. Government Type
    def classify_government(grantee):
        if grantee in STATE_GOVERNMENTS:
            return 'State'
        elif grantee in LOCAL_GOVERNMENTS:
            return 'Local'
        else:
            # Default to Local for unknown grantees
            warnings.warn(f"Unknown grantee '{grantee}', classifying as Local")
            return 'Local'

    panel['Government_Type'] = panel['Grantee'].apply(classify_government)
    panel['Government_Type_State'] = (panel['Government_Type'] == 'State').astype(int)

    # 2. Log Obligated (use QPR Fund Obligated $ from quarterly data aggregation)
    obligated_col = None
    for col in ['QPR Fund Obligated $', 'Max_Obligated', 'Total_Obligated']:
        if col in panel.columns:
            obligated_col = col
            break

    if obligated_col:
        panel['Log_Obligated'] = np.log1p(panel[obligated_col].fillna(0))
    else:
        warnings.warn("No obligated column found. Log_Obligated not created.")
        panel['Log_Obligated'] = 0

    # 3. Prior Experience (should now exist from experience computation above)
    if 'Prior_Grant_Count' in panel.columns:
        panel['Prior_Grant_Count'] = panel['Prior_Grant_Count'].fillna(0).astype(int)
    else:
        warnings.warn("Prior_Grant_Count missing - this is unexpected after experience computation")
        panel['Prior_Grant_Count'] = 0

    if 'Prior_Grant_Dollars' in panel.columns:
        panel['Prior_Grant_Dollars_log'] = np.log1p(panel['Prior_Grant_Dollars'].fillna(0))
    else:
        warnings.warn("Prior_Grant_Dollars missing - this is unexpected after experience computation")
        panel['Prior_Grant_Dollars_log'] = 0

    # 4. Disaster Year (extract from disaster type string)
    # Most disaster types are formatted like "Hurricane Harvey (2017)"
    def extract_year(disaster_str):
        import re
        if pd.isna(disaster_str):
            return None
        match = re.search(r'\((\d{4})\)', str(disaster_str))
        if match:
            return int(match.group(1))
        # If no year in parentheses, check for 4-digit year anywhere
        match = re.search(r'(\d{4})', str(disaster_str))
        if match:
            return int(match.group(1))
        return None

    if 'Disaster_Year' not in panel.columns:
        panel['Disaster_Year'] = panel['Disaster Type'].apply(extract_year)

    # 5. Population log (should already exist from s01_link)
    if 'Population' in panel.columns and 'Population_log' not in panel.columns:
        panel['Population_log'] = np.log1p(panel['Population'].fillna(0))

    return panel


def build_standardized_features() -> None:
    """
    Main function to build standardized features panel.

    Reads standardized quarterly data from s00b and base panel from s01,
    aggregates velocity and timeliness features, and outputs a complete
    analysis-ready panel.
    """
    print("=" * 80)
    print("Stage 01b: Build Standardized Features")
    print("=" * 80)
    print()

    # =========================================================================
    # Load data
    # =========================================================================

    print("Loading standardized quarterly data...")
    qpr_std = load_standardized_qpr()
    print(f"  Loaded {len(qpr_std):,} quarterly observations")
    print(f"  {qpr_std['Grantee'].nunique()} grantees")
    print(f"  {qpr_std.groupby(['Grantee', 'Disaster Type']).ngroups} grantee-disaster pairs")
    print()

    print("Collapsing standardized activity rows to quarter-end panel...")
    qpr_std_quarterly = collapse_to_quarterly_panel(qpr_std)
    duplicate_quarters = (
        qpr_std.groupby(['Grantee', 'Disaster Type', 'QPR Actual Quarter']).size().gt(1).sum()
        if 'QPR Actual Quarter' in qpr_std.columns
        else 0
    )
    print(f"  Reduced {len(qpr_std):,} rows to {len(qpr_std_quarterly):,} quarter-level observations")
    print(f"  {duplicate_quarters:,} grantee-disaster-quarter groups had multiple activity rows")
    print()

    print("Loading base panel...")
    base_panel = load_base_panel()
    print(f"  Loaded {len(base_panel)} grantee-disaster pairs")
    print()

    # =========================================================================
    # Aggregate velocity features
    # =========================================================================

    print("Aggregating standardized velocity features...")
    velocity_panel = aggregate_standardized_velocity(qpr_std_quarterly)
    print(f"  Created {len(velocity_panel)} records")
    print(f"  {len([c for c in velocity_panel.columns if 'Velocity' in c])} velocity columns")
    print()

    # =========================================================================
    # Compute timeliness features
    # =========================================================================

    print("Computing timeliness features...")
    timeliness_panel = compute_timeliness_features_std(qpr_std_quarterly)
    print(f"  Created {len(timeliness_panel)} records")
    print(f"  {len([c for c in timeliness_panel.columns if 'Duration' in c])} duration columns")
    print()

    # =========================================================================
    # Compute multi-stage lag features (Phase 2 Week 3)
    # =========================================================================

    print("Computing multi-stage lag features (obligate→disburse→expend)...")
    stage_lag_panel = compute_stage_lags(qpr_std)
    print(f"  Created {len(stage_lag_panel)} records")
    lag_cols = [c for c in stage_lag_panel.columns if 'Lag_' in c or 'Stage' in c]
    print(f"  {len(lag_cols)} stage lag/efficiency columns: {', '.join(lag_cols)}")
    print()

    # =========================================================================
    # Compute phase-specific velocity features (Phase 2 Week 5)
    # =========================================================================

    print("Computing phase-specific velocity (early/mid/late)...")
    phase_velocity_panel = compute_phase_specific_velocity(qpr_std_quarterly)
    print(f"  Created {len(phase_velocity_panel)} records")
    phase_cols = [c for c in phase_velocity_panel.columns if 'Velocity_' in c]
    print(f"  {len(phase_cols)} phase velocity columns: {', '.join(phase_cols)}")
    print()

    # =========================================================================
    # Merge features with base panel
    # =========================================================================

    print("Merging features with base panel...")
    panel = base_panel.merge(
        velocity_panel,
        on=['Grantee', 'Disaster Type'],
        how='left'
    )
    panel = panel.merge(
        timeliness_panel,
        on=['Grantee', 'Disaster Type'],
        how='left'
    )
    panel = panel.merge(
        stage_lag_panel,
        on=['Grantee', 'Disaster Type'],
        how='left'
    )
    panel = panel.merge(
        phase_velocity_panel,
        on=['Grantee', 'Disaster Type'],
        how='left'
    )
    print(f"  Panel size: {len(panel)} records")
    print(f"  {len(panel.columns)} total columns")
    print()

    # =========================================================================
    # Compute experience features
    # =========================================================================

    print("Computing experience indicators...")
    experience_panel = compute_experience_features(qpr_std)
    print(f"  Created {len(experience_panel)} records")
    if 'Prior_Grant_Count' in experience_panel.columns:
        print(f"  Mean prior grants: {experience_panel['Prior_Grant_Count'].mean():.2f}")
    print()

    # Merge with panel
    panel = panel.merge(
        experience_panel,
        on=['Grantee', 'Disaster Type'],
        how='left'
    )

    # Fill missing values (first-time grantees)
    panel['Years_Experience'] = panel['Years_Experience'].fillna(0)
    panel['Prior_Grant_Count'] = panel['Prior_Grant_Count'].fillna(0).astype(int)
    panel['Prior_Grant_Dollars'] = panel['Prior_Grant_Dollars'].fillna(0)
    panel['Experience_Index'] = panel['Experience_Index'].fillna(0)

    print("Experience indicators merged to panel")
    print()

    # =========================================================================
    # Add derived features
    # =========================================================================

    print("Adding derived features...")

    panel = add_capacity_ratios(panel)
    print("  ✓ Capacity ratios")

    panel = add_scaled_velocity_features(panel)
    print("  ✓ Scaled velocity")

    panel = add_winsorized_velocity_features(panel)
    print("  ✓ Winsorized velocity (aggregated level)")

    panel = add_capacity_indices(panel)
    print("  ✓ Capacity indices")

    panel = add_capacity_quartiles(panel)
    print("  ✓ Capacity quartiles")

    panel = add_interaction_terms(panel)
    print("  ✓ Interaction terms")

    panel = add_survival_covariates(panel)
    print("  ✓ Survival covariates")

    # Recompute QA_High_Flag_Program with Q3 thresholds (more appropriate than fixed threshold)
    if 'Flag_Count_Extreme_Velocity' in panel.columns and 'Flag_Count_Obligated_Jump' in panel.columns:
        extreme_q3 = panel['Flag_Count_Extreme_Velocity'].quantile(0.75)
        obligated_q3 = panel['Flag_Count_Obligated_Jump'].quantile(0.75)

        panel['QA_High_Flag_Program'] = (
            (panel['Flag_Count_Extreme_Velocity'] > extreme_q3) |
            (panel['Flag_Count_Obligated_Jump'] > obligated_q3)
        )
        print(f"  ✓ QA High Flag Program (Q3 thresholds: extreme>{extreme_q3:.0f}, obligated>{obligated_q3:.0f})")
    elif 'QA_High_Flag_Program_Temp' in panel.columns:
        # Fallback to temp version if recalculation not possible
        panel['QA_High_Flag_Program'] = panel['QA_High_Flag_Program_Temp']
        panel = panel.drop(columns=['QA_High_Flag_Program_Temp'])

    print()

    # =========================================================================
    # Save standardized features panel
    # =========================================================================

    output_path = DATA_WORK_DIR / "panel_features_std.parquet"
    print(f"Saving standardized features panel: {output_path}")

    # Add column aliases for backward compatibility
    if 'Duration' in panel.columns and 'Duration_of_completion' not in panel.columns:
        panel['Duration_of_completion'] = panel['Duration']
    if 'N_Quarters' not in panel.columns and 'QPR Actual Quarter' in panel.columns:
        # N_Quarters is needed for time-varying analysis
        panel['N_Quarters'] = panel.groupby(['Grantee', 'Disaster Type'])['QPR Actual Quarter'].transform('nunique')

    safe_to_parquet(panel, output_path)

    print(f"  Saved {len(panel)} records")
    print(f"  {len(panel.columns)} total columns")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()

    # =========================================================================
    # Summary
    # =========================================================================

    print("=" * 80)
    print("Feature Engineering Complete")
    print("=" * 80)
    print()
    print("Outputs:")
    print(f"  - {output_path}")
    print()

    # Show key feature counts
    velocity_cols = [c for c in panel.columns if 'Velocity' in c]
    duration_cols = [c for c in panel.columns if 'Duration' in c]
    interaction_cols = [c for c in panel.columns if '_x_' in c]

    print(f"Feature summary:")
    print(f"  Velocity features: {len(velocity_cols)}")
    print(f"  Duration features: {len(duration_cols)}")
    print(f"  Interaction features: {len(interaction_cols)}")
    print(f"  Total features: {len(panel.columns)}")
    print()


if __name__ == "__main__":
    build_standardized_features()
