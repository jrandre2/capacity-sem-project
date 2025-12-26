"""
Stage 02: Feature Engineering

Compute timeliness metrics, experience indicators, and program stratification.

Commands:
    python src/pipeline.py compute_features

Outputs:
    data_work/panel_features.parquet  - Panel with all computed features
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from config import DATA_WORK_DIR, COMPLETION_THRESHOLD, QPR_DOLLAR_FIELDS_ARE_FLOW
from stages._io_utils import safe_to_parquet, safe_read_parquet

# Import from existing modules
from capacity_sem.features.timeliness import (
    calculate_duration_of_completion,
    calculate_timeliness,
    calculate_quarter_variance,
    calculate_all_timeliness_metrics,
    calculate_all_timeliness_metrics_multi_threshold,
    calculate_multi_threshold_duration,
)

from capacity_sem.features.experience_indicators import (
    compute_years_of_experience,
    compute_prior_grant_count,
    compute_cumulative_prior_dollars,
    compute_experience_index,
    compute_grantee_experience,
    build_experience_dataset,
    get_experience_summary,
)

from capacity_sem.features.program_stratification import (
    PROGRAM_TYPE_MAPPING,
    ACTIVITY_TO_PROGRAM,
    map_activity_to_program_type,
    add_program_type_column,
    get_program_type_distribution,
    filter_by_program_type,
    compute_indicators_by_program_type,
    build_program_stratified_dataset,
)
from capacity_sem.utils.date_utils import quarter_to_date
from capacity_sem.data.loader import build_qpr_quarterly


def load_panel() -> pd.DataFrame:
    """Load analysis panel."""
    panel_path = DATA_WORK_DIR / "panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Panel not found at {panel_path}. Run build_panel first."
        )
    return safe_read_parquet(panel_path)


def load_qpr_raw() -> pd.DataFrame:
    """Load QPR data for feature computation (cleaned if available)."""
    clean_path = DATA_WORK_DIR / "qpr_clean.parquet"
    if clean_path.exists():
        return safe_read_parquet(clean_path)
    qpr_path = DATA_WORK_DIR / "qpr_raw.parquet"
    if not qpr_path.exists():
        raise FileNotFoundError(
            f"QPR data not found at {qpr_path}. Run ingest_data first."
        )
    return safe_read_parquet(qpr_path)


def load_qpr_quarterly() -> pd.DataFrame:
    """Load or build quarterly QPR data with cumulative series."""
    quarterly_path = DATA_WORK_DIR / "qpr_quarterly.parquet"
    if quarterly_path.exists():
        return safe_read_parquet(quarterly_path)
    qpr_raw = load_qpr_raw()
    return build_qpr_quarterly(qpr_raw, flows_are_net=QPR_DOLLAR_FIELDS_ARE_FLOW)


def compute_timeliness_features(
    qpr: pd.DataFrame,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type',
    use_multi_threshold: bool = True
) -> pd.DataFrame:
    """
    Compute timeliness features for each grantee-disaster pair.

    Supports multi-threshold duration analysis (30% to 100% in 5% increments)
    for robustness testing and to avoid arbitrary threshold choice.

    Parameters
    ----------
    qpr : pd.DataFrame
        Raw QPR data with quarterly observations.
    grantee_col : str
        Column name for grantee identifier.
    disaster_col : str
        Column name for disaster identifier.
    use_multi_threshold : bool
        Whether to compute duration at multiple thresholds (default True).

    Returns
    -------
    pd.DataFrame
        DataFrame with timeliness metrics at grantee-disaster level.
        Includes Duration_Xpct columns for each threshold if use_multi_threshold=True.
    """
    results = []

    pairs = qpr.groupby([grantee_col, disaster_col]).size().reset_index()

    for _, row in pairs.iterrows():
        grantee = row[grantee_col]
        disaster = row[disaster_col]

        # Filter to this grantee-disaster
        mask = (qpr[grantee_col] == grantee) & (qpr[disaster_col] == disaster)
        subset = qpr[mask].copy()

        # Compute metrics (with or without multi-threshold)
        if use_multi_threshold:
            metrics = calculate_all_timeliness_metrics_multi_threshold(subset)
        else:
            metrics = calculate_all_timeliness_metrics(subset, COMPLETION_THRESHOLD)

        metrics[grantee_col] = grantee
        metrics[disaster_col] = disaster

        results.append(metrics)

    return pd.DataFrame(results)


def compute_additional_timeliness_metrics(
    qpr: pd.DataFrame,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type'
) -> pd.DataFrame:
    """
    Compute additional timeliness metrics for SEM models.

    These include alternative measures that avoid mathematical coupling
    between capacity and outcome indicators.

    Parameters
    ----------
    qpr : pd.DataFrame
        Raw QPR data.
    grantee_col : str
        Column name for grantee identifier.
    disaster_col : str
        Column name for disaster identifier.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional metrics.
    """
    results = []

    pairs = qpr.groupby([grantee_col, disaster_col]).size().reset_index()

    def gini_coefficient(values: np.ndarray) -> float:
        values = values.astype(float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            return np.nan
        values = np.abs(values)
        if np.all(values == 0):
            return 0.0
        values = np.sort(values)
        n = values.size
        cum_values = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cum_values) / cum_values[-1]) / n

    for _, row in pairs.iterrows():
        grantee = row[grantee_col]
        disaster = row[disaster_col]

        mask = (qpr[grantee_col] == grantee) & (qpr[disaster_col] == disaster)
        subset = qpr[mask].copy()

        if subset.empty:
            continue

        record = {
            grantee_col: grantee,
            disaster_col: disaster
        }

        # Order by quarter for time-based metrics
        if 'QPR Actual Quarter' in subset.columns:
            subset['QPR_Date'] = subset['QPR Actual Quarter'].apply(quarter_to_date)
            subset = subset.sort_values('QPR_Date').reset_index(drop=True)

        # Number of quarters
        n_quarters = subset['QPR Actual Quarter'].nunique() if 'QPR Actual Quarter' in subset.columns else 0
        record['N_Quarters'] = n_quarters

        # Completion percentage
        final_obligated = subset['QPR Fund Obligated $'].iloc[-1] if 'QPR Fund Obligated $' in subset.columns else 0
        final_expended = subset['QPR Fund Expended $'].iloc[-1] if 'QPR Fund Expended $' in subset.columns else 0

        if final_obligated > 0:
            completion_pct = final_expended / final_obligated
        else:
            completion_pct = 0
        record['Completion_Pct'] = completion_pct
        record['Ratio_obligated_funds_fully_expended'] = completion_pct

        # Progress Rate = Completion % per quarter
        if n_quarters > 0:
            record['Progress_Rate'] = completion_pct / n_quarters
        else:
            record['Progress_Rate'] = np.nan

        # Spending CV (coefficient of variation)
        if 'QPR Fund Expended Q $' in subset.columns:
            expended_q = subset['QPR Fund Expended Q $'].values
            if len(expended_q) > 1 and np.mean(expended_q) > 0:
                record['Spending_CV'] = np.std(expended_q) / np.mean(expended_q)
            else:
                record['Spending_CV'] = np.nan

            record['Spending_Gini'] = gini_coefficient(expended_q)

            if len(expended_q) > 2:
                diffs = np.diff(expended_q)
                denom = np.mean(np.abs(expended_q))
                if denom > 0:
                    record['Spending_Acceleration'] = np.mean(diffs) / denom
                else:
                    record['Spending_Acceleration'] = np.nan

        # Duration log-transformed
        duration = calculate_duration_of_completion(subset, COMPLETION_THRESHOLD)
        if not np.isnan(duration) and duration > 0:
            record['Duration_log'] = np.log(duration)
        else:
            record['Duration_log'] = np.nan

        # Time to 50% completion (normalized by total quarters)
        time_to_50pct = np.nan
        if final_obligated > 0 and 'QPR Fund Expended $' in subset.columns and n_quarters > 1:
            completion_series = subset['QPR Fund Expended $'] / final_obligated
            milestone = completion_series[completion_series >= 0.5]
            if not milestone.empty:
                milestone_idx = milestone.index[0]
                time_to_50pct = milestone_idx / max(n_quarters - 1, 1)
        record['Time_to_50pct'] = time_to_50pct

        # Startup lag: quarters before first expenditure
        startup_lag = np.nan
        if 'QPR Fund Expended $' in subset.columns:
            first_positive = subset.index[subset['QPR Fund Expended $'] > 0]
            if len(first_positive) > 0:
                startup_lag = int(first_positive[0])
        record['Startup_Lag'] = startup_lag

        # Completion velocity: mean quarterly change in completion %
        completion_velocity = np.nan
        if final_obligated > 0 and 'QPR Fund Expended $' in subset.columns and n_quarters > 1:
            completion_series = subset['QPR Fund Expended $'] / final_obligated
            diffs = completion_series.diff().dropna()
            if not diffs.empty:
                completion_velocity = diffs.mean()
        record['Completion_Velocity'] = completion_velocity

        results.append(record)

    return pd.DataFrame(results)


def merge_features_to_panel(
    panel: pd.DataFrame,
    timeliness: pd.DataFrame,
    experience: pd.DataFrame,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type'
) -> pd.DataFrame:
    """
    Merge computed features into the panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Base panel.
    timeliness : pd.DataFrame
        Timeliness metrics.
    experience : pd.DataFrame
        Experience indicators.
    grantee_col : str
        Column name for grantee identifier.
    disaster_col : str
        Column name for disaster identifier.

    Returns
    -------
    pd.DataFrame
        Panel with all features merged.
    """
    # Merge timeliness
    if timeliness is not None and not timeliness.empty:
        panel = panel.merge(
            timeliness,
            on=[grantee_col, disaster_col],
            how='left',
            suffixes=('', '_timeliness')
        )

    # Merge experience
    if experience is not None and not experience.empty:
        # Rename if needed
        if 'Disaster_Type' in experience.columns and disaster_col not in experience.columns:
            experience = experience.rename(columns={'Disaster_Type': disaster_col})

        panel = panel.merge(
            experience,
            on=[grantee_col, disaster_col],
            how='left',
            suffixes=('', '_experience')
        )

    return panel


def main():
    """Main entry point for feature engineering stage."""
    print("=" * 60)
    print("Stage 02: Feature Engineering")
    print("=" * 60)

    DATA_WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    try:
        panel = load_panel()
        qpr_raw = load_qpr_raw()
        qpr_quarterly = load_qpr_quarterly()
        print(f"  Panel: {len(panel):,} observations")
        print(f"  QPR raw data: {len(qpr_raw):,} rows")
        print(f"  QPR quarterly data: {len(qpr_quarterly):,} rows")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return

    # Compute timeliness metrics
    print("\nComputing timeliness metrics...")
    timeliness = compute_timeliness_features(qpr_quarterly)
    print(f"  Computed for {len(timeliness):,} grantee-disaster pairs")

    # Compute additional timeliness metrics
    print("Computing additional timeliness metrics...")
    additional = compute_additional_timeliness_metrics(qpr_quarterly)

    # Merge additional into timeliness
    timeliness = timeliness.merge(
        additional,
        on=['Grantee', 'Disaster Type'],
        how='outer',
        suffixes=('', '_additional')
    )

    # Compute experience indicators
    print("\nComputing experience indicators...")
    experience = build_experience_dataset(qpr_raw)
    print(f"  Computed for {len(experience):,} grantee-disaster pairs")

    # Merge features to panel
    print("\nMerging features to panel...")
    panel_features = merge_features_to_panel(panel, timeliness, experience)

    if 'Experience_Index' in panel_features.columns:
        mean = panel_features['Experience_Index'].mean()
        std = panel_features['Experience_Index'].std()
        if std > 0:
            panel_features['Experience_Index_scaled'] = (
                panel_features['Experience_Index'] - mean
            ) / std
        else:
            panel_features['Experience_Index_scaled'] = 0

    # Capacity index for formative model support
    if {'Ratio_disbursed_to_obligated', 'Ratio_expended_to_disbursed'}.issubset(panel_features.columns):
        panel_features['Capacity_Index'] = panel_features[
            ['Ratio_disbursed_to_obligated', 'Ratio_expended_to_disbursed']
        ].mean(axis=1, skipna=True)

    # Add program type
    if 'Activity Type' in qpr_raw.columns:
        print("Adding program type classification...")
        # Get dominant program type for each grantee-disaster
        qpr_typed = add_program_type_column(qpr_raw)
        program_types = qpr_typed.groupby(['Grantee', 'Disaster Type'])['Program_Type'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 'Other'
        ).reset_index()
        panel_features = panel_features.merge(
            program_types,
            on=['Grantee', 'Disaster Type'],
            how='left'
        )

    # Save
    output_path = DATA_WORK_DIR / "panel_features.parquet"
    safe_to_parquet(panel_features, output_path)
    print(f"\n  Saved panel with features → {output_path}")
    print(f"  Total columns: {len(panel_features.columns)}")

    # Summary statistics
    print("\n  Feature summary:")
    for col in ['Duration_of_completion', 'Timeliness', 'Experience_Index', 'Progress_Rate']:
        if col in panel_features.columns:
            print(f"    {col}: mean={panel_features[col].mean():.3f}, "
                  f"std={panel_features[col].std():.3f}")

    # Multi-threshold duration summary
    duration_cols = [c for c in panel_features.columns if c.startswith('Duration_') and 'pct' in c and '_log' not in c]
    if duration_cols:
        print("\n  Multi-threshold duration sample sizes:")
        for col in sorted(duration_cols, key=lambda x: int(x.replace('Duration_', '').replace('pct', ''))):
            n_valid = panel_features[col].notna().sum()
            pct_valid = 100 * n_valid / len(panel_features)
            print(f"    {col}: {n_valid}/{len(panel_features)} ({pct_valid:.1f}% available)")

    print("\n✓ Feature engineering complete")


if __name__ == "__main__":
    main()
