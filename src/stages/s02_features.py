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

from config import DATA_WORK_DIR, COMPLETION_THRESHOLD

# Import from existing modules
from capacity_sem.features.timeliness import (
    calculate_duration_of_completion,
    calculate_timeliness,
    calculate_quarter_variance,
    calculate_all_timeliness_metrics,
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


def load_panel() -> pd.DataFrame:
    """Load analysis panel."""
    panel_path = DATA_WORK_DIR / "panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Panel not found at {panel_path}. Run build_panel first."
        )
    return pd.read_parquet(panel_path)


def load_qpr_data() -> pd.DataFrame:
    """Load raw QPR data for feature computation."""
    qpr_path = DATA_WORK_DIR / "qpr_raw.parquet"
    if not qpr_path.exists():
        raise FileNotFoundError(
            f"QPR data not found at {qpr_path}. Run ingest_data first."
        )
    return pd.read_parquet(qpr_path)


def compute_timeliness_features(
    qpr: pd.DataFrame,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type'
) -> pd.DataFrame:
    """
    Compute timeliness features for each grantee-disaster pair.

    Parameters
    ----------
    qpr : pd.DataFrame
        Raw QPR data with quarterly observations.
    grantee_col : str
        Column name for grantee identifier.
    disaster_col : str
        Column name for disaster identifier.

    Returns
    -------
    pd.DataFrame
        DataFrame with timeliness metrics at grantee-disaster level.
    """
    results = []

    pairs = qpr.groupby([grantee_col, disaster_col]).size().reset_index()

    for _, row in pairs.iterrows():
        grantee = row[grantee_col]
        disaster = row[disaster_col]

        # Filter to this grantee-disaster
        mask = (qpr[grantee_col] == grantee) & (qpr[disaster_col] == disaster)
        subset = qpr[mask].copy()

        # Compute metrics
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

        # Number of quarters
        n_quarters = subset['QPR Actual Quarter'].nunique() if 'QPR Actual Quarter' in subset.columns else 0
        record['N_Quarters'] = n_quarters

        # Completion percentage
        total_obligated = subset['QPR Fund Obligated $'].sum() if 'QPR Fund Obligated $' in subset.columns else 0
        total_expended = subset['QPR Fund Expended $'].sum() if 'QPR Fund Expended $' in subset.columns else 0

        if total_obligated > 0:
            completion_pct = total_expended / total_obligated
        else:
            completion_pct = 0
        record['Completion_Pct'] = completion_pct

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

        # Duration log-transformed
        duration = calculate_duration_of_completion(subset, COMPLETION_THRESHOLD)
        if not np.isnan(duration) and duration > 0:
            record['Duration_log'] = np.log(duration)
        else:
            record['Duration_log'] = np.nan

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
        qpr = load_qpr_data()
        print(f"  Panel: {len(panel):,} observations")
        print(f"  QPR data: {len(qpr):,} rows")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return

    # Compute timeliness metrics
    print("\nComputing timeliness metrics...")
    timeliness = compute_timeliness_features(qpr)
    print(f"  Computed for {len(timeliness):,} grantee-disaster pairs")

    # Compute additional timeliness metrics
    print("Computing additional timeliness metrics...")
    additional = compute_additional_timeliness_metrics(qpr)

    # Merge additional into timeliness
    timeliness = timeliness.merge(
        additional,
        on=['Grantee', 'Disaster Type'],
        how='outer',
        suffixes=('', '_additional')
    )

    # Compute experience indicators
    print("\nComputing experience indicators...")
    experience = build_experience_dataset(qpr)
    print(f"  Computed for {len(experience):,} grantee-disaster pairs")

    # Merge features to panel
    print("\nMerging features to panel...")
    panel_features = merge_features_to_panel(panel, timeliness, experience)

    # Add program type
    if 'Activity Type' in qpr.columns:
        print("Adding program type classification...")
        # Get dominant program type for each grantee-disaster
        qpr_typed = add_program_type_column(qpr)
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
    panel_features.to_parquet(output_path, index=False)
    print(f"\n  Saved panel with features → {output_path}")
    print(f"  Total columns: {len(panel_features.columns)}")

    # Summary statistics
    print("\n  Feature summary:")
    for col in ['Duration_of_completion', 'Timeliness', 'Experience_Index', 'Progress_Rate']:
        if col in panel_features.columns:
            print(f"    {col}: mean={panel_features[col].mean():.3f}, "
                  f"std={panel_features[col].std():.3f}")

    print("\n✓ Feature engineering complete")


if __name__ == "__main__":
    main()
