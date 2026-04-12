"""
Stage 01c: Aggregate Program Type Characteristics

Aggregate activity-level data to grantee-disaster level to create
program portfolio features.

Commands:
    python src/pipeline.py aggregate_program_types

Inputs:
    data_work/qpr_standardized.parquet  - Quarterly data with Activity Type
    data_work/panel_features_std.parquet - Base panel

Outputs:
    data_work/panel_program_types.parquet - Panel with program type features
"""

from pathlib import Path
import pandas as pd
import numpy as np
from config import PROGRAM_TYPE_MAPPING, DATA_WORK_DIR
from ._io_utils import safe_read_parquet


def map_activity_to_category(activity: str) -> str:
    """
    Map raw activity type to 6-category classification.

    Parameters
    ----------
    activity : str
        Raw activity type from QPR data

    Returns
    -------
    str
        Category: Housing, Infrastructure, Economic Development, Acquisition, Administration, or Other
    """
    for category, activities in PROGRAM_TYPE_MAPPING.items():
        if activity in activities:
            return category
    return 'Other'


def aggregate_program_types(qpr_std: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate activity-level data to grantee-disaster level.

    Creates program portfolio features:
    - Obligated dollars by category (Housing, Infrastructure, etc.)
    - Category percentages
    - Primary program type (category with highest obligation)
    - Program diversity index (Herfindahl index)

    Parameters
    ----------
    qpr_std : pd.DataFrame
        Standardized QPR data with Activity Type column

    Returns
    -------
    pd.DataFrame
        Grantee-disaster panel with program type features
    """

    # Check if Activity Type exists
    if 'Activity Type' not in qpr_std.columns:
        raise ValueError("Activity Type column not found in qpr_std")

    # Map activities to categories
    print("  Mapping activity types to program categories...")
    qpr_std = qpr_std.copy()
    qpr_std['Program_Category'] = qpr_std['Activity Type'].apply(map_activity_to_category)

    # Check category distribution
    print("  Category distribution:")
    for cat, count in qpr_std['Program_Category'].value_counts().items():
        print(f"    {cat}: {count:,} observations")

    # For each grantee-disaster, get the maximum obligated amount per activity
    # (to avoid double-counting across quarters)
    print("  Aggregating obligated dollars by category...")

    # Group by grantee-disaster-activity and get max final obligated
    # Use Obligated_Final which is the final denominator amount for each activity
    obligated_col = 'Obligated_Final' if 'Obligated_Final' in qpr_std.columns else 'Obligated_Clean'

    activity_max = qpr_std.groupby(['Grantee', 'Disaster Type', 'Activity Type', 'Program_Category'])[
        obligated_col
    ].max().reset_index()

    # Now aggregate by category
    category_dollars = activity_max.groupby(['Grantee', 'Disaster Type', 'Program_Category'])[
        obligated_col
    ].sum().reset_index()

    # Pivot to wide format
    program_dollars = category_dollars.pivot_table(
        index=['Grantee', 'Disaster Type'],
        columns='Program_Category',
        values=obligated_col,
        fill_value=0
    ).reset_index()

    # Ensure all categories exist (even if zero)
    for cat in ['Housing', 'Infrastructure', 'Economic Development', 'Acquisition', 'Administration', 'Other']:
        if cat not in program_dollars.columns:
            program_dollars[cat] = 0

    # Total obligated across all categories
    cat_cols = ['Housing', 'Infrastructure', 'Economic Development', 'Acquisition', 'Administration', 'Other']
    program_dollars['Total_Obligated_by_Category'] = program_dollars[cat_cols].sum(axis=1)

    # Percentages
    for cat in cat_cols:
        program_dollars[f'{cat}_Pct'] = (
            program_dollars[cat] / program_dollars['Total_Obligated_by_Category']
        ).fillna(0)

    # Primary program type (largest category)
    program_dollars['Primary_Program_Type'] = program_dollars[cat_cols].idxmax(axis=1)

    # Diversity index (Herfindahl index: 1 - sum of squared shares)
    # Higher values = more diverse portfolio
    pct_cols = [f'{cat}_Pct' for cat in cat_cols]
    program_dollars['Program_Diversity_Index'] = 1 - (program_dollars[pct_cols] ** 2).sum(axis=1)

    # Number of active program categories (categories with >5% of obligated)
    program_dollars['N_Active_Categories'] = (program_dollars[pct_cols] > 0.05).sum(axis=1)

    print(f"  Created {len(program_dollars)} grantee-disaster records")
    print(f"  Primary program type distribution:")
    for ptype, count in program_dollars['Primary_Program_Type'].value_counts().items():
        print(f"    {ptype}: {count}")

    return program_dollars


def run_aggregate_program_types():
    """Main entry point for Stage 01c: Aggregate Program Types."""

    print("=" * 80)
    print("Stage 01c: Aggregate Program Types")
    print("=" * 80)
    print()

    # Load standardized QPR data
    qpr_path = DATA_WORK_DIR / "qpr_standardized.parquet"
    print(f"Loading standardized QPR data: {qpr_path}")
    qpr_std = safe_read_parquet(qpr_path)
    print(f"  Loaded {len(qpr_std):,} quarterly observations")
    print()

    # Aggregate program types
    program_types = aggregate_program_types(qpr_std)

    # Ensure proper data types for parquet serialization
    # Reset index to avoid pyarrow serialization issues with multi-index
    program_types = program_types.copy()
    program_types['Grantee'] = program_types['Grantee'].astype(str)
    program_types['Disaster Type'] = program_types['Disaster Type'].astype(str)
    program_types['Primary_Program_Type'] = program_types['Primary_Program_Type'].astype(str)

    # Save (use fastparquet engine to avoid pyarrow issues)
    output_path = DATA_WORK_DIR / "panel_program_types.parquet"
    try:
        program_types.to_parquet(output_path, index=False, engine='fastparquet')
    except:
        # Fallback to pyarrow with explicit schema
        program_types.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"\n✓ Saved program type panel: {output_path}")
    print(f"  {len(program_types)} grantee-disaster pairs")
    print(f"  {len(program_types.columns)} columns")

    # Summary statistics
    print("\n" + "=" * 80)
    print("Program Type Summary Statistics")
    print("=" * 80)
    print()

    print("Diversity Index:")
    print(f"  Mean: {program_types['Program_Diversity_Index'].mean():.4f}")
    print(f"  Median: {program_types['Program_Diversity_Index'].median():.4f}")
    print(f"  Std: {program_types['Program_Diversity_Index'].std():.4f}")
    print()

    print("Active Categories per Program:")
    print(program_types['N_Active_Categories'].value_counts().sort_index())
    print()

    print("Top 10 Grantees by Program Diversity:")
    top_diverse = program_types.nlargest(10, 'Program_Diversity_Index')[
        ['Grantee', 'Disaster Type', 'Program_Diversity_Index', 'N_Active_Categories', 'Primary_Program_Type']
    ]
    print(top_diverse.to_string(index=False))

    return program_types


if __name__ == "__main__":
    run_aggregate_program_types()
