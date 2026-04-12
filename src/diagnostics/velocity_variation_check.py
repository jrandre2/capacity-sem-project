"""
Task 1: Velocity Variation Check

Verifies that velocity measures actually vary within grantee-disasters.
This is the CRITICAL first check to rule out data quality issues.

If velocity has zero variance within units, the time-varying Cox model
will produce null results by design.

Usage:
    python src/diagnostics/velocity_variation_check.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
from config import DATA_WORK_DIR


def check_velocity_variation():
    """
    Check within-unit variation in velocity measures from time-varying panel.
    """
    print("=" * 80)
    print("VELOCITY VARIATION CHECK")
    print("=" * 80)
    print()

    # Load time-varying panel
    tv_panel_path = DATA_WORK_DIR / "panel_time_varying.parquet"
    print(f"Loading: {tv_panel_path}")

    if not tv_panel_path.exists():
        print(f"ERROR: File not found: {tv_panel_path}")
        print("You may need to run the time-varying survival analysis first.")
        return None

    try:
        tv_panel = pd.read_parquet(tv_panel_path, engine='pyarrow')
    except (ValueError, Exception) as e:
        print(f"Warning: PyArrow engine failed ({e}), trying fastparquet...")
        try:
            tv_panel = pd.read_parquet(tv_panel_path, engine='fastparquet')
        except Exception as e2:
            print(f"ERROR: Both parquet engines failed: {e2}")
            print("Trying to read with pandas-only approach...")
            import pyarrow.parquet as pq
            table = pq.read_table(tv_panel_path)
            tv_panel = table.to_pandas(self_destruct=True, ignore_metadata=True)

    print(f"Loaded {len(tv_panel)} intervals from {tv_panel['Grantee'].nunique()} grantees")
    print()

    # Identify velocity columns
    velocity_cols = [col for col in tv_panel.columns if 'Velocity' in col and 'lag' in col]
    print(f"Found {len(velocity_cols)} lagged velocity columns:")
    for col in velocity_cols[:10]:  # Show first 10
        print(f"  - {col}")
    if len(velocity_cols) > 10:
        print(f"  ... and {len(velocity_cols) - 10} more")
    print()

    # Key velocity measures to check
    key_measures = [
        'Disbursement_Velocity_pp_lag1',
        'Expenditure_Velocity_pp_lag1',
        'Capacity_Velocity_Index_pp_lag1',
    ]

    # Filter to available measures
    key_measures = [m for m in key_measures if m in tv_panel.columns]

    if not key_measures:
        print("WARNING: No key velocity measures found in panel!")
        print("Available columns:", tv_panel.columns.tolist())
        return None

    results = []

    print("=" * 80)
    print("WITHIN-UNIT VARIATION ANALYSIS")
    print("=" * 80)
    print()

    for measure in key_measures:
        print(f"\nAnalyzing: {measure}")
        print("-" * 80)

        # Overall statistics
        overall_stats = tv_panel[measure].describe()
        print(f"\nOverall distribution:")
        print(overall_stats)

        # Within-unit statistics
        unit_stats = tv_panel.groupby(['Grantee', 'Disaster Type'])[measure].agg([
            'count',  # Number of observations
            'mean',   # Average velocity within unit
            'std',    # Standard deviation within unit
            'min',    # Minimum velocity
            'max'     # Maximum velocity
        ]).reset_index()

        # Classify units by variation
        unit_stats['has_variation'] = unit_stats['std'] > 0
        unit_stats['zero_variance'] = unit_stats['std'] == 0
        unit_stats['missing_all'] = unit_stats['std'].isna()

        # Summary statistics
        n_total = len(unit_stats)
        n_variation = unit_stats['has_variation'].sum()
        n_zero_var = unit_stats['zero_variance'].sum()
        n_missing = unit_stats['missing_all'].sum()

        pct_variation = (n_variation / n_total) * 100 if n_total > 0 else 0
        pct_zero_var = (n_zero_var / n_total) * 100 if n_total > 0 else 0
        pct_missing = (n_missing / n_total) * 100 if n_total > 0 else 0

        print(f"\nWithin-unit variation summary:")
        print(f"  Total grantee-disasters: {n_total}")
        print(f"  Units with variation (std > 0): {n_variation} ({pct_variation:.1f}%)")
        print(f"  Units with zero variance: {n_zero_var} ({pct_zero_var:.1f}%)")
        print(f"  Units with all missing: {n_missing} ({pct_missing:.1f}%)")

        # Average standard deviation across units
        mean_std = unit_stats[unit_stats['has_variation']]['std'].mean()
        median_std = unit_stats[unit_stats['has_variation']]['std'].median()
        print(f"\nAmong units with variation:")
        print(f"  Mean std: {mean_std:.4f}")
        print(f"  Median std: {median_std:.4f}")

        # Range of velocity values
        mean_range = (unit_stats['max'] - unit_stats['min']).mean()
        print(f"  Mean range (max - min): {mean_range:.4f}")

        # Collect results
        results.append({
            'Measure': measure,
            'N_Total': n_total,
            'N_Variation': n_variation,
            'Pct_Variation': pct_variation,
            'N_Zero_Variance': n_zero_var,
            'Pct_Zero_Variance': pct_zero_var,
            'N_Missing': n_missing,
            'Pct_Missing': pct_missing,
            'Mean_Std_Within_Unit': mean_std,
            'Median_Std_Within_Unit': median_std,
            'Mean_Range': mean_range,
            'Overall_Mean': overall_stats['mean'],
            'Overall_Std': overall_stats['std'],
            'Overall_Min': overall_stats['min'],
            'Overall_Max': overall_stats['max'],
        })

        # Show examples of high and low variance units
        print(f"\nExamples of HIGH within-unit variation:")
        high_var_examples = unit_stats[unit_stats['has_variation']].nlargest(3, 'std')[
            ['Grantee', 'Disaster Type', 'count', 'mean', 'std', 'min', 'max']
        ]
        print(high_var_examples.to_string(index=False))

        print(f"\nExamples of ZERO variance units (if any):")
        if n_zero_var > 0:
            zero_var_examples = unit_stats[unit_stats['zero_variance']].head(3)[
                ['Grantee', 'Disaster Type', 'count', 'mean', 'std', 'min', 'max']
            ]
            print(zero_var_examples.to_string(index=False))
        else:
            print("  (None found - this is GOOD)")

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Save results
    output_path = DATA_WORK_DIR / "diagnostics" / "velocity_variation_stats.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")

    # Final assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    # Check if variation exists
    min_pct_variation = results_df['Pct_Variation'].min()
    max_pct_zero_var = results_df['Pct_Zero_Variance'].max()

    if min_pct_variation > 80:
        print("✅ PASS: Velocity measures show substantial within-unit variation")
        print(f"   At least {min_pct_variation:.1f}% of units have non-zero variance")
        print("   → Time-varying null results are NOT due to lack of variation")
        print("   → Proceed to Phase 2 diagnostics (model specification checks)")
    elif max_pct_zero_var > 50:
        print("❌ FAIL: Many units have zero variance in velocity")
        print(f"   Up to {max_pct_zero_var:.1f}% of units have constant velocity")
        print("   → This explains the time-varying null results")
        print("   → Investigate computational error in velocity calculation")
    else:
        print("⚠️  MIXED: Some units have variation, some don't")
        print(f"   {min_pct_variation:.1f}%-{results_df['Pct_Variation'].max():.1f}% have variation")
        print("   → May indicate heterogeneity or data quality issues")
        print("   → Review individual unit patterns")

    print()

    return results_df


if __name__ == "__main__":
    results = check_velocity_variation()

    if results is not None:
        print("\nVariation check complete!")
        print("Review the output above and the saved CSV for details.")
    else:
        print("\nVariation check failed - see errors above.")
