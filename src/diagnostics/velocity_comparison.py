"""
Task 3: Static vs Time-Varying Velocity Comparison

Directly compares static velocity (mean across all quarters) with
time-varying velocity aggregates to ensure they're measuring the same thing.

Usage:
    python src/diagnostics/velocity_comparison.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
from config import DATA_WORK_DIR


def compare_velocities():
    """
    Compare static and time-varying velocity measures.
    """
    print("=" * 80)
    print("STATIC VS TIME-VARYING VELOCITY COMPARISON")
    print("=" * 80)
    print()

    # Load static panel with velocity features
    static_path = DATA_WORK_DIR / "panel_features.parquet"
    print(f"Loading static panel: {static_path}")

    try:
        static_panel = pd.read_parquet(static_path, engine='fastparquet')
    except Exception as e:
        print(f"ERROR loading static panel: {e}")
        return None

    print(f"Loaded {len(static_panel)} grantee-disasters\n")

    # Load time-varying panel
    tv_path = DATA_WORK_DIR / "panel_time_varying.parquet"
    print(f"Loading time-varying panel: {tv_path}")

    try:
        tv_panel = pd.read_parquet(tv_path, engine='fastparquet')
    except Exception as e:
        print(f"ERROR loading time-varying panel: {e}")
        return None

    print(f"Loaded {len(tv_panel)} intervals\n")

    # Velocity columns to compare
    # Note: Time-varying panel only has lagged versions, use lag0 as "base"
    velocity_cols = {
        'Disbursement': {
            'static': 'Disbursement_Velocity',
            'tv_base': 'Disbursement_Velocity_pp_lag0',
            'tv_lag1': 'Disbursement_Velocity_pp_lag1',
        },
        'Expenditure': {
            'static': 'Expenditure_Velocity',
            'tv_base': 'Expenditure_Velocity_pp_lag0',
            'tv_lag1': 'Expenditure_Velocity_pp_lag1',
        },
    }

    results = []

    for vel_type, cols in velocity_cols.items():
        print(f"\n{'='*80}")
        print(f"Comparing {vel_type} Velocity")
        print(f"{'='*80}\n")

        # Check if columns exist
        static_col = cols['static']
        tv_base_col = cols['tv_base']
        tv_lag_col = cols['tv_lag1']

        if static_col not in static_panel.columns:
            print(f"WARNING: Static column '{static_col}' not found")
            continue

        if tv_base_col not in tv_panel.columns:
            print(f"WARNING: Time-varying column '{tv_base_col}' not found")
            continue

        # Compute mean time-varying velocity for each grantee-disaster
        print("Computing aggregated time-varying velocity (should match static)...")

        agg_dict = {tv_base_col: ['mean', 'std', 'count', 'min', 'max']}
        if tv_lag_col in tv_panel.columns:
            agg_dict[tv_lag_col] = ['mean', 'std', 'count']

        tv_agg = tv_panel.groupby(['Grantee', 'Disaster Type']).agg(agg_dict).reset_index()

        # Flatten multi-level column names
        new_cols = ['Grantee', 'Disaster Type']
        for col in tv_agg.columns[2:]:
            if isinstance(col, tuple):
                new_cols.append(f'{col[0]}_{col[1]}')
            else:
                new_cols.append(col)
        tv_agg.columns = new_cols

        # Rename to more readable names
        rename_dict = {}
        for col in tv_agg.columns:
            if tv_base_col in col:
                if '_mean' in col:
                    rename_dict[col] = f'{vel_type}_TV_Mean'
                elif '_std' in col:
                    rename_dict[col] = f'{vel_type}_TV_Std'
                elif '_count' in col:
                    rename_dict[col] = f'{vel_type}_TV_Count'
                elif '_min' in col:
                    rename_dict[col] = f'{vel_type}_TV_Min'
                elif '_max' in col:
                    rename_dict[col] = f'{vel_type}_TV_Max'
            elif tv_lag_col in col:
                if '_mean' in col:
                    rename_dict[col] = f'{vel_type}_TV_Lag1_Mean'
                elif '_std' in col:
                    rename_dict[col] = f'{vel_type}_TV_Lag1_Std'
                elif '_count' in col:
                    rename_dict[col] = f'{vel_type}_TV_Lag1_Count'

        tv_agg = tv_agg.rename(columns=rename_dict)

        # Merge with static panel
        comparison = static_panel[['Grantee', 'Disaster Type', static_col]].merge(
            tv_agg,
            on=['Grantee', 'Disaster Type'],
            how='inner'
        )

        # Note: Static velocity might be in different units (raw vs pp)
        # Check if static is in pp or needs conversion
        static_mean = comparison[static_col].mean()
        tv_mean = comparison[f'{vel_type}_TV_Mean'].mean()

        print(f"Static velocity (from features): mean = {static_mean:.4f}")
        print(f"Time-varying velocity (mean): mean = {tv_mean:.4f}")

        # Check if they need unit conversion
        if abs(static_mean - tv_mean) / max(abs(static_mean), abs(tv_mean), 0.01) > 10:
            print(f"\n⚠️  WARNING: Large difference detected!")
            print(f"   Static and time-varying may be in different units")
            print(f"   Checking if static needs *100 conversion...")

            # Try multiplying static by 100
            if abs(static_mean * 100 - tv_mean) / max(abs(tv_mean), 0.01) < 0.5:
                print(f"   ✓ Yes! Static * 100 = {static_mean * 100:.4f} ≈ {tv_mean:.4f}")
                comparison[f'{static_col}_pp'] = comparison[static_col] * 100
                static_col_compare = f'{static_col}_pp'
            else:
                print(f"   ✗ No, they're genuinely different")
                static_col_compare = static_col
        else:
            print(f"   Units appear consistent")
            static_col_compare = static_col

        # Correlation analysis
        print(f"\nCorrelation analysis:")
        corr_static_tv = comparison[static_col_compare].corr(comparison[f'{vel_type}_TV_Mean'])
        print(f"  Static vs Time-varying mean: r = {corr_static_tv:.4f}")

        # Scatter plot statistics
        diff = comparison[static_col_compare] - comparison[f'{vel_type}_TV_Mean']
        mean_diff = diff.mean()
        std_diff = diff.std()
        max_abs_diff = diff.abs().max()

        print(f"\nDifference (Static - TV_Mean):")
        print(f"  Mean difference: {mean_diff:.4f}")
        print(f"  Std difference: {std_diff:.4f}")
        print(f"  Max absolute difference: {max_abs_diff:.4f}")

        # Classify agreement
        if corr_static_tv > 0.95 and abs(mean_diff) < 1:
            print(f"\n✅ EXCELLENT AGREEMENT: Static and TV measures are nearly identical")
        elif corr_static_tv > 0.85:
            print(f"\n✅ GOOD AGREEMENT: Static and TV measures are highly correlated")
        elif corr_static_tv > 0.70:
            print(f"\n⚠️  MODERATE AGREEMENT: Some differences exist")
        else:
            print(f"\n❌ POOR AGREEMENT: Static and TV measures differ substantially")

        # Show examples of large discrepancies
        if max_abs_diff > 5:
            print(f"\nLargest discrepancies:")
            large_diff = comparison.copy()
            large_diff['Diff'] = diff.abs()
            top_diff = large_diff.nlargest(5, 'Diff')[
                ['Grantee', 'Disaster Type', static_col_compare, f'{vel_type}_TV_Mean', 'Diff']
            ]
            print(top_diff.to_string(index=False))

        # Check variation in time-varying velocity
        print(f"\nTime-varying velocity variation:")
        print(f"  Mean within-unit std: {comparison[f'{vel_type}_TV_Std'].mean():.4f}")
        print(f"  Median within-unit std: {comparison[f'{vel_type}_TV_Std'].median():.4f}")

        high_std = comparison[f'{vel_type}_TV_Std'] > 10
        n_high_std = high_std.sum()
        pct_high_std = (n_high_std / len(comparison)) * 100
        print(f"  Units with high variation (std > 10): {n_high_std} ({pct_high_std:.1f}%)")

        # Collect results
        results.append({
            'Velocity_Type': vel_type,
            'N_Grantee_Disasters': len(comparison),
            'Static_Mean': static_mean,
            'TV_Mean': tv_mean,
            'Correlation': corr_static_tv,
            'Mean_Difference': mean_diff,
            'Std_Difference': std_diff,
            'Max_Abs_Difference': max_abs_diff,
            'Mean_Within_Unit_Std': comparison[f'{vel_type}_TV_Std'].mean(),
            'Median_Within_Unit_Std': comparison[f'{vel_type}_TV_Std'].median(),
        })

        # Save detailed comparison
        output_path = DATA_WORK_DIR / "diagnostics" / f"velocity_comparison_{vel_type.lower()}.csv"
        comparison.to_csv(output_path, index=False)
        print(f"\nDetailed comparison saved to: {output_path}")

    # Summary results
    results_df = pd.DataFrame(results)
    summary_path = DATA_WORK_DIR / "diagnostics" / "velocity_static_tv_comparison.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\n\nSummary saved to: {summary_path}")

    # Final assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    min_corr = results_df['Correlation'].min()
    max_diff_std = results_df['Std_Difference'].max()

    if min_corr > 0.95:
        print("✅ VALIDATED: Static and time-varying velocity measures are consistent")
        print(f"   Correlation: {min_corr:.4f}")
        print("   → No computational errors detected")
    elif min_corr > 0.80:
        print("✅ ACCEPTABLE: Static and time-varying measures are generally consistent")
        print(f"   Correlation: {min_corr:.4f}")
        print("   → Minor differences likely due to missing values or sample differences")
    else:
        print("❌ PROBLEM: Static and time-varying measures diverge substantially")
        print(f"   Correlation: {min_corr:.4f}")
        print("   → Investigate computational differences or sample mismatches")

    # Check if they're measuring different constructs
    mean_within_std = results_df['Mean_Within_Unit_Std'].mean()
    print(f"\nWithin-unit variation:")
    print(f"  Average std across units: {mean_within_std:.4f}")

    if mean_within_std < 2:
        print("   ⚠️  Low within-unit variation - velocity relatively stable")
    elif mean_within_std < 10:
        print("   ✓ Moderate within-unit variation - velocity fluctuates normally")
    else:
        print("   ⚠️  High within-unit variation - substantial quarterly fluctuations")

    print()

    return results_df


if __name__ == "__main__":
    results = compare_velocities()

    if results is not None:
        print("\nVelocity comparison complete!")
    else:
        print("\nVelocity comparison failed - see errors above.")
