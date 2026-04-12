"""
Task 2: Velocity Distribution Analysis

Examines distributions of velocity measures to identify:
- Outliers that might wash out signal
- Scaling issues
- Data anomalies

Usage:
    python src/diagnostics/velocity_distribution_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
from config import DATA_WORK_DIR, FIGURES_DIR


def analyze_velocity_distributions():
    """
    Analyze distributions of velocity measures and identify outliers.
    """
    print("=" * 80)
    print("VELOCITY DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print()

    # Load time-varying panel
    tv_panel_path = DATA_WORK_DIR / "panel_time_varying.parquet"
    print(f"Loading: {tv_panel_path}")

    try:
        tv_panel = pd.read_parquet(tv_panel_path, engine='fastparquet')
    except Exception as e:
        print(f"ERROR loading panel: {e}")
        return None

    print(f"Loaded {len(tv_panel)} intervals\n")

    # Key velocity measures
    key_measures = [
        'Disbursement_Velocity_pp_lag1',
        'Expenditure_Velocity_pp_lag1',
        'Capacity_Velocity_Index_pp_lag1',
    ]

    # Filter to available
    key_measures = [m for m in key_measures if m in tv_panel.columns]

    if not key_measures:
        print("ERROR: No key velocity measures found")
        return None

    # Results storage
    summary_stats = []
    outlier_cases = []

    for measure in key_measures:
        print(f"\n{'='*80}")
        print(f"Analyzing: {measure}")
        print(f"{'='*80}\n")

        # Get non-null values
        values = tv_panel[measure].dropna()
        n_total = len(tv_panel[measure])
        n_valid = len(values)
        n_missing = n_total - n_valid

        print(f"Sample size: {n_valid:,} valid / {n_total:,} total ({n_missing:,} missing)")
        print()

        # Summary statistics
        stats = values.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        print("Distribution summary:")
        print(stats)
        print()

        # Identify outliers using multiple methods
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 3 * iqr  # Use 3*IQR for extreme outliers
        upper_fence = q3 + 3 * iqr

        outliers_iqr = (values < lower_fence) | (values > upper_fence)
        n_outliers_iqr = outliers_iqr.sum()
        pct_outliers_iqr = (n_outliers_iqr / n_valid) * 100

        # Percentile-based outliers
        p01, p99 = values.quantile(0.01), values.quantile(0.99)
        outliers_pct = (values < p01) | (values > p99)
        n_outliers_pct = outliers_pct.sum()
        pct_outliers_pct = (n_outliers_pct / n_valid) * 100

        print(f"Outlier detection:")
        print(f"  IQR method (3×IQR): {n_outliers_iqr} outliers ({pct_outliers_iqr:.1f}%)")
        print(f"  Range: [{lower_fence:.2f}, {upper_fence:.2f}]")
        print(f"  1st/99th percentile: {n_outliers_pct} outliers ({pct_outliers_pct:.1f}%)")
        print(f"  Range: [{p01:.2f}, {p99:.2f}]")
        print()

        # Extreme values (beyond 3 std from mean)
        mean_val, std_val = values.mean(), values.std()
        extreme_threshold = 3 * std_val
        extreme_low = values < (mean_val - extreme_threshold)
        extreme_high = values > (mean_val + extreme_threshold)
        n_extreme = (extreme_low | extreme_high).sum()
        pct_extreme = (n_extreme / n_valid) * 100

        print(f"Extreme values (>3 std from mean):")
        print(f"  {n_extreme} extreme values ({pct_extreme:.1f}%)")
        print(f"  Range: [{mean_val - extreme_threshold:.2f}, {mean_val + extreme_threshold:.2f}]")
        print()

        # Zero velocity
        n_zero = (values == 0).sum()
        pct_zero = (n_zero / n_valid) * 100
        print(f"Zero velocity: {n_zero} observations ({pct_zero:.1f}%)")
        print()

        # Collect summary stats
        summary_stats.append({
            'Measure': measure,
            'N_Valid': n_valid,
            'N_Missing': n_missing,
            'Mean': mean_val,
            'Median': values.median(),
            'Std': std_val,
            'Min': values.min(),
            'Max': values.max(),
            'Q01': p01,
            'Q99': p99,
            'N_Zero': n_zero,
            'Pct_Zero': pct_zero,
            'N_Outliers_IQR': n_outliers_iqr,
            'Pct_Outliers_IQR': pct_outliers_iqr,
            'N_Extreme_3std': n_extreme,
            'Pct_Extreme_3std': pct_extreme,
        })

        # Identify specific extreme cases
        tv_panel_copy = tv_panel.copy()
        tv_panel_copy['is_extreme'] = (tv_panel_copy[measure] < (mean_val - extreme_threshold)) | \
                                       (tv_panel_copy[measure] > (mean_val + extreme_threshold))

        extreme_cases = tv_panel_copy[tv_panel_copy['is_extreme']][
            ['Grantee', 'Disaster Type', 'Quarter_Index', measure]
        ].copy()
        extreme_cases['Measure'] = measure

        if len(extreme_cases) > 0:
            print(f"Most extreme cases (top 10):")
            top_extreme = extreme_cases.nlargest(5, measure)[['Grantee', 'Disaster Type', 'Quarter_Index', measure]]
            print("\nHighest:")
            print(top_extreme.to_string(index=False))

            bottom_extreme = extreme_cases.nsmallest(5, measure)[['Grantee', 'Disaster Type', 'Quarter_Index', measure]]
            print("\nLowest:")
            print(bottom_extreme.to_string(index=False))
            print()

            outlier_cases.append(extreme_cases)

    # Create summary dataframe
    summary_df = pd.DataFrame(summary_stats)

    # Save summary stats
    output_path = DATA_WORK_DIR / "diagnostics" / "velocity_distribution_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"\nSummary stats saved to: {output_path}")

    # Save outlier cases
    if outlier_cases:
        outliers_df = pd.concat(outlier_cases, ignore_index=True)
        outliers_path = DATA_WORK_DIR / "diagnostics" / "velocity_extreme_outliers.csv"
        outliers_df.to_csv(outliers_path, index=False)
        print(f"Extreme outlier cases saved to: {outliers_path}")

    # Create diagnostic plots
    print("\nGenerating diagnostic plots...")
    create_diagnostic_plots(tv_panel, key_measures, summary_df)

    # Final assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    max_extreme_pct = summary_df['Pct_Extreme_3std'].max()
    max_outlier_pct = summary_df['Pct_Outliers_IQR'].max()

    if max_extreme_pct > 5:
        print(f"⚠️  WARNING: Up to {max_extreme_pct:.1f}% of observations are extreme outliers (>3 std)")
        print("   → These could wash out the signal in time-varying Cox models")
        print("   → Consider winsorizing or investigating data quality issues")
    elif max_outlier_pct > 10:
        print(f"⚠️  CAUTION: Up to {max_outlier_pct:.1f}% of observations are IQR outliers")
        print("   → May introduce noise but not necessarily problematic")
    else:
        print(f"✅ ACCEPTABLE: Outliers are within reasonable bounds")
        print(f"   Extreme outliers: {max_extreme_pct:.1f}%")
        print(f"   IQR outliers: {max_outlier_pct:.1f}%")

    # Check for many zeros
    max_zero_pct = summary_df['Pct_Zero'].max()
    if max_zero_pct > 30:
        print(f"\n⚠️  WARNING: Up to {max_zero_pct:.1f}% of observations have zero velocity")
        print("   → Many programs not changing quarter-to-quarter")
        print("   → May explain weak time-varying signal")
    elif max_zero_pct > 10:
        print(f"\n⚠️  NOTE: {max_zero_pct:.1f}% of observations have zero velocity")
        print("   → Some programs stall, but not unusual")
    else:
        print(f"\n✅ Zero velocity is rare ({max_zero_pct:.1f}%)")

    print()

    return summary_df


def create_diagnostic_plots(tv_panel, measures, summary_df):
    """
    Create distribution plots for velocity measures.
    """
    n_measures = len(measures)
    fig, axes = plt.subplots(n_measures, 3, figsize=(15, 5 * n_measures))

    if n_measures == 1:
        axes = axes.reshape(1, -1)

    for i, measure in enumerate(measures):
        values = tv_panel[measure].dropna()

        # Histogram
        ax1 = axes[i, 0]
        ax1.hist(values, bins=100, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Velocity (pp/quarter)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{measure}\nHistogram (all values)')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.5)

        # Box plot
        ax2 = axes[i, 1]
        ax2.boxplot(values, vert=True)
        ax2.set_ylabel('Velocity (pp/quarter)')
        ax2.set_title('Box Plot (shows outliers)')
        ax2.axhline(0, color='red', linestyle='--', alpha=0.5)

        # Histogram without extreme outliers (for better visualization)
        ax3 = axes[i, 2]
        q01, q99 = values.quantile(0.01), values.quantile(0.99)
        trimmed_values = values[(values >= q01) & (values <= q99)]
        ax3.hist(trimmed_values, bins=50, edgecolor='black', alpha=0.7, color='green')
        ax3.set_xlabel('Velocity (pp/quarter)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Histogram (trimmed 1%-99%)')
        ax3.axvline(0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save figure
    fig_path = FIGURES_DIR / "velocity_distribution_diagnostics.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Diagnostic plots saved to: {fig_path}")
    plt.close()


if __name__ == "__main__":
    results = analyze_velocity_distributions()

    if results is not None:
        print("\nDistribution analysis complete!")
    else:
        print("\nDistribution analysis failed - see errors above.")
