#!/usr/bin/env python3
"""
Phase 4 Week 10: Meta-Analysis of Velocity Effects

Aggregates all velocity HR estimates from Phases 1-3 analyses and creates
comprehensive forest plot and meta-analytic summary.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

print("=" * 80)
print("Phase 4 Week 10: Meta-Analysis of Velocity Effects")
print("=" * 80)
print()

# ============================================================================
# Load All Results Files
# ============================================================================
diagnostics_dir = Path("../data_work/diagnostics")

results_files = {
    'Phase-Specific Velocity': diagnostics_dir / 'phase_specific_velocity.csv',
    'Trajectory Clustering': diagnostics_dir / 'temporal_dynamics_trajectory_clusters.csv',
    'Learning Curves': diagnostics_dir / 'learning_curves_experience_velocity.csv',
    'Program Type': diagnostics_dir / 'program_type_heterogeneity.csv',
    'Disaster Context': diagnostics_dir / 'disaster_context_heterogeneity.csv',
}

all_estimates = []

print("Loading results from all analyses...")
print()

# ============================================================================
# Phase-Specific Velocity
# ============================================================================
if results_files['Phase-Specific Velocity'].exists():
    print("Phase 2 Week 5: Phase-Specific Velocity")
    df = pd.read_csv(results_files['Phase-Specific Velocity'])

    for idx, row in df.iterrows():
        model = row['Model']

        # Overall velocity
        if pd.notna(row.get('Overall_Velocity_HR')):
            all_estimates.append({
                'Analysis': 'Phase-Specific',
                'Context': f'{model} (Overall)',
                'Phase': 2,
                'Week': 5,
                'HR': row['Overall_Velocity_HR'],
                'p': row['Overall_Velocity_p'],
                'N': row['N'],
                'Events': row['N_Events'],
            })

        # Early velocity
        if pd.notna(row.get('Early_Velocity_HR')):
            all_estimates.append({
                'Analysis': 'Phase-Specific',
                'Context': f'{model} (Early)',
                'Phase': 2,
                'Week': 5,
                'HR': row['Early_Velocity_HR'],
                'p': row['Early_Velocity_p'],
                'N': row['N'],
                'Events': row['N_Events'],
            })

        # Mid velocity
        if pd.notna(row.get('Mid_Velocity_HR')):
            all_estimates.append({
                'Analysis': 'Phase-Specific',
                'Context': f'{model} (Mid)',
                'Phase': 2,
                'Week': 5,
                'HR': row['Mid_Velocity_HR'],
                'p': row['Mid_Velocity_p'],
                'N': row['N'],
                'Events': row['N_Events'],
            })

        # Late velocity
        if pd.notna(row.get('Late_Velocity_HR')):
            all_estimates.append({
                'Analysis': 'Phase-Specific',
                'Context': f'{model} (Late)',
                'Phase': 2,
                'Week': 5,
                'HR': row['Late_Velocity_HR'],
                'p': row['Late_Velocity_p'],
                'N': row['N'],
                'Events': row['N_Events'],
            })

        # Acceleration
        if pd.notna(row.get('Acceleration_HR')):
            all_estimates.append({
                'Analysis': 'Phase-Specific',
                'Context': f'{model} (Acceleration)',
                'Phase': 2,
                'Week': 5,
                'HR': row['Acceleration_HR'],
                'p': row['Acceleration_p'],
                'N': row['N'],
                'Events': row['N_Events'],
            })

    print(f"  Loaded {len(df)} models")

# ============================================================================
# Learning Curves (Experience)
# ============================================================================
if results_files['Learning Curves'].exists():
    print("Phase 2 Week 6: Learning Curves & Experience")
    df = pd.read_csv(results_files['Learning Curves'])

    for idx, row in df.iterrows():
        if pd.notna(row.get('Velocity_HR')):
            all_estimates.append({
                'Analysis': 'Experience',
                'Context': row['Experience_Group'],
                'Phase': 2,
                'Week': 6,
                'HR': row['Velocity_HR'],
                'p': row['Velocity_p'],
                'N': row['N'],
                'Events': row['N_Events'],
                'CI_lower': row.get('Velocity_CI_lower'),
                'CI_upper': row.get('Velocity_CI_upper'),
            })

    print(f"  Loaded {len(df)} experience groups")

# ============================================================================
# Program Type
# ============================================================================
if results_files['Program Type'].exists():
    print("Phase 3 Week 8: Program Type Heterogeneity")
    df = pd.read_csv(results_files['Program Type'])

    for idx, row in df.iterrows():
        if pd.notna(row.get('Velocity_HR')):
            all_estimates.append({
                'Analysis': 'Program Type',
                'Context': row['Program_Type'],
                'Phase': 3,
                'Week': 8,
                'HR': row['Velocity_HR'],
                'p': row['Velocity_p'],
                'N': row['N'],
                'Events': row['N_Events'],
                'CI_lower': row.get('Velocity_CI_lower'),
                'CI_upper': row.get('Velocity_CI_upper'),
            })

    print(f"  Loaded {len(df)} program types")

# ============================================================================
# Disaster Context
# ============================================================================
if results_files['Disaster Context'].exists():
    print("Phase 3 Week 9: Disaster Context")
    df = pd.read_csv(results_files['Disaster Context'])

    for idx, row in df.iterrows():
        if pd.notna(row.get('Velocity_HR')) and row['Velocity_HR'] > 1e-10:  # Filter out near-zero HRs
            all_estimates.append({
                'Analysis': 'Disaster Context',
                'Context': row['Disaster_Context'],
                'Phase': 3,
                'Week': 9,
                'HR': row['Velocity_HR'],
                'p': row['Velocity_p'],
                'N': row['N'],
                'Events': row['N_Events'],
                'CI_lower': row.get('Velocity_CI_lower'),
                'CI_upper': row.get('Velocity_CI_upper'),
            })

    print(f"  Loaded {len(df)} disaster contexts")

print()
print(f"Total estimates collected: {len(all_estimates)}")
print()

# ============================================================================
# Create Master Results DataFrame
# ============================================================================
meta_df = pd.DataFrame(all_estimates)

# Fill missing CIs with approximate values based on p-value if needed
for idx, row in meta_df.iterrows():
    if pd.isna(row.get('CI_lower')) or pd.isna(row.get('CI_upper')):
        # Approximate CI from HR and p-value (rough estimate)
        hr = row['HR']
        p = row['p']

        # Very rough approximation: assume normal distribution on log scale
        if p < 1.0 and hr > 0:
            from scipy.stats import norm
            z = norm.ppf(1 - p/2)  # Two-tailed
            log_hr = np.log(hr)
            se_log_hr = abs(log_hr) / z if z > 0 else 0.5

            meta_df.loc[idx, 'CI_lower'] = np.exp(log_hr - 1.96 * se_log_hr)
            meta_df.loc[idx, 'CI_upper'] = np.exp(log_hr + 1.96 * se_log_hr)

# ============================================================================
# Summary Statistics
# ============================================================================
print("=" * 80)
print("META-ANALYSIS SUMMARY")
print("=" * 80)
print()

print("Overall Summary:")
print(f"  Total estimates: {len(meta_df)}")
print(f"  Significant (p < 0.05): {(meta_df['p'] < 0.05).sum()} ({(meta_df['p'] < 0.05).sum() / len(meta_df) * 100:.1f}%)")
print(f"  HR > 1 (velocity accelerates completion): {(meta_df['HR'] > 1).sum()} ({(meta_df['HR'] > 1).sum() / len(meta_df) * 100:.1f}%)")
print()

print("Summary by Analysis Type:")
for analysis in meta_df['Analysis'].unique():
    subset = meta_df[meta_df['Analysis'] == analysis]
    sig_count = (subset['p'] < 0.05).sum()
    print(f"  {analysis}: {len(subset)} estimates, {sig_count} significant ({sig_count/len(subset)*100:.1f}%)")
print()

print("Distribution of Effect Sizes (HR):")
print(f"  Mean: {meta_df['HR'].mean():.3f}")
print(f"  Median: {meta_df['HR'].median():.3f}")
print(f"  Min: {meta_df['HR'].min():.3f}")
print(f"  Max: {meta_df['HR'].max():.3f}")
print(f"  25th percentile: {meta_df['HR'].quantile(0.25):.3f}")
print(f"  75th percentile: {meta_df['HR'].quantile(0.75):.3f}")
print()

# Save master results
output_path = Path("../data_work/diagnostics/meta_analysis_all_estimates.csv")
meta_df.to_csv(output_path, index=False)
print(f"✓ Saved master results: {output_path}")
print()

# ============================================================================
# Comprehensive Forest Plot
# ============================================================================
print("Creating comprehensive forest plot...")

# Sort by HR for better visualization
plot_df = meta_df.sort_values('HR', ascending=True).reset_index(drop=True)

# Create figure
fig, ax = plt.subplots(figsize=(14, max(12, len(plot_df) * 0.4)), dpi=300)

y_positions = np.arange(len(plot_df))

# Color by analysis type
colors_by_analysis = {
    'Phase-Specific': '#3498db',
    'Experience': '#2ecc71',
    'Program Type': '#9b59b6',
    'Disaster Context': '#e67e22',
}

# Plot each estimate
for idx, row in plot_df.iterrows():
    hr = row['HR']
    ci_lower = row.get('CI_lower', hr * 0.5)
    ci_upper = row.get('CI_upper', hr * 2.0)
    p_value = row['p']
    context = row['Context']
    analysis = row['Analysis']

    color = colors_by_analysis.get(analysis, '#95a5a6')

    # Point estimate
    marker_size = 10 if p_value < 0.05 else 6
    ax.plot(hr, y_positions[idx], 'o', markersize=marker_size, color=color,
            zorder=3, alpha=0.8)

    # CI
    ax.plot([ci_lower, ci_upper], [y_positions[idx], y_positions[idx]],
            '-', linewidth=2 if p_value < 0.05 else 1, color=color, zorder=2, alpha=0.6)

# Null line
ax.axvline(x=1, color='black', linestyle='--', linewidth=2, alpha=0.7, zorder=1)

# X-axis (log scale)
ax.set_xscale('log')
ax.set_xlabel('Hazard Ratio (95% CI)\nper 1 pp/quarter increase in expenditure velocity',
              fontsize=12, fontweight='bold')
ax.set_xlim(0.01, max(plot_df['HR'].max() * 1.5, 100))

# Y-axis
ax.set_yticks(y_positions)
# Create compact labels
labels = []
for idx, row in plot_df.iterrows():
    context = row['Context'][:30]  # Truncate long labels
    sig = '*' if row['p'] < 0.05 else ''
    labels.append(f"{context}{sig}")
ax.set_yticklabels(labels, fontsize=8)
ax.set_ylim(-0.5, len(plot_df) - 0.5)

# Title
ax.set_title('Meta-Analysis: Velocity Effects Across All Contexts\nPhases 2-3 Research Extension',
             fontsize=14, fontweight='bold', pad=20)

# Grid
ax.grid(axis='x', alpha=0.3, linestyle=':', zorder=0)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors_by_analysis['Phase-Specific'], label='Phase-Specific Velocity'),
    Patch(facecolor=colors_by_analysis['Experience'], label='Experience Effects'),
    Patch(facecolor=colors_by_analysis['Program Type'], label='Program Type'),
    Patch(facecolor=colors_by_analysis['Disaster Context'], label='Disaster Context'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

# Notes
note_text = (
    f"Total estimates: {len(plot_df)}\n"
    f"Significant (p<0.05): {(plot_df['p'] < 0.05).sum()} marked with *\n"
    f"Median HR: {plot_df['HR'].median():.2f}\n"
    "Vertical dashed line: Null effect (HR=1)"
)
ax.text(0.02, 0.98, note_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

plt.tight_layout()
forest_path = Path("../figures/meta_analysis_all_velocity_effects.png")
forest_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(forest_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved comprehensive forest plot: {forest_path}")
plt.close()

# ============================================================================
# Highlight Top Effects
# ============================================================================
print()
print("=" * 80)
print("TOP 10 STRONGEST VELOCITY EFFECTS")
print("=" * 80)
print()

top_effects = meta_df.nlargest(10, 'HR')[['Analysis', 'Context', 'HR', 'p', 'N', 'Events']]
print(top_effects.to_string(index=False))
print()

print("=" * 80)
print("Meta-Analysis Complete")
print("=" * 80)
