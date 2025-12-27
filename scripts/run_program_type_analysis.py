#!/usr/bin/env python3
"""
Phase 3 Week 8: Program Type Heterogeneity Analysis

Tests if velocity effects vary by program type (Housing, Infrastructure, Economic Development).
Uses stratified Cox PH and interaction models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from stages._io_utils import safe_read_parquet

print("=" * 80)
print("Phase 3 Week 8: Program Type Heterogeneity Analysis")
print("=" * 80)
print()

# Load panel and program types
panel_path = Path("../data_work/panel_features_std.parquet")
program_types_path = Path("../data_work/panel_program_types.parquet")

print("Loading data...")
panel = safe_read_parquet(panel_path)
program_types = safe_read_parquet(program_types_path)
print(f"  Panel: {len(panel)} grantee-disaster pairs")
print(f"  Program types: {len(program_types)} grantee-disaster pairs")
print()

# Merge
panel_with_types = panel.merge(
    program_types,
    on=['Grantee', 'Disaster Type'],
    how='left'
)
print(f"Merged panel: {len(panel_with_types)} records")
print(f"  Missing program types: {panel_with_types['Primary_Program_Type'].isna().sum()}")
print()

# Prepare survival data
panel_with_types['Duration_Surv'] = panel_with_types['Duration'].fillna(panel_with_types['N_Quarters'])
panel_with_types['Event'] = (panel_with_types['Duration'].notna() & (panel_with_types['Duration'] > 0)).astype(int)
panel_with_types['Velocity_scaled'] = panel_with_types['Expenditure_Velocity_pp'] * 100

# Check program type distribution
print("Primary program type distribution:")
print(panel_with_types['Primary_Program_Type'].value_counts())
print()

# ============================================================================
# Analysis 1: Stratified Cox PH by Program Type
# ============================================================================
print("=" * 80)
print("Analysis 1: Stratified Cox PH by Program Type")
print("=" * 80)
print()

results = []

# Focus on top 3 program types (Housing, Infrastructure, Economic Development)
for prog_type in ['Housing', 'Infrastructure', 'Administration']:
    print(f"\n{prog_type} Programs")
    print("-" * 80)

    subset = panel_with_types[panel_with_types['Primary_Program_Type'] == prog_type].copy()

    # Drop missing
    subset_clean = subset[['Duration_Surv', 'Event', 'Velocity_scaled', 'Government_Type_State']].dropna()

    n_events = subset_clean['Event'].sum()
    print(f"  Sample: N={len(subset_clean)}, Events={n_events}")

    if n_events < 5:
        print(f"  ⚠ Too few events ({n_events}), skipping Cox PH")
        results.append({
            'Program_Type': prog_type,
            'N': len(subset_clean),
            'N_Events': int(n_events),
            'Velocity_HR': np.nan,
            'Velocity_p': np.nan,
            'Velocity_CI_lower': np.nan,
            'Velocity_CI_upper': np.nan,
        })
        continue

    # Fit Cox PH
    cph = CoxPHFitter(penalizer=0.01)
    try:
        cph.fit(
            subset_clean,
            duration_col='Duration_Surv',
            event_col='Event'
        )

        hr = np.exp(cph.params_['Velocity_scaled'])
        p = cph.summary.loc['Velocity_scaled', 'p']
        ci_lower = np.exp(cph.confidence_intervals_.loc['Velocity_scaled', '95% lower-bound'])
        ci_upper = np.exp(cph.confidence_intervals_.loc['Velocity_scaled', '95% upper-bound'])

        results.append({
            'Program_Type': prog_type,
            'N': len(subset_clean),
            'N_Events': int(n_events),
            'Velocity_HR': hr,
            'Velocity_p': p,
            'Velocity_CI_lower': ci_lower,
            'Velocity_CI_upper': ci_upper,
        })

        print(f"  Velocity HR: {hr:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f}), p={p:.4f}")
        print()
        print(cph.summary[['coef', 'exp(coef)', 'p']])

    except Exception as e:
        print(f"  ✗ Model failed: {e}")
        results.append({
            'Program_Type': prog_type,
            'N': len(subset_clean),
            'N_Events': int(n_events),
            'Velocity_HR': np.nan,
            'Velocity_p': np.nan,
            'Velocity_CI_lower': np.nan,
            'Velocity_CI_upper': np.nan,
        })

# ============================================================================
# Analysis 2: Program Type × Velocity Interaction
# ============================================================================
print("\n" + "=" * 80)
print("Analysis 2: Program Type × Velocity Interaction")
print("=" * 80)
print()

# Create dummy variables for program types
panel_with_types['Housing_Dummy'] = (panel_with_types['Primary_Program_Type'] == 'Housing').astype(int)
panel_with_types['Infrastructure_Dummy'] = (panel_with_types['Primary_Program_Type'] == 'Infrastructure').astype(int)

# Interaction terms
panel_with_types['Housing_x_Velocity'] = panel_with_types['Housing_Dummy'] * panel_with_types['Velocity_scaled']
panel_with_types['Infrastructure_x_Velocity'] = panel_with_types['Infrastructure_Dummy'] * panel_with_types['Velocity_scaled']

# Fit interaction model
subset_interact = panel_with_types[[
    'Duration_Surv', 'Event', 'Velocity_scaled',
    'Housing_Dummy', 'Infrastructure_Dummy',
    'Housing_x_Velocity', 'Infrastructure_x_Velocity',
    'Government_Type_State'
]].dropna()

print(f"Sample: N={len(subset_interact)}, Events={subset_interact['Event'].sum()}")
print()

if subset_interact['Event'].sum() >= 10:
    cph_interact = CoxPHFitter(penalizer=0.01)
    cph_interact.fit(
        subset_interact,
        duration_col='Duration_Surv',
        event_col='Event'
    )

    print("Interaction Model Results:")
    print(cph_interact.summary[['coef', 'exp(coef)', 'p']])
    print()

    # Extract interaction results
    housing_interaction_p = cph_interact.summary.loc['Housing_x_Velocity', 'p']
    infra_interaction_p = cph_interact.summary.loc['Infrastructure_x_Velocity', 'p']

    print("Interaction Interpretation:")
    if housing_interaction_p < 0.05:
        housing_hr = np.exp(cph_interact.params_['Housing_x_Velocity'])
        print(f"  ✓ Housing × Velocity interaction: HR={housing_hr:.3f}, p={housing_interaction_p:.4f}")
        if housing_hr > 1.0:
            print("    → Housing programs have STRONGER velocity effects")
        else:
            print("    → Housing programs have WEAKER velocity effects")
    else:
        print(f"  ○ Housing × Velocity interaction not significant (p={housing_interaction_p:.4f})")

    if infra_interaction_p < 0.05:
        infra_hr = np.exp(cph_interact.params_['Infrastructure_x_Velocity'])
        print(f"  ✓ Infrastructure × Velocity interaction: HR={infra_hr:.3f}, p={infra_interaction_p:.4f}")
        if infra_hr > 1.0:
            print("    → Infrastructure programs have STRONGER velocity effects")
        else:
            print("    → Infrastructure programs have WEAKER velocity effects")
    else:
        print(f"  ○ Infrastructure × Velocity interaction not significant (p={infra_interaction_p:.4f})")

else:
    print("⚠ Insufficient events for interaction model")

# ============================================================================
# Save Results
# ============================================================================
results_df = pd.DataFrame(results)
output_path = Path("../data_work/diagnostics/program_type_heterogeneity.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)
print(f"\n✓ Saved program type heterogeneity results: {output_path}")

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print()
print(results_df.to_string(index=False))
print()

# ============================================================================
# Visualization: Forest Plot by Program Type
# ============================================================================
if len(results_df.dropna(subset=['Velocity_HR'])) >= 2:
    print("Creating forest plot of velocity effects by program type...")

    plot_df = results_df.dropna(subset=['Velocity_HR', 'Velocity_CI_lower', 'Velocity_CI_upper'])

    if len(plot_df) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        y_positions = np.arange(len(plot_df))

        colors = {'Housing': '#2ecc71', 'Infrastructure': '#3498db', 'Administration': '#9b59b6',
                  'Economic Development': '#e67e22', 'Acquisition': '#e74c3c'}

        for idx, (i, row) in enumerate(plot_df.iterrows()):
            prog_type = row['Program_Type']
            hr = row['Velocity_HR']
            ci_lower = row['Velocity_CI_lower']
            ci_upper = row['Velocity_CI_upper']
            p_value = row['Velocity_p']
            n = int(row['N'])
            n_events = int(row['N_Events'])

            color = colors.get(prog_type, '#95a5a6')

            # Point estimate
            ax.plot(hr, y_positions[idx], 'o', markersize=12, color=color, zorder=3)

            # CI
            ax.plot([ci_lower, ci_upper], [y_positions[idx], y_positions[idx]],
                    '-', linewidth=3, color=color, zorder=2)

            # Significance
            sig_marker = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))

            # Label
            label_text = f"HR={hr:.2f} {sig_marker}\n(N={n}, Events={n_events})"
            ax.text(max(ci_upper * 1.15, 2.0), y_positions[idx], label_text,
                    va='center', ha='left', fontsize=10)

        # Null line
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

        # X-axis (log scale)
        ax.set_xscale('log')
        ax.set_xlabel('Hazard Ratio (95% CI)\nper 1 pp/quarter increase in expenditure velocity', fontsize=11)

        # Y-axis
        ax.set_yticks(y_positions)
        ax.set_yticklabels(plot_df['Program_Type'], fontsize=11)
        ax.set_ylim(-0.5, len(plot_df) - 0.5)

        # Title
        ax.set_title('Velocity Effects by Primary Program Type',
                     fontsize=13, fontweight='bold', pad=20)

        # Grid
        ax.grid(axis='x', alpha=0.3, linestyle=':', zorder=0)

        # Notes
        note_text = "*** p<0.001, ** p<0.01, * p<0.05\nHR>1: Higher velocity → faster completion\nVertical dashed line: Null effect (HR=1)"
        ax.text(0.02, 0.02, note_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        forest_path = Path("../figures/velocity_effect_by_program_type.png")
        forest_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(forest_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved forest plot: {forest_path}")
        plt.close()

print("\n" + "=" * 80)
print("Program Type Heterogeneity Analysis Complete")
print("=" * 80)
