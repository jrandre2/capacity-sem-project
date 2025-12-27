#!/usr/bin/env python3
"""
Phase 2 Week 6: Learning Curves & Experience Effects

Tests if prior CDBG-DR experience amplifies velocity effects through:
1. Experience × Velocity interaction
2. Stratified analysis: Novice vs Experienced grantees
3. Learning curves: Velocity improvement over successive grants
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from stages._io_utils import safe_read_parquet

print("=" * 80)
print("Phase 2 Week 6: Learning Curves & Experience Effects")
print("=" * 80)
print()

# Load panel
panel_path = Path("../data_work/panel_features_std.parquet")
panel = safe_read_parquet(panel_path)
print(f"Loaded {len(panel)} grantee-disaster pairs")
print()

# Prepare survival data
panel['Duration_Surv'] = panel['Duration'].fillna(panel['N_Quarters'])
panel['Event'] = (panel['Duration'].notna() & (panel['Duration'] > 0)).astype(int)

# Scale velocity for pp/quarter interpretation
panel['Velocity_scaled'] = panel['Expenditure_Velocity_pp'] * 100

# Check experience variables
print("Experience variable availability:")
for col in ['Prior_Grant_Count', 'Experience_Index']:
    if col in panel.columns:
        print(f"  {col}: N={panel[col].notna().sum()}, Mean={panel[col].mean():.4f}, Std={panel[col].std():.4f}")
    else:
        print(f"  {col}: NOT FOUND")
print()

# Create experience groups
if 'Prior_Grant_Count' in panel.columns:
    panel['Experience_Group'] = panel['Prior_Grant_Count'].apply(
        lambda x: 'Novice' if x == 0 else 'Experienced'
    )

    print("Experience group distribution:")
    print(panel['Experience_Group'].value_counts())
    print()
else:
    print("⚠ Prior_Grant_Count not found - cannot create experience groups")
    sys.exit(1)

# ============================================================================
# Analysis 1: Stratified Cox PH by Experience Level
# ============================================================================
print("=" * 80)
print("Analysis 1: Stratified Cox PH by Experience Level")
print("=" * 80)
print()

results = []

for group in ['Novice', 'Experienced']:
    print(f"\n{group} Grantees")
    print("-" * 80)

    subset = panel[panel['Experience_Group'] == group].copy()

    # Drop missing
    subset_clean = subset[['Duration_Surv', 'Event', 'Velocity_scaled', 'Government_Type_State']].dropna()

    n_events = subset_clean['Event'].sum()
    print(f"  Sample: N={len(subset_clean)}, Events={n_events}")

    if n_events < 5:
        print(f"  ⚠ Too few events ({n_events}), skipping Cox PH")
        results.append({
            'Experience_Group': group,
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
            'Experience_Group': group,
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
            'Experience_Group': group,
            'N': len(subset_clean),
            'N_Events': int(n_events),
            'Velocity_HR': np.nan,
            'Velocity_p': np.nan,
            'Velocity_CI_lower': np.nan,
            'Velocity_CI_upper': np.nan,
        })

# ============================================================================
# Analysis 2: Experience × Velocity Interaction
# ============================================================================
print("\n" + "=" * 80)
print("Analysis 2: Experience × Velocity Interaction")
print("=" * 80)
print()

if 'Experience_Index' in panel.columns:
    # Create interaction term
    panel['Experience_x_Velocity'] = panel['Experience_Index'] * panel['Velocity_scaled']

    # Fit Cox PH with interaction
    subset_interact = panel[['Duration_Surv', 'Event', 'Velocity_scaled',
                               'Experience_Index', 'Experience_x_Velocity',
                               'Government_Type_State']].dropna()

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
        interaction_hr = np.exp(cph_interact.params_['Experience_x_Velocity'])
        interaction_p = cph_interact.summary.loc['Experience_x_Velocity', 'p']
        velocity_hr = np.exp(cph_interact.params_['Velocity_scaled'])
        velocity_p = cph_interact.summary.loc['Velocity_scaled', 'p']

        results.append({
            'Experience_Group': 'Interaction',
            'N': len(subset_interact),
            'N_Events': int(subset_interact['Event'].sum()),
            'Velocity_HR': velocity_hr,
            'Velocity_p': velocity_p,
            'Interaction_HR': interaction_hr,
            'Interaction_p': interaction_p,
        })

        print(f"Velocity main effect: HR={velocity_hr:.3f}, p={velocity_p:.4f}")
        print(f"Experience × Velocity interaction: HR={interaction_hr:.3f}, p={interaction_p:.4f}")

        if interaction_p < 0.05:
            if interaction_hr > 1.0:
                print("  ✓ Significant positive interaction: Experienced grantees leverage velocity more effectively")
            else:
                print("  ⚠ Significant negative interaction: Experience dampens velocity effect")
        else:
            print("  ○ No significant interaction: Velocity effect is similar across experience levels")
    else:
        print("⚠ Insufficient events for interaction model")
else:
    print("⚠ Experience_Index not found - cannot test interaction")

# ============================================================================
# Analysis 3: Learning Curves (Multi-Grant Grantees)
# ============================================================================
print("\n" + "=" * 80)
print("Analysis 3: Learning Curves (Multi-Grant Grantees)")
print("=" * 80)
print()

# Identify grantees with multiple grants
if 'Prior_Grant_Count' in panel.columns:
    multi_grant_grantees = panel[panel['Prior_Grant_Count'] > 0]['Grantee'].unique()
    print(f"Grantees with multiple CDBG-DR grants: {len(multi_grant_grantees)}")

    # For each multi-grant grantee, check if velocity improves over time
    learning_curves = []

    for grantee in multi_grant_grantees:
        grantee_grants = panel[panel['Grantee'] == grantee].copy()

        # Sort by disaster year (proxy for grant sequence)
        if 'Disaster_Year' in grantee_grants.columns:
            grantee_grants = grantee_grants.sort_values('Disaster_Year')

        # Extract velocity for each grant
        for idx, (i, row) in enumerate(grantee_grants.iterrows()):
            learning_curves.append({
                'Grantee': grantee,
                'Grant_Sequence': idx + 1,
                'Disaster_Type': row['Disaster Type'],
                'Velocity': row['Expenditure_Velocity_pp'],
                'Duration': row['Duration_Surv'],
                'Completed': row['Event'],
            })

    learning_df = pd.DataFrame(learning_curves)

    if len(learning_df) > 0:
        print(f"Total grant observations from multi-grant grantees: {len(learning_df)}")
        print()

        # Summary by grant sequence
        print("Velocity by Grant Sequence:")
        sequence_summary = learning_df.groupby('Grant_Sequence').agg({
            'Velocity': ['count', 'mean', 'std'],
            'Duration': 'mean',
            'Completed': 'mean',
        }).round(4)
        print(sequence_summary)
        print()

        # Test if velocity increases with experience (regression)
        from scipy.stats import pearsonr

        valid_learning = learning_df.dropna(subset=['Grant_Sequence', 'Velocity'])
        if len(valid_learning) >= 10:
            corr, p_value = pearsonr(valid_learning['Grant_Sequence'], valid_learning['Velocity'])
            print(f"Correlation between Grant Sequence and Velocity: r={corr:.4f}, p={p_value:.4f}")

            if p_value < 0.05:
                if corr > 0:
                    print("  ✓ Significant positive correlation: Velocity improves with experience")
                else:
                    print("  ⚠ Significant negative correlation: Velocity declines with experience")
            else:
                print("  ○ No significant correlation: Velocity does not systematically change with experience")

        # Save learning curve data
        learning_output_path = Path("../data_work/diagnostics/learning_curves.csv")
        learning_df.to_csv(learning_output_path, index=False)
        print(f"\n✓ Saved learning curve data: {learning_output_path}")
    else:
        print("⚠ No multi-grant grantees found with valid velocity data")
else:
    print("⚠ Cannot analyze learning curves without Prior_Grant_Count")

# ============================================================================
# Save Results
# ============================================================================
results_df = pd.DataFrame(results)
output_path = Path("../data_work/diagnostics/learning_curves_experience_velocity.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)
print(f"\n✓ Saved experience analysis results: {output_path}")

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print()
print(results_df.to_string(index=False))
print()

# ============================================================================
# Visualization: Kaplan-Meier by Experience Group
# ============================================================================
print("Creating Kaplan-Meier survival curves by experience group...")
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

colors = {'Novice': '#e74c3c', 'Experienced': '#2ecc71'}

for group in ['Novice', 'Experienced']:
    subset = panel[panel['Experience_Group'] == group].copy()
    subset_clean = subset[['Duration_Surv', 'Event']].dropna()

    if len(subset_clean) >= 5 and subset_clean['Event'].sum() >= 3:
        kmf = KaplanMeierFitter()
        kmf.fit(subset_clean['Duration_Surv'], subset_clean['Event'], label=group)

        n = len(subset_clean)
        events = subset_clean['Event'].sum()
        kmf.plot_survival_function(
            ax=ax,
            color=colors[group],
            linewidth=2.5,
            label=f"{group} (N={n}, Events={events})"
        )

ax.set_xlabel('Time to Completion (Quarters)', fontsize=11)
ax.set_ylabel('Survival Probability (Not Yet Completed)', fontsize=11)
ax.set_title('Kaplan-Meier Survival Curves by Experience Level',
             fontsize=13, fontweight='bold', pad=20)
ax.grid(alpha=0.3, linestyle=':')
ax.legend(loc='best', fontsize=10, framealpha=0.9)

km_path = Path("../figures/kaplan_meier_by_experience.png")
km_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(km_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved Kaplan-Meier curves: {km_path}")
plt.close()

# ============================================================================
# Visualization: Forest Plot of Experience-Stratified HRs
# ============================================================================
if len(results_df[results_df['Experience_Group'].isin(['Novice', 'Experienced'])]) >= 2:
    print("Creating forest plot of experience-stratified hazard ratios...")

    plot_df = results_df[results_df['Experience_Group'].isin(['Novice', 'Experienced'])].copy()
    plot_df = plot_df.dropna(subset=['Velocity_HR', 'Velocity_CI_lower', 'Velocity_CI_upper'])

    if len(plot_df) >= 2:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

        y_positions = np.arange(len(plot_df))

        for idx, (i, row) in enumerate(plot_df.iterrows()):
            group = row['Experience_Group']
            hr = row['Velocity_HR']
            ci_lower = row['Velocity_CI_lower']
            ci_upper = row['Velocity_CI_upper']
            p_value = row['Velocity_p']
            n = int(row['N'])
            n_events = int(row['N_Events'])

            color = colors[group]

            # Point estimate
            ax.plot(hr, y_positions[idx], 'o', markersize=12, color=color, zorder=3)

            # CI
            ax.plot([ci_lower, ci_upper], [y_positions[idx], y_positions[idx]],
                    '-', linewidth=3, color=color, zorder=2)

            # Significance
            sig_marker = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))

            # Label
            label_text = f"HR={hr:.2f} {sig_marker}\\n(N={n}, Events={n_events})"
            ax.text(ci_upper * 1.2, y_positions[idx], label_text,
                    va='center', ha='left', fontsize=10)

        # Null line
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

        # X-axis (log scale)
        ax.set_xscale('log')
        ax.set_xlabel('Hazard Ratio (95% CI)\\nper 1 pp/quarter increase in expenditure velocity', fontsize=11)

        # Y-axis
        ax.set_yticks(y_positions)
        ax.set_yticklabels(plot_df['Experience_Group'], fontsize=11)
        ax.set_ylim(-0.5, len(plot_df) - 0.5)

        # Title
        ax.set_title('Velocity Effects by Experience Level',
                     fontsize=13, fontweight='bold', pad=20)

        # Grid
        ax.grid(axis='x', alpha=0.3, linestyle=':', zorder=0)

        # Notes
        note_text = "*** p<0.001, ** p<0.01, * p<0.05\\nHR>1: Higher velocity → faster completion\\nVertical dashed line: Null effect (HR=1)"
        ax.text(0.02, 0.02, note_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        forest_path = Path("../figures/velocity_effect_by_experience.png")
        plt.savefig(forest_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved forest plot: {forest_path}")
        plt.close()

print("\n" + "=" * 80)
print("Learning Curves & Experience Analysis Complete")
print("=" * 80)
