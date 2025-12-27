#!/usr/bin/env python3
"""
Phase 3 Week 9: Disaster Context Analysis

Tests if velocity effects vary by disaster characteristics:
- Disaster type (Hurricane, Flood, Fire, Other)
- Disaster timing (Pre-2010, 2010-2020, Post-2020)
- Disaster magnitude (based on obligated dollars)
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
print("Phase 3 Week 9: Disaster Context Analysis")
print("=" * 80)
print()

def classify_disaster_type_only(disaster: str) -> dict:
    """
    Classify disaster type (Hurricane, Flood, Fire, Other).

    Parameters
    ----------
    disaster : str
        Disaster Type string

    Returns
    -------
    dict
        Dictionary with disaster type classifications
    """
    disaster_lower = disaster.lower()

    # Type classification
    is_hurricane = any(x in disaster_lower for x in [
        'katrina', 'sandy', 'harvey', 'irma', 'maria', 'ian',
        'hurricane', 'rita', 'wilma', 'gustav', 'ike', 'matthew',
        'florence', 'michael', 'dorian', 'sally', 'laura', 'delta',
        'zeta', 'eta', 'iota', 'joaquin', 'patricia'
    ])
    is_flood = 'flood' in disaster_lower and not is_hurricane
    is_fire = 'fire' in disaster_lower or 'wildfire' in disaster_lower

    return {
        'Disaster_Type_Hurricane': is_hurricane,
        'Disaster_Type_Flood': is_flood,
        'Disaster_Type_Fire': is_fire,
        'Disaster_Type_Other': not (is_hurricane or is_flood or is_fire),
    }


# Load panel
panel_path = Path("../data_work/panel_features_std.parquet")
print(f"Loading panel: {panel_path}")
panel = safe_read_parquet(panel_path)
print(f"  Loaded {len(panel)} grantee-disaster pairs")
print()

# Prepare survival data
panel['Duration_Surv'] = panel['Duration'].fillna(panel['N_Quarters'])
panel['Event'] = (panel['Duration'].notna() & (panel['Duration'] > 0)).astype(int)
panel['Velocity_scaled'] = panel['Expenditure_Velocity_pp'] * 100

# Add disaster type characteristics
print("Classifying disaster types...")
disaster_type_features = panel['Disaster Type'].apply(classify_disaster_type_only).apply(pd.Series)
panel = pd.concat([panel, disaster_type_features], axis=1)

# Add disaster era based on existing Disaster_Year column
if 'Disaster_Year' in panel.columns:
    panel['Disaster_Era'] = panel['Disaster_Year'].apply(
        lambda y: 'Pre-2010' if y < 2010 else ('2010-2020' if y <= 2020 else 'Post-2020')
    )

print("Disaster type distribution:")
for dtype in ['Hurricane', 'Flood', 'Fire', 'Other']:
    col = f'Disaster_Type_{dtype}'
    count = panel[col].sum()
    print(f"  {dtype}: {count}")
print()

print("Disaster era distribution:")
print(panel['Disaster_Era'].value_counts())
print()

# Add magnitude classification (based on total obligated)
if 'Total_Obligated' in panel.columns:
    # Major disaster: >$1B obligated
    panel['Disaster_Magnitude'] = panel['Total_Obligated'].apply(
        lambda x: 'Major' if x > 1e9 else 'Moderate'
    )
    print("Disaster magnitude distribution:")
    print(panel['Disaster_Magnitude'].value_counts())
    print()

# ============================================================================
# Analysis 1: Stratified Cox PH by Disaster Type
# ============================================================================
print("=" * 80)
print("Analysis 1: Stratified Cox PH by Disaster Type")
print("=" * 80)
print()

results = []

for dtype in ['Hurricane', 'Flood', 'Fire', 'Other']:
    col = f'Disaster_Type_{dtype}'
    subset = panel[panel[col] == True].copy()

    print(f"\n{dtype} Disasters")
    print("-" * 80)

    subset_clean = subset[['Duration_Surv', 'Event', 'Velocity_scaled', 'Government_Type_State']].dropna()

    n_events = subset_clean['Event'].sum()
    print(f"  Sample: N={len(subset_clean)}, Events={n_events}")

    if n_events < 5:
        print(f"  ⚠ Too few events ({n_events}), skipping Cox PH")
        results.append({
            'Disaster_Context': dtype,
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
            'Disaster_Context': dtype,
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
            'Disaster_Context': dtype,
            'N': len(subset_clean),
            'N_Events': int(n_events),
            'Velocity_HR': np.nan,
            'Velocity_p': np.nan,
            'Velocity_CI_lower': np.nan,
            'Velocity_CI_upper': np.nan,
        })

# ============================================================================
# Analysis 2: Stratified Cox PH by Disaster Era
# ============================================================================
print("\n" + "=" * 80)
print("Analysis 2: Stratified Cox PH by Disaster Era")
print("=" * 80)
print()

for era in ['Pre-2010', '2010-2020', 'Post-2020']:
    subset = panel[panel['Disaster_Era'] == era].copy()

    print(f"\n{era} Disasters")
    print("-" * 80)

    subset_clean = subset[['Duration_Surv', 'Event', 'Velocity_scaled', 'Government_Type_State']].dropna()

    n_events = subset_clean['Event'].sum()
    print(f"  Sample: N={len(subset_clean)}, Events={n_events}")

    if n_events < 5:
        print(f"  ⚠ Too few events ({n_events}), skipping Cox PH")
        results.append({
            'Disaster_Context': era,
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
            'Disaster_Context': era,
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
            'Disaster_Context': era,
            'N': len(subset_clean),
            'N_Events': int(n_events),
            'Velocity_HR': np.nan,
            'Velocity_p': np.nan,
            'Velocity_CI_lower': np.nan,
            'Velocity_CI_upper': np.nan,
        })

# ============================================================================
# Analysis 3: Stratified Cox PH by Disaster Magnitude
# ============================================================================
if 'Disaster_Magnitude' in panel.columns:
    print("\n" + "=" * 80)
    print("Analysis 3: Stratified Cox PH by Disaster Magnitude")
    print("=" * 80)
    print()

    for magnitude in ['Major', 'Moderate']:
        subset = panel[panel['Disaster_Magnitude'] == magnitude].copy()

        print(f"\n{magnitude} Disasters (>$1B obligated)" if magnitude == 'Major' else f"\n{magnitude} Disasters (<$1B obligated)")
        print("-" * 80)

        subset_clean = subset[['Duration_Surv', 'Event', 'Velocity_scaled', 'Government_Type_State']].dropna()

        n_events = subset_clean['Event'].sum()
        print(f"  Sample: N={len(subset_clean)}, Events={n_events}")

        if n_events < 5:
            print(f"  ⚠ Too few events ({n_events}), skipping Cox PH")
            results.append({
                'Disaster_Context': f'{magnitude}_Disaster',
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
                'Disaster_Context': f'{magnitude}_Disaster',
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
                'Disaster_Context': f'{magnitude}_Disaster',
                'N': len(subset_clean),
                'N_Events': int(n_events),
                'Velocity_HR': np.nan,
                'Velocity_p': np.nan,
                'Velocity_CI_lower': np.nan,
                'Velocity_CI_upper': np.nan,
            })

# ============================================================================
# Save Results
# ============================================================================
results_df = pd.DataFrame(results)
output_path = Path("../data_work/diagnostics/disaster_context_heterogeneity.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)
print(f"\n✓ Saved disaster context results: {output_path}")

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print()
print(results_df.to_string(index=False))
print()

# ============================================================================
# Visualization: Forest Plot by Disaster Context
# ============================================================================
if len(results_df.dropna(subset=['Velocity_HR'])) >= 2:
    print("Creating forest plot of velocity effects by disaster context...")

    plot_df = results_df.dropna(subset=['Velocity_HR', 'Velocity_CI_lower', 'Velocity_CI_upper'])

    if len(plot_df) >= 2:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        y_positions = np.arange(len(plot_df))

        colors = {
            'Hurricane': '#e74c3c',
            'Flood': '#3498db',
            'Fire': '#e67e22',
            'Other': '#95a5a6',
            'Pre-2010': '#9b59b6',
            '2010-2020': '#1abc9c',
            'Post-2020': '#f39c12',
            'Major_Disaster': '#2ecc71',
            'Moderate_Disaster': '#34495e',
        }

        for idx, (i, row) in enumerate(plot_df.iterrows()):
            context = row['Disaster_Context']
            hr = row['Velocity_HR']
            ci_lower = row['Velocity_CI_lower']
            ci_upper = row['Velocity_CI_upper']
            p_value = row['Velocity_p']
            n = int(row['N'])
            n_events = int(row['N_Events'])

            color = colors.get(context, '#95a5a6')

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
                    va='center', ha='left', fontsize=9)

        # Null line
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

        # X-axis (log scale)
        ax.set_xscale('log')
        ax.set_xlabel('Hazard Ratio (95% CI)\nper 1 pp/quarter increase in expenditure velocity', fontsize=11)

        # Y-axis
        ax.set_yticks(y_positions)
        ax.set_yticklabels([c.replace('_', ' ') for c in plot_df['Disaster_Context']], fontsize=10)
        ax.set_ylim(-0.5, len(plot_df) - 0.5)

        # Title
        ax.set_title('Velocity Effects by Disaster Context',
                     fontsize=13, fontweight='bold', pad=20)

        # Grid
        ax.grid(axis='x', alpha=0.3, linestyle=':', zorder=0)

        # Notes
        note_text = "*** p<0.001, ** p<0.01, * p<0.05\nHR>1: Higher velocity → faster completion\nVertical dashed line: Null effect (HR=1)"
        ax.text(0.02, 0.02, note_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        forest_path = Path("../figures/velocity_effect_by_disaster_context.png")
        forest_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(forest_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved forest plot: {forest_path}")
        plt.close()

print("\n" + "=" * 80)
print("Disaster Context Analysis Complete")
print("=" * 80)
