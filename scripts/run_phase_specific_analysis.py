#!/usr/bin/env python3
"""
Phase 2 Week 5: Phase-Specific Velocity Analysis

Tests if velocity effects differ across program phases (early/mid/late).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from stages._io_utils import safe_read_parquet

# Load panel
panel_path = Path("../data_work/panel_features_std.parquet")
panel = safe_read_parquet(panel_path)

print("=" * 80)
print("Phase 2 Week 5: Phase-Specific Velocity Analysis")
print("=" * 80)
print()

# Prepare survival data
panel['Duration_Surv'] = panel['Duration'].fillna(panel['N_Quarters'])
panel['Event'] = (panel['Duration'].notna() & (panel['Duration'] > 0)).astype(int)

# Scale phase velocities by 100 for pp/quarter interpretation
for phase in ['Early', 'Mid', 'Late']:
    col = f'Velocity_{phase}'
    if col in panel.columns:
        panel[f'{col}_scaled'] = panel[col] * 100

panel['Velocity_Acceleration_scaled'] = panel['Velocity_Acceleration'] * 100 if 'Velocity_Acceleration' in panel.columns else np.nan

# Check phase velocity availability
print("Phase velocity statistics:")
for phase in ['Early', 'Mid', 'Late', 'Acceleration']:
    col = f'Velocity_{phase}_scaled'
    if col in panel.columns:
        print(f"  {phase:15s}: N={panel[col].notna().sum():3d}, Mean={panel[col].mean():.4f}, Std={panel[col].std():.4f}")
print()

results = []

# Model 1: Overall velocity (baseline)
print("Model 1: Overall Velocity (baseline)")
print("-" * 80)

subset1 = panel[['Duration_Surv', 'Event', 'Expenditure_Velocity_pp', 'Government_Type_State']].copy()
subset1['Velocity_scaled'] = subset1['Expenditure_Velocity_pp'] * 100
subset1 = subset1.drop(columns=['Expenditure_Velocity_pp']).dropna()

cph1 = CoxPHFitter(penalizer=0.01)
cph1.fit(subset1, duration_col='Duration_Surv', event_col='Event')

results.append({
    'Model': 'Overall_Velocity',
    'N': len(subset1),
    'N_Events': int(subset1['Event'].sum()),
    'Overall_Velocity_HR': np.exp(cph1.params_['Velocity_scaled']),
    'Overall_Velocity_p': cph1.summary.loc['Velocity_scaled', 'p'],
    'Early_Velocity_HR': np.nan,
    'Early_Velocity_p': np.nan,
    'Mid_Velocity_HR': np.nan,
    'Mid_Velocity_p': np.nan,
    'Late_Velocity_HR': np.nan,
    'Late_Velocity_p': np.nan,
    'Acceleration_HR': np.nan,
    'Acceleration_p': np.nan,
})

print(cph1.summary[['coef', 'exp(coef)', 'p']])
print()

# Model 2: Early velocity only
print("Model 2: Early Velocity Only")
print("-" * 80)

subset2 = panel[['Duration_Surv', 'Event', 'Velocity_Early_scaled', 'Government_Type_State']].dropna()

if len(subset2) >= 20 and subset2['Event'].sum() >= 5:
    cph2 = CoxPHFitter(penalizer=0.01)
    cph2.fit(subset2, duration_col='Duration_Surv', event_col='Event')

    results.append({
        'Model': 'Early_Velocity_Only',
        'N': len(subset2),
        'N_Events': int(subset2['Event'].sum()),
        'Overall_Velocity_HR': np.nan,
        'Overall_Velocity_p': np.nan,
        'Early_Velocity_HR': np.exp(cph2.params_['Velocity_Early_scaled']),
        'Early_Velocity_p': cph2.summary.loc['Velocity_Early_scaled', 'p'],
        'Mid_Velocity_HR': np.nan,
        'Mid_Velocity_p': np.nan,
        'Late_Velocity_HR': np.nan,
        'Late_Velocity_p': np.nan,
        'Acceleration_HR': np.nan,
        'Acceleration_p': np.nan,
    })

    print(cph2.summary[['coef', 'exp(coef)', 'p']])
else:
    print(f"  ⚠ Insufficient sample (N={len(subset2)}, Events={subset2['Event'].sum()})")

print()

# Model 3: All three phases
print("Model 3: Early + Mid + Late Velocity")
print("-" * 80)

subset3 = panel[['Duration_Surv', 'Event', 'Velocity_Early_scaled', 'Velocity_Mid_scaled',
                  'Velocity_Late_scaled', 'Government_Type_State']].dropna()

if len(subset3) >= 20 and subset3['Event'].sum() >= 5:
    cph3 = CoxPHFitter(penalizer=0.01)
    cph3.fit(subset3, duration_col='Duration_Surv', event_col='Event')

    results.append({
        'Model': 'Three_Phase',
        'N': len(subset3),
        'N_Events': int(subset3['Event'].sum()),
        'Overall_Velocity_HR': np.nan,
        'Overall_Velocity_p': np.nan,
        'Early_Velocity_HR': np.exp(cph3.params_['Velocity_Early_scaled']),
        'Early_Velocity_p': cph3.summary.loc['Velocity_Early_scaled', 'p'],
        'Mid_Velocity_HR': np.exp(cph3.params_['Velocity_Mid_scaled']),
        'Mid_Velocity_p': cph3.summary.loc['Velocity_Mid_scaled', 'p'],
        'Late_Velocity_HR': np.exp(cph3.params_['Velocity_Late_scaled']),
        'Late_Velocity_p': cph3.summary.loc['Velocity_Late_scaled', 'p'],
        'Acceleration_HR': np.nan,
        'Acceleration_p': np.nan,
    })

    print(cph3.summary[['coef', 'exp(coef)', 'p']])
else:
    print(f"  ⚠ Insufficient sample (N={len(subset3)}, Events={subset3['Event'].sum()})")

print()

# Model 4: Velocity Acceleration
print("Model 4: Velocity Acceleration (Late - Early)")
print("-" * 80)

subset4 = panel[['Duration_Surv', 'Event', 'Velocity_Acceleration_scaled', 'Government_Type_State']].dropna()

if len(subset4) >= 20 and subset4['Event'].sum() >= 5:
    cph4 = CoxPHFitter(penalizer=0.01)
    cph4.fit(subset4, duration_col='Duration_Surv', event_col='Event')

    results.append({
        'Model': 'Acceleration',
        'N': len(subset4),
        'N_Events': int(subset4['Event'].sum()),
        'Overall_Velocity_HR': np.nan,
        'Overall_Velocity_p': np.nan,
        'Early_Velocity_HR': np.nan,
        'Early_Velocity_p': np.nan,
        'Mid_Velocity_HR': np.nan,
        'Mid_Velocity_p': np.nan,
        'Late_Velocity_HR': np.nan,
        'Late_Velocity_p': np.nan,
        'Acceleration_HR': np.exp(cph4.params_['Velocity_Acceleration_scaled']),
        'Acceleration_p': cph4.summary.loc['Velocity_Acceleration_scaled', 'p'],
    })

    print(cph4.summary[['coef', 'exp(coef)', 'p']])
else:
    print(f"  ⚠ Insufficient sample (N={len(subset4)}, Events={subset4['Event'].sum()})")

print()

# Save results
results_df = pd.DataFrame(results)
output_path = Path("../data_work/diagnostics/phase_specific_velocity.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)
print(f"\n✓ Saved results: {output_path}")

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print()
print(results_df[['Model', 'N', 'N_Events', 'Early_Velocity_HR', 'Early_Velocity_p',
                   'Mid_Velocity_HR', 'Mid_Velocity_p', 'Late_Velocity_HR', 'Late_Velocity_p']].to_string(index=False))
print()

print("\n" + "=" * 80)
print("Phase-Specific Velocity Analysis Complete")
print("=" * 80)
