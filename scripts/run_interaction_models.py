#!/usr/bin/env python3
"""
Phase 2 Week 4: Interaction Models

Tests Velocity × Stage1_Efficiency interaction to determine if disbursement
capacity moderates velocity effects.
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
print("Phase 2 Week 4: Interaction Models (Velocity × Stage1_Efficiency)")
print("=" * 80)
print()

# Prepare survival data
panel['Duration_Surv'] = panel['Duration'].fillna(panel['N_Quarters'])
panel['Velocity_scaled'] = panel['Expenditure_Velocity_pp'] * 100
panel['Event'] = (panel['Duration'].notna() & (panel['Duration'] > 0)).astype(int)

# Create interaction term
panel['Velocity_x_Stage1'] = panel['Velocity_scaled'] * panel['Stage1_Efficiency']

# Subset to complete cases
subset = panel[
    ['Duration_Surv', 'Event', 'Velocity_scaled', 'Stage1_Efficiency',
     'Velocity_x_Stage1', 'Government_Type_State']
].dropna()

print(f"Sample: N={len(subset)}, Events={subset['Event'].sum()}")
print()

results = []

# Model 1: Velocity only (baseline)
print("Model 1: Velocity only (baseline)")
print("-" * 80)

cph1 = CoxPHFitter(penalizer=0.01)
cph1.fit(
    subset[['Duration_Surv', 'Event', 'Velocity_scaled', 'Government_Type_State']],
    duration_col='Duration_Surv',
    event_col='Event'
)

results.append({
    'Model': 'Velocity_Only',
    'N': len(subset),
    'N_Events': int(subset['Event'].sum()),
    'Velocity_HR': np.exp(cph1.params_['Velocity_scaled']),
    'Velocity_p': cph1.summary.loc['Velocity_scaled', 'p'],
    'Stage1_HR': np.nan,
    'Stage1_p': np.nan,
    'Interaction_HR': np.nan,
    'Interaction_p': np.nan,
})

print(cph1.summary[['coef', 'exp(coef)', 'p']])
print()

# Model 2: Velocity + Stage1 (additive)
print("Model 2: Velocity + Stage1_Efficiency (additive)")
print("-" * 80)

cph2 = CoxPHFitter(penalizer=0.01)
cph2.fit(
    subset[['Duration_Surv', 'Event', 'Velocity_scaled', 'Stage1_Efficiency', 'Government_Type_State']],
    duration_col='Duration_Surv',
    event_col='Event'
)

results.append({
    'Model': 'Additive',
    'N': len(subset),
    'N_Events': int(subset['Event'].sum()),
    'Velocity_HR': np.exp(cph2.params_['Velocity_scaled']),
    'Velocity_p': cph2.summary.loc['Velocity_scaled', 'p'],
    'Stage1_HR': np.exp(cph2.params_['Stage1_Efficiency']),
    'Stage1_p': cph2.summary.loc['Stage1_Efficiency', 'p'],
    'Interaction_HR': np.nan,
    'Interaction_p': np.nan,
})

print(cph2.summary[['coef', 'exp(coef)', 'p']])
print()

# Model 3: Velocity × Stage1 (interaction)
print("Model 3: Velocity × Stage1_Efficiency (interaction)")
print("-" * 80)

cph3 = CoxPHFitter(penalizer=0.01)
cph3.fit(
    subset[['Duration_Surv', 'Event', 'Velocity_scaled', 'Stage1_Efficiency',
            'Velocity_x_Stage1', 'Government_Type_State']],
    duration_col='Duration_Surv',
    event_col='Event'
)

results.append({
    'Model': 'Interaction',
    'N': len(subset),
    'N_Events': int(subset['Event'].sum()),
    'Velocity_HR': np.exp(cph3.params_['Velocity_scaled']),
    'Velocity_p': cph3.summary.loc['Velocity_scaled', 'p'],
    'Stage1_HR': np.exp(cph3.params_['Stage1_Efficiency']),
    'Stage1_p': cph3.summary.loc['Stage1_Efficiency', 'p'],
    'Interaction_HR': np.exp(cph3.params_['Velocity_x_Stage1']),
    'Interaction_p': cph3.summary.loc['Velocity_x_Stage1', 'p'],
})

print(cph3.summary[['coef', 'exp(coef)', 'p']])
print()

# Compare models with likelihood ratio test
from lifelines.statistics import multivariate_logrank_test

print("=" * 80)
print("Model Comparison")
print("=" * 80)
print()

print("AIC Comparison (partial AIC for Cox PH):")
print(f"  Model 1 (Velocity only):     AIC = {cph1.AIC_partial_:.2f}")
print(f"  Model 2 (Additive):          AIC = {cph2.AIC_partial_:.2f}")
print(f"  Model 3 (Interaction):       AIC = {cph3.AIC_partial_:.2f}")
print()

# Interpretation of interaction
interaction_coef = cph3.params_['Velocity_x_Stage1']
interaction_p = cph3.summary.loc['Velocity_x_Stage1', 'p']

print("Interaction Interpretation:")
if interaction_p < 0.05:
    if interaction_coef > 0:
        print("  ✓ SIGNIFICANT POSITIVE interaction (p<0.05)")
        print("  → Velocity effects INCREASE with higher Stage1_Efficiency")
    else:
        print("  ✓ SIGNIFICANT NEGATIVE interaction (p<0.05)")
        print("  → Velocity effects DECREASE with higher Stage1_Efficiency")
else:
    print("  ✗ Interaction NOT significant (p≥0.05)")
    print("  → Velocity effects do not significantly vary by Stage1_Efficiency")

print()

# Save results
results_df = pd.DataFrame(results)
output_path = Path("../data_work/diagnostics/velocity_stage1_interaction.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)
print(f"\n✓ Saved results: {output_path}")

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print()
print(results_df.to_string(index=False))
print()

print("\n" + "=" * 80)
print("Interaction Analysis Complete")
print("=" * 80)
